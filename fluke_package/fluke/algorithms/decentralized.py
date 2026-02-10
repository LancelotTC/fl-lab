"""Decentralized federated learning algorithms."""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Callable, Collection, Sequence

import numpy as np
import torch

from .. import DDict, FlukeENV, ObserverSubject  # NOQA
from ..client import Client  # NOQA
from ..comm import Channel, ChannelObserver, Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import DataSplitter, FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..utils import ClientObserver, ServerObserver, get_loss, get_model  # NOQA
from ..utils.model import aggregate_models  # NOQA

__all__ = ["DecentralizedFedAvg"]


class DecentralizedFedAvg:
    """Decentralized FedAvg with random peer exchanges at each round."""

    def __init__(
        self,
        n_clients: int,
        data_splitter: DataSplitter,
        hyper_params: DDict | dict[str, Any],
        **kwargs,
    ):
        if isinstance(hyper_params, dict):
            hyper_params = DDict(hyper_params)

        self._id = str(uuid.uuid4().hex)
        FlukeENV().open_cache(self._id)

        self.hyper_params = hyper_params
        self.n_clients = n_clients
        self.rounds = 0

        (clients_tr_data, clients_te_data), shared_test = data_splitter.assign(
            n_clients, hyper_params.client.batch_size
        )
        self.shared_test: FastDataLoader | None = shared_test
        self.clients = self._init_clients(clients_tr_data, clients_te_data, hyper_params.client)

        base_model = (
            get_model(
                mname=hyper_params.model,
                **hyper_params.net_args if "net_args" in hyper_params else {},
            )
            if isinstance(hyper_params.model, str)
            else hyper_params.model
        )
        for client in self.clients:
            client.model = deepcopy(base_model)

        self._channel = Channel()
        for client in self.clients:
            client.set_channel(self._channel)

        self._coordinator = ObserverSubject()

    @property
    def id(self) -> str:
        return str(self._id)

    def set_callbacks(self, callbacks: Callable | Collection[Callable]) -> None:
        if not isinstance(callbacks, Collection):
            callbacks = [callbacks]
        self._coordinator.attach([c for c in callbacks if isinstance(c, ServerObserver)])
        self._channel.attach([c for c in callbacks if isinstance(c, ChannelObserver)])
        for client in self.clients:
            client.attach([c for c in callbacks if isinstance(c, ClientObserver)])

    def _init_clients(
        self,
        clients_tr_data: list[FastDataLoader],
        clients_te_data: list[FastDataLoader],
        config: DDict,
    ) -> Sequence[Client]:
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=config.optimizer, scheduler_cfg=config.scheduler
        )
        loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss()
        return [
            Client(
                index=i,
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=deepcopy(loss),
                **config.exclude("optimizer", "loss", "batch_size", "scheduler"),
            )
            for i in range(self.n_clients)
        ]

    def _get_eligible_clients(self, eligible_perc: float) -> Sequence[Client]:
        if eligible_perc == 1:
            return self.clients
        n = max(1, int(self.n_clients * eligible_perc))
        return np.random.choice(self.clients, n, replace=False)

    def _local_update(self, client: Client, current_round: int) -> None:
        client._last_round = current_round
        client._load_from_cache()

        fluke_env = FlukeENV()
        if fluke_env.get_eval_cfg().pre_fit:
            metrics = client.evaluate(fluke_env.get_evaluator(), client.test_set)
            if metrics:
                client.notify(
                    event="client_evaluation",
                    round=current_round,
                    client_id=client.index,
                    phase="pre-fit",
                    evals=metrics,
                )

        client.notify("start_fit", round=current_round, client_id=client.index, model=client.model)
        loss = client.fit()
        client.notify(
            "end_fit",
            round=current_round,
            client_id=client.index,
            model=client.model,
            loss=loss,
        )

        if fluke_env.get_eval_cfg().post_fit:
            metrics = client.evaluate(fluke_env.get_evaluator(), client.test_set)
            if metrics:
                client.notify(
                    event="client_evaluation",
                    round=current_round,
                    client_id=client.index,
                    phase="post-fit",
                    evals=metrics,
                )

        client._save_to_cache()

    def _exchange_and_aggregate(self, eligible: Sequence[Client]) -> None:
        eligible_ids = [client.index for client in eligible]
        id_to_client = {client.index: client for client in eligible}
        steps = (
            self.hyper_params.server.consensus_steps
            if "consensus_steps" in self.hyper_params.server
            else 1
        )

        for _ in range(steps):
            neighbors = self._sample_neighbors(eligible_ids)

            for sender_id in eligible_ids:
                sender = id_to_client[sender_id]
                msg = Message(sender.model, "model", sender.index, inmemory=True)
                for receiver_id in neighbors[sender_id]:
                    self._channel.send(msg, receiver_id)

            for receiver_id in eligible_ids:
                receiver = id_to_client[receiver_id]
                sender_ids = [sid for sid, nbs in neighbors.items() if receiver_id in nbs]
                if not sender_ids:
                    continue
                received_models = []
                received_clients = []
                for sender_id in sender_ids:
                    payload = self._channel.receive(receiver_id, sender_id, "model").payload
                    received_models.append(payload)
                    received_clients.append(id_to_client[sender_id])
                self._aggregate_local_multi(receiver, received_models, received_clients)

    def _sample_neighbors(self, eligible_ids: list[int]) -> dict[int, list[int]]:
        n = len(eligible_ids)
        max_neighbors = max(1, n - 1)
        k = self.hyper_params.server.neighbors if "neighbors" in self.hyper_params.server else 3
        k = max(1, min(k, max_neighbors))

        neighbors = {}
        for sender_id in eligible_ids:
            candidates = [cid for cid in eligible_ids if cid != sender_id]
            if k >= len(candidates):
                chosen = candidates
            else:
                chosen = list(np.random.choice(candidates, k, replace=False))
            neighbors[sender_id] = chosen
        return neighbors

    def _aggregate_local_multi(
        self,
        client: Client,
        peer_models: list[torch.nn.Module],
        peers: list[Client],
    ) -> None:
        models = [client.model] + peer_models
        if "weighted" in self.hyper_params.server.keys() and self.hyper_params.server.weighted:
            weights = [client.n_examples] + [peer.n_examples for peer in peers]
            total = sum(weights)
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(models)] * len(models)
        eta = self.hyper_params.server.lr if "lr" in self.hyper_params.server else 1.0
        aggregate_models(client.model, models, weights, eta, inplace=True)

    def _compute_evaluation(self, round: int, eligible: Sequence[Client]) -> None:
        evaluator: Evaluator = FlukeENV().get_evaluator()

        if FlukeENV().get_eval_cfg().locals and self.shared_test is not None:
            client_evals = {
                client.index: client.evaluate(evaluator, self.shared_test) for client in eligible
            }
            self._coordinator.notify(
                event="server_evaluation", round=round + 1, eval_type="locals", evals=client_evals
            )

        if FlukeENV().get_eval_cfg().server and self.shared_test is not None:
            weights = []
            if "weighted" in self.hyper_params.server.keys() and self.hyper_params.server.weighted:
                total = sum(client.n_examples for client in self.clients)
                weights = [client.n_examples / total for client in self.clients]
            else:
                weights = [1.0 / len(self.clients)] * len(self.clients)

            global_model = deepcopy(self.clients[0].model)
            aggregate_models(global_model, [c.model for c in self.clients], weights, 1.0, True)
            evals = evaluator.evaluate(round + 1, global_model, self.shared_test, loss_fn=None)
            self._coordinator.notify(
                event="server_evaluation", round=round + 1, eval_type="global", evals=evals
            )

    def run(self, n_rounds: int, eligible_perc: float, finalize: bool = True, **kwargs) -> None:
        with FlukeENV().get_live_renderer():
            progress_fl = FlukeENV().get_progress_bar("FL")
            progress_client = FlukeENV().get_progress_bar("clients")
            client_x_round = int(self.n_clients * eligible_perc)
            if client_x_round == 0:
                client_x_round = 1
            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds * client_x_round)
            task_local = progress_client.add_task("[green]Local Training", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            for rnd in range(self.rounds, total_rounds):
                self._coordinator.notify(event="start_round", round=rnd + 1, global_model=None)
                eligible = self._get_eligible_clients(eligible_perc)
                self._coordinator.notify(event="selected_clients", round=rnd + 1, clients=eligible)

                for c, client in enumerate(eligible):
                    self._local_update(client, rnd + 1)
                    progress_client.update(task_id=task_local, completed=c + 1)
                    progress_fl.update(task_id=task_rounds, advance=1)

                self._exchange_and_aggregate(eligible)
                self._compute_evaluation(rnd, eligible)
                self._coordinator.notify(event="end_round", round=rnd + 1)
                self.rounds += 1

            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

        if finalize:
            pass
        self._coordinator.notify(event="finished", round=self.rounds + 1)
