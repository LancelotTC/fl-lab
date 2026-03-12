"""True vertical federated learning for aligned tabular data."""

from __future__ import annotations

import sys
import uuid
from collections.abc import Collection
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

sys.path.append(".")
sys.path.append("..")

from .. import DDict, FlukeENV, ObserverSubject  # NOQA
from ..comm import Channel, Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import DataSplitter, FastDataLoader  # NOQA
from ..utils import get_loss, get_model, safe_train_test_split  # NOQA

__all__ = ["VerticalFL"]


@dataclass
class _VerticalParty:
    index: int
    feature_idx: torch.Tensor
    train_X: torch.Tensor
    test_X: torch.Tensor | None
    encoder: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


class _VerticalInferenceModel(nn.Module):
    def __init__(
        self,
        parties: Sequence[_VerticalParty],
        head: nn.Module,
        active_parties: set[int] | None = None,
    ):
        super().__init__()
        self.parties = nn.ModuleList([party.encoder for party in parties])
        self._feature_idx = [party.feature_idx.clone().long() for party in parties]
        self.head = head
        self.active_parties = active_parties

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for idx, encoder in enumerate(self.parties):
            local_x = x[:, self._feature_idx[idx].to(x.device)]
            z = encoder(local_x)
            if self.active_parties is not None and idx not in self.active_parties:
                z = torch.zeros_like(z)
            embeddings.append(z)
        return self.head(torch.cat(embeddings, dim=1))


class VerticalFL(ObserverSubject):
    """True VFL coordinator for aligned tabular datasets.

    Each party owns a subset of columns for the same aligned samples. Parties train
    local encoders, the coordinator trains the server head, and embeddings/embedding
    gradients are exchanged through the standard channel for communication accounting.
    """

    def __init__(
        self,
        n_clients: int,
        data_splitter: DataSplitter,
        hyper_params: DDict | dict[str, Any],
        **kwargs,
    ):
        super().__init__()
        if isinstance(hyper_params, dict):
            hyper_params = DDict(hyper_params)

        self._id = str(uuid.uuid4().hex)
        self.n_clients = int(n_clients)
        self.hyper_params = hyper_params
        self.data_splitter = data_splitter
        self.channel = Channel()
        self.device = FlukeENV().get_device()
        FlukeENV().open_cache(self._id)

        if self.n_clients < 2:
            raise ValueError("VerticalFL requires at least two parties.")
        if not str(self.data_splitter.distribution).startswith("vertical"):
            raise ValueError("VerticalFL requires a vertical data distribution.")

        self._loss_fn = get_loss(hyper_params.client.loss)
        if hasattr(self._loss_fn, "to"):
            self._loss_fn = self._loss_fn.to(self.device)

        feature_splits = self._resolve_feature_splits()
        (
            self.train_full_X,
            self.train_full_y,
            self.client_test_full_X,
            self.client_test_full_y,
            self.client_test_metadata,
            self.server_test,
        ) = self._prepare_vertical_tensors()

        model_factory = (
            hyper_params.model
            if isinstance(hyper_params.model, nn.Module)
            else get_model(
                mname=hyper_params.model,
                **hyper_params.net_args if "net_args" in hyper_params else {},
            )
        )
        if not hasattr(model_factory, "make_client_encoder") or not hasattr(
            model_factory, "make_server_head"
        ):
            raise ValueError(
                "VerticalFL expects a model factory exposing make_client_encoder() "
                "and make_server_head()."
            )

        self._optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=hyper_params.client.optimizer,
            scheduler_cfg=hyper_params.client.scheduler,
        )

        self.parties: list[_VerticalParty] = []
        for index, split in enumerate(feature_splits):
            encoder = model_factory.make_client_encoder(input_dim=len(split)).to(self.device)
            optimizer, scheduler = self._optimizer_cfg(encoder)
            self.parties.append(
                _VerticalParty(
                    index=index,
                    feature_idx=torch.tensor(split, dtype=torch.long),
                    train_X=self.train_full_X[:, split].clone(),
                    test_X=(None if self.client_test_full_X is None else self.client_test_full_X[:, split].clone()),
                    encoder=encoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
            )

        self.head = model_factory.make_server_head(n_parties=self.n_clients).to(self.device)
        self.head_optimizer, self.head_scheduler = self._optimizer_cfg(self.head)
        self.model = _VerticalInferenceModel(self.parties, self.head)
        self.rounds = 0

    @property
    def id(self) -> str:
        return self._id

    def set_callbacks(self, callbacks: Any | Collection[Any]) -> None:
        if not isinstance(callbacks, Collection) or isinstance(callbacks, (str, bytes)):
            callbacks = [callbacks]
        self.attach(callbacks)
        self.channel.attach(callbacks)

    def _resolve_feature_splits(self) -> list[list[int]]:
        n_features = int(self.data_splitter.data_container.num_features)
        if "feature_splits" in self.data_splitter.dist_args:
            splits = [list(map(int, split)) for split in self.data_splitter.dist_args.feature_splits]
        else:
            splits = [list(chunk.astype(int)) for chunk in np.array_split(np.arange(n_features), self.n_clients)]

        if len(splits) != self.n_clients:
            raise ValueError(
                f"vertical feature_splits must contain one split per client ({len(splits)} != {self.n_clients})."
            )

        for idx, split in enumerate(splits):
            if not split:
                raise ValueError(f"vertical split for client {idx} is empty.")
            invalid = [v for v in split if v < 0 or v >= n_features]
            if invalid:
                raise ValueError(
                    f"vertical split for client {idx} contains invalid indexes {invalid}; valid range is [0, {n_features - 1}]."
                )
        return splits

    def _prepare_vertical_tensors(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, torch.Tensor],
        FastDataLoader | None,
    ]:
        container = self.data_splitter.data_container
        train_metadata = getattr(container, "train_metadata", {})
        test_metadata = getattr(container, "test_metadata", {})
        if self.data_splitter.server_test and self.data_splitter.keep_test:
            server_X, server_Y = container.test
            server_metadata = test_metadata
            client_X, client_Y = container.train
            client_Xtr, client_Xte, client_Ytr, client_Yte, _, client_test_metadata = (
                DataSplitter._safe_train_test_split_with_metadata(
                    client_X,
                    client_Y,
                    train_metadata,
                    test_size=self.data_splitter.client_split,
                )
            )
        elif not self.data_splitter.keep_test:
            Xtr, ytr = container.train
            Xte, yte = container.test
            X = torch.cat((Xtr, Xte), dim=0)
            Y = torch.cat((ytr, yte), dim=0)
            idx = torch.randperm(X.size(0))
            X, Y = X[idx], Y[idx]
            merged_metadata = DataSplitter._merge_metadata(train_metadata, test_metadata)
            merged_metadata = {key: value[idx] for key, value in merged_metadata.items()}
            if self.data_splitter.server_test:
                (
                    client_X,
                    server_X,
                    client_Y,
                    server_Y,
                    _,
                    server_metadata,
                ) = DataSplitter._safe_train_test_split_with_metadata(
                    X,
                    Y,
                    merged_metadata,
                    test_size=self.data_splitter.server_split,
                )
            else:
                client_X, client_Y = X, Y
                server_X, server_Y = None, None
                server_metadata = {}
            client_Xtr, client_Xte, client_Ytr, client_Yte, _, client_test_metadata = (
                DataSplitter._safe_train_test_split_with_metadata(
                    client_X,
                    client_Y,
                    client_metadata if self.data_splitter.server_test else merged_metadata,
                    test_size=self.data_splitter.client_split,
                )
            )
        else:
            server_X, server_Y = None, None
            server_metadata = {}
            client_Xtr, client_Ytr = container.train
            client_Xte, client_Yte = container.test
            client_test_metadata = test_metadata

        server_te = (
            FastDataLoader(
                server_X,
                server_Y,
                num_labels=container.num_classes,
                batch_size=128,
                shuffle=False,
                percentage=self.data_splitter.sampling_perc,
                metadata=server_metadata,
            )
            if self.data_splitter.server_test and server_X is not None and server_Y is not None
            else None
        )
        return client_Xtr, client_Ytr, client_Xte, client_Yte, client_test_metadata, server_te

    def _iter_aligned_batches(self, batch_size: int, shuffle: bool = True):
        n_samples = int(self.train_full_X.shape[0])
        order = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = order[start : start + batch_size]
            if batch_idx.numel() == 0:
                continue
            yield batch_idx

    def _send_tensor(self, payload: torch.Tensor, msg_type: str, sender: Any, receiver: Any) -> None:
        tensor = payload.detach().cpu()
        self.channel.send(Message(tensor, msg_type=msg_type, sender=sender), receiver)
        self.channel.receive(receiver, sender, msg_type=msg_type)

    def _evaluate_global(self, round_id: int) -> dict[str, float]:
        evaluator = FlukeENV().get_evaluator()
        self.model.active_parties = None
        if self.server_test is not None:
            return evaluator.evaluate(round_id, self.model, self.server_test, loss_fn=None, device=self.device)

        if self.client_test_full_X is None or self.client_test_full_y is None:
            return {}
        full_loader = FastDataLoader(
            self.client_test_full_X,
            self.client_test_full_y,
            num_labels=self.data_splitter.num_classes,
            batch_size=128,
            shuffle=False,
            metadata=self.client_test_metadata,
        )
        return evaluator.evaluate(round_id, self.model, full_loader, loss_fn=None, device=self.device)

    def _evaluate_local_views(self, round_id: int) -> dict[int, dict[str, float]]:
        evaluator = FlukeENV().get_evaluator()
        if self.client_test_full_X is None or self.client_test_full_y is None:
            return {}
        full_loader = FastDataLoader(
            self.client_test_full_X,
            self.client_test_full_y,
            num_labels=self.data_splitter.num_classes,
            batch_size=128,
            shuffle=False,
            metadata=self.client_test_metadata,
        )
        evals: dict[int, dict[str, float]] = {}
        for party in self.parties:
            self.model.active_parties = {party.index}
            metrics = evaluator.evaluate(round_id, self.model, full_loader, loss_fn=None, device=self.device)
            if metrics:
                evals[party.index] = metrics
        self.model.active_parties = None
        return evals

    def run(self, n_rounds: int, eligible_perc: float, finalize: bool = True, **kwargs) -> None:
        if eligible_perc != 1.0:
            raise ValueError("VerticalFL requires eligible_perc=1.0 so all parties stay aligned.")

        batch_size = int(self.hyper_params.client.batch_size)
        local_epochs = int(self.hyper_params.client.local_epochs)

        for round_id in range(1, n_rounds + 1):
            self.notify(event="start_round", round=round_id, global_model=self.model)

            for party in self.parties:
                party.encoder.train()
            self.head.train()

            running_loss = 0.0
            n_steps = 0
            for _ in range(local_epochs):
                for batch_idx in self._iter_aligned_batches(batch_size=batch_size, shuffle=True):
                    y = self.train_full_y[batch_idx].to(self.device)
                    self.head_optimizer.zero_grad()
                    for party in self.parties:
                        party.optimizer.zero_grad()

                    embeddings: list[torch.Tensor] = []
                    for party in self.parties:
                        local_x = party.train_X[batch_idx].to(self.device)
                        z = party.encoder(local_x)
                        z.retain_grad()
                        embeddings.append(z)
                        self._send_tensor(z, "embedding", sender=party.index, receiver="server")

                    logits = self.head(torch.cat(embeddings, dim=1))
                    loss = self._loss_fn(logits, y)
                    loss.backward()

                    for party, z in zip(self.parties, embeddings):
                        if z.grad is not None:
                            self._send_tensor(z.grad, "embedding_grad", sender="server", receiver=party.index)

                    self.head_optimizer.step()
                    for party in self.parties:
                        party.optimizer.step()

                    running_loss += float(loss.item())
                    n_steps += 1

                self.head_scheduler.step()
                for party in self.parties:
                    party.scheduler.step()

            if n_steps > 0:
                self.notify(
                    event="track_item",
                    round=round_id,
                    item="vfl/train_loss",
                    value=float(running_loss / n_steps),
                )

            if FlukeENV().get_eval_cfg().server:
                global_metrics = self._evaluate_global(round_id)
                if global_metrics:
                    self.notify(
                        event="server_evaluation",
                        round=round_id,
                        eval_type="global",
                        evals=global_metrics,
                    )

            if FlukeENV().get_eval_cfg().locals:
                local_metrics = self._evaluate_local_views(round_id)
                if local_metrics:
                    self.notify(
                        event="server_evaluation",
                        round=round_id,
                        eval_type="locals",
                        evals=local_metrics,
                    )

            self.rounds = round_id
            self.notify(event="end_round", round=round_id)

        if finalize:
            self.notify(event="finished", round=self.rounds + 1)
        FlukeENV().close_cache()

    def __str__(self) -> str:
        return (
            f"VerticalFL[{self._id}](model={self.hyper_params.model}, "
            f"n_clients={self.n_clients}, batch_size={self.hyper_params.client.batch_size}, "
            f"local_epochs={self.hyper_params.client.local_epochs})"
        )

    def __repr__(self) -> str:
        return self.__str__()
