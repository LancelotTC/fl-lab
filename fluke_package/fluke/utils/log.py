"""This submodule provides logging utilities."""

import csv
import logging
import os
import sys
from typing import Any, Collection, Literal, Union

import json
import clearml
import wandb
from psutil import Process
from rich import print as rich_print
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import Pretty
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
sys.path.append("..")

from .. import DDict  # NOQA
from ..comm import ChannelObserver, Message  # NOQA
from ..evaluation import PerformanceTracker  # NOQA
from ..utils import bytes2human, get_class_from_qualified_name  # NOQA
from . import ClientObserver, ServerObserver, get_class_from_str  # NOQA

__all__ = ["Log", "DebugLog", "TensorboardLog", "WandBLog", "ClearMLLog", "CsvLog", "get_logger"]


class Log(ServerObserver, ChannelObserver, ClientObserver):
    """Basic logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process. The logging happens in the console.

    Attributes:
        tracker (PerformanceTracker): The performance tracker.
        current_round (int): The current round.
    """

    def __init__(self, **kwargs):
        self.tracker = PerformanceTracker()
        self.current_round: int = 0
        self.custom_fields: dict = {}

    def log(self, message: str) -> None:
        """Log a message.

        Args:
            message (str): The message to log.
        """
        rich_print(message)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        """Add a scalar to the logger.

        Args:
            key (Any): The key of the scalar.
            value (float): The value of the scalar.
            round (int): The round.
        """
        if round not in self.custom_fields:
            self.custom_fields[round] = {}
        self.custom_fields[round][key] = value

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        """Add scalars to the logger.

        Args:
            key (Any): The main key of the scalars.
            values (dict[str, float]): The key-value pairs of the scalars.
            round (int): The round.
        """
        if round not in self.custom_fields:
            self.custom_fields[round] = {}
        for k, v in values.items():
            self.custom_fields[round][f"{key}/{k}"] = v

    def pretty_log(self, data: Any, title: str) -> None:
        """Log a pretty-printed data.

        Args:
            data (Any): The data to log.
            title (str): The title of the data.
        """
        rich_print(Panel(Pretty(data, expand_all=True), title=title, width=100))

    def init(self, **kwargs) -> None:
        """Initialize the logger.
        The initialization is done by printing the configuration in the console.

        Args:
            **kwargs: The configuration.
        """
        if kwargs:
            rich_print(Panel(Pretty(kwargs, expand_all=True), title="Configuration", width=100))

    def start_round(self, round: int, global_model: Module) -> None:
        self.tracker.add(perf_type="comm", metrics=0, round=round)
        self.current_round = round

        if round == 1 and self.tracker.get("comm", round=0) > 0:
            rich_print(
                Panel(
                    Pretty({"comm_costs": self.tracker.get("comm", round=0)}),
                    title=f"Round: {round - 1}",
                    width=100,
                )
            )

    def end_round(self, round: int) -> None:
        stats = {
            "pre-fit": self.tracker.summary("pre-fit", round=round, force_round=False),
            "locals": self.tracker.summary("locals", round=round, force_round=False),
            "post-fit": self.tracker.summary("post-fit", round=round, force_round=False),
            "global": self.tracker.summary("global", round=round, force_round=False),
            "comm_cost": self.tracker.get("comm", round=round),
        }

        proc = Process(os.getpid())

        self.tracker.add(
            perf_type="mem",
            metrics=proc.memory_full_info().uss,
            round=round,
        )

        if self.custom_fields and round in self.custom_fields:
            stats.update(self.custom_fields[round])

        to_skip = [k for k, v in stats.items() if v is None or (isinstance(v, dict) and not v)]
        stats = {k: v for k, v in stats.items() if k not in to_skip}

        rich_print(Panel(Pretty(stats, expand_all=True), title=f"Round: {round}", width=100))
        rich_print(
            f"  Memory usage: {bytes2human(self.tracker.get('mem', round=round))} "
            + f"[{proc.memory_percent():.2f} %]"
        )

    def client_evaluation(
        self,
        round: int,
        client_id: int,
        phase: Literal["pre-fit", "post-fit"],
        evals: dict[str, float],
        **kwargs,
    ) -> None:
        if round == -1:
            round = self.current_round + 1
        self.tracker.add(perf_type=phase, metrics=evals, round=round, client_id=client_id)

    def server_evaluation(
        self,
        round: int,
        eval_type: Literal["global", "locals"],
        evals: Union[dict[str, float], dict[int, dict[str, float]]],
        **kwargs,
    ) -> None:

        self.tracker.add(perf_type=eval_type, metrics=evals, round=round)

    def message_received(self, by: Any, message: Message) -> None:
        """Update the communication costs.

        Args:
            by (Any): The sender of the message.
            message (Message): The message received.
        """
        self.tracker.add(perf_type="comm", metrics=message.size, round=self.current_round)

    def finished(self, round: int) -> None:
        stats = {
            "pre-fit": self.tracker.summary("pre-fit", round=round),
            "locals": self.tracker.summary("locals", round=round),
            "post-fit": self.tracker.summary("post-fit", round=round),
            "global": self.tracker.summary("global", round=round),
            "comm_costs": self.tracker.summary("comm", round),
        }

        to_skip = [k for k, v in stats.items() if v is None or (isinstance(v, dict) and not v)]
        stats = DDict(stats).exclude(*to_skip)

        rich_print(
            Panel(
                Pretty(stats, expand_all=True),
                title="Overall Performance",
                width=100,
            )
        )

    def interrupted(self) -> None:
        rich_print("\n[bold italic yellow]The experiment has been interrupted by the user.")

    def early_stop(self, round: int) -> None:
        return self.end_round(round)

    def track_item(self, round: int, item: str, value: Any) -> None:
        self.add_scalar(item, value, round)

    def save(self, path: str) -> None:
        """Save the logger's history to a JSON file.

        Args:
            path (str): The path to the JSON file.
        """
        json_to_save = {
            "perf_global": self.tracker["global"],
            "comm_costs": self.tracker["comm"],
            "perf_locals": self.tracker["locals"],
            "perf_prefit": self.tracker["pre-fit"],
            "perf_postfit": self.tracker["post-fit"],
            "custom_fields": self.custom_fields,
        }
        with open(path, "w") as f:
            json.dump(json_to_save, f, indent=4)

    def close(self) -> None:
        """Close the logger."""
        pass


class DebugLog(Log):
    """Debug Logger.
    This type of logger extends the basic logger by adding debug information.
    It can be seen as a more verbose version of the basic logger.

    Args:
        **kwargs: The configuration for the logger.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            # datefmt="[%X]",
            datefmt="[%Y-%m-%d %H:%M:%S]",
            handlers=[RichHandler(rich_tracebacks=False, show_path=False, markup=True)],
        )

        self.logger = logging.getLogger("rich")

        # Example logs
        # logger.debug("This is a [bold blue]debug[/] message.")
        # logger.info("This is a [green]info[/] message.")
        # logger.warning("This is a [yellow]warning[/] message.")
        # logger.error("This is a [red]error[/] message.")
        # logger.critical("This is a [bold red]critical[/] message.")

    def init(self, **kwargs) -> None:
        self.logger.debug("Debug logging enabled")
        super().init(**kwargs)

    def start_round(self, round: int, global_model: Module) -> None:
        self.logger.debug("Starting round %d", round)
        super().start_round(round, global_model)

    def end_round(self, round: int) -> None:
        self.logger.debug("Ending round %d", round)
        super().end_round(round)

    def selected_clients(self, round: int, clients: Collection) -> None:
        clients_idx = [client.index for client in clients]
        self.logger.debug(f"Selected {len(clients_idx)} clients for round {round}: {clients_idx}")
        super().selected_clients(round, clients)

    def server_evaluation(
        self,
        round: int,
        eval_type: Literal["global", "locals"],
        evals: Union[dict[str, float], dict[int, dict[str, float]]],
        **kwargs,
    ) -> None:
        if eval_type == "global":
            self.logger.debug(f"Global evaluation for round {round}")
        elif eval_type == "locals":
            self.logger.debug(f"Local models evaluated on server's test set for round {round}")
        super().server_evaluation(round, eval_type, evals, **kwargs)

    def finished(self, round: int) -> None:
        self.logger.debug(f"Federation ended successfully after round {round - 1}")
        return super().finished(round)

    def interrupted(self) -> None:
        self.logger.debug("Experiment interrupted by the user")
        return super().interrupted()

    def early_stop(self, round: int) -> None:
        self.logger.debug(f"Early stopping fired for round {round}")
        return super().early_stop(round)

    def start_fit(self, round: int, client_id: int, model: Module, **kwargs) -> None:
        self.logger.debug(f"Starting fit for client {client_id}")
        return super().start_fit(round, client_id, model, **kwargs)

    def end_fit(self, round: int, client_id: int, model: Module, loss: float, **kwargs) -> None:
        self.logger.debug(f"Fit for Client[{client_id}] ended with loss {loss}")
        return super().end_fit(round, client_id, model, loss, **kwargs)

    def client_evaluation(
        self,
        round: int,
        client_id: int,
        phase: Literal["pre-fit", "post-fit"],
        evals: dict[str, float],
        **kwargs,
    ) -> None:
        if round >= 0:
            self.logger.debug(f"Client[{client_id}] {phase} evaluation for round {round}")
        else:
            self.logger.debug(f"Client[{client_id}] {phase} evaluation after final round")
        self.logger.debug(evals)
        return super().client_evaluation(round, client_id, phase, evals, **kwargs)

    def message_received(self, by: Any, message: Message) -> None:
        sender = str(message.sender).split("(")[0]
        receiver = str(by).split("(")[0]
        self.logger.debug(f"Message {message.id} ({sender} -> {receiver}) received")
        return super().message_received(by, message)

    def message_sent(self, to: Any, message: Message) -> None:
        sender = str(message.sender).split("(")[0]
        receiver = str(to).split("(")[0]
        self.logger.debug(f"Message {message.id} ({sender} -> {receiver}) sent")
        return super().message_sent(to, message)

    def message_broadcasted(self, to: list[Any], message: Message) -> None:
        sender = str(message.sender).split("(")[0]
        self.logger.debug(f"Message {message.id} from {sender} broadcasted ")
        return super().message_broadcasted(to, message)


class TensorboardLog(Log):
    """TensorBoard logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process on TensorBoard

    See Also:
        For more information on TensorBoard, see the `official documentation
        <https://www.tensorflow.org/tensorboard>`_.

    Args:
        **config: The configuration for TensorBoard.
    """

    def __init__(self, **config):
        super().__init__(**config)
        self._config: dict = DDict(**config)
        if "log_dir" not in self._config:
            self._config["log_dir"] = "./logs"
        self._writer: SummaryWriter | None = None

    def init(self, **kwargs) -> None:
        exp_name = self._config.name
        self._config["log_dir"] = os.path.join(self._config["log_dir"], f"{exp_name}")
        self._writer = SummaryWriter(**self._config.exclude("name"))

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        super().add_scalar(key, value, round)
        return self._writer.add_scalar(key, value, round)

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        super().add_scalars(key, values, round)
        return self._writer.add_scalars(key, values, round)

    def start_round(self, round: int, global_model: Module) -> None:
        super().start_round(round, global_model)
        if round == 1 and self.tracker.get("comm", round) > 0:
            self._writer.add_scalar("comm_costs", self.tracker.get("comm", round), round)
        self._writer.flush()

    def _report(self, prefix: str, evals: dict[str, float], round: int) -> None:
        for metric, value in evals.items():
            self._writer.add_scalar(f"{prefix}/{metric}", value, round)
            self._writer.flush()

    def end_round(self, round: int) -> None:
        super().end_round(round)
        global_perf = self.tracker.get("global", round)
        if global_perf is not None:
            self._report("global", global_perf, round)
        self._writer.add_scalar("comm_costs", self.tracker.get("comm", round), round)
        self._writer.flush()

        prefit_perf = self.tracker.summary("pre-fit", round=round, include_round=True)
        if prefit_perf:
            self._report("pre-fit", prefit_perf, prefit_perf["round"])

        postfit_perf = self.tracker.summary("post-fit", round=round, include_round=True)
        if postfit_perf:
            self._report("post-fit", postfit_perf, postfit_perf["round"])

        locals_perf = self.tracker.summary("locals", round=round, include_round=True)
        if locals_perf:
            self._report("locals", locals_perf, locals_perf["round"])

        self._writer.flush()

    def finished(self, round: int) -> None:
        super().finished(round)
        prefit_perf = self.tracker.summary("pre-fit", round=round, include_round=True)
        if prefit_perf and round == prefit_perf["round"]:
            self._report("pre-fit", prefit_perf, round)

        locals_perf = self.tracker.summary("locals", round=round, include_round=True)
        if locals_perf and round == locals_perf["round"]:
            self._report("locals", locals_perf, round)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


class WandBLog(Log):
    """Weights and Biases logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process on Weights and Biases.

    See Also:
        For more information on Weights and Biases, see the `Weights and Biases documentation
        <https://docs.wandb.ai/>`_.

    Args:
        **config: The configuration for Weights and Biases.
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.run = None

    def init(self, **kwargs) -> None:
        super().init(**kwargs)
        self.config["config"] = kwargs
        self.run = wandb.init(**self.config)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        super().add_scalar(key, value, round)
        return self.run.log({key: value}, step=round)

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        super().add_scalars(key, values, round)
        return self.run.log({f"{key}/{k}": v for k, v in values.items()}, step=round)

    def start_round(self, round: int, global_model: Module) -> None:
        super().start_round(round, global_model)
        if round == 1 and self.tracker.get("comm", round) > 0:
            self.run.log({"comm_costs": self.tracker.get("comm", round)})

    def end_round(self, round: int) -> None:
        super().end_round(round)
        global_perf = self.tracker.get("global", round)
        if global_perf is not None:
            self.run.log({"global": global_perf}, step=round)
        self.run.log({"comm_cost": self.tracker.get("comm", round)}, step=round)

        prefit_perf = self.tracker.summary("pre-fit", round=round, include_round=True)
        if prefit_perf:
            self.run.log({"pre-fit": prefit_perf}, step=prefit_perf["round"])

        postfit_perf = self.tracker.summary("post-fit", round=round, include_round=True)
        if postfit_perf:
            self.run.log({"post-fit": postfit_perf}, step=postfit_perf["round"])

        locals_perf = self.tracker.summary("locals", round=round, include_round=True)
        if locals_perf:
            self.run.log({"locals": locals_perf}, step=locals_perf["round"])

    def finished(self, round: int) -> None:
        super().finished(round)

        server_last_round = max(self.tracker["global"])
        self.run.log(
            {"global": self.tracker.get("global", server_last_round)}, step=server_last_round
        )

        postfit_eval_summary = self.tracker.summary("post-fit", round=round, include_round=True)
        if postfit_eval_summary:
            last_round = postfit_eval_summary["round"]
            # avoid warning
            if server_last_round <= last_round:
                self.run.log({"post-fit": postfit_eval_summary}, step=last_round)

        prefit_eval_summary = self.tracker.summary("pre-fit", round=round, include_round=True)
        if prefit_eval_summary:
            last_round = prefit_eval_summary["round"]
            # avoid warning
            if server_last_round <= last_round:
                self.run.log({"pre-fit": prefit_eval_summary}, step=last_round)

        locals_eval_summary = self.tracker.summary("locals", round=round, include_round=True)
        if locals_eval_summary:
            last_round = locals_eval_summary["round"]
            # avoid warning
            if server_last_round <= last_round:
                self.run.log({"locals": locals_eval_summary}, step=last_round)

    def close(self) -> None:
        self.run.finish()

    def save(self, path: str) -> None:
        super().save(path)


class ClearMLLog(TensorboardLog):
    """ClearML logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process on ClearML.

    Note:
        The ClearML logger takes advantage of the TensorBoard logger, thus the logging also happens
        on TensorBoard. The logging folder is "./runs/{experiment_name}_{timestamp}".

    See Also:
        For more information on ClearML, see the `official documentation
        <https://clear.ml/docs/latest/docs/>`_.

    Args:
        **config: The configuration for ClearML.
    """

    def __init__(self, **config):
        super().__init__(name=config["name"])
        self.config: DDict = DDict(**config)
        self.task: clearml.task.Task | None = None

    def init(self, **kwargs) -> None:
        super().init(**kwargs)
        # imported here to avoid issues with requests
        from clearml import Task

        self.task = Task.init(task_name=self.config.name, **self.config.exclude("name"))
        self.task.connect(kwargs)

    def close(self) -> None:
        super().close()
        if self.task is not None:
            self.task.close()
            self.task = None


class NewLog(Log):
    """Custom logger that saves both global and client-wise metrics to a JSON file."""

    def __init__(self, **config):
        super().__init__(**config)
        self._config = config
        if "log_dir" not in self._config:
            self._config["log_dir"] = "./runs"
        self._history = {
            "global": {},
            "locals": {},
            "post-fit": {},
            "pre-fit": {},
            "comm_costs": {},
        }

    def end_round(self, round: int) -> None:
        super().end_round(round)

        # Global evaluation
        global_perf = self.tracker.get("global", round)
        if global_perf is not None:
            self._history["global"][round] = global_perf
        # Communication costs
        self._history["comm_costs"][round] = self.tracker.get("comm", round)
        # Local models evaluation (server side)
        locals_perf = self.tracker.get("locals", round)
        if locals_perf:
            self._history["locals"][round] = locals_perf
        # Client post-fit metrics
        postfit_perf = self.tracker.get("post-fit", round)
        if postfit_perf:
            self._history["post-fit"][round] = postfit_perf
        # Client pre-fit metrics
        prefit_perf = self.tracker.get("pre-fit", round)
        if prefit_perf:
            self._history["pre-fit"][round] = prefit_perf

    def close(self) -> None:
        """Save all client-wise and global metrics in JSON format."""
        os.makedirs(self._config["log_dir"], exist_ok=True)
        path = os.path.join(self._config["log_dir"], "history.json")
        with open(path, "w") as f:
            json.dump(self._history, f, indent=4)
        rich_print(f"[green]Saved metrics to[/green] {path}")


class CsvLog(Log):
    """Custom logger that saves both global and client-wise metrics to CSV files."""

    def __init__(self, **config):
        super().__init__(**config)
        self._config = DDict(**config)
        if "log_dir" not in self._config:
            self._config["log_dir"] = "./runs"

    def _collect_metric_keys(self, data_by_round: dict, include_client: bool) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for round_id in sorted(data_by_round.keys()):
            metrics = data_by_round[round_id]
            if include_client:
                for evals in metrics.values():
                    for key in evals.keys():
                        if key not in seen:
                            keys.append(key)
                            seen.add(key)
            else:
                for key in metrics.keys():
                        if key not in seen:
                            keys.append(key)
                            seen.add(key)
        return keys

    def _write_csv(self, path: str, fieldnames: list[str], rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _normalize_locals_round(self, round_data: dict) -> dict:
        if list(round_data.keys()) == [None] and isinstance(round_data.get(None), dict):
            return round_data[None]
        return round_data

    def _normalize_epoch_prefix(self, round_id: int, key: str) -> str:
        prefix = f"Epoch {round_id}/"
        return key[len(prefix) :] if key.startswith(prefix) else key

    def close(self) -> None:
        """Save all client-wise and global metrics in CSV format."""
        log_dir = self._config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)

        global_data = self.tracker["global"]
        global_keys = self._collect_metric_keys(global_data, include_client=False)
        global_rows = [
            {"round": round_id, **global_data[round_id]} for round_id in sorted(global_data.keys())
        ]
        self._write_csv(
            os.path.join(log_dir, "global_metrics.csv"),
            ["round"] + global_keys,
            global_rows,
        )

        locals_data = {
            round_id: self._normalize_locals_round(round_data)
            for round_id, round_data in self.tracker["locals"].items()
        }
        locals_keys = self._collect_metric_keys(locals_data, include_client=True)
        locals_rows: list[dict] = []
        for round_id in sorted(locals_data.keys()):
            for client_id in sorted(locals_data[round_id].keys()):
                row = {"round": round_id, "client": client_id}
                row.update(locals_data[round_id][client_id])
                locals_rows.append(row)
        self._write_csv(
            os.path.join(log_dir, "locals_metrics.csv"),
            ["round", "client"] + locals_keys,
            locals_rows,
        )

        prefit_data = self.tracker["pre-fit"]
        prefit_keys = self._collect_metric_keys(prefit_data, include_client=True)
        prefit_rows: list[dict] = []
        for round_id in sorted(prefit_data.keys()):
            for client_id in sorted(prefit_data[round_id].keys()):
                row = {"round": round_id, "client": client_id}
                row.update(prefit_data[round_id][client_id])
                prefit_rows.append(row)
        self._write_csv(
            os.path.join(log_dir, "prefit_metrics.csv"),
            ["round", "client"] + prefit_keys,
            prefit_rows,
        )

        postfit_data = self.tracker["post-fit"]
        postfit_keys = self._collect_metric_keys(postfit_data, include_client=True)
        postfit_rows: list[dict] = []
        for round_id in sorted(postfit_data.keys()):
            for client_id in sorted(postfit_data[round_id].keys()):
                row = {"round": round_id, "client": client_id}
                row.update(postfit_data[round_id][client_id])
                postfit_rows.append(row)
        self._write_csv(
            os.path.join(log_dir, "postfit_metrics.csv"),
            ["round", "client"] + postfit_keys,
            postfit_rows,
        )

        comm_data = self.tracker["comm"]
        comm_rows = [
            {"round": round_id, "comm_costs": comm_data[round_id]}
            for round_id in sorted(comm_data.keys())
            if round_id > 0
        ]
        self._write_csv(
            os.path.join(log_dir, "comm_costs.csv"),
            ["round", "comm_costs"],
            comm_rows,
        )
        if self.custom_fields:
            local_rows: dict[tuple[int, int], dict] = {}
            local_keys: list[str] = []
            local_seen: set[str] = set()
            shared_rows: dict[tuple[int, int], dict] = {}
            shared_keys: list[str] = []
            shared_seen: set[str] = set()
            client_rows: dict[tuple[int, int], dict] = {}
            client_keys: list[str] = []
            client_seen: set[str] = set()
            global_rows: list[dict] = []
            global_keys: list[str] = []
            global_seen: set[str] = set()
            run_time_seconds: float | None = None

            for round_id in sorted(self.custom_fields.keys()):
                global_row: dict = {"round": round_id}
                for key, value in self.custom_fields[round_id].items():
                    if key.startswith("Client[") and "]" in key:
                        client_part, metric_part = key.split("]", 1)
                        client_id = int(client_part[len("Client[") :])
                        metric_key = metric_part.lstrip(".")
                        row_key = (round_id, client_id)
                        if metric_key.startswith("local_test/"):
                            metric_name = metric_key[len("local_test/") :]
                            row = local_rows.setdefault(
                                row_key, {"round": round_id, "client": client_id}
                            )
                            row[metric_name] = value
                            if metric_name not in local_seen:
                                local_keys.append(metric_name)
                                local_seen.add(metric_name)
                        elif metric_key.startswith("shared_test/"):
                            metric_name = metric_key[len("shared_test/") :]
                            row = shared_rows.setdefault(
                                row_key, {"round": round_id, "client": client_id}
                            )
                            row[metric_name] = value
                            if metric_name not in shared_seen:
                                shared_keys.append(metric_name)
                                shared_seen.add(metric_name)
                        else:
                            row = client_rows.setdefault(
                                row_key, {"round": round_id, "client": client_id}
                            )
                            row[metric_key] = value
                            if metric_key not in client_seen:
                                client_keys.append(metric_key)
                                client_seen.add(metric_key)
                    else:
                        normalized_key = self._normalize_epoch_prefix(round_id, key)
                        if normalized_key == "run_time_seconds":
                            run_time_seconds = float(value)
                        else:
                            global_row[normalized_key] = value
                            if normalized_key not in global_seen:
                                global_keys.append(normalized_key)
                                global_seen.add(normalized_key)
                if len(global_row) > 1:
                    global_rows.append(global_row)

            if local_rows:
                local_data = [
                    local_rows[key]
                    for key in sorted(local_rows.keys(), key=lambda k: (k[0], k[1]))
                ]
                self._write_csv(
                    os.path.join(log_dir, "local_test_metrics.csv"),
                    ["round", "client"] + local_keys,
                    local_data,
                )
            if shared_rows:
                shared_data = [
                    shared_rows[key]
                    for key in sorted(shared_rows.keys(), key=lambda k: (k[0], k[1]))
                ]
                self._write_csv(
                    os.path.join(log_dir, "shared_test_metrics.csv"),
                    ["round", "client"] + shared_keys,
                    shared_data,
                )
            if client_rows:
                client_data = [
                    client_rows[key]
                    for key in sorted(client_rows.keys(), key=lambda k: (k[0], k[1]))
                ]
                self._write_csv(
                    os.path.join(log_dir, "metrics.csv"),
                    ["round", "client"] + client_keys,
                    client_data,
                )
            if local_rows or shared_rows or client_rows:
                if global_rows:
                    self._write_csv(
                        os.path.join(log_dir, "metrics_global.csv"),
                        ["round"] + global_keys,
                        global_rows,
                    )
            else:
                self._write_csv(
                    os.path.join(log_dir, "metrics.csv"),
                    ["round"] + global_keys,
                    global_rows,
                )
            if run_time_seconds is not None:
                self._write_csv(
                    os.path.join(log_dir, "run_metrics.csv"),
                    ["metric", "value"],
                    [{"metric": "run_time_seconds", "value": run_time_seconds}],
                )
        rich_print(f"[green]Saved CSV metrics to[/green] {log_dir}")


def get_logger(lname: str, **kwargs) -> Log:
    """Get a logger from its name.
    This function is used to get a logger from its name. It is used to dynamically import loggers.
    The supported loggers are the ones defined in the ``fluke.utils.log`` module, but it can handle
    any logger defined by the user.

    Note:
        To use a custom logger, it must be defined in a module and the full model name must be
        provided in the configuration file. For example, if the logger ``MyLogger`` is defined in
        the module ``my_module`` (i.e., a file called ``my_module.py``), the logger name must be
        ``my_module.MyLogger``. The other logger's parameters must be passed as in the following
        example:

        .. code-block:: yaml

            logger:
                name: my_module.MyLogger
                param1: value1
                param2: value2
                ...


    Args:
        lname (str): The name of the logger.
        **kwargs: The keyword arguments to pass to the logger's constructor.

    Returns:
        Log | DebugLog | WandBLog | ClearMLLog | TensorboardLog: The logger.
    """
    if "." in lname and not lname.startswith("fluke.utils.log"):
        return get_class_from_qualified_name(lname)(**kwargs)
    return get_class_from_str("fluke.utils.log", lname)(**kwargs)
