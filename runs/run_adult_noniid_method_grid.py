"""
Run Adult non-IID federated experiments to compare FedAvg and SCAFFOLD.

Outputs are written to:
    runs/adult_noniid_methods/{FedAvg|SCAFFOLD}/{LogReg|MLP}/...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


GRID = (
    (16, 5, 100),
    (16, 10, 100),
    (16, 15, 100),
    (32, 5, 100),
    (32, 10, 100),
    (32, 15, 100),
    (64, 5, 100),
    (64, 10, 100),
    (64, 15, 100),
    (64, 15, 50),
    (64, 15, 25),
)

METHODS = {
    "FedAvg": {
        "algo_name": "fluke.algorithms.fedavg.FedAVG",
        "server_block": "    weighted: true\n    lr: 1.0\n",
    },
    "SCAFFOLD": {
        "algo_name": "fluke.algorithms.scaffold.SCAFFOLD",
        "server_block": "    weighted: true\n    global_step: 1.0\n",
    },
}

MODELS = {
    "LogReg": {
        "model_name": "Adult_LogReg",
        "net_args": {
            "input_dim": 14,
        },
    },
    "MLP": {
        "model_name": "Adult_MLP",
        "net_args": {
            "input_dim": 14,
            "hidden1": 64,
            "hidden2": 32,
        },
    },
}


def make_exp_yaml(log_dir: str, eligible_perc: float) -> str:
    return f"""data:
  client_split: 0.2
  dataset:
    name: adult
    path: ./data
  distribution:
    name: dir
    beta: 0.02
  keep_test: true
  sampling_perc: 1.0
  server_split: 0.0
  server_test: true
  uniform_test: false
eval:
  eval_every: 1
  locals: true
  post_fit: true
  pre_fit: false
  server: true
exp:
  device: cpu
  seed: 42
  inmemory: true
logger:
  name: CsvLog
  log_dir: {log_dir}
protocol:
  eligible_perc: {eligible_perc}
  n_clients: 10
  n_rounds: 50
"""


def make_alg_yaml(
    algo_name: str,
    server_block: str,
    model_name: str,
    net_args: dict[str, int],
    batch_size: int,
    local_epochs: int,
) -> str:
    net_args_yaml = "\n".join(f"    {key}: {value}" for key, value in net_args.items())
    return f"""hyperparameters:
  client:
    batch_size: {batch_size}
    local_epochs: {local_epochs}
    loss: CrossEntropyLoss
    optimizer:
      lr: 0.001
      momentum: 0
      weight_decay: 1.0e-05
    persistency: false
    scheduler:
      gamma: 1
      step_size: 1
  model: {model_name}
  net_args:
{net_args_yaml}
  server:
{server_block}name: {algo_name}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Adult non-IID FedAvg vs SCAFFOLD grid.")
    parser.add_argument("--fluke-cmd", default="fluke", help="Command used to invoke the Fluke CLI.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs") / "adult_noniid_methods",
        help="Root directory where run folders will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS.keys()),
        default=list(METHODS.keys()),
        help="Subset of methods to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Subset of models to run.",
    )
    parser.add_argument("--batch-size", type=int, help="Run only this batch size.")
    parser.add_argument("--local-epochs", type=int, help="Run only this local epoch count.")
    parser.add_argument("--client-pct", type=int, help="Run only this selected client percentage.")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_grid = [
        cfg
        for cfg in GRID
        if (args.batch_size is None or cfg[0] == args.batch_size)
        and (args.local_epochs is None or cfg[1] == args.local_epochs)
        and (args.client_pct is None or cfg[2] == args.client_pct)
    ]
    if not selected_grid:
        print("No configurations match the provided filters.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="adult_noniid_methods_", dir=output_root) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for method_name in args.methods:
            method_cfg = METHODS[method_name]
            for model_name in args.models:
                model_cfg = MODELS[model_name]
                for batch_size, local_epochs, client_pct in selected_grid:
                    run_dir = (
                        output_root
                        / method_name
                        / model_name
                        / f"adult_{model_name}_{method_name}-{batch_size}-{local_epochs}-{client_pct}"
                    )
                    run_dir.parent.mkdir(parents=True, exist_ok=True)

                    exp_cfg = tmp_dir / f"exp_{method_name}_{model_name}_{batch_size}_{local_epochs}_{client_pct}.yaml"
                    alg_cfg = tmp_dir / f"alg_{method_name}_{model_name}_{batch_size}_{local_epochs}_{client_pct}.yaml"

                    exp_cfg.write_text(
                        make_exp_yaml(log_dir=run_dir.as_posix(), eligible_perc=client_pct / 100.0),
                        encoding="utf-8",
                    )
                    alg_cfg.write_text(
                        make_alg_yaml(
                            algo_name=method_cfg["algo_name"],
                            server_block=method_cfg["server_block"],
                            model_name=model_cfg["model_name"],
                            net_args=model_cfg["net_args"],
                            batch_size=batch_size,
                            local_epochs=local_epochs,
                        ),
                        encoding="utf-8",
                    )

                    cmd = [args.fluke_cmd, "federation", str(exp_cfg), str(alg_cfg)]
                    print(" ".join(cmd))
                    if args.dry_run:
                        continue

                    completed = subprocess.run(cmd, check=False)
                    if completed.returncode != 0:
                        print(
                            f"Run failed for {method_name} {model_name} "
                            f"{batch_size}-{local_epochs}-{client_pct}",
                            file=sys.stderr,
                        )
                        return completed.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
