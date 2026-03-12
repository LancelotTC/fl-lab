"""
Run the Adult non-IID decentralized grid for LogReg and MLP only.

This reproduces the same parameter sweep used for the existing Adult decentralized
runs, but stores outputs under runs/adult_noniid/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import yaml


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


def make_exp_yaml(base_cfg: dict, log_dir: str, eligible_perc: float) -> str:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["logger"]["log_dir"] = log_dir
    cfg["protocol"]["eligible_perc"] = eligible_perc
    return yaml.safe_dump(cfg, sort_keys=False)


def make_alg_yaml(model_name: str, net_args: dict[str, int], batch_size: int, local_epochs: int) -> str:
    net_args_yaml = "\n".join(f"    {key}: {value}" for key, value in net_args.items())
    return f"""hyperparameters:
  client:
    batch_size: {batch_size}
    local_epochs: {local_epochs}
    loss: CrossEntropyLoss
    optimizer:
      lr: 0.001
      momentum: 0.9
      weight_decay: 1.0e-05
    persistency: true
    scheduler:
      gamma: 1
      step_size: 1
  model: {model_name}
  net_args:
{net_args_yaml}
  server:
    weighted: true
    lr: 1.0
    neighbors: 4
    consensus_steps: 5
name: fluke.algorithms.decentralized.DecentralizedFedAvg
"""


def iter_runs(models: Iterable[str]) -> Iterable[tuple[str, int, int, int]]:
    for model in models:
        for batch_size, local_epochs, client_pct in GRID:
            yield model, batch_size, local_epochs, client_pct


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Adult non-IID decentralized grid for LogReg and MLP.")
    parser.add_argument(
        "--fluke-cmd",
        default="fluke",
        help="Command used to invoke the Fluke CLI.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs") / "adult_noniid",
        help="Root directory where run folders will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_exp_cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config" / "exp-adult-noniid.yaml").read_text(
            encoding="utf-8"
        )
    )

    with tempfile.TemporaryDirectory(prefix="adult_noniid_grid_", dir=output_root) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for model, batch_size, local_epochs, client_pct in iter_runs(MODELS):
            model_cfg = MODELS[model]
            run_dir = output_root / model / f"adult_{model}_decentralized-{batch_size}-{local_epochs}-{client_pct}"
            run_dir.parent.mkdir(parents=True, exist_ok=True)

            exp_cfg = tmp_dir / f"exp_{model}_{batch_size}_{local_epochs}_{client_pct}.yaml"
            alg_cfg = tmp_dir / f"alg_{model}_{batch_size}_{local_epochs}_{client_pct}.yaml"

            exp_cfg.write_text(
                make_exp_yaml(
                    base_cfg=base_exp_cfg,
                    log_dir=run_dir.as_posix(),
                    eligible_perc=client_pct / 100.0,
                ),
                encoding="utf-8",
            )
            alg_cfg.write_text(
                make_alg_yaml(
                    model_name=model_cfg["model_name"],
                    net_args=model_cfg["net_args"],
                    batch_size=batch_size,
                    local_epochs=local_epochs,
                ),
                encoding="utf-8",
            )

            cmd = [args.fluke_cmd, "decentralized", str(exp_cfg), str(alg_cfg)]
            print(" ".join(cmd))
            if args.dry_run:
                continue

            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                print(f"Run failed for {model} {batch_size}-{local_epochs}-{client_pct}", file=sys.stderr)
                return completed.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
