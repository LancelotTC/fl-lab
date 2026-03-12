"""
Run medical FL experiments with differential privacy across IID/non-IID splits
and multiple DP epsilon levels using the real `fluke` CLI.

Usage:
    python runs/run_medical_dp_sweep.py

Optional:
    python runs/run_medical_dp_sweep.py --epsilon-levels 0.2 0.5 1.0 5.0 --rounds 50
    python runs/run_medical_dp_sweep.py --fluke C:\\Users\\lance\\miniconda3\\envs\\fluke310\\Scripts\\fluke.exe
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run medical DP-FL sweep with fluke CLI.")
    parser.add_argument(
        "--fluke",
        type=str,
        default="",
        help="Path to `fluke` executable. If omitted, auto-detection is used.",
    )
    parser.add_argument(
        "--epsilon-levels",
        nargs="+",
        type=float,
        default=[0.2, 0.5, 1.0, 5.0],
        help="Target epsilon values to test.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Target delta used by Opacus.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm used by Opacus.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of FL rounds.",
    )
    return parser.parse_args()


def _fluke_exe_name() -> str:
    return "fluke.exe" if os.name == "nt" else "fluke"


def resolve_fluke_executable(user_fluke: str) -> Path:
    if user_fluke:
        candidate = Path(user_fluke).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Provided --fluke path does not exist: {candidate}")
        return candidate

    in_path = shutil.which("fluke")
    if in_path:
        return Path(in_path).resolve()

    current = Path(sys.executable).resolve()
    exe_name = _fluke_exe_name()
    candidates = [
        current.parent / exe_name,
        current.parent / "Scripts" / exe_name,
        current.parent.parent / "envs" / "fluke310" / "Scripts" / exe_name,
        Path.home() / "miniconda3" / "envs" / "fluke310" / "Scripts" / exe_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find `fluke` executable. Activate fluke310 or pass --fluke explicitly."
    )


def run_one(
    fluke_exe: Path,
    project_root: Path,
    exp_cfg: str,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    rounds: int,
) -> int:
    dist_tag = "iid" if "iid" in exp_cfg and "non-iid" not in exp_cfg else "non-iid"
    dp_total_epochs = rounds * 10
    run_dir = (
        f"runs/medical-dp-svm-{dist_tag}-eps-{epsilon}-delta-{delta}-mgn-{max_grad_norm}"
        .replace(".", "p")
    )
    cmd = [
        str(fluke_exe),
        "federation",
        exp_cfg,
        "config/fl-config-svm-dp.yaml",
        f"method.hyperparameters.client.target_epsilon={epsilon}",
        f"method.hyperparameters.client.target_delta={delta}",
        f"method.hyperparameters.client.max_grad_norm={max_grad_norm}",
        f"method.hyperparameters.client.dp_total_epochs={dp_total_epochs}",
        f"protocol.n_rounds={rounds}",
        f"logger.log_dir={run_dir}",
    ]

    print("\nRunning:", " ".join(cmd))
    env = os.environ.copy()
    # Prefer UTF-8 on Windows to avoid banner encoding issues.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        completed = subprocess.run(cmd, check=False, cwd=project_root, env=env)
        return completed.returncode
    except FileNotFoundError as exc:
        print(f"Failed to start process: {exc}")
        return 127


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    fluke_exe = resolve_fluke_executable(args.fluke)
    print(f"Using fluke executable: {fluke_exe}")

    exp_cfgs = ["config/medical-data-iid.yaml", "config/medical-data-non-iid.yaml"]
    failures = 0
    for exp_cfg in exp_cfgs:
        for epsilon in args.epsilon_levels:
            rc = run_one(
                fluke_exe=fluke_exe,
                project_root=project_root,
                exp_cfg=exp_cfg,
                epsilon=epsilon,
                delta=args.delta,
                max_grad_norm=args.max_grad_norm,
                rounds=args.rounds,
            )
            if rc != 0:
                failures += 1
                print(f"Failed run: exp_cfg={exp_cfg}, epsilon={epsilon}, rc={rc}")

    if failures:
        print(f"\nSweep finished with {failures} failed run(s).")
        return 1

    print("\nSweep finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
