"""Compatibility wrapper around runs.generate_fl_comparisons.

The progression plotting implementation now lives in generate_fl_comparisons.py.
This shim preserves the old CLI shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_fl_comparisons import main as generate_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for progression plots.")
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots" / "iid_noise_progression",
    )
    parser.add_argument("--dataset", type=str, default="medical")
    parser.add_argument("--epsilon-levels", type=float, nargs="+", default=None)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    forwarded = [
        "--runs-dir", str(args.runs_dir),
        "--out-dir", str(args.out_dir.parent),
        "--progression-out-dir", str(args.out_dir),
        "--dataset", args.dataset,
    ]
    if args.epsilon_levels is not None:
        forwarded.append("--epsilon-levels")
        forwarded.extend(str(v) for v in args.epsilon_levels)
    if args.noise_levels is not None:
        forwarded.append("--noise-levels")
        forwarded.extend(str(v) for v in args.noise_levels)
    return generate_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
