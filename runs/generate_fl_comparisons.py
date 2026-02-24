#!/usr/bin/env python3
"""
Generate federated learning comparison metrics/plots across LogReg, MLP, and SVM runs.

Outputs:
- Per-common-config F1 line plots and round-wise CSV tables.
- Summary CSV with final-round F1 per model/config.
- Time comparison grouped bar chart for 64-15-{100,50,25}.
- Time metrics CSV backing the time chart.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


MODELS: Sequence[str] = ("LogReg", "MLP", "SVM")
TIME_TARGET_CONFIGS: Sequence[Tuple[int, int, int]] = (
    (64, 15, 100),
    (64, 15, 50),
    (64, 15, 25),
)


def parse_config_from_dirname(dirname: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse trailing config from run directory names such as:
    - adult_MLP_decentralized-64-15-100
    - adult_LogReg_decentralized_64_15_100
    """
    match = re.search(r"(\d+)[_-](\d+)[_-](\d+)$", dirname)
    if not match:
        return None
    return tuple(int(match.group(i)) for i in range(1, 4))  # type: ignore[return-value]


def read_global_f1(global_metrics_path: Path) -> Tuple[List[int], List[float], str]:
    rounds: List[int] = []
    f1_values: List[float] = []
    with global_metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No headers in {global_metrics_path}")
        if "macro_f1" in reader.fieldnames:
            f1_column = "macro_f1"
        elif "micro_f1" in reader.fieldnames:
            f1_column = "micro_f1"
        else:
            raise ValueError(
                f"No macro_f1/micro_f1 column found in {global_metrics_path}"
            )
        for row in reader:
            rounds.append(int(float(row["round"])))
            f1_values.append(float(row[f1_column]))
    return rounds, f1_values, f1_column


def read_run_time_seconds(run_metrics_path: Path) -> Optional[float]:
    with run_metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("metric") == "run_time_seconds":
                return float(row["value"])
    return None


def discover_runs(
    runs_dir: Path,
) -> Dict[str, Dict[Tuple[int, int, int], Path]]:
    discovered: Dict[str, Dict[Tuple[int, int, int], Path]] = {m: {} for m in MODELS}

    for model in MODELS:
        model_dir = runs_dir / model
        if not model_dir.exists():
            continue
        for entry in sorted(model_dir.iterdir()):
            if not entry.is_dir():
                continue
            config = parse_config_from_dirname(entry.name)
            if config is None:
                continue
            # Keep first seen per model/config (deterministic due sorting).
            discovered[model].setdefault(config, entry)
    return discovered


def common_configs(run_index: Dict[str, Dict[Tuple[int, int, int], Path]]) -> List[Tuple[int, int, int]]:
    if not MODELS:
        return []
    config_sets = []
    for model in MODELS:
        config_sets.append(set(run_index.get(model, {}).keys()))
    return sorted(set.intersection(*config_sets))


def save_roundwise_f1_csv(
    out_path: Path,
    rounds: Sequence[int],
    model_to_round_f1: Dict[str, Dict[int, float]],
) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", *MODELS])
        for rnd in rounds:
            writer.writerow([rnd, *[model_to_round_f1[m][rnd] for m in MODELS]])


def make_f1_plot(
    out_path: Path,
    config: Tuple[int, int, int],
    rounds: Sequence[int],
    model_to_round_f1: Dict[str, Dict[int, float]],
    f1_column_name: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        ys = [model_to_round_f1[model][rnd] for rnd in rounds]
        plt.plot(rounds, ys, marker="o", linewidth=1.8, markersize=3.5, label=model)

    batch, epochs, frac = config
    plt.title(
        f"F1 vs Round | config={batch}-{epochs}-{frac} ({f1_column_name})",
        fontsize=12,
    )
    plt.xlabel("Round")
    plt.ylabel(f1_column_name)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_time_plot(
    out_path: Path,
    time_table: Dict[str, Dict[Tuple[int, int, int], Optional[float]]],
) -> None:
    model_positions = list(range(len(MODELS)))
    bar_width = 0.24
    offsets = (-bar_width, 0.0, bar_width)
    labels = [f"{cfg[0]}-{cfg[1]}-{cfg[2]}" for cfg in TIME_TARGET_CONFIGS]

    plt.figure(figsize=(10, 6))
    for idx, cfg in enumerate(TIME_TARGET_CONFIGS):
        values = []
        for model in MODELS:
            value = time_table.get(model, {}).get(cfg)
            values.append(float("nan") if value is None else value)
        x_positions = [x + offsets[idx] for x in model_positions]
        plt.bar(x_positions, values, width=bar_width, label=labels[idx], alpha=0.9)

    plt.xticks(model_positions, MODELS)
    plt.ylabel("run_time_seconds")
    plt.xlabel("Model")
    plt.title("Run Time Comparison | configs 64-15-100 vs 64-15-50 vs 64-15-25")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend(title="Configuration")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate FL model comparison metrics and plots.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing model run folders (default: this script directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Output directory for generated CSVs and plots.",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_index = discover_runs(runs_dir)
    shared_configs = common_configs(run_index)

    if not shared_configs:
        print("No common configurations found across LogReg, MLP, and SVM.")
        return 1

    final_f1_summary_rows: List[List[object]] = []
    generated_f1_plots = 0

    for config in shared_configs:
        model_to_round_f1: Dict[str, Dict[int, float]] = {}
        per_model_rounds: Dict[str, List[int]] = {}
        f1_column_name = "macro_f1"
        valid_config = True

        for model in MODELS:
            run_dir = run_index[model].get(config)
            if run_dir is None:
                valid_config = False
                break
            global_metrics_path = run_dir / "global_metrics.csv"
            if not global_metrics_path.exists():
                print(f"Skipping config {config}: missing {global_metrics_path}")
                valid_config = False
                break
            rounds, f1_values, f1_column_name = read_global_f1(global_metrics_path)
            per_model_rounds[model] = rounds
            model_to_round_f1[model] = dict(zip(rounds, f1_values))

        if not valid_config:
            continue

        # Align all models to rounds they all share.
        aligned_rounds = sorted(set.intersection(*[set(v) for v in per_model_rounds.values()]))
        if not aligned_rounds:
            print(f"Skipping config {config}: no common rounds across models.")
            continue

        batch, epochs, frac = config
        csv_out = out_dir / f"f1_roundwise_{batch}-{epochs}-{frac}.csv"
        plot_out = out_dir / f"f1_comparison_{batch}-{epochs}-{frac}.png"

        save_roundwise_f1_csv(csv_out, aligned_rounds, model_to_round_f1)
        make_f1_plot(plot_out, config, aligned_rounds, model_to_round_f1, f1_column_name)
        generated_f1_plots += 1

        last_round = aligned_rounds[-1]
        for model in MODELS:
            final_f1_summary_rows.append(
                [batch, epochs, frac, model, last_round, model_to_round_f1[model][last_round]]
            )

    final_summary_csv = out_dir / "final_round_f1_summary.csv"
    with final_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_size", "epochs", "client_fraction", "model", "round", "f1"])
        writer.writerows(final_f1_summary_rows)

    # Build time table for target configurations.
    time_table: Dict[str, Dict[Tuple[int, int, int], Optional[float]]] = {m: {} for m in MODELS}
    for model in MODELS:
        for cfg in TIME_TARGET_CONFIGS:
            run_dir = run_index.get(model, {}).get(cfg)
            if run_dir is None:
                time_table[model][cfg] = None
                continue
            run_metrics_path = run_dir / "run_metrics.csv"
            if not run_metrics_path.exists():
                time_table[model][cfg] = None
                continue
            time_table[model][cfg] = read_run_time_seconds(run_metrics_path)

    time_metrics_csv = out_dir / "run_time_64-15-100_50_25.csv"
    with time_metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "config", "run_time_seconds"])
        for model in MODELS:
            for cfg in TIME_TARGET_CONFIGS:
                cfg_label = f"{cfg[0]}-{cfg[1]}-{cfg[2]}"
                writer.writerow([model, cfg_label, time_table[model].get(cfg)])

    time_plot_out = out_dir / "time_comparison_64-15-100_50_25.png"
    make_time_plot(time_plot_out, time_table)

    print(f"Generated {generated_f1_plots} F1 comparison plot(s) in: {out_dir}")
    print(f"Wrote summary metrics: {final_summary_csv}")
    print(f"Wrote time metrics: {time_metrics_csv}")
    print(f"Wrote time comparison plot: {time_plot_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
