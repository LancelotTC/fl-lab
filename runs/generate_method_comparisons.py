"""
Generate FedAvg vs SCAFFOLD comparison plots for Adult non-IID runs.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


METHODS: Sequence[str] = ("FedAvg", "SCAFFOLD")
MODELS: Sequence[str] = ("LogReg", "MLP")
PARAMETER_IMPACT_SPECS = (
    {
        "key": "batch_size",
        "title": "Batch Size Impact",
        "x_label": "batch_size",
        "x_getter": lambda cfg: cfg[0],
        "varying_configs": [(16, 15, 100), (32, 15, 100), (64, 15, 100)],
        "fixed_label": "local_epochs=15, clients=100%",
    },
    {
        "key": "local_epochs",
        "title": "Local Epochs Impact",
        "x_label": "local_epochs",
        "x_getter": lambda cfg: cfg[1],
        "varying_configs": [(64, 5, 100), (64, 10, 100), (64, 15, 100)],
        "fixed_label": "batch_size=64, clients=100%",
    },
    {
        "key": "client_percentage",
        "title": "Client Percentage Impact",
        "x_label": "selected_clients_percent",
        "x_getter": lambda cfg: cfg[2],
        "varying_configs": [(64, 15, 25), (64, 15, 50), (64, 15, 100)],
        "fixed_label": "batch_size=64, local_epochs=15",
    },
)


def parse_config_from_dirname(dirname: str) -> Optional[Tuple[int, int, int]]:
    match = re.search(r"(\d+)[_-](\d+)[_-](\d+)$", dirname)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def run_quality_score(run_dir: Path) -> Tuple[int, float]:
    required = ("global_metrics.csv", "run_metrics.csv", "locals_metrics.csv", "postfit_metrics.csv")
    score = sum(1 for name in required if (run_dir / name).exists())
    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        mtime = 0.0
    return score, mtime


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
            raise ValueError(f"No macro_f1/micro_f1 column found in {global_metrics_path}")
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
) -> Dict[str, Dict[str, Dict[Tuple[int, int, int], Path]]]:
    discovered: Dict[str, Dict[str, Dict[Tuple[int, int, int], Path]]] = {
        model: {method: {} for method in METHODS} for model in MODELS
    }
    for method in METHODS:
        for model in MODELS:
            model_dir = runs_dir / method / model
            if not model_dir.exists():
                continue
            candidates_by_config: Dict[Tuple[int, int, int], List[Path]] = defaultdict(list)
            for entry in sorted(model_dir.iterdir()):
                if not entry.is_dir():
                    continue
                config = parse_config_from_dirname(entry.name)
                if config is None:
                    continue
                candidates_by_config[config].append(entry)

            for config, candidates in candidates_by_config.items():
                discovered[model][method][config] = max(candidates, key=run_quality_score)
    return discovered


def common_configs(run_index: Dict[str, Dict[str, Dict[Tuple[int, int, int], Path]]], model: str) -> List[Tuple[int, int, int]]:
    config_sets = [set(run_index.get(model, {}).get(method, {}).keys()) for method in METHODS]
    return sorted(set.intersection(*config_sets))


def make_method_plot(
    out_path: Path,
    model: str,
    config: Tuple[int, int, int],
    rounds: Sequence[int],
    method_to_round_f1: Dict[str, Dict[int, float]],
    f1_column_name: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for method in METHODS:
        ys = [method_to_round_f1[method][rnd] for rnd in rounds]
        plt.plot(rounds, ys, marker="o", linewidth=1.8, markersize=3.5, label=method)

    batch, epochs, frac = config
    plt.title(f"{model} | FedAvg vs SCAFFOLD | config={batch}-{epochs}-{frac} ({f1_column_name})", fontsize=12)
    plt.xlabel("Round")
    plt.ylabel(f1_column_name)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_parameter_impact_plot(
    out_path: Path,
    model: str,
    title: str,
    x_label: str,
    y_label: str,
    fixed_label: str,
    x_values: Sequence[int],
    series_by_method: Dict[str, Sequence[float]],
) -> None:
    plt.figure(figsize=(10, 6))
    for method in METHODS:
        ys = series_by_method.get(method)
        if ys is None:
            continue
        plt.plot(x_values, ys, marker="o", linewidth=1.8, markersize=4, label=method)

    plt.title(f"{model} | {title} | {fixed_label}", fontsize=12)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(list(x_values))
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate FedAvg vs SCAFFOLD comparison plots.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "adult_noniid_methods",
        help="Directory containing method/model run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots_adult_noniid_methods",
        help="Output directory for generated CSVs and plots.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_index = discover_runs(runs_dir)
    generated_round_plots = 0
    summary_rows: List[List[object]] = []

    for model in MODELS:
        shared_configs = common_configs(run_index, model)
        final_f1_lookup: Dict[str, Dict[Tuple[int, int, int], float]] = {method: {} for method in METHODS}
        time_lookup: Dict[str, Dict[Tuple[int, int, int], Optional[float]]] = {method: {} for method in METHODS}

        for config in shared_configs:
            method_to_round_f1: Dict[str, Dict[int, float]] = {}
            per_method_rounds: Dict[str, List[int]] = {}
            f1_column_name = "macro_f1"
            valid = True

            for method in METHODS:
                run_dir = run_index[model][method].get(config)
                if run_dir is None:
                    valid = False
                    break
                global_metrics_path = run_dir / "global_metrics.csv"
                if not global_metrics_path.exists():
                    valid = False
                    break
                rounds, f1_values, f1_column_name = read_global_f1(global_metrics_path)
                method_to_round_f1[method] = dict(zip(rounds, f1_values))
                per_method_rounds[method] = rounds

                run_metrics_path = run_dir / "run_metrics.csv"
                time_lookup[method][config] = (
                    read_run_time_seconds(run_metrics_path) if run_metrics_path.exists() else None
                )

            if not valid:
                continue

            aligned_rounds = sorted(set.intersection(*[set(v) for v in per_method_rounds.values()]))
            if not aligned_rounds:
                continue

            batch, epochs, frac = config
            plot_out = out_dir / f"{model.lower()}_fedavg_vs_scaffold_{batch}-{epochs}-{frac}.png"
            make_method_plot(plot_out, model, config, aligned_rounds, method_to_round_f1, f1_column_name)
            generated_round_plots += 1

            last_round = aligned_rounds[-1]
            for method in METHODS:
                final_f1 = method_to_round_f1[method][last_round]
                final_f1_lookup[method][config] = final_f1
                summary_rows.append([model, method, batch, epochs, frac, last_round, final_f1, time_lookup[method][config]])

        for spec in PARAMETER_IMPACT_SPECS:
            configs = spec["varying_configs"]
            x_values = [spec["x_getter"](cfg) for cfg in configs]

            f1_series_by_method: Dict[str, Sequence[float]] = {}
            time_series_by_method: Dict[str, Sequence[float]] = {}
            for method in METHODS:
                if not all(cfg in final_f1_lookup[method] for cfg in configs):
                    continue
                if not all(time_lookup[method].get(cfg) is not None for cfg in configs):
                    continue
                f1_series_by_method[method] = [final_f1_lookup[method][cfg] for cfg in configs]
                time_series_by_method[method] = [time_lookup[method][cfg] for cfg in configs]  # type: ignore[list-item]

            if f1_series_by_method:
                make_parameter_impact_plot(
                    out_dir / f"{model.lower()}_impact_{spec['key']}_f1.png",
                    model,
                    f"{spec['title']} on Final F1",
                    spec["x_label"],
                    "final_f1",
                    spec["fixed_label"],
                    x_values,
                    f1_series_by_method,
                )

            if time_series_by_method:
                make_parameter_impact_plot(
                    out_dir / f"{model.lower()}_impact_{spec['key']}_time.png",
                    model,
                    f"{spec['title']} on Run Time",
                    spec["x_label"],
                    "run_time_seconds",
                    spec["fixed_label"],
                    x_values,
                    time_series_by_method,
                )

    summary_csv = out_dir / "fedavg_vs_scaffold_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "method", "batch_size", "epochs", "client_fraction", "round", "f1", "run_time_seconds"])
        writer.writerows(summary_rows)

    print(f"Generated {generated_round_plots} FedAvg vs SCAFFOLD round plot(s) in: {out_dir}")
    print(f"Wrote summary metrics: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
