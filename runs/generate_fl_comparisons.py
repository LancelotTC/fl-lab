"""
Generate federated learning comparison metrics/plots across LogReg, MLP, and SVM runs.

Outputs:
- Per-common-config F1 line plots and round-wise CSV tables.
- Summary CSV with final-round F1 per model/config.
- Time comparison grouped bar chart for 64-15-{100,50,25}.
- Time metrics CSV backing the time chart.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


MODELS: Sequence[str] = ("LogReg", "MLP", "SVM")
TIME_TARGET_CONFIGS: Sequence[Tuple[int, int, int]] = (
    (64, 15, 100),
    (64, 15, 50),
    (64, 15, 25),
)
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
    """
    Parse trailing config from run directory names such as:
    - adult_MLP_decentralized-64-15-100
    - adult_LogReg_decentralized_64_15_100
    """
    match = re.search(r"(\d+)[_-](\d+)[_-](\d+)$", dirname)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def run_quality_score(run_dir: Path) -> Tuple[int, float]:
    """
    Rank candidate run folders for the same config.
    Higher score means more complete/usable run data.
    """
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


def read_local_metric_by_client(
    locals_metrics_path: Path,
    metric_name: str = "macro_f1",
) -> Dict[int, Dict[int, float]]:
    client_to_round_metric: Dict[int, Dict[int, float]] = defaultdict(dict)
    with locals_metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or metric_name not in reader.fieldnames:
            raise ValueError(f"No {metric_name} column found in {locals_metrics_path}")
        for row in reader:
            round_id = int(float(row["round"]))
            client_id = int(float(row["client"]))
            client_to_round_metric[client_id][round_id] = float(row[metric_name])
    return dict(client_to_round_metric)


def discover_runs(
    runs_dir: Path,
) -> Dict[str, Dict[Tuple[int, int, int], Path]]:
    discovered: Dict[str, Dict[Tuple[int, int, int], Path]] = {m: {} for m in MODELS}

    for model in MODELS:
        model_dir = runs_dir / model
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
            # Pick the most complete run; tie-break by latest modification time.
            discovered[model][config] = max(candidates, key=run_quality_score)
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


def make_parameter_impact_plot(
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    fixed_label: str,
    x_values: Sequence[int],
    series_by_model: Dict[str, Sequence[float]],
) -> None:
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        ys = series_by_model.get(model)
        if ys is None:
            continue
        plt.plot(x_values, ys, marker="o", linewidth=1.8, markersize=4, label=model)

    plt.title(f"{title} | {fixed_label}", fontsize=12)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(list(x_values))
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_local_client_plot(
    out_path: Path,
    model: str,
    config: Tuple[int, int, int],
    client_to_round_metric: Dict[int, Dict[int, float]],
    metric_name: str = "macro_f1",
) -> None:
    plt.figure(figsize=(11, 7))
    for client_id in sorted(client_to_round_metric):
        round_to_metric = client_to_round_metric[client_id]
        rounds = sorted(round_to_metric)
        ys = [round_to_metric[rnd] for rnd in rounds]
        plt.plot(rounds, ys, linewidth=1.2, alpha=0.8, label=f"client_{client_id}")

    batch, epochs, frac = config
    plt.title(f"Local {metric_name} by Client | {model} | config={batch}-{epochs}-{frac}", fontsize=12)
    plt.xlabel("Round")
    plt.ylabel(metric_name)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(title="Client", ncol=2, fontsize=8)
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
    generated_local_client_plots = 0

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
            final_f1_summary_rows.append([batch, epochs, frac, model, last_round, model_to_round_f1[model][last_round]])

        for model in MODELS:
            run_dir = run_index[model].get(config)
            if run_dir is None:
                continue
            locals_metrics_path = run_dir / "locals_metrics.csv"
            if not locals_metrics_path.exists():
                continue
            client_to_round_f1 = read_local_metric_by_client(locals_metrics_path, metric_name=f1_column_name)
            local_plot_out = out_dir / f"local_clients_{model}_{batch}-{epochs}-{frac}.png"
            make_local_client_plot(local_plot_out, model, config, client_to_round_f1, metric_name=f1_column_name)
            generated_local_client_plots += 1

    final_summary_csv = out_dir / "final_round_f1_summary.csv"
    with final_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_size", "epochs", "client_fraction", "model", "round", "f1"])
        writer.writerows(final_f1_summary_rows)

    final_f1_lookup: Dict[str, Dict[Tuple[int, int, int], float]] = {m: {} for m in MODELS}
    for batch, epochs, frac, model, _round, f1 in final_f1_summary_rows:
        final_f1_lookup[model][(int(batch), int(epochs), int(frac))] = float(f1)

    # Build run time lookup for all shared configurations.
    time_lookup: Dict[str, Dict[Tuple[int, int, int], Optional[float]]] = {m: {} for m in MODELS}
    for model in MODELS:
        for cfg in shared_configs:
            run_dir = run_index.get(model, {}).get(cfg)
            if run_dir is None:
                time_lookup[model][cfg] = None
                continue
            run_metrics_path = run_dir / "run_metrics.csv"
            if not run_metrics_path.exists():
                time_lookup[model][cfg] = None
                continue
            time_lookup[model][cfg] = read_run_time_seconds(run_metrics_path)

    time_table: Dict[str, Dict[Tuple[int, int, int], Optional[float]]] = {m: {} for m in MODELS}
    for model in MODELS:
        for cfg in TIME_TARGET_CONFIGS:
            time_table[model][cfg] = time_lookup[model].get(cfg)

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

    for spec in PARAMETER_IMPACT_SPECS:
        configs = spec["varying_configs"]
        x_values = [spec["x_getter"](cfg) for cfg in configs]

        f1_series_by_model: Dict[str, Sequence[float]] = {}
        time_series_by_model: Dict[str, Sequence[float]] = {}
        for model in MODELS:
            if not all(cfg in final_f1_lookup[model] for cfg in configs):
                continue
            if not all(time_lookup.get(model, {}).get(cfg) is not None for cfg in configs):
                continue
            f1_series_by_model[model] = [final_f1_lookup[model][cfg] for cfg in configs]
            time_series_by_model[model] = [time_lookup[model][cfg] for cfg in configs]  # type: ignore[list-item]

        if f1_series_by_model:
            f1_out = out_dir / f"impact_{spec['key']}_f1.png"
            make_parameter_impact_plot(
                f1_out,
                f"{spec['title']} on Final F1",
                spec["x_label"],
                "final_f1",
                spec["fixed_label"],
                x_values,
                f1_series_by_model,
            )

        if time_series_by_model:
            time_out = out_dir / f"impact_{spec['key']}_time.png"
            make_parameter_impact_plot(
                time_out,
                f"{spec['title']} on Run Time",
                spec["x_label"],
                "run_time_seconds",
                spec["fixed_label"],
                x_values,
                time_series_by_model,
            )

    print(f"Generated {generated_f1_plots} F1 comparison plot(s) in: {out_dir}")
    print(f"Generated {generated_local_client_plots} local client plot(s) in: {out_dir}")
    print(f"Wrote summary metrics: {final_summary_csv}")
    print(f"Wrote time metrics: {time_metrics_csv}")
    print(f"Wrote time comparison plot: {time_plot_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
