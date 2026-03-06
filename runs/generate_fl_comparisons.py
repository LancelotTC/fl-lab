"""
Generate FL comparison artifacts focused on:
- quality (global metrics),
- fairness (cross-client disparity),
- cost (runtime + communication),
- privacy tags (DP noise inferred from run naming).

Expected run folders are under `runs/` and contain at least one of:
- `global_metrics.csv` (federated/decentralized),
- `metrics.csv` (centralized).

Recommended DP run naming:
- `...-noise-0.3...` or `..._noise_0.3...`
- `...-mgn-1.0...` for max_grad_norm (optional)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLOBAL_METRICS_FILE = "global_metrics.csv"
CENTRALIZED_METRICS_FILE = "metrics.csv"
LOCALS_METRICS_FILE = "locals_metrics.csv"
RUN_METRICS_FILE = "run_metrics.csv"
COMM_COSTS_FILE = "comm_costs.csv"

QUALITY_METRICS = ("macro_f1", "micro_f1", "accuracy")
KNOWN_DATASETS = (
    "medical",
    "adult",
    "mnist",
    "femnist",
    "emnist",
    "cifar10",
    "cifar100",
    "fashion_mnist",
    "cinic10",
    "tiny_imagenet",
    "svhn",
    "mnistm",
    "fcube",
    "shakespeare",
)
KNOWN_MODELS = ("svm", "logreg", "mlp", "2nn", "cnn", "resnet", "lstm")


@dataclass
class RunRecord:
    run_label: str
    run_path: Path
    dataset: str
    model: str
    setting: str
    is_dp: bool
    dp_noise_mul: Optional[float]
    dp_max_grad_norm: Optional[float]
    n_rounds: int
    final_round: int
    final_accuracy: Optional[float]
    final_macro_f1: Optional[float]
    final_micro_f1: Optional[float]
    best_round_by_macro_f1: Optional[int]
    best_macro_f1: Optional[float]
    final_client_accuracy_mean: Optional[float]
    final_client_macro_f1_mean: Optional[float]
    final_client_micro_f1_mean: Optional[float]
    fairness_final_acc_mean: Optional[float]
    fairness_final_acc_std: Optional[float]
    fairness_final_acc_gap: Optional[float]
    fairness_final_acc_jain: Optional[float]
    fairness_final_macro_f1_mean: Optional[float]
    fairness_final_macro_f1_std: Optional[float]
    fairness_final_macro_f1_gap: Optional[float]
    fairness_final_spd_mean: Optional[float]
    fairness_final_spd_std: Optional[float]
    fairness_final_spd_gap: Optional[float]
    fairness_final_eod_mean: Optional[float]
    fairness_final_eod_std: Optional[float]
    fairness_final_eod_gap: Optional[float]
    run_time_seconds: Optional[float]
    total_comm_cost: Optional[float]
    comm_cost_per_round: Optional[float]
    run_time_per_round: Optional[float]


def to_float_or_none(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def tokenize(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def first_existing_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_dataset(name: str) -> str:
    lower = name.lower()
    for ds in KNOWN_DATASETS:
        if ds in lower:
            return ds
    return "unknown"


def infer_model(name: str) -> str:
    lower = name.lower()
    for mdl in KNOWN_MODELS:
        if mdl in lower:
            return mdl
    return "unknown"


def infer_setting(name: str) -> str:
    tokens = tokenize(name)
    lower = name.lower()
    if "decentralized" in lower:
        return "decentralized"
    if "centralized" in lower:
        return "centralized"
    if "non" in tokens and "iid" in tokens:
        return "non-iid"
    if "noniid" in lower:
        return "non-iid"
    if "iid" in tokens:
        return "iid"
    return "federated"


def parse_float_tag(name: str, key: str) -> Optional[float]:
    # Examples captured:
    # - noise-0.3
    # - noise_0.3
    # - noise-0p3
    # - mgn-1.0
    pattern = rf"{key}[._-]?([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, name.lower())
    if not match:
        return None
    raw = match.group(1).replace("p", ".")
    return to_float_or_none(raw)


def is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / GLOBAL_METRICS_FILE).exists() or (path / CENTRALIZED_METRICS_FILE).exists()


def discover_run_dirs(runs_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if is_run_dir(entry):
            run_dirs.append(entry)
            continue
        for sub in sorted(entry.iterdir()):
            if is_run_dir(sub):
                run_dirs.append(sub)
    return run_dirs


def build_run_label(run_dir: Path, runs_dir: Path) -> str:
    return str(run_dir.relative_to(runs_dir)).replace("\\", "/")


def choose_global_metrics_path(run_dir: Path) -> Path:
    global_path = run_dir / GLOBAL_METRICS_FILE
    if global_path.exists():
        return global_path
    return run_dir / CENTRALIZED_METRICS_FILE


def read_run_time_seconds(run_dir: Path) -> Optional[float]:
    path = run_dir / RUN_METRICS_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "metric" not in df.columns or "value" not in df.columns:
        return None
    rows = df[df["metric"] == "run_time_seconds"]
    if rows.empty:
        return None
    return to_float_or_none(rows.iloc[0]["value"])


def read_total_comm_cost(run_dir: Path) -> Optional[float]:
    path = run_dir / COMM_COSTS_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    col = "comm_costs" if "comm_costs" in df.columns else ("comm_cost" if "comm_cost" in df.columns else None)
    if col is None:
        return None
    return to_float_or_none(df[col].sum())


def jain_index(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    denom = values.size * np.square(values).sum()
    if denom <= 0:
        return None
    return to_float_or_none(np.square(values.sum()) / denom)


def compute_fairness_from_locals(run_dir: Path) -> tuple[dict[str, Optional[float]], Optional[pd.DataFrame]]:
    locals_path = run_dir / LOCALS_METRICS_FILE
    defaults = {
        "fairness_final_acc_mean": None,
        "fairness_final_acc_std": None,
        "fairness_final_acc_gap": None,
        "fairness_final_acc_jain": None,
        "fairness_final_macro_f1_mean": None,
        "fairness_final_macro_f1_std": None,
        "fairness_final_macro_f1_gap": None,
        "fairness_final_spd_mean": None,
        "fairness_final_spd_std": None,
        "fairness_final_spd_gap": None,
        "fairness_final_eod_mean": None,
        "fairness_final_eod_std": None,
        "fairness_final_eod_gap": None,
    }
    if not locals_path.exists():
        return defaults, None

    df = pd.read_csv(locals_path)
    if df.empty or "round" not in df.columns:
        return defaults, None

    metric_cols = [c for c in ("accuracy", "macro_f1") if c in df.columns]
    spd_col = first_existing_col(
        df,
        (
            "statistical_parity_difference",
            "statistical_parity_diff",
            "spd",
        ),
    )
    eod_col = first_existing_col(
        df,
        (
            "equal_opportunity_difference",
            "equal_opportunity_diff",
            "eod",
        ),
    )
    if not metric_cols and spd_col is None and eod_col is None:
        return defaults, None

    by_round = []
    for rnd, grp in df.groupby("round", sort=True):
        row = {"round": int(rnd)}
        if "accuracy" in grp.columns:
            acc = grp["accuracy"].dropna().to_numpy(dtype=float)
            if acc.size > 0:
                row["acc_mean"] = float(acc.mean())
                row["acc_std"] = float(acc.std(ddof=0))
                row["acc_gap"] = float(acc.max() - acc.min())
                row["acc_jain"] = jain_index(acc)
        if "macro_f1" in grp.columns:
            f1 = grp["macro_f1"].dropna().to_numpy(dtype=float)
            if f1.size > 0:
                row["macro_f1_mean"] = float(f1.mean())
                row["macro_f1_std"] = float(f1.std(ddof=0))
                row["macro_f1_gap"] = float(f1.max() - f1.min())
        if spd_col is not None:
            spd = grp[spd_col].dropna().to_numpy(dtype=float)
            if spd.size > 0:
                row["spd_mean"] = float(spd.mean())
                row["spd_std"] = float(spd.std(ddof=0))
                row["spd_gap"] = float(spd.max() - spd.min())
        if eod_col is not None:
            eod = grp[eod_col].dropna().to_numpy(dtype=float)
            if eod.size > 0:
                row["eod_mean"] = float(eod.mean())
                row["eod_std"] = float(eod.std(ddof=0))
                row["eod_gap"] = float(eod.max() - eod.min())
        by_round.append(row)

    fairness_by_round = pd.DataFrame(by_round).sort_values("round")
    if fairness_by_round.empty:
        return defaults, None

    last = fairness_by_round.iloc[-1]
    result = {
        "fairness_final_acc_mean": to_float_or_none(last.get("acc_mean")),
        "fairness_final_acc_std": to_float_or_none(last.get("acc_std")),
        "fairness_final_acc_gap": to_float_or_none(last.get("acc_gap")),
        "fairness_final_acc_jain": to_float_or_none(last.get("acc_jain")),
        "fairness_final_macro_f1_mean": to_float_or_none(last.get("macro_f1_mean")),
        "fairness_final_macro_f1_std": to_float_or_none(last.get("macro_f1_std")),
        "fairness_final_macro_f1_gap": to_float_or_none(last.get("macro_f1_gap")),
        "fairness_final_spd_mean": to_float_or_none(last.get("spd_mean")),
        "fairness_final_spd_std": to_float_or_none(last.get("spd_std")),
        "fairness_final_spd_gap": to_float_or_none(last.get("spd_gap")),
        "fairness_final_eod_mean": to_float_or_none(last.get("eod_mean")),
        "fairness_final_eod_std": to_float_or_none(last.get("eod_std")),
        "fairness_final_eod_gap": to_float_or_none(last.get("eod_gap")),
    }
    return result, fairness_by_round


def compute_client_quality_from_locals(
    run_dir: Path,
) -> tuple[dict[str, Optional[float]], Optional[pd.DataFrame]]:
    locals_path = run_dir / LOCALS_METRICS_FILE
    defaults = {
        "final_client_accuracy_mean": None,
        "final_client_macro_f1_mean": None,
        "final_client_micro_f1_mean": None,
    }
    if not locals_path.exists():
        return defaults, None

    df = pd.read_csv(locals_path)
    if df.empty or "round" not in df.columns:
        return defaults, None

    metric_cols = [c for c in QUALITY_METRICS if c in df.columns]
    if not metric_cols:
        return defaults, None

    by_round = []
    for rnd, grp in df.groupby("round", sort=True):
        row = {"round": int(rnd)}
        for metric in metric_cols:
            vals = grp[metric].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            row[f"client_{metric}_mean"] = float(vals.mean())
            row[f"client_{metric}_std"] = float(vals.std(ddof=0))
            row[f"client_{metric}_gap"] = float(vals.max() - vals.min())
        by_round.append(row)

    client_by_round = pd.DataFrame(by_round).sort_values("round")
    if client_by_round.empty:
        return defaults, None

    last = client_by_round.iloc[-1]
    result = {
        "final_client_accuracy_mean": to_float_or_none(last.get("client_accuracy_mean")),
        "final_client_macro_f1_mean": to_float_or_none(last.get("client_macro_f1_mean")),
        "final_client_micro_f1_mean": to_float_or_none(last.get("client_micro_f1_mean")),
    }
    return result, client_by_round


def read_round_loss(run_dir: Path) -> Optional[pd.DataFrame]:
    global_path = run_dir / GLOBAL_METRICS_FILE
    if global_path.exists():
        gdf = pd.read_csv(global_path)
        if "loss" in gdf.columns and not gdf.empty:
            if "round" not in gdf.columns:
                gdf["round"] = np.arange(1, len(gdf) + 1)
            out = gdf[["round", "loss"]].dropna()
            if not out.empty:
                out["round"] = out["round"].astype(int)
                return out.sort_values("round")

    sources = [
        (run_dir / "metrics.csv", ("train_loss", "loss", "fit_loss", "client_loss")),
        (run_dir / "local_test_metrics.csv", ("loss",)),
        (run_dir / "postfit_metrics.csv", ("loss",)),
        (run_dir / "prefit_metrics.csv", ("loss",)),
        (run_dir / LOCALS_METRICS_FILE, ("loss",)),
    ]
    for path, cols in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "round" not in df.columns:
            continue
        col = first_existing_col(df, cols)
        if col is None:
            continue
        sdf = df[["round", col] + (["client"] if "client" in df.columns else [])].dropna()
        if sdf.empty:
            continue
        if "client" in sdf.columns:
            sdf = sdf.groupby("round", as_index=False)[col].mean()
        else:
            sdf = sdf[["round", col]]
        sdf = sdf.rename(columns={col: "loss"})
        sdf["round"] = sdf["round"].astype(int)
        return sdf.sort_values("round")

    return None


def parse_run(
    run_dir: Path, runs_dir: Path
) -> tuple[RunRecord, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    global_path = choose_global_metrics_path(run_dir)
    gdf = pd.read_csv(global_path)
    if gdf.empty:
        raise ValueError(f"Empty metrics file: {global_path}")

    if "round" not in gdf.columns:
        gdf["round"] = np.arange(1, len(gdf) + 1)
    gdf = gdf.sort_values("round").reset_index(drop=True)

    if "loss" not in gdf.columns:
        loss_by_round = read_round_loss(run_dir)
        if loss_by_round is not None and not loss_by_round.empty:
            gdf = gdf.merge(loss_by_round, on="round", how="left")

    final = gdf.iloc[-1]
    final_round = int(final["round"])
    n_rounds = int(gdf["round"].nunique())

    final_accuracy = to_float_or_none(final["accuracy"]) if "accuracy" in gdf.columns else None
    final_macro_f1 = to_float_or_none(final["macro_f1"]) if "macro_f1" in gdf.columns else None
    final_micro_f1 = to_float_or_none(final["micro_f1"]) if "micro_f1" in gdf.columns else None

    best_round_by_macro_f1 = None
    best_macro_f1 = None
    if "macro_f1" in gdf.columns:
        best_idx = int(gdf["macro_f1"].astype(float).idxmax())
        best_row = gdf.loc[best_idx]
        best_round_by_macro_f1 = int(best_row["round"])
        best_macro_f1 = to_float_or_none(best_row["macro_f1"])

    client_quality, client_by_round = compute_client_quality_from_locals(run_dir)
    fairness, fairness_by_round = compute_fairness_from_locals(run_dir)

    run_time_seconds = read_run_time_seconds(run_dir)
    total_comm_cost = read_total_comm_cost(run_dir)
    comm_cost_per_round = (
        None if (total_comm_cost is None or n_rounds <= 0) else float(total_comm_cost / n_rounds)
    )
    run_time_per_round = (
        None if (run_time_seconds is None or n_rounds <= 0) else float(run_time_seconds / n_rounds)
    )

    name = run_dir.name
    noise = parse_float_tag(name, "noise")
    mgn = parse_float_tag(name, "mgn")
    if mgn is None:
        mgn = parse_float_tag(name, "maxgradnorm")

    run_label = build_run_label(run_dir, runs_dir)
    rec = RunRecord(
        run_label=run_label,
        run_path=run_dir,
        dataset=infer_dataset(name),
        model=infer_model(name),
        setting=infer_setting(name),
        is_dp=(noise is not None) or ("dp" in name.lower()),
        dp_noise_mul=noise,
        dp_max_grad_norm=mgn,
        n_rounds=n_rounds,
        final_round=final_round,
        final_accuracy=final_accuracy,
        final_macro_f1=final_macro_f1,
        final_micro_f1=final_micro_f1,
        best_round_by_macro_f1=best_round_by_macro_f1,
        best_macro_f1=best_macro_f1,
        final_client_accuracy_mean=client_quality["final_client_accuracy_mean"],
        final_client_macro_f1_mean=client_quality["final_client_macro_f1_mean"],
        final_client_micro_f1_mean=client_quality["final_client_micro_f1_mean"],
        fairness_final_acc_mean=fairness["fairness_final_acc_mean"],
        fairness_final_acc_std=fairness["fairness_final_acc_std"],
        fairness_final_acc_gap=fairness["fairness_final_acc_gap"],
        fairness_final_acc_jain=fairness["fairness_final_acc_jain"],
        fairness_final_macro_f1_mean=fairness["fairness_final_macro_f1_mean"],
        fairness_final_macro_f1_std=fairness["fairness_final_macro_f1_std"],
        fairness_final_macro_f1_gap=fairness["fairness_final_macro_f1_gap"],
        fairness_final_spd_mean=fairness["fairness_final_spd_mean"],
        fairness_final_spd_std=fairness["fairness_final_spd_std"],
        fairness_final_spd_gap=fairness["fairness_final_spd_gap"],
        fairness_final_eod_mean=fairness["fairness_final_eod_mean"],
        fairness_final_eod_std=fairness["fairness_final_eod_std"],
        fairness_final_eod_gap=fairness["fairness_final_eod_gap"],
        run_time_seconds=run_time_seconds,
        total_comm_cost=total_comm_cost,
        comm_cost_per_round=comm_cost_per_round,
        run_time_per_round=run_time_per_round,
    )
    return rec, gdf, fairness_by_round, client_by_round


def make_line_plot(
    out_path: Path,
    title: str,
    ylabel: str,
    series: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    if not series:
        return
    plt.figure(figsize=(11, 6))
    for label, x, y in series:
        plt.plot(x, y, marker="o", linewidth=1.6, markersize=3.2, label=label)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_bar_plot(
    out_path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    values: list[float],
) -> None:
    if not labels:
        return
    plt.figure(figsize=(11, 6))
    xs = np.arange(len(labels))
    plt.bar(xs, values, alpha=0.9)
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_noise_scatter(out_path: Path, title: str, ylabel: str, pairs: list[tuple[float, float, str]]) -> None:
    if len(pairs) < 2:
        return
    plt.figure(figsize=(8, 6))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    plt.scatter(xs, ys, s=52, alpha=0.9)
    for x, y, label in pairs:
        plt.annotate(label, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.title(title)
    plt.xlabel("dp_noise_mul")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def concat_drop_all_na(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cleaned = [df.loc[:, df.notna().any(axis=0)] for df in frames if not df.empty]
    if not cleaned:
        return pd.DataFrame()
    return pd.concat(cleaned, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quality/fairness/cost FL comparisons.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base runs directory (default: runs/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Output directory for generated CSV/PNG artifacts.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset substring filter (e.g. medical).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(runs_dir)
    if not run_dirs:
        print(f"No run directories found under {runs_dir}")
        return 1

    records: list[RunRecord] = []
    roundwise_rows: list[pd.DataFrame] = []
    fairness_rows: list[pd.DataFrame] = []
    client_roundwise_rows: list[pd.DataFrame] = []
    errors: list[str] = []

    for run_dir in run_dirs:
        if args.dataset and args.dataset.lower() not in run_dir.name.lower():
            continue
        try:
            record, gdf, fairness, client_by_round = parse_run(run_dir, runs_dir)
            records.append(record)

            local_global = gdf.copy()
            local_global["run_label"] = record.run_label
            local_global["dataset"] = record.dataset
            local_global["model"] = record.model
            local_global["setting"] = record.setting
            local_global["dp_noise_mul"] = record.dp_noise_mul
            roundwise_rows.append(local_global)

            if fairness is not None and not fairness.empty:
                local_fair = fairness.copy()
                local_fair["run_label"] = record.run_label
                local_fair["dataset"] = record.dataset
                local_fair["model"] = record.model
                local_fair["setting"] = record.setting
                local_fair["dp_noise_mul"] = record.dp_noise_mul
                fairness_rows.append(local_fair)

            if client_by_round is not None and not client_by_round.empty:
                local_client = client_by_round.copy()
                local_client["run_label"] = record.run_label
                local_client["dataset"] = record.dataset
                local_client["model"] = record.model
                local_client["setting"] = record.setting
                local_client["dp_noise_mul"] = record.dp_noise_mul
                client_roundwise_rows.append(local_client)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{run_dir}: {exc}")

    if not records:
        print("No runs parsed successfully.")
        if errors:
            print("\nParse errors:")
            for err in errors:
                print(f"- {err}")
        return 1

    summary_df = pd.DataFrame([r.__dict__ for r in records]).sort_values(
        by=["dataset", "model", "setting", "dp_noise_mul", "run_label"],
        na_position="last",
    )
    summary_csv = out_dir / "run_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    roundwise_df = concat_drop_all_na(roundwise_rows)
    roundwise_csv = out_dir / "roundwise_global_metrics.csv"
    roundwise_df.to_csv(roundwise_csv, index=False)

    fairness_df = concat_drop_all_na(fairness_rows)
    fairness_csv = out_dir / "roundwise_fairness_metrics.csv"
    fairness_df.to_csv(fairness_csv, index=False)

    client_roundwise_df = concat_drop_all_na(client_roundwise_rows)
    client_roundwise_csv = out_dir / "roundwise_client_metrics.csv"
    client_roundwise_df.to_csv(client_roundwise_csv, index=False)

    # Quality plot: macro_f1 if available, else accuracy.
    quality_series = []
    for r in records:
        run_slice = roundwise_df[roundwise_df["run_label"] == r.run_label]
        if run_slice.empty:
            continue
        y_col = "macro_f1" if "macro_f1" in run_slice.columns else ("accuracy" if "accuracy" in run_slice.columns else None)
        if y_col is None:
            continue
        s = run_slice.dropna(subset=["round", y_col]).sort_values("round")
        if s.empty:
            continue
        quality_series.append((r.run_label, s["round"].to_numpy(dtype=float), s[y_col].to_numpy(dtype=float)))
    make_line_plot(
        out_dir / "quality_metric_vs_round.png",
        "Quality vs Round",
        "metric value",
        quality_series,
    )

    # Utility plot: loss by round (server loss if available, otherwise train/client mean loss).
    loss_series = []
    for r in records:
        run_slice = roundwise_df[roundwise_df["run_label"] == r.run_label]
        if run_slice.empty or "loss" not in run_slice.columns:
            continue
        s = run_slice.dropna(subset=["round", "loss"]).sort_values("round")
        if s.empty:
            continue
        loss_series.append(
            (r.run_label, s["round"].to_numpy(dtype=float), s["loss"].to_numpy(dtype=float))
        )
    make_line_plot(
        out_dir / "utility_loss_vs_round.png",
        "Loss vs Round",
        "loss",
        loss_series,
    )

    # Client quality plot: average across clients by round.
    client_quality_series = []
    if not client_roundwise_df.empty:
        for r in records:
            s = client_roundwise_df[client_roundwise_df["run_label"] == r.run_label]
            if s.empty:
                continue
            y_col = (
                "client_macro_f1_mean"
                if "client_macro_f1_mean" in s.columns
                else ("client_accuracy_mean" if "client_accuracy_mean" in s.columns else None)
            )
            if y_col is None:
                continue
            s = s.dropna(subset=["round", y_col]).sort_values("round")
            if s.empty:
                continue
            client_quality_series.append(
                (
                    r.run_label,
                    s["round"].to_numpy(dtype=float),
                    s[y_col].to_numpy(dtype=float),
                )
            )
    make_line_plot(
        out_dir / "quality_client_mean_vs_round.png",
        "Client Mean Quality vs Round",
        "client mean metric value",
        client_quality_series,
    )

    # Overlay plot: server metric vs client mean metric.
    server_client_series = []
    if not client_roundwise_df.empty and not roundwise_df.empty:
        for r in records:
            s_server = roundwise_df[roundwise_df["run_label"] == r.run_label]
            s_client = client_roundwise_df[client_roundwise_df["run_label"] == r.run_label]
            if s_server.empty or s_client.empty:
                continue
            if "macro_f1" in s_server.columns and "client_macro_f1_mean" in s_client.columns:
                s_server = s_server.dropna(subset=["round", "macro_f1"]).sort_values("round")
                s_client = s_client.dropna(subset=["round", "client_macro_f1_mean"]).sort_values(
                    "round"
                )
                if not s_server.empty:
                    server_client_series.append(
                        (
                            f"{r.run_label} [server]",
                            s_server["round"].to_numpy(dtype=float),
                            s_server["macro_f1"].to_numpy(dtype=float),
                        )
                    )
                if not s_client.empty:
                    server_client_series.append(
                        (
                            f"{r.run_label} [clients-mean]",
                            s_client["round"].to_numpy(dtype=float),
                            s_client["client_macro_f1_mean"].to_numpy(dtype=float),
                        )
                    )
            elif "accuracy" in s_server.columns and "client_accuracy_mean" in s_client.columns:
                s_server = s_server.dropna(subset=["round", "accuracy"]).sort_values("round")
                s_client = s_client.dropna(subset=["round", "client_accuracy_mean"]).sort_values(
                    "round"
                )
                if not s_server.empty:
                    server_client_series.append(
                        (
                            f"{r.run_label} [server]",
                            s_server["round"].to_numpy(dtype=float),
                            s_server["accuracy"].to_numpy(dtype=float),
                        )
                    )
                if not s_client.empty:
                    server_client_series.append(
                        (
                            f"{r.run_label} [clients-mean]",
                            s_client["round"].to_numpy(dtype=float),
                            s_client["client_accuracy_mean"].to_numpy(dtype=float),
                        )
                    )
    make_line_plot(
        out_dir / "quality_server_vs_clients_mean_vs_round.png",
        "Server vs Clients-Mean Quality vs Round",
        "metric value",
        server_client_series,
    )

    # Fairness plot: accuracy gap across clients by round.
    fairness_series = []
    if not fairness_df.empty and "acc_gap" in fairness_df.columns:
        for r in records:
            s = fairness_df[fairness_df["run_label"] == r.run_label].dropna(subset=["round", "acc_gap"]).sort_values("round")
            if s.empty:
                continue
            fairness_series.append((r.run_label, s["round"].to_numpy(dtype=float), s["acc_gap"].to_numpy(dtype=float)))
    make_line_plot(
        out_dir / "fairness_acc_gap_vs_round.png",
        "Fairness (Client Accuracy Gap) vs Round",
        "max(client_acc) - min(client_acc)",
        fairness_series,
    )

    spd_series = []
    if not fairness_df.empty and "spd_mean" in fairness_df.columns:
        for r in records:
            s = (
                fairness_df[fairness_df["run_label"] == r.run_label]
                .dropna(subset=["round", "spd_mean"])
                .sort_values("round")
            )
            if s.empty:
                continue
            spd_series.append(
                (r.run_label, s["round"].to_numpy(dtype=float), s["spd_mean"].to_numpy(dtype=float))
            )
    make_line_plot(
        out_dir / "fairness_spd_vs_round.png",
        "Statistical Parity Difference vs Round",
        "SPD (mean across clients)",
        spd_series,
    )

    eod_series = []
    if not fairness_df.empty and "eod_mean" in fairness_df.columns:
        for r in records:
            s = (
                fairness_df[fairness_df["run_label"] == r.run_label]
                .dropna(subset=["round", "eod_mean"])
                .sort_values("round")
            )
            if s.empty:
                continue
            eod_series.append(
                (r.run_label, s["round"].to_numpy(dtype=float), s["eod_mean"].to_numpy(dtype=float))
            )
    make_line_plot(
        out_dir / "fairness_eod_vs_round.png",
        "Equal Opportunity Difference vs Round",
        "EOD (mean across clients)",
        eod_series,
    )

    # Bar charts for final values.
    labels = summary_df["run_label"].tolist()
    if "final_macro_f1" in summary_df.columns:
        vals = [v for v in summary_df["final_macro_f1"].tolist()]
        if any(pd.notna(v) for v in vals):
            make_bar_plot(
                out_dir / "final_macro_f1_bar.png",
                "Final Macro-F1 by Run",
                "final_macro_f1",
                labels,
                [0.0 if pd.isna(v) else float(v) for v in vals],
            )

    if "fairness_final_acc_gap" in summary_df.columns and any(pd.notna(v) for v in summary_df["fairness_final_acc_gap"].tolist()):
        make_bar_plot(
            out_dir / "final_fairness_acc_gap_bar.png",
            "Final Fairness Gap (Accuracy) by Run",
            "final fairness acc gap",
            labels,
            [0.0 if pd.isna(v) else float(v) for v in summary_df["fairness_final_acc_gap"].tolist()],
        )

    if "fairness_final_spd_mean" in summary_df.columns and any(pd.notna(v) for v in summary_df["fairness_final_spd_mean"].tolist()):
        make_bar_plot(
            out_dir / "final_fairness_spd_bar.png",
            "Final Statistical Parity Difference by Run",
            "final SPD",
            labels,
            [0.0 if pd.isna(v) else float(v) for v in summary_df["fairness_final_spd_mean"].tolist()],
        )

    if "fairness_final_eod_mean" in summary_df.columns and any(pd.notna(v) for v in summary_df["fairness_final_eod_mean"].tolist()):
        make_bar_plot(
            out_dir / "final_fairness_eod_bar.png",
            "Final Equal Opportunity Difference by Run",
            "final EOD",
            labels,
            [0.0 if pd.isna(v) else float(v) for v in summary_df["fairness_final_eod_mean"].tolist()],
        )

    runtime_vals = summary_df["run_time_seconds"].tolist()
    if any(pd.notna(v) for v in runtime_vals):
        make_bar_plot(
            out_dir / "run_time_seconds_bar.png",
            "Run Time by Run",
            "run_time_seconds",
            labels,
            [0.0 if pd.isna(v) else float(v) for v in runtime_vals],
        )

    comm_vals = summary_df["total_comm_cost"].tolist()
    if any(pd.notna(v) for v in comm_vals):
        make_bar_plot(
            out_dir / "total_comm_cost_bar.png",
            "Total Communication Cost by Run",
            "total_comm_cost",
            labels,
            [0.0 if pd.isna(v) else float(v) for v in comm_vals],
        )

    # DP trade-off scatter plots.
    noise_quality_pairs = []
    noise_fair_pairs = []
    noise_cost_pairs = []
    for _, row in summary_df.iterrows():
        noise = to_float_or_none(row.get("dp_noise_mul"))
        if noise is None:
            continue
        label = str(row["run_label"])
        q = to_float_or_none(row.get("final_macro_f1"))
        fgap = to_float_or_none(row.get("fairness_final_acc_gap"))
        cost = to_float_or_none(row.get("run_time_seconds"))
        if q is not None:
            noise_quality_pairs.append((noise, q, label))
        if fgap is not None:
            noise_fair_pairs.append((noise, fgap, label))
        if cost is not None:
            noise_cost_pairs.append((noise, cost, label))

    make_noise_scatter(
        out_dir / "dp_noise_vs_quality_macro_f1.png",
        "DP Noise vs Final Macro-F1",
        "final_macro_f1",
        noise_quality_pairs,
    )
    make_noise_scatter(
        out_dir / "dp_noise_vs_fairness_acc_gap.png",
        "DP Noise vs Final Fairness Gap (Accuracy)",
        "final fairness acc gap",
        noise_fair_pairs,
    )
    make_noise_scatter(
        out_dir / "dp_noise_vs_runtime.png",
        "DP Noise vs Run Time",
        "run_time_seconds",
        noise_cost_pairs,
    )

    print(f"Parsed runs: {len(records)}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {roundwise_csv}")
    print(f"Wrote: {fairness_csv}")
    print(f"Wrote: {client_roundwise_csv}")
    print(f"Wrote plots in: {out_dir}")

    if errors:
        print("\nSkipped runs:")
        for err in errors:
            print(f"- {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
