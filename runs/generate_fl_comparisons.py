"""
Generate FL comparison artifacts focused on:
- predictive metrics (Macro-F1/Micro-F1/Accuracy),
- fairness (SPD/EOD),
- cost (runtime + communication),
- privacy tags (DP epsilon inferred from run naming).

Expected run folders are under `runs/` and contain at least one of:
- `global_metrics.csv` (federated/decentralized),
- `metrics.csv` (centralized).

Recommended DP run naming:
- `...-eps-1.5...` or `..._epsilon_1.5...`
- `...-mgn-1.0...` for max_grad_norm (optional)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLOBAL_METRICS_FILE = "global_metrics.csv"
CENTRALIZED_METRICS_FILE = "metrics.csv"
LOCALS_METRICS_FILE = "locals_metrics.csv"
RUN_METRICS_FILE = "run_metrics.csv"
COMM_COSTS_FILE = "comm_costs.csv"

QUALITY_METRICS = ("macro_f1", "micro_f1", "accuracy")
SERVER_CLIENT_QUALITY_METRIC_PAIRS = (
    ("macro_f1", "client_macro_f1_mean"),
    ("micro_f1", "client_micro_f1_mean"),
    ("accuracy", "client_accuracy_mean"),
)
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
KNOWN_MODELS = ("vfl", "svm", "logreg", "mlp", "2nn", "cnn", "resnet", "lstm")

LINE_FIGSIZE = (11, 6)
BAR_FIGSIZE = (11, 6)
SCATTER_FIGSIZE = (8, 6)
LINE_MARGINS = {"left": 0.11, "right": 0.97, "bottom": 0.13, "top": 0.90}
BAR_MARGINS = {"left": 0.11, "right": 0.97, "bottom": 0.27, "top": 0.90}
SCATTER_MARGINS = {"left": 0.14, "right": 0.97, "bottom": 0.13, "top": 0.90}

PROGRESSION_SETTING_META = {
    "horizontal-iid": {"label": "Horizontal-IID", "dir": "Horizontal-IID"},
    "horizontal-non-iid": {"label": "Horizontal-NonIID", "dir": "Horizontal-NonIID"},
    "vertical-disjoint": {"label": "Vertical-Disjoint", "dir": "Vertical-Disjoint"},
    "vertical-overlap": {"label": "Vertical-Overlap", "dir": "Vertical-Overlap"},
}
PROGRESSION_SETTING_COLORS = {
    "horizontal-iid": "#1f77b4",
    "horizontal-non-iid": "#ff7f0e",
    "vertical-disjoint": "#2ca02c",
    "vertical-overlap": "#d62728",
}
PROGRESSION_COMPARISON_GROUPS = {
    "Vertical-Comparison": {
        "title": "Vertical Comparison",
        "settings": ["vertical-disjoint", "vertical-overlap"],
    },
}
PROGRESSION_METRICS = {
    "accuracy": {"plot_label": "Accuracy", "file_label": "Accuracy"},
    "macro_f1": {"plot_label": "F1 score (macro)", "file_label": "F1 score"},
    "macro_precision": {"plot_label": "Precision (macro)", "file_label": "Precision"},
    "macro_recall": {"plot_label": "Recall (macro)", "file_label": "Recall"},
    "loss": {"plot_label": "Loss", "file_label": "Loss"},
}
PROGRESSION_BASELINE_KEY = "baseline"
PROGRESSION_RUN_KEY_PREFIX = "run:"


def _new_axes(figsize: tuple[float, float], margins: dict[str, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(**margins)
    return fig, ax


@dataclass
class RunRecord:
    run_label: str
    run_path: Path
    dataset: str
    model: str
    setting: str
    is_dp: bool
    dp_epsilon: Optional[float]
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


def pretty_metric_name(metric: str) -> str:
    mapping = {
        "macro_f1": "Macro-F1",
        "micro_f1": "Micro-F1",
        "accuracy": "Accuracy",
        "client_macro_f1_mean": "Client Mean Macro-F1",
        "client_micro_f1_mean": "Client Mean Micro-F1",
        "client_accuracy_mean": "Client Mean Accuracy",
    }
    return mapping.get(metric, metric)


def choose_metric_for_df(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
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
    if "vertical" in lower:
        if "overlap" in lower or "shared" in lower:
            return "vertical-overlap"
        if "disjoint" in lower or "without-overlap" in lower:
            return "vertical-disjoint"
        if "non-iid" in lower or "noniid" in lower or ("non" in tokens and "iid" in tokens):
            return "vertical-overlap"
        if "iid" in tokens:
            return "vertical-disjoint"
        return "vertical"
    if "non" in tokens and "iid" in tokens:
        return "non-iid"
    if "noniid" in lower:
        return "non-iid"
    if "iid" in tokens:
        return "iid"
    return "federated"


def parse_float_tag(name: str, key: str) -> Optional[float]:
    # Examples captured:
    # - eps-1.5
    # - eps_8.0
    # - epsilon-8p0
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
        (run_dir / "metrics.csv", ("train_loss", "vfl/train_loss", "loss", "fit_loss", "client_loss")),
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
    epsilon = parse_float_tag(name, "epsilon")
    if epsilon is None:
        epsilon = parse_float_tag(name, "eps")
    if epsilon is None:
        epsilon = parse_float_tag(name, "noise")
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
        is_dp=(epsilon is not None) or ("dp" in name.lower()),
        dp_epsilon=epsilon,
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
    fig, ax = _new_axes(LINE_FIGSIZE, LINE_MARGINS)
    for label, x, y in series:
        ax.plot(x, y, marker="o", linewidth=1.6, markersize=3.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_bar_plot(
    out_path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    values: list[float],
    yscale: str = "linear",
) -> None:
    if not labels:
        return
    fig, ax = _new_axes(BAR_FIGSIZE, BAR_MARGINS)
    xs = np.arange(len(labels))
    ax.bar(xs, values, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_grouped_bar_plot(
    out_path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    series: list[tuple[str, list[float]]],
) -> None:
    if not labels or not series:
        return
    fig, ax = _new_axes(BAR_FIGSIZE, BAR_MARGINS)
    xs = np.arange(len(labels))
    width = 0.35 if len(series) == 2 else 0.8 / max(1, len(series))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    for idx, (series_label, values) in enumerate(series):
        ax.bar(xs + offsets[idx], values, width=width, alpha=0.9, label=series_label)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_epsilon_scatter(out_path: Path, title: str, ylabel: str, pairs: list[tuple[float, float, str]]) -> None:
    if len(pairs) < 2:
        return
    fig, ax = _new_axes(SCATTER_FIGSIZE, SCATTER_MARGINS)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    ax.scatter(xs, ys, s=52, alpha=0.9)
    for x, y, label in pairs:
        ax.annotate(label, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("dp_epsilon")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def concat_drop_all_na(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cleaned = [df.loc[:, df.notna().any(axis=0)] for df in frames if not df.empty]
    if not cleaned:
        return pd.DataFrame()
    return pd.concat(cleaned, ignore_index=True)


def progression_parse_privacy_level(name: str) -> Optional[float]:
    patterns = (
        r"epsilon[._-]?([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)",
        r"eps[._-]?([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)",
        r"noise[._-]?([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, name.lower())
        if not match:
            continue
        try:
            return float(match.group(1).replace("p", "."))
        except ValueError:
            return None
    return None


def progression_parse_setting(name: str) -> Optional[str]:
    lower = name.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", lower) if t]
    if "vertical" in lower:
        if "overlap" in lower or "shared" in lower:
            return "vertical-overlap"
        if "disjoint" in lower or "without-overlap" in lower:
            return "vertical-disjoint"
        if "non-iid" in lower or "noniid" in lower or ("non" in tokens and "iid" in tokens):
            return "vertical-overlap"
        if "iid" in tokens:
            return "vertical-disjoint"
        return None
    if "non-iid" in lower or "noniid" in lower or ("non" in tokens and "iid" in tokens):
        return "horizontal-non-iid"
    if "iid" in tokens:
        return "horizontal-iid"
    return None


def canonical_vertical_run_label(name: str) -> Optional[str]:
    lower = name.lower()
    if "vertical-overlap" in lower:
        setting = "vertical-overlap"
    elif "vertical-disjoint" in lower:
        setting = "vertical-disjoint"
    else:
        return None
    if "medical" not in lower:
        return None
    patterns = (
        r"(?:^|[^0-9])(\d+)[-_]?clients?(?:[^a-z0-9]|$)",
        r"clients?[-_ ]?(\d+)",
        r"n[-_ ]?clients?[-_ ]?(\d+)",
    )
    client_count = None
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            client_count = int(match.group(1))
            break
    if client_count is None:
        return None
    return f"medical-vfl-{setting}-{client_count}-clients"


def progression_infer_privacy_label(name: str) -> Optional[str]:
    lower = name.lower()
    if "epsilon" in lower or "eps" in lower:
        return "epsilon"
    if "noise" in lower:
        return "noise"
    return None


def progression_discover_privacy_levels(runs_dir: Path, dataset_filter: str) -> tuple[list[float], str]:
    levels: set[float] = set()
    labels: set[str] = set()
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name.lower()
        if dataset_filter and dataset_filter.lower() not in name:
            continue
        level = progression_parse_privacy_level(name)
        if level is None:
            continue
        levels.add(level)
        label = progression_infer_privacy_label(name)
        if label is not None:
            labels.add(label)
    privacy_label = labels.pop() if len(labels) == 1 else "privacy"
    return sorted(levels), privacy_label


def progression_load_round_metric_series(run_dir: Path, metric_col: str) -> Optional[pd.DataFrame]:
    gpath = run_dir / GLOBAL_METRICS_FILE
    if gpath.exists():
        gdf = pd.read_csv(gpath)
        if metric_col in gdf.columns:
            if "round" not in gdf.columns:
                gdf["round"] = np.arange(1, len(gdf) + 1)
            sdf = gdf[["round", metric_col]].dropna().copy()
            if not sdf.empty:
                sdf["round"] = sdf["round"].astype(int)
                return sdf.sort_values("round")
    if metric_col != "loss":
        return None
    loss_by_round = read_round_loss(run_dir)
    if loss_by_round is None or loss_by_round.empty:
        return None
    return loss_by_round[["round", "loss"]].rename(columns={"loss": metric_col}).sort_values("round")


def progression_is_close_to_any(value: float, targets: list[float], tol: float = 1e-9) -> bool:
    return any(math.isclose(value, target, abs_tol=tol) for target in targets)


def progression_aggregate_runs_for_metric(
    runs_dir: Path,
    metric_col: str,
    privacy_levels: list[float],
    dataset_filter: str,
) -> dict[tuple[str, str], pd.DataFrame]:
    series_by_key: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name.lower()
        if dataset_filter and dataset_filter.lower() not in name:
            continue
        setting = progression_parse_setting(name)
        if setting is None:
            continue
        privacy_level = progression_parse_privacy_level(name)
        if privacy_level is None:
            series_key = (
                f"{PROGRESSION_RUN_KEY_PREFIX}{run_dir.name}"
                if setting.startswith("vertical")
                else PROGRESSION_BASELINE_KEY
            )
        else:
            if privacy_levels and not progression_is_close_to_any(privacy_level, privacy_levels):
                continue
            series_key = f"{privacy_level:g}"
        sdf = progression_load_round_metric_series(run_dir, metric_col)
        if sdf is None or sdf.empty:
            continue
        series_by_key.setdefault((setting, series_key), []).append(sdf)
    aggregated: dict[tuple[str, str], pd.DataFrame] = {}
    for key, frames in series_by_key.items():
        merged = pd.concat(frames, ignore_index=True)
        aggregated[key] = merged.groupby("round", as_index=False)[metric_col].mean().sort_values("round")
    return aggregated


def progression_ordered_series_keys(aggregated: dict[tuple[str, str], pd.DataFrame], setting: str) -> list[str]:
    keys = {series_key for current_setting, series_key in aggregated if current_setting == setting}
    ordered: list[str] = []
    if PROGRESSION_BASELINE_KEY in keys:
        ordered.append(PROGRESSION_BASELINE_KEY)
    run_keys = sorted(k for k in keys if k.startswith(PROGRESSION_RUN_KEY_PREFIX))
    ordered.extend(run_keys)
    numeric_keys = sorted((k for k in keys if k not in ordered), key=float)
    ordered.extend(numeric_keys)
    return ordered


def progression_series_display_label(series_key: str, privacy_label: str) -> str:
    if series_key == PROGRESSION_BASELINE_KEY:
        return "baseline"
    if series_key.startswith(PROGRESSION_RUN_KEY_PREFIX):
        run_name = series_key[len(PROGRESSION_RUN_KEY_PREFIX):]
        return canonical_vertical_run_label(run_name) or run_name
    return f"{privacy_label}={series_key}"


def progression_comparison_series_label(setting: str, series_key: str, privacy_label: str) -> str:
    setting_label = PROGRESSION_SETTING_META[setting]["label"]
    if series_key == PROGRESSION_BASELINE_KEY:
        return setting_label
    return f"{setting_label} | {progression_series_display_label(series_key, privacy_label)}"


def progression_comparison_style_map(
    aggregated: dict[tuple[str, str], pd.DataFrame], settings: list[str]
) -> dict[tuple[str, str], tuple[object, str]]:
    style_cycle = ["-", "--", ":", "-."]
    style_map: dict[tuple[str, str], tuple[object, str]] = {}
    for setting in settings:
        keys = progression_ordered_series_keys(aggregated, setting)
        if not keys:
            continue
        cmap = plt.cm.Greens if setting == "vertical-disjoint" else plt.cm.Reds
        n_keys = max(1, len(keys))
        for idx, series_key in enumerate(keys):
            if series_key == PROGRESSION_BASELINE_KEY:
                color = "#444444"
            else:
                denom = max(1, n_keys - 1)
                color = cmap(0.45 + 0.45 * (idx / denom))
            linestyle = style_cycle[idx % len(style_cycle)]
            style_map[(setting, series_key)] = (color, linestyle)
    return style_map


def progression_plot_metric_for_setting(
    out_path: Path,
    title: str,
    y_label: str,
    metric_col: str,
    aggregated: dict[tuple[str, str], pd.DataFrame],
    setting: str,
    privacy_label: str,
) -> bool:
    if not aggregated:
        return False
    keys = progression_ordered_series_keys(aggregated, setting)
    if not keys:
        return False
    colors: dict[str, object] = {}
    run_keys = [k for k in keys if k.startswith(PROGRESSION_RUN_KEY_PREFIX)]
    numeric_keys = [k for k in keys if k not in run_keys and k != PROGRESSION_BASELINE_KEY]
    for idx, series_key in enumerate(run_keys):
        colors[series_key] = plt.cm.tab10(idx % 10)
    for idx, series_key in enumerate(numeric_keys):
        colors[series_key] = plt.cm.viridis(idx / max(1, len(numeric_keys) - 1))
    fig, ax = _new_axes(LINE_FIGSIZE, LINE_MARGINS)
    plotted = 0
    for series_key in keys:
        key = (setting, series_key)
        if key not in aggregated:
            continue
        series_df = aggregated[key]
        if series_df.empty:
            continue
        color = "#444444" if series_key == PROGRESSION_BASELINE_KEY else colors[series_key]
        linestyle = "--" if series_key == PROGRESSION_BASELINE_KEY else "-"
        ax.plot(
            series_df["round"].to_numpy(dtype=float),
            series_df[metric_col].to_numpy(dtype=float),
            label=progression_series_display_label(series_key, privacy_label),
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("FL round")
    ax.set_ylabel(y_label)
    if metric_col != "loss":
        ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def progression_plot_metric_for_comparison_group(
    out_path: Path,
    title: str,
    y_label: str,
    metric_col: str,
    aggregated: dict[tuple[str, str], pd.DataFrame],
    settings: list[str],
    privacy_label: str,
) -> bool:
    series: list[tuple[str, str, pd.DataFrame]] = []
    for setting in settings:
        for series_key in progression_ordered_series_keys(aggregated, setting):
            key = (setting, series_key)
            if key not in aggregated:
                continue
            series_df = aggregated[key]
            if series_df.empty:
                continue
            series.append((setting, series_key, series_df))
    if not series:
        return False
    fig, ax = _new_axes(LINE_FIGSIZE, LINE_MARGINS)
    style_map = progression_comparison_style_map(aggregated, settings)
    plotted = 0
    for idx, (setting, series_key, series_df) in enumerate(series):
        color, linestyle = style_map.get(
            (setting, series_key),
            (PROGRESSION_SETTING_COLORS.get(setting, plt.cm.tab10(idx % 10)), "-"),
        )
        ax.plot(
            series_df["round"].to_numpy(dtype=float),
            series_df[metric_col].to_numpy(dtype=float),
            label=progression_comparison_series_label(setting, series_key, privacy_label),
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("FL round")
    ax.set_ylabel(y_label)
    if metric_col != "loss":
        ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def generate_progression_artifacts(
    runs_dir: Path,
    out_dir: Path,
    dataset: str,
    epsilon_levels: Optional[list[float]],
    noise_levels: Optional[list[float]],
) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    if noise_levels is not None:
        privacy_levels = noise_levels
        privacy_label = "noise"
    elif epsilon_levels is not None:
        privacy_levels = epsilon_levels
        privacy_label = "epsilon"
    else:
        privacy_levels, privacy_label = progression_discover_privacy_levels(runs_dir, dataset)
        if privacy_levels:
            print(f"Inferred {privacy_label} levels: {', '.join(str(v) for v in privacy_levels)}")
        else:
            print("No privacy levels were inferred from the available run folders. Plotting baseline runs only where available.")
    wrote_any = False
    roundwise_rows: list[pd.DataFrame] = []
    for metric_col, metric_meta in PROGRESSION_METRICS.items():
        aggregated = progression_aggregate_runs_for_metric(runs_dir, metric_col, privacy_levels, dataset)
        if not aggregated:
            print(f"Skip progression {metric_col}: no matching runs or metric column.")
            continue
        for (setting, series_key), df in aggregated.items():
            tmp = df.copy()
            tmp["metric"] = metric_col
            tmp["setting"] = setting
            tmp["series_key"] = series_key
            tmp["privacy_level"] = (
                np.nan
                if series_key == PROGRESSION_BASELINE_KEY or series_key.startswith(PROGRESSION_RUN_KEY_PREFIX)
                else float(series_key)
            )
            tmp["series_label"] = progression_series_display_label(series_key, privacy_label)
            tmp["privacy_label"] = (
                "baseline"
                if series_key == PROGRESSION_BASELINE_KEY or series_key.startswith(PROGRESSION_RUN_KEY_PREFIX)
                else privacy_label
            )
            roundwise_rows.append(tmp)
        for setting, meta in PROGRESSION_SETTING_META.items():
            out_path = out_dir / meta["dir"] / f"{meta['label']} - {metric_meta['file_label']}.png"
            if progression_plot_metric_for_setting(out_path, f"{meta['label']} | {metric_meta['plot_label']} ({dataset})", metric_meta['plot_label'], metric_col, aggregated, setting, privacy_label):
                wrote_any = True
                print(f"Wrote progression plot: {out_path}")
        for group_dir, group_meta in PROGRESSION_COMPARISON_GROUPS.items():
            out_path = out_dir / group_dir / f"{group_meta['title']} - {metric_meta['file_label']}.png"
            if progression_plot_metric_for_comparison_group(out_path, f"{group_meta['title']} | {metric_meta['plot_label']} ({dataset})", metric_meta['plot_label'], metric_col, aggregated, group_meta['settings'], privacy_label):
                wrote_any = True
                print(f"Wrote progression plot: {out_path}")
    if roundwise_rows:
        out_csv = out_dir / "roundwise_setting_privacy_metrics.csv"
        pd.concat(roundwise_rows, ignore_index=True).to_csv(out_csv, index=False)
        print(f"Wrote progression CSV: {out_csv}")
    return wrote_any


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictive-metric/fairness/cost FL comparisons.")
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
        help="Output directory for summary comparison artifacts.",
    )
    parser.add_argument(
        "--progression-out-dir",
        type=Path,
        default=None,
        help="Optional output directory for progression plots. Defaults to <out-dir>/iid_noise_progression.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset substring filter (e.g. medical).",
    )
    parser.add_argument(
        "--epsilon-levels",
        type=float,
        nargs="+",
        default=None,
        help="Optional epsilon levels to include in progression plots.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Legacy alias for old run names using noise levels in progression plots.",
    )
    parser.add_argument(
        "--skip-progression",
        action="store_true",
        help="Generate only the summary comparison artifacts and skip the progression plots.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progression_out_dir = (args.progression_out_dir.resolve() if args.progression_out_dir is not None else out_dir / "iid_noise_progression")

    # Remove deprecated artifacts so old non-SPD/EOD fairness and old naming do not linger.
    stale_plot_files = (
        "fairness_acc_gap_vs_round.png",
        "final_fairness_acc_gap_bar.png",
        "dp_epsilon_vs_fairness_acc_gap.png",
        "quality_metric_vs_round.png",
        "quality_client_mean_vs_round.png",
        "quality_server_vs_clients_mean_vs_round.png",
        "dp_epsilon_vs_quality_macro_f1.png",
    )
    for stale_name in stale_plot_files:
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

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
            local_global["dp_epsilon"] = record.dp_epsilon
            roundwise_rows.append(local_global)

            if fairness is not None and not fairness.empty:
                local_fair = fairness.copy()
                local_fair["run_label"] = record.run_label
                local_fair["dataset"] = record.dataset
                local_fair["model"] = record.model
                local_fair["setting"] = record.setting
                local_fair["dp_epsilon"] = record.dp_epsilon
                fairness_rows.append(local_fair)

            if client_by_round is not None and not client_by_round.empty:
                local_client = client_by_round.copy()
                local_client["run_label"] = record.run_label
                local_client["dataset"] = record.dataset
                local_client["model"] = record.model
                local_client["setting"] = record.setting
                local_client["dp_epsilon"] = record.dp_epsilon
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
        by=["dataset", "model", "setting", "dp_epsilon", "run_label"],
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

    # Predictive metric plot: Macro-F1 first, then Micro-F1, then Accuracy (per run fallback).
    quality_series = []
    quality_metrics_used: set[str] = set()
    for r in records:
        run_slice = roundwise_df[roundwise_df["run_label"] == r.run_label]
        if run_slice.empty:
            continue
        y_col = choose_metric_for_df(run_slice, QUALITY_METRICS)
        if y_col is None:
            continue
        s = run_slice.dropna(subset=["round", y_col]).sort_values("round")
        if s.empty:
            continue
        quality_metrics_used.add(y_col)
        quality_series.append(
            (
                f"{r.run_label} [{pretty_metric_name(y_col)}]",
                s["round"].to_numpy(dtype=float),
                s[y_col].to_numpy(dtype=float),
            )
        )
    if len(quality_metrics_used) == 1:
        only_metric = next(iter(quality_metrics_used))
        quality_title = f"{pretty_metric_name(only_metric)} vs Round"
        quality_ylabel = pretty_metric_name(only_metric)
    else:
        quality_title = "Predictive Metric vs Round (mixed by run)"
        quality_ylabel = "metric value"
    make_line_plot(
        out_dir / "predictive_metric_vs_round.png",
        quality_title,
        quality_ylabel,
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

    # Client mean predictive metric by round (same fallback order per run).
    client_quality_series = []
    client_metrics_used: set[str] = set()
    if not client_roundwise_df.empty:
        for r in records:
            s = client_roundwise_df[client_roundwise_df["run_label"] == r.run_label]
            if s.empty:
                continue
            y_col = choose_metric_for_df(
                s,
                ("client_macro_f1_mean", "client_micro_f1_mean", "client_accuracy_mean"),
            )
            if y_col is None:
                continue
            s = s.dropna(subset=["round", y_col]).sort_values("round")
            if s.empty:
                continue
            client_metrics_used.add(y_col)
            client_quality_series.append(
                (
                    f"{r.run_label} [{pretty_metric_name(y_col)}]",
                    s["round"].to_numpy(dtype=float),
                    s[y_col].to_numpy(dtype=float),
                )
            )
    if len(client_metrics_used) == 1:
        only_metric = next(iter(client_metrics_used))
        client_title = f"{pretty_metric_name(only_metric)} by Round (Clients Mean)"
        client_ylabel = pretty_metric_name(only_metric)
    else:
        client_title = "Clients Mean Predictive Metric vs Round (mixed by run)"
        client_ylabel = "metric value"
    make_line_plot(
        out_dir / "predictive_client_mean_vs_round.png",
        client_title,
        client_ylabel,
        client_quality_series,
    )

    # Overlay: server metric vs clients-mean metric, with per-run fallback.
    server_client_series = []
    server_client_metrics_used: set[str] = set()
    if not client_roundwise_df.empty and not roundwise_df.empty:
        for r in records:
            s_server = roundwise_df[roundwise_df["run_label"] == r.run_label]
            s_client = client_roundwise_df[client_roundwise_df["run_label"] == r.run_label]
            if s_server.empty or s_client.empty:
                continue

            selected_pair = None
            for server_col, client_col in SERVER_CLIENT_QUALITY_METRIC_PAIRS:
                if (
                    server_col in s_server.columns
                    and client_col in s_client.columns
                    and s_server[server_col].notna().any()
                    and s_client[client_col].notna().any()
                ):
                    selected_pair = (server_col, client_col)
                    break
            if selected_pair is None:
                continue

            server_col, client_col = selected_pair
            server_client_metrics_used.add(server_col)
            s_server = s_server.dropna(subset=["round", server_col]).sort_values("round")
            s_client = s_client.dropna(subset=["round", client_col]).sort_values("round")

            if not s_server.empty:
                server_client_series.append(
                    (
                        f"{r.run_label} [server {pretty_metric_name(server_col)}]",
                        s_server["round"].to_numpy(dtype=float),
                        s_server[server_col].to_numpy(dtype=float),
                    )
                )
            if not s_client.empty:
                server_client_series.append(
                    (
                        f"{r.run_label} [clients-mean {pretty_metric_name(server_col)}]",
                        s_client["round"].to_numpy(dtype=float),
                        s_client[client_col].to_numpy(dtype=float),
                    )
                )
    if len(server_client_metrics_used) == 1:
        only_metric = next(iter(server_client_metrics_used))
        server_client_title = f"Server vs Clients Mean {pretty_metric_name(only_metric)}"
        server_client_ylabel = pretty_metric_name(only_metric)
    else:
        server_client_title = "Server vs Clients Mean Predictive Metric (mixed by run)"
        server_client_ylabel = "metric value"
    make_line_plot(
        out_dir / "predictive_server_vs_clients_mean_vs_round.png",
        server_client_title,
        server_client_ylabel,
        server_client_series,
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

    spd_vals = [0.0 if pd.isna(v) else float(v) for v in summary_df.get("fairness_final_spd_mean", pd.Series(dtype=float)).tolist()]
    eod_vals = [0.0 if pd.isna(v) else float(v) for v in summary_df.get("fairness_final_eod_mean", pd.Series(dtype=float)).tolist()]
    if labels and (any(pd.notna(v) for v in summary_df.get("fairness_final_spd_mean", pd.Series(dtype=float)).tolist()) or any(pd.notna(v) for v in summary_df.get("fairness_final_eod_mean", pd.Series(dtype=float)).tolist())):
        make_grouped_bar_plot(
            out_dir / "final_fairness_spd_eod_bar.png",
            "Final SPD and EOD by Run",
            "fairness metric value",
            labels,
            [("SPD", spd_vals), ("EOD", eod_vals)],
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
        comm_numeric = [0.0 if pd.isna(v) else float(v) for v in comm_vals]
        make_bar_plot(
            out_dir / "total_comm_cost_bar.png",
            "Total Communication Cost by Run",
            "total_comm_cost",
            labels,
            comm_numeric,
        )
        if any(v > 0 for v in comm_numeric):
            make_bar_plot(
                out_dir / "total_comm_cost_bar_log.png",
                "Total Communication Cost by Run (log scale)",
                "total_comm_cost",
                labels,
                [max(v, 1.0) for v in comm_numeric],
                yscale="log",
            )

    # DP trade-off scatter plots.
    epsilon_quality_pairs = []
    epsilon_quality_metrics_used: set[str] = set()
    epsilon_cost_pairs = []
    for _, row in summary_df.iterrows():
        epsilon = to_float_or_none(row.get("dp_epsilon"))
        if epsilon is None:
            continue
        label = str(row["run_label"])
        q = None
        q_name = None
        for key, metric_name in (
            ("final_macro_f1", "Macro-F1"),
            ("final_micro_f1", "Micro-F1"),
            ("final_accuracy", "Accuracy"),
        ):
            value = to_float_or_none(row.get(key))
            if value is not None:
                q = value
                q_name = metric_name
                break

        cost = to_float_or_none(row.get("run_time_seconds"))
        if q is not None and q_name is not None:
            epsilon_quality_pairs.append((epsilon, q, f"{label} [{q_name}]"))
            epsilon_quality_metrics_used.add(q_name)
        if cost is not None:
            epsilon_cost_pairs.append((epsilon, cost, label))

    if len(epsilon_quality_metrics_used) == 1:
        only_metric = next(iter(epsilon_quality_metrics_used))
        eps_quality_title = f"DP Epsilon vs Final {only_metric}"
        eps_quality_ylabel = f"final {only_metric}"
    else:
        eps_quality_title = "DP Epsilon vs Final Predictive Metric (mixed by run)"
        eps_quality_ylabel = "final metric value"

    make_epsilon_scatter(
        out_dir / "dp_epsilon_vs_predictive_metric.png",
        eps_quality_title,
        eps_quality_ylabel,
        epsilon_quality_pairs,
    )
    make_epsilon_scatter(
        out_dir / "dp_epsilon_vs_runtime.png",
        "DP Epsilon vs Run Time",
        "run_time_seconds",
        epsilon_cost_pairs,
    )

    print(f"Parsed runs: {len(records)}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {roundwise_csv}")
    print(f"Wrote: {fairness_csv}")
    progression_wrote_any = False
    if not args.skip_progression:
        progression_wrote_any = generate_progression_artifacts(
            runs_dir=runs_dir,
            out_dir=progression_out_dir,
            dataset=args.dataset or "medical",
            epsilon_levels=args.epsilon_levels,
            noise_levels=args.noise_levels,
        )

    print(f"Wrote: {client_roundwise_csv}")
    print(f"Wrote summary plots in: {out_dir}")
    if not args.skip_progression:
        if progression_wrote_any:
            print(f"Wrote progression plots in: {progression_out_dir}")
        else:
            print(f"No progression plots were generated in: {progression_out_dir}")

    if errors:
        print("\nSkipped runs:")
        for err in errors:
            print(f"- {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
