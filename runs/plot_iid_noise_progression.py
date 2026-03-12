"""Plot FL round-wise metric progression by setting and privacy level.

Generates one chart per (setting, metric), with privacy-tagged runs overlaid as
separate lines and baseline runs shown when available.

Expected run folders may include tags such as:
- horizontal: `iid` or `non-iid`
- vertical: `vertical-disjoint` / `vertical-overlap`
- privacy: `eps-<value>` / `epsilon-<value>` or legacy `noise-<value>`
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLOBAL_METRICS_FILE = "global_metrics.csv"

LINE_FIGSIZE = (11, 6)
LINE_MARGINS = {"left": 0.11, "right": 0.97, "bottom": 0.13, "top": 0.90}

SETTING_META = {
    "horizontal-iid": {"label": "Horizontal-IID", "dir": "Horizontal-IID"},
    "horizontal-non-iid": {"label": "Horizontal-NonIID", "dir": "Horizontal-NonIID"},
    "vertical-disjoint": {"label": "Vertical-Disjoint", "dir": "Vertical-Disjoint"},
    "vertical-overlap": {"label": "Vertical-Overlap", "dir": "Vertical-Overlap"},
}

SETTING_COLORS = {
    "horizontal-iid": "#1f77b4",
    "horizontal-non-iid": "#ff7f0e",
    "vertical-disjoint": "#2ca02c",
    "vertical-overlap": "#d62728",
}

COMPARISON_GROUPS = {
    "Vertical-Comparison": {
        "title": "Vertical Comparison",
        "settings": ["vertical-disjoint", "vertical-overlap"],
    },
}

BASELINE_KEY = "baseline"


def _new_axes(figsize: tuple[float, float], margins: dict[str, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(**margins)
    return fig, ax


METRICS = {
    "accuracy": {"plot_label": "Accuracy", "file_label": "Accuracy"},
    "macro_f1": {"plot_label": "F1 score (macro)", "file_label": "F1 score"},
    "macro_precision": {"plot_label": "Precision (macro)", "file_label": "Precision"},
    "macro_recall": {"plot_label": "Recall (macro)", "file_label": "Recall"},
    "loss": {"plot_label": "Loss", "file_label": "Loss"},
}


def first_existing_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_privacy_level(name: str) -> float | None:
    patterns = (
        r"epsilon[._-]?([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)",
        r"eps[._-]?([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)",
        r"noise[._-]?([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        m = re.search(pattern, name.lower())
        if not m:
            continue
        try:
            return float(m.group(1).replace("p", "."))
        except ValueError:
            return None
    return None


def parse_setting(name: str) -> str | None:
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


def is_close_to_any(v: float, targets: list[float], tol: float = 1e-9) -> bool:
    return any(math.isclose(v, t, abs_tol=tol) for t in targets)


def infer_privacy_label(name: str) -> str | None:
    lower = name.lower()
    if "epsilon" in lower or "eps" in lower:
        return "epsilon"
    if "noise" in lower:
        return "noise"
    return None


def discover_privacy_levels(runs_dir: Path, dataset_filter: str) -> tuple[list[float], str]:
    levels: set[float] = set()
    labels: set[str] = set()
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name.lower()
        if dataset_filter and dataset_filter.lower() not in name:
            continue
        level = parse_privacy_level(name)
        if level is None:
            continue
        levels.add(level)
        label = infer_privacy_label(name)
        if label is not None:
            labels.add(label)

    privacy_label = labels.pop() if len(labels) == 1 else "privacy"
    return sorted(levels), privacy_label


def load_round_metric_series(run_dir: Path, metric_col: str) -> pd.DataFrame | None:
    gpath = run_dir / GLOBAL_METRICS_FILE
    if gpath.exists():
        gdf = pd.read_csv(gpath)
        if metric_col in gdf.columns:
            if "round" not in gdf.columns:
                gdf["round"] = np.arange(1, len(gdf) + 1)
            sdf = gdf[["round", metric_col]].dropna().copy()
            if not sdf.empty:
                sdf["round"] = sdf["round"].astype(int)
                return sdf

    if metric_col != "loss":
        return None

    fallback_sources: list[tuple[str, tuple[str, ...]]] = [
        ("metrics.csv", ("train_loss", "loss", "fit_loss", "client_loss")),
        ("local_test_metrics.csv", ("loss",)),
        ("postfit_metrics.csv", ("loss",)),
        ("prefit_metrics.csv", ("loss",)),
        ("locals_metrics.csv", ("loss",)),
    ]
    for fname, candidates in fallback_sources:
        path = run_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "round" not in df.columns:
            continue
        source_col = first_existing_col(df, candidates)
        if source_col is None:
            continue
        sdf = df[["round", source_col] + (["client"] if "client" in df.columns else [])].dropna()
        if sdf.empty:
            continue
        if "client" in sdf.columns:
            sdf = sdf.groupby("round", as_index=False)[source_col].mean()
        else:
            sdf = sdf[["round", source_col]].copy()
        sdf = sdf.rename(columns={source_col: metric_col})
        sdf["round"] = sdf["round"].astype(int)
        return sdf.sort_values("round")

    return None


def aggregate_runs_for_metric(
    runs_dir: Path,
    metric_col: str,
    privacy_levels: list[float],
    dataset_filter: str,
) -> dict[tuple[str, str], pd.DataFrame]:
    series_by_key: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue

        name = d.name.lower()
        if dataset_filter and dataset_filter.lower() not in name:
            continue

        setting = parse_setting(name)
        if setting is None:
            continue

        privacy_level = parse_privacy_level(name)
        if privacy_level is None:
            privacy_key = BASELINE_KEY
        else:
            if privacy_levels and not is_close_to_any(privacy_level, privacy_levels):
                continue
            privacy_key = f"{privacy_level:g}"

        sdf = load_round_metric_series(d, metric_col)
        if sdf is None or sdf.empty:
            continue

        key = (setting, privacy_key)
        series_by_key.setdefault(key, []).append(sdf)

    aggregated: dict[tuple[str, str], pd.DataFrame] = {}
    for key, frames in series_by_key.items():
        merged = pd.concat(frames, ignore_index=True)
        agg = merged.groupby("round", as_index=False)[metric_col].mean().sort_values("round")
        aggregated[key] = agg

    return aggregated


def ordered_privacy_keys(aggregated: dict[tuple[str, str], pd.DataFrame], setting: str) -> list[str]:
    keys = sorted({privacy_key for s, privacy_key in aggregated if s == setting and privacy_key != BASELINE_KEY}, key=float)
    if any(s == setting and privacy_key == BASELINE_KEY for s, privacy_key in aggregated):
        return [BASELINE_KEY, *keys]
    return keys


def privacy_display_label(privacy_key: str, privacy_label: str) -> str:
    if privacy_key == BASELINE_KEY:
        return "baseline"
    return f"{privacy_label}={privacy_key}"


def comparison_series_label(setting: str, privacy_key: str, privacy_label: str) -> str:
    setting_label = SETTING_META[setting]["label"]
    if privacy_key == BASELINE_KEY:
        return setting_label
    return f"{setting_label} | {privacy_display_label(privacy_key, privacy_label)}"


def plot_metric_for_setting(
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

    keys = ordered_privacy_keys(aggregated, setting)
    if not keys:
        return False

    numeric_keys = [k for k in keys if k != BASELINE_KEY]
    colors = {
        privacy_key: plt.cm.viridis(i / max(1, len(numeric_keys) - 1))
        for i, privacy_key in enumerate(numeric_keys)
    }

    fig, ax = _new_axes(LINE_FIGSIZE, LINE_MARGINS)
    plotted = 0
    for privacy_key in keys:
        key = (setting, privacy_key)
        if key not in aggregated:
            continue
        s = aggregated[key]
        if s.empty:
            continue
        color = "#444444" if privacy_key == BASELINE_KEY else colors[privacy_key]
        linestyle = "--" if privacy_key == BASELINE_KEY else "-"
        ax.plot(
            s["round"].to_numpy(dtype=float),
            s[metric_col].to_numpy(dtype=float),
            label=privacy_display_label(privacy_key, privacy_label),
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


def plot_metric_for_comparison_group(
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
        for privacy_key in ordered_privacy_keys(aggregated, setting):
            key = (setting, privacy_key)
            if key not in aggregated:
                continue
            s = aggregated[key]
            if s.empty:
                continue
            series.append((setting, privacy_key, s))

    if not series:
        return False

    fig, ax = _new_axes(LINE_FIGSIZE, LINE_MARGINS)
    style_cycle = ["-", "--", ":", "-."]
    plotted = 0
    for idx, (setting, privacy_key, s) in enumerate(series):
        if privacy_key == BASELINE_KEY:
            linestyle = "-"
        else:
            privacy_idx = max(0, ordered_privacy_keys(aggregated, setting).index(privacy_key) - 1)
            linestyle = style_cycle[privacy_idx % len(style_cycle)]
        ax.plot(
            s["round"].to_numpy(dtype=float),
            s[metric_col].to_numpy(dtype=float),
            label=comparison_series_label(setting, privacy_key, privacy_label),
            color=SETTING_COLORS.get(setting, plt.cm.tab10(idx % 10)),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metric progression by setting and DP privacy level.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Runs directory (default: runs/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots" / "iid_noise_progression",
        help="Output directory for plots and CSVs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="medical",
        help="Dataset substring filter in run folder names (default: medical).",
    )
    parser.add_argument(
        "--epsilon-levels",
        type=float,
        nargs="+",
        default=None,
        help="Optional epsilon levels to include. If omitted, infer from available runs.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Legacy alias for old run names using noise levels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.noise_levels is not None:
        privacy_levels = args.noise_levels
        privacy_label = "noise"
    elif args.epsilon_levels is not None:
        privacy_levels = args.epsilon_levels
        privacy_label = "epsilon"
    else:
        privacy_levels, privacy_label = discover_privacy_levels(runs_dir, args.dataset)
        if privacy_levels:
            print(f"Inferred {privacy_label} levels: {', '.join(str(v) for v in privacy_levels)}")
        else:
            print("No privacy levels were inferred from the available run folders. Plotting baseline runs only where available.")

    wrote_any = False
    roundwise_rows: list[pd.DataFrame] = []

    for metric_col, metric_meta in METRICS.items():
        metric_plot_label = metric_meta["plot_label"]
        metric_file_label = metric_meta["file_label"]
        aggregated = aggregate_runs_for_metric(
            runs_dir=runs_dir,
            metric_col=metric_col,
            privacy_levels=privacy_levels,
            dataset_filter=args.dataset,
        )
        if not aggregated:
            print(f"Skip {metric_col}: no matching runs or metric column.")
            continue

        for (setting, privacy_key), df in aggregated.items():
            tmp = df.copy()
            tmp["metric"] = metric_col
            tmp["setting"] = setting
            tmp["privacy_key"] = privacy_key
            tmp["privacy_level"] = np.nan if privacy_key == BASELINE_KEY else float(privacy_key)
            tmp["privacy_label"] = "baseline" if privacy_key == BASELINE_KEY else privacy_label
            roundwise_rows.append(tmp)

        for setting, meta in SETTING_META.items():
            out_path = out_dir / meta["dir"] / f"{meta['label']} - {metric_file_label}.png"
            ok = plot_metric_for_setting(
                out_path=out_path,
                title=f"{meta['label']} | {metric_plot_label} ({args.dataset})",
                y_label=metric_plot_label,
                metric_col=metric_col,
                aggregated=aggregated,
                setting=setting,
                privacy_label=privacy_label,
            )
            if ok:
                wrote_any = True
                print(f"Wrote plot: {out_path}")

        for group_dir, group_meta in COMPARISON_GROUPS.items():
            out_path = out_dir / group_dir / f"{group_meta['title']} - {metric_file_label}.png"
            ok = plot_metric_for_comparison_group(
                out_path=out_path,
                title=f"{group_meta['title']} | {metric_plot_label} ({args.dataset})",
                y_label=metric_plot_label,
                metric_col=metric_col,
                aggregated=aggregated,
                settings=group_meta["settings"],
                privacy_label=privacy_label,
            )
            if ok:
                wrote_any = True
                print(f"Wrote plot: {out_path}")

    if roundwise_rows:
        out_csv = out_dir / "roundwise_setting_privacy_metrics.csv"
        pd.concat(roundwise_rows, ignore_index=True).to_csv(out_csv, index=False)
        print(f"Wrote CSV: {out_csv}")

    if not wrote_any:
        print("No plots were generated. Check run names, metric availability, and privacy levels.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
