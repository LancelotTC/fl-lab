"""Plot FL round-wise metric progression by IID-ness and DP noise level.

Generates one chart per (iidness, metric), with all selected noise levels
overlaid as separate lines.

Expected run folders include tags in the name:
- `iid` or `non-iid`
- `noise-<value>` (e.g. noise-0p3 or noise-0.3)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLOBAL_METRICS_FILE = "global_metrics.csv"

METRICS = {
    "accuracy": {"plot_label": "Accuracy", "file_label": "Accuracy"},
    "macro_f1": {"plot_label": "F1 score (macro)", "file_label": "F1 score"},
    "macro_precision": {"plot_label": "Precision (macro)", "file_label": "Precision"},
    "macro_recall": {"plot_label": "Recall (macro)", "file_label": "Recall"},
    "loss": {"plot_label": "Loss", "file_label": "Loss"},
}


def iidness_label(iidness: str) -> str:
    return "IID" if iidness == "iid" else "NonIID"


def first_existing_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_noise(name: str) -> float | None:
    m = re.search(r"noise[._-]?([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)", name.lower())
    if not m:
        return None
    try:
        return float(m.group(1).replace("p", "."))
    except ValueError:
        return None


def parse_iidness(name: str) -> str | None:
    s = name.lower()
    if "non-iid" in s or "noniid" in s:
        return "non-iid"
    tokens = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    if "iid" in tokens:
        return "iid"
    return None


def is_close_to_any(v: float, targets: list[float], tol: float = 1e-9) -> bool:
    return any(math.isclose(v, t, abs_tol=tol) for t in targets)


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

    # Fallback for loss evolution when global_metrics.csv has no loss column.
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
    noise_levels: list[float],
    dataset_filter: str,
) -> dict[tuple[str, float], pd.DataFrame]:
    series_by_key: dict[tuple[str, float], list[pd.DataFrame]] = {}

    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue

        name = d.name.lower()
        if dataset_filter and dataset_filter.lower() not in name:
            continue

        iidness = parse_iidness(name)
        noise = parse_noise(name)
        if iidness is None or noise is None:
            continue
        if not is_close_to_any(noise, noise_levels):
            continue

        sdf = load_round_metric_series(d, metric_col)
        if sdf is None or sdf.empty:
            continue

        key = (iidness, noise)
        series_by_key.setdefault(key, []).append(sdf)

    aggregated: dict[tuple[str, float], pd.DataFrame] = {}
    for key, frames in series_by_key.items():
        merged = pd.concat(frames, ignore_index=True)
        agg = merged.groupby("round", as_index=False)[metric_col].mean().sort_values("round")
        aggregated[key] = agg

    return aggregated


def plot_metric_for_iidness(
    out_path: Path,
    title: str,
    y_label: str,
    metric_col: str,
    aggregated: dict[tuple[str, float], pd.DataFrame],
    noise_levels: list[float],
    iidness: str,
) -> bool:
    if not aggregated:
        return False

    colors = {
        noise: plt.cm.viridis(i / max(1, len(noise_levels) - 1))
        for i, noise in enumerate(sorted(noise_levels))
    }

    plt.figure(figsize=(11, 6))
    plotted = 0
    for noise in sorted(noise_levels):
        key = (iidness, noise)
        if key not in aggregated:
            continue
        s = aggregated[key]
        if s.empty:
            continue
        plt.plot(
            s["round"].to_numpy(dtype=float),
            s[metric_col].to_numpy(dtype=float),
            label=f"noise={noise:g}",
            color=colors[noise],
            linestyle="-",
            linewidth=1.9,
        )
        plotted += 1

    if plotted == 0:
        plt.close()
        return False

    plt.title(title)
    plt.xlabel("FL round")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot metric progression by IID-ness and noise level."
    )
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
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.6, 1.0],
        help="Noise levels to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote_any = False
    roundwise_rows: list[pd.DataFrame] = []

    for metric_col, metric_meta in METRICS.items():
        metric_plot_label = metric_meta["plot_label"]
        metric_file_label = metric_meta["file_label"]
        aggregated = aggregate_runs_for_metric(
            runs_dir=runs_dir,
            metric_col=metric_col,
            noise_levels=args.noise_levels,
            dataset_filter=args.dataset,
        )
        if not aggregated:
            print(f"Skip {metric_col}: no matching runs or metric column.")
            continue

        # Save roundwise data used for plotting.
        for (iidness, noise), df in aggregated.items():
            tmp = df.copy()
            tmp["metric"] = metric_col
            tmp["iidness"] = iidness
            tmp["noise"] = noise
            roundwise_rows.append(tmp)

        for iidness in ("iid", "non-iid"):
            iid_label = iidness_label(iidness)
            out_path = out_dir / iid_label / f"{iid_label} - {metric_file_label}.png"
            ok = plot_metric_for_iidness(
                out_path=out_path,
                title=f"{iid_label} | {metric_plot_label} ({args.dataset})",
                y_label=metric_plot_label,
                metric_col=metric_col,
                aggregated=aggregated,
                noise_levels=args.noise_levels,
                iidness=iidness,
            )
            if ok:
                wrote_any = True
                print(f"Wrote plot: {out_path}")

    if roundwise_rows:
        out_csv = out_dir / "roundwise_iid_noise_metrics.csv"
        pd.concat(roundwise_rows, ignore_index=True).to_csv(out_csv, index=False)
        print(f"Wrote CSV: {out_csv}")

    if not wrote_any:
        print("No plots were generated. Check run names, metric availability, and noise levels.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
