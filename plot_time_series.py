from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).parent / "data"
DATASETS = ["AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo"]

COLOR_SIGNAL = "#2563EB"
COLOR_ANOMALY = "#DC2626"
COLOR_ANOMALY_BG = "#FEE2E2"

def parse_timestamp(ts_series: pd.Series) -> pd.Series:
    """Convert unix-ms, unix-s, or datetime strings to pandas Timestamps."""
    sample = ts_series.iloc[0]
    try:
        val = float(sample)
        if val > 1e12:
            return pd.to_datetime(ts_series.astype(float), unit="ms")
        else:
            return pd.to_datetime(ts_series.astype(float), unit="s")
    except (ValueError, TypeError):
        return pd.to_datetime(ts_series)

def find_anomaly_spans(is_anomaly: np.ndarray, times: pd.Series):
    """Return list of (start_time, end_time) for contiguous anomaly runs."""
    spans = []
    in_span = False
    start = None
    for i, flag in enumerate(is_anomaly):
        if flag and not in_span:
            in_span = True
            start = times.iloc[i]
        elif not flag and in_span:
            in_span = False
            spans.append((start, times.iloc[i - 1]))
    if in_span:
        spans.append((start, times.iloc[-1]))
    return spans

def format_yaxis(ax):
    """Force plain integer/float notation on Y axis (no scientific notation)."""
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.4g}"
    ))

def plot_series(csv_path: Path, ax: plt.Axes, value_col: str = "value_0"):
    df = pd.read_csv(csv_path)
    times = parse_timestamp(df["timestamp"])
    values = df[value_col].to_numpy(dtype=float)
    is_anomaly = df["is_anomaly"].to_numpy(dtype=int)

    for t_start, t_end in find_anomaly_spans(is_anomaly, times):
        ax.axvspan(t_start, t_end, color=COLOR_ANOMALY_BG, alpha=0.6, zorder=1)

    ax.plot(times, values, color=COLOR_SIGNAL, linewidth=0.8, zorder=2, label="value")

    anom_mask = is_anomaly.astype(bool)
    if anom_mask.any():
        ax.scatter(
            times[anom_mask], values[anom_mask],
            color=COLOR_ANOMALY, s=18, zorder=3, label="anomaly"
        )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel(value_col, fontsize=9)
    format_yaxis(ax)

    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_facecolor("white")

    n_anom = int(anom_mask.sum())
    ax.set_title(
        f"{csv_path.parent.name} / {csv_path.name}   "
        f"[n={len(df)}, anomalies={n_anom} ({100*n_anom/len(df):.1f}%)]",
        fontsize=9, pad=6,
    )

    if anom_mask.any():
        ax.legend(fontsize=7, loc="upper right")

def collect_csv_files(datasets: list[str]) -> list[Path]:
    files = []
    for ds in datasets:
        ds_dir = DATA_ROOT / ds
        if not ds_dir.exists():
            print(f"[warn] Dataset directory not found: {ds_dir}")
            continue
        found = sorted(ds_dir.glob("*.csv"))
        print(f"{ds}: {len(found)} files")
        files.extend(found)
    return files


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark time series")
    parser.add_argument(
        "--datasets", default=",".join(DATASETS),
        help="Comma-separated list of dataset names (default: all)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save each figure as PNG instead of (or in addition to) showing it"
    )
    parser.add_argument(
        "--save_dir", default="plots/forecasting",
        help="Root directory for saved PNGs (default: plots/forecasting/)"
    )
    parser.add_argument(
        "--show", action="store_true", default=True,
        help="Show interactive window (default: True)"
    )
    parser.add_argument(
        "--no_show", action="store_true",
        help="Suppress interactive window (useful with --save)"
    )
    args = parser.parse_args()

    if args.no_show:
        args.show = False

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"Datasets: {datasets}")
    csv_files = collect_csv_files(datasets)
    print(f"Total files: {len(csv_files)}\n")

    if not csv_files:
        print("No CSV files found. Check --datasets and data/ directory.")
        return

    save_dir = Path(args.save_dir)
    if args.save:
        print(f"Saving PNGs to: {save_dir.resolve()}/\n")

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.5,
        "axes.spines.top":False,
        "axes.spines.right": False,
    })

    for i, csv_path in enumerate(csv_files):
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor("white")

        try:
            plot_series(csv_path, ax)
        except Exception as e:
            ax.set_title(f"ERROR: {csv_path.name} — {e}", color="red")
            print(f"[error] {csv_path.name}: {e}")

        fig.tight_layout()

        if args.save:
            out_path = save_dir / csv_path.parent.name / f"{csv_path.stem}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[{i+1}/{len(csv_files)}] saved → {out_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
