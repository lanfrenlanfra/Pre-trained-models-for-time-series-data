from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


DATASETS = ["AIOPS", "TODS", "WSD", "Yahoo"]

WINDOW_POINTS = {
    "AIOPS": 4000,
    "TODS": 2000,
    "WSD": 4000,
    "Yahoo": 1500,
}


def pick_representative_file(data_root: Path, dataset: str) -> Path:
    """Return a CSV file that has at least one anomaly and a typical
    anomaly count for the dataset (median over the dataset)."""
    files = sorted((data_root / dataset).glob("*.csv"))
    counts = []
    for f in files:
        df = pd.read_csv(f, usecols=["is_anomaly"])
        counts.append((f, int(df["is_anomaly"].sum())))
    with_anom = [(f, c) for f, c in counts if c > 0]
    if not with_anom:
        return files[0]
    with_anom.sort(key=lambda x: x[1])
    return with_anom[len(with_anom) // 2][0]


def load_fragment(path: Path, window: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    if len(df) <= window:
        return df.reset_index(drop=True)
    anom_idx = np.where(df["is_anomaly"].values > 0)[0]
    if len(anom_idx) == 0:
        start = 0
    else:
        center = int(anom_idx[0])
        start = max(0, center - window // 2)
        start = min(start, len(df) - window)
    return df.iloc[start:start + window].reset_index(drop=True)


def plot_one(ax, df: pd.DataFrame, title: str) -> None:
    ax.plot(df["timestamp"], df["value_0"], lw=0.7, color="#1f4e79")
    anom_mask = df["is_anomaly"].values > 0
    if anom_mask.any():
        ax.scatter(
            df.loc[anom_mask, "timestamp"],
            df.loc[anom_mask, "value_0"],
            color="#c0392b",
            s=10,
            zorder=3,
            label="is_anomaly = 1",
        )
        ax.legend(
            loc="upper right",
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="0.7",
            framealpha=1.0,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Время", fontsize=9)
    ax.set_ylabel("Значение метрики", fontsize=9)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--copy-to", type=Path, default=None,
                    help="Optional: also copy the PNG into this folder "
                         "(e.g. Diploma/graphics).")
    ap.add_argument("--files", type=str, default=None,
                    help="Override file picks as DATASET=path,DATASET=path,...")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.files:
        for token in args.files.split(","):
            k, v = token.split("=", 1)
            overrides[k.strip()] = Path(v.strip())

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    axes = axes.flatten()

    for ax, ds in zip(axes, DATASETS):
        path = overrides.get(ds) or pick_representative_file(args.data_root, ds)
        df = load_fragment(path, WINDOW_POINTS[ds])
        plot_one(ax, df, f"{ds}  ({path.name})")

    fig.suptitle(
        "Примеры фрагментов рядов из используемых датасетов",
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = args.out_dir / "dataset_examples.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"saved: {out_path}")

    if args.copy_to is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        dst = args.copy_to / out_path.name
        shutil.copy2(out_path, dst)
        print(f"copied to: {dst}")


if __name__ == "__main__":
    main()
