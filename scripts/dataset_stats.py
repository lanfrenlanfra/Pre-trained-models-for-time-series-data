from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

def summarise_dir(d: Path) -> dict:
    files = sorted(d.glob("*.csv"))
    lengths: list[int] = []
    anom_counts: list[int] = []
    sampling_ms = None
    total_points = 0
    total_anoms = 0
    files_with_anom = 0
    for f in files:
        df = pd.read_csv(f, usecols=["timestamp", "is_anomaly"])
        n = len(df)
        a = int((df["is_anomaly"] > 0).sum())
        lengths.append(n)
        anom_counts.append(a)
        total_points += n
        total_anoms += a
        if a > 0:
            files_with_anom += 1
        if sampling_ms is None and n >= 2:
            sampling_ms = int(df["timestamp"].iloc[1] - df["timestamp"].iloc[0])
    return {
        "n_files": len(files),
        "min_len": int(np.min(lengths)) if lengths else 0,
        "median_len": int(np.median(lengths)) if lengths else 0,
        "max_len": int(np.max(lengths)) if lengths else 0,
        "sampling_seconds": (sampling_ms or 0) / 1000.0,
        "total_points": total_points,
        "total_anoms": total_anoms,
        "global_anom_pct": 100.0 * total_anoms / total_points if total_points else 0.0,
        "files_with_anom": files_with_anom,
    }

def print_summary(name: str, s: dict) -> None:
    print(f"\n{name}")
    print(f"series: {s['n_files']}")
    print(f"length min: {s['min_len']:,}")
    print(f"length median: {s['median_len']:,}")
    print(f"length max: {s['max_len']:,}")
    print(f"sampling step: {s['sampling_seconds']:.0f} s "
          f"(~{s['sampling_seconds']/60:.1f} min)")
    print(f"total points: {s['total_points']:,}")
    print(f"total anomalies: {s['total_anoms']:,}")
    print(f"global anomaly%: {s['global_anom_pct']:.3f}")
    print(f"series w/ anom: {s['files_with_anom']} / {s['n_files']}")

def yahoo_subcollections(data_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for f in sorted((data_root / "Yahoo").glob("*.csv")):
        name = f.name
        if re.match(r"^A3", name):
            key = "Yahoo/A3Benchmark"
        elif re.match(r"^A4", name):
            key = "Yahoo/A4Benchmark"
        elif name.startswith("real_"):
            key = "Yahoo/real"
        elif name.startswith("synthetic_"):
            key = "Yahoo/synthetic"
        else:
            key = "Yahoo/other"
        out[key].append(f)
    return out

def summarise_files(files: list[Path]) -> dict:
    lengths: list[int] = []
    total_points = 0
    total_anoms = 0
    files_with_anom = 0
    sampling_ms = None
    for f in files:
        df = pd.read_csv(f, usecols=["timestamp", "is_anomaly"])
        n = len(df)
        a = int((df["is_anomaly"] > 0).sum())
        lengths.append(n)
        total_points += n
        total_anoms += a
        if a > 0:
            files_with_anom += 1
        if sampling_ms is None and n >= 2:
            sampling_ms = int(df["timestamp"].iloc[1] - df["timestamp"].iloc[0])
    return {
        "n_files": len(files),
        "min_len": int(np.min(lengths)) if lengths else 0,
        "median_len": int(np.median(lengths)) if lengths else 0,
        "max_len": int(np.max(lengths)) if lengths else 0,
        "sampling_seconds": (sampling_ms or 0) / 1000.0,
        "total_points": total_points,
        "total_anoms": total_anoms,
        "global_anom_pct": 100.0 * total_anoms / total_points if total_points else 0.0,
        "files_with_anom": files_with_anom,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args()

    for ds in ["AIOPS", "TODS", "WSD", "Yahoo"]:
        d = args.data_root / ds
        if not d.is_dir():
            print(f"skip missing dir: {d}")
            continue
        s = summarise_dir(d)
        print_summary(ds, s)

    print("\nYahoo breakdown")
    for name, files in yahoo_subcollections(args.data_root).items():
        if not files:
            continue
        s = summarise_files(files)
        print_summary(name, s)

if __name__ == "__main__":
    main()
