from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import json5
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_ROOT = Path(__file__).parent / "data"
MODELS_FILE = Path(__file__).parent / "models.json5"
DATASETS = ["AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo"]

C_SIGNAL = "#3B82F6"
C_EXPECT = "#6B7280"
C_BOUNDS = "#D1D5DB"
C_TP = "#16A34A"
C_FN = "#F97316"
C_FP = "#DC2626"
C_SCORE = "#7C3AED"
C_THRESH = "#DC2626"
C_GT_BG = "#FEF9C3"

def parse_timestamp(ts_series: pd.Series) -> pd.Series:
    sample = ts_series.iloc[0]
    try:
        val = float(sample)
        return pd.to_datetime(ts_series.astype(float), unit="ms" if val > 1e12 else "s")
    except (ValueError, TypeError):
        return pd.to_datetime(ts_series)

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.index = parse_timestamp(df["timestamp"])
    return df

def run_detection(csv_path: Path, ad_config: dict):
    """Run AnomalyDetectionSystem on one file; return (df_result, detection_result)."""
    from anomaly_detection_forecasting import AnomalyDetectionSystem

    df = load_csv(csv_path)
    value_cols = [c for c in df.columns if c.startswith("value_")]

    values_df = df[value_cols].copy()
    values_df.index = df.index

    detector = AnomalyDetectionSystem(**ad_config)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        result = detector.detect(values_df)

    out = pd.DataFrame(index=df.index)
    out["value"] = df[value_cols[0]].values
    out["ground_truth"] = df["is_anomaly"].astype(int).values
    out["predicted"] = result.is_anomaly.astype(int)
    out["score"] = result.anomaly_scores

    if result.expected_value is not None:
        ev = np.asarray(result.expected_value)
        out["expected"] = ev[0] if ev.ndim == 2 else ev
    else:
        out["expected"] = np.nan

    if result.expected_bounds is not None:
        eb = np.asarray(result.expected_bounds)
        if eb.ndim == 2 and eb.shape[1] == 2:
            out["bound_lo"] = eb[:, 0]
            out["bound_hi"] = eb[:, 1]
        else:
            out["bound_lo"] = np.nan
            out["bound_hi"] = np.nan
    else:
        out["bound_lo"] = np.nan
        out["bound_hi"] = np.nan

    return out

def plain_number_formatter(ax_obj, which="y"):
    """No scientific notation; plain integers for large numbers."""
    fmt = ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.4g}"
    )
    if which == "y":
        ax_obj.yaxis.set_major_formatter(fmt)
    else:
        ax_obj.xaxis.set_major_formatter(fmt)


def find_spans(flags: np.ndarray, times: pd.Index):
    """Yield (t_start, t_end) for each contiguous True run."""
    flags = np.asarray(flags, dtype=bool)
    diffs = np.diff(np.concatenate([[False], flags, [False]]).astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    for s, e in zip(starts, ends):
        yield times[s], times[min(e, len(times) - 1)]

def plot_one(df: pd.DataFrame, threshold: float, title: str, ax_ts: plt.Axes, ax_sc: plt.Axes):
    """
    Two-panel figure:
      ax_ts — top: signal + TP / FN / FP scatter, x-axis labelled
      ax_sc — bottom: anomaly score curve + threshold line, x-axis labelled
    Ground-truth anomaly regions shown as yellow background bands in both panels.
    """
    times = df.index
    v = df["value"].to_numpy(dtype=float)
    gt = df["ground_truth"].to_numpy(dtype=int)
    pred = df["predicted"].to_numpy(dtype=int)
    score = df["score"].to_numpy(dtype=float)

    first_gt = True
    for t0, t1 in find_spans(gt, times):
        ax_ts.axvspan(t0, t1, color=C_GT_BG, alpha=0.7, zorder=1,
                      label="ground truth anomaly" if first_gt else "_nolegend_")
        first_gt = False

    ax_ts.plot(times, v, color=C_SIGNAL, linewidth=0.8, zorder=4, label="signal")

    tp_mask = (gt == 1) & (pred == 1)
    fn_mask = (gt == 1) & (pred == 0)
    fp_mask = (gt == 0) & (pred == 1)

    scatter_kw = dict(zorder=6, s=35, linewidths=0.4, edgecolors="white")
    if tp_mask.any():
        ax_ts.scatter(times[tp_mask], v[tp_mask], color=C_TP,
                      label=f"TP ({tp_mask.sum()})", **scatter_kw)
    if fn_mask.any():
        ax_ts.scatter(times[fn_mask], v[fn_mask], color=C_FN,
                      label=f"FN ({fn_mask.sum()})", **scatter_kw)
    if fp_mask.any():
        ax_ts.scatter(times[fp_mask], v[fp_mask], color=C_FP,
                      label=f"FP ({fp_mask.sum()})", **scatter_kw)

    ax_ts.set_ylim(bottom=0)
    ax_ts.set_ylabel("value", fontsize=8)
    ax_ts.set_xlabel("time", fontsize=8)
    plain_number_formatter(ax_ts, "y")
    ax_ts.tick_params(axis="x", labelsize=7, rotation=15)
    ax_ts.tick_params(axis="y", labelsize=7)
    ax_ts.set_facecolor("white")
    ax_ts.set_title(title, fontsize=8, pad=5)
    ax_ts.legend(fontsize=6, loc="upper right", ncol=3, framealpha=0.85)

    for t0, t1 in find_spans(gt, times):
        ax_sc.axvspan(t0, t1, color=C_GT_BG, alpha=0.5, zorder=1)

    ax_sc.plot(times, score, color=C_SCORE, linewidth=0.7, zorder=3, label="anomaly score")
    ax_sc.axhline(threshold, color=C_THRESH, linewidth=1.1, linestyle="--",
                  zorder=5, label=f"threshold = {threshold:.2g}")
    ax_sc.fill_between(
        times, score, threshold,
        where=(score > threshold),
        color=C_SCORE, alpha=0.2, zorder=2, label="score > threshold",
    )

    sc_max = max(float(np.nanmax(score)) if np.isfinite(score).any() else threshold,
                 threshold * 1.1)
    ax_sc.set_ylim(bottom=0, top=sc_max * 1.15)
    ax_sc.set_ylabel("anomaly score", fontsize=8)
    ax_sc.set_xlabel("time", fontsize=8)
    plain_number_formatter(ax_sc, "y")
    ax_sc.tick_params(axis="x", labelsize=7, rotation=15)
    ax_sc.tick_params(axis="y", labelsize=7)
    ax_sc.set_facecolor("white")
    ax_sc.legend(fontsize=6, loc="upper right", framealpha=0.85)

def collect_files(datasets: list[str]) -> list[Path]:
    files = []
    for ds in datasets:
        d = DATA_ROOT / ds
        if d.exists():
            files.extend(sorted(d.glob("*.csv")))
        else:
            print(f"  [warn] not found: {d}")
    return files

def main():
    parser = argparse.ArgumentParser(description="Plot anomaly detection results")
    parser.add_argument("--models",   nargs="+", default=None,
                        help="Model names to plot (default: all in models.json5)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Dataset names (default: all)")
    parser.add_argument("--save",     action="store_true",
                        help="Save figures as PNGs")
    parser.add_argument("--save_dir", default="plots/anomaly_detection",
                        help="Root directory for saved PNGs (default: plots/anomaly_detection/)")
    parser.add_argument("--no_show",  action="store_true",
                        help="Suppress interactive GUI (use with --save)")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Limit files per dataset (0 = no limit)")
    args = parser.parse_args()

    show = not args.no_show

    with open(MODELS_FILE) as f:
        all_configs = json5.load(f)

    model_names = args.models or list(all_configs.keys())
    print(f"Models: {model_names}")
    print(f"Datasets: {args.datasets}")

    csv_files = collect_files(args.datasets)
    if args.max_files > 0:
        grouped = {}
        for p in csv_files:
            grouped.setdefault(p.parent.name, []).append(p)
        csv_files = []
        for ds, files in grouped.items():
            csv_files.extend(files[:args.max_files])

    print(f"Files: {len(csv_files)} total\n")

    save_dir = Path(args.save_dir)
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {save_dir.resolve()}\n")

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    total = len(model_names) * len(csv_files)
    done  = 0

    for model_name in model_names:
        if model_name not in all_configs:
            print(f"[skip] model '{model_name}' not in models.json5")
            continue

        cfg = all_configs[model_name]
        ad_config = {k: v for k, v in cfg.items() if k != "forecasting_model_params"}
        threshold = float(cfg["detection_model_params"].get("threshold", 3.0))

        for csv_path in csv_files:
            done += 1
            label = f"{csv_path.parent.name}/{csv_path.name}"
            print(f"[{done}/{total}] {model_name} | {label}")

            try:
                df = run_detection(csv_path, ad_config)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            n = len(df)
            n_gt = int(df["ground_truth"].sum())
            n_pred = int(df["predicted"].sum())
            tp = int(((df["ground_truth"] == 1) & (df["predicted"] == 1)).sum())
            fp = int(((df["ground_truth"] == 0) & (df["predicted"] == 1)).sum())
            fn = int(((df["ground_truth"] == 1) & (df["predicted"] == 0)).sum())
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)

            title = (
                f"{model_name} | {label} |"
                f"n={n} gt={n_gt} pred={n_pred}"
                f"TP={tp} FP={fp} FN={fn}"
                f"P={prec:.2f} R={rec:.2f} F1={f1:.2f} threshold={threshold:.2g}"
            )

            fig, (ax_ts, ax_sc) = plt.subplots(
                nrows=2, ncols=1, figsize=(16, 7),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            fig.patch.set_facecolor("white")

            try:
                plot_one(df, threshold, title, ax_ts, ax_sc)
            except Exception as e:
                ax_ts.set_title(f"PLOT ERROR: {e}", color="red")
                print(f"  PLOT ERROR: {e}")

            fig.tight_layout()

            if args.save:
                out_path = save_dir / csv_path.parent.name / model_name / f"{csv_path.stem}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"saved {out_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
