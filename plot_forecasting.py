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

C_ACTUAL = "#3B82F6"
C_FORECAST = "#F97316"
C_CI_95 = "#FED7AA"
C_CI_80 = "#FDBA74"
C_WARMUP = "#E5E7EB"
C_RESID = "#7C3AED"
C_GT_BG = "#FEF9C3"
C_ZERO = "#6B7280"

def plain_number_formatter(ax_obj, which: str = "y") -> None:
    """No scientific notation; plain integers for large numbers."""
    fmt = ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"
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

def parse_timestamp(ts_series: pd.Series) -> pd.Series:
    sample = ts_series.iloc[0]
    try:
        val = float(sample)
        return pd.to_datetime(ts_series.astype(float),
                              unit="ms" if val > 1e12 else "s")
    except (ValueError, TypeError):
        return pd.to_datetime(ts_series)


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.index = parse_timestamp(df["timestamp"])
    return df

def run_forecast(csv_path: Path, fc_config: dict, warmup_points: int):
    """
    Instantiate the forecaster described in *fc_config* and run it on one file.
    Returns
    times: pd.DatetimeIndex
    y_true: np.ndarray  [T]
    y_pred: np.ndarray  [T]  (NaN in warmup / gap positions)
    is_anomaly: np.ndarray | None  [T]
    warmup_points: int
    """
    from run_forecasting import build_detector, normalize_forecast_array
    from anomaly_detection_forecasting.core import TimeSeriesWrapper

    df = load_csv(csv_path)
    value_cols  = [c for c in df.columns if c.startswith("value_")]
    is_anomaly  = (
        df["is_anomaly"].to_numpy(dtype=int).astype(bool)
        if "is_anomaly" in df.columns else None
    )

    ts_df = df[value_cols].copy()
    ts_df.index = df.index
    time_series = TimeSeriesWrapper(ts_df)

    detector = build_detector(fc_config.get("_model_name", ""), fc_config)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        result = detector(time_series, file_label=csv_path.name) \
            if hasattr(detector, "__call__") and "file_label" in \
               __import__("inspect").signature(detector.__call__).parameters \
            else detector(time_series)

    forecast = normalize_forecast_array(result.expected_value)
    y_true = ts_df[value_cols[0]].to_numpy(dtype=float)
    y_pred = forecast[0, :]

    return df.index, y_true, y_pred, is_anomaly, warmup_points

def estimate_ci(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Return (residuals, sigma, ci95_lo, ci95_hi, ci80_lo, ci80_hi).
    σ is estimated from the global residual distribution with the top 10 %
    of absolute errors trimmed (robust to anomalous spikes).
    """
    valid = ~np.isnan(y_pred)
    if not valid.any():
        nan = np.full(len(y_true), np.nan)
        return nan, 0.0, nan, nan, nan, nan

    residuals = np.where(valid, y_true - y_pred, np.nan)
    finite = residuals[np.isfinite(residuals)]

    if len(finite) >= 10:
        trim_n = max(1, int(0.10 * len(finite)))
        sigma = float(np.sort(np.abs(finite))[:-trim_n].std())
        if sigma == 0:
            sigma = float(np.abs(finite).mean())
    else:
        sigma = float(np.nanstd(residuals)) if np.isfinite(residuals).any() else 0.0

    ci95_lo = np.where(valid, y_pred - 1.96 * sigma, np.nan)
    ci95_hi = np.where(valid, y_pred + 1.96 * sigma, np.nan)
    ci80_lo = np.where(valid, y_pred - 1.28 * sigma, np.nan)
    ci80_hi = np.where(valid, y_pred + 1.28 * sigma, np.nan)

    return residuals, sigma, ci95_lo, ci95_hi, ci80_lo, ci80_hi

def _g(v): return f"{v:.3g}" if np.isfinite(v) else "n/a"
def _gp(v): return f"{v:.1f}%" if np.isfinite(v) else "n/a"
def _gf(v): return f"{v:.3f}" if np.isfinite(v) else "n/a"


def quick_metrics(y_true: np.ndarray, y_pred: np.ndarray, warmup: int) -> dict:
    """Compute summary metrics on the post-warmup, non-NaN portion."""
    yt = y_true[warmup:]
    yp = y_pred[warmup:]
    mask = ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return {k: float("nan") for k in
                ["mae", "rmse", "smape", "r2", "wape", "mase", "max_ae", "bias", "nrmse"]}

    mae_v = float(np.mean(np.abs(yt - yp)))
    rmse_v = float(np.sqrt(np.mean((yt - yp) ** 2)))
    denom_s = np.maximum(np.abs(yt) + np.abs(yp), 1e-8)
    smape_v = float(np.mean(2.0 * np.abs(yt - yp) / denom_s) * 100.0)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2_v = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    wape_v = float(np.sum(np.abs(yt - yp)) / max(np.sum(np.abs(yt)), 1e-8) * 100.0)
    naive = np.abs(np.diff(yt))
    mase_v = (mae_v / float(np.mean(naive))) if len(naive) and np.mean(naive) > 1e-8 \
              else float("nan")
    max_ae_v = float(np.max(np.abs(yt - yp)))
    bias_v = float(np.mean(yp - yt))
    rang = float(np.max(yt) - np.min(yt))
    nrmse_v = (rmse_v / rang * 100.0) if rang > 1e-8 else float("nan")

    return dict(mae=mae_v, rmse=rmse_v, smape=smape_v, r2=r2_v,
                wape=wape_v, mase=mase_v, max_ae=max_ae_v,
                bias=bias_v, nrmse=nrmse_v)

def plot_one(
    times: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_anomaly,
    warmup_points: int,
    model_name: str,
    label: str,
    ax_ts: plt.Axes,
    ax_res: plt.Axes,
) -> None:
    """
    Fill two pre-created Axes:
      ax_ts  — time series panel (actual + forecast + CI bands)
      ax_res — residual panel
    """
    T = len(y_true)

    residuals, sigma, ci95_lo, ci95_hi, ci80_lo, ci80_hi = estimate_ci(y_true, y_pred)
    m    = quick_metrics(y_true, y_pred, warmup_points)
    n_pred = int((~np.isnan(y_pred[warmup_points:])).sum())

    title = (
        f"{model_name}  |  {label}  |  n_eval={n_pred}\n"
        f"MAE={_g(m['mae'])}  RMSE={_g(m['rmse'])}  SMAPE={_gp(m['smape'])}  "
        f"R²={_gf(m['r2'])}  WAPE={_gp(m['wape'])}  MASE={_g(m['mase'])}  "
        f"MaxAE={_g(m['max_ae'])}  Bias={_g(m['bias'])}  NRMSE={_gp(m['nrmse'])}"
    )

    if warmup_points > 0 and warmup_points < T:
        ax_ts.axvspan(
            times[0], times[min(warmup_points, T - 1)],
            color=C_WARMUP, alpha=0.8, zorder=1,
            label=f"warmup ({warmup_points} pts, excluded)",
        )

    if is_anomaly is not None and np.any(is_anomaly):
        first_gt = True
        for t0, t1 in find_spans(is_anomaly, times):
            ax_ts.axvspan(
                t0, t1, color=C_GT_BG, alpha=0.55, zorder=2,
                label="anomaly region" if first_gt else "_nolegend_",
            )
            first_gt = False

    ax_ts.fill_between(
        times, ci95_lo, ci95_hi,
        color=C_CI_95, alpha=0.55, zorder=3,
        label=f"95 % CI  (±1.96σ, σ={sigma:.3g})",
    )
    ax_ts.fill_between(
        times, ci80_lo, ci80_hi,
        color=C_CI_80, alpha=0.65, zorder=4,
        label="80 % CI  (±1.28σ)",
    )
    ax_ts.plot(times, y_true,
               color=C_ACTUAL, linewidth=0.9, zorder=5, label="actual")
    ax_ts.plot(times, y_pred,
               color=C_FORECAST, linewidth=1.1, linestyle="--", zorder=6,
               label="forecast")

    ax_ts.set_title(title, fontsize=8, pad=6)
    ax_ts.set_ylabel("value", fontsize=9)
    ax_ts.legend(fontsize=7, loc="upper right", ncol=3, framealpha=0.85)
    plain_number_formatter(ax_ts, "y")
    ax_ts.tick_params(axis="y", labelsize=7)
    ax_ts.set_facecolor("white")

    if is_anomaly is not None and np.any(is_anomaly):
        for t0, t1 in find_spans(is_anomaly, times):
            ax_res.axvspan(t0, t1, color=C_GT_BG, alpha=0.4, zorder=1)

    ax_res.axhline(0.0, color=C_ZERO, linewidth=0.9, zorder=2)

    ax_res.plot(times, residuals,
                color=C_RESID, linewidth=0.75, zorder=3,
                label="residual  (actual − forecast)")
    ax_res.fill_between(times, 0.0, residuals,
                        color=C_RESID, alpha=0.18, zorder=2)

    if sigma > 0:
        ax_res.axhline( 2 * sigma, color=C_FORECAST, linewidth=0.9,
                        linestyle=":", zorder=4, label=f"+2σ ({2*sigma:.3g})")
        ax_res.axhline(-2 * sigma, color=C_FORECAST, linewidth=0.9,
                        linestyle=":", zorder=4, label=f"−2σ ({-2*sigma:.3g})")

    ax_res.set_ylabel("residual", fontsize=9)
    ax_res.set_xlabel("time", fontsize=9)
    ax_res.legend(fontsize=7, loc="upper right", framealpha=0.85)
    plain_number_formatter(ax_res, "y")
    ax_res.tick_params(axis="x", labelsize=7, rotation=15)
    ax_res.tick_params(axis="y", labelsize=7)
    ax_res.set_facecolor("white")

def collect_files(datasets: list[str]) -> list[Path]:
    files = []
    for ds in datasets:
        d = DATA_ROOT / ds
        if d.exists():
            files.extend(sorted(d.glob("*.csv")))
        else:
            print(f"  [warn] dataset folder not found: {d}")
    return files

def main():
    parser = argparse.ArgumentParser(
        description="Plot forecasting results (actual vs. forecast, residuals)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models",   nargs="+", default=None,
                        help="Model names to plot (default: all in models.json5)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Dataset names (default: all)")
    parser.add_argument("--save",     action="store_true",
                        help="Save figures as PNGs")
    parser.add_argument("--save_dir", default="plots/forecasting",
                        help="Root directory for saved PNGs "
                             "(default: plots/forecasting/)")
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
        grouped: dict[str, list[Path]] = {}
        for p in csv_files:
            grouped.setdefault(p.parent.name, []).append(p)
        csv_files = []
        for files in grouped.values():
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
        fc_params = cfg.get("forecasting_model_params", {})
        fc_params = {**fc_params, "_model_name": model_name}
        warmup = int(
            fc_params.get("warmup_points")
            or fc_params.get("context_length", 512)
        )

        _detector_cache: list = []

        for csv_path in csv_files:
            done += 1
            dataset_name = csv_path.parent.name
            label = f"{dataset_name}/{csv_path.name}"
            print(f"[{done}/{total}] {model_name} | {label}")

            try:
                times, y_true, y_pred, is_anomaly, warmup_pts = run_forecast(
                    csv_path, fc_params, warmup
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            fig, (ax_ts, ax_res) = plt.subplots(
                nrows=2, ncols=1, figsize=(18, 8),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
            fig.patch.set_facecolor("white")

            try:
                plot_one(
                    times=times,
                    y_true=y_true,
                    y_pred=y_pred,
                    is_anomaly=is_anomaly,
                    warmup_points=warmup_pts,
                    model_name=model_name,
                    label=label,
                    ax_ts=ax_ts,
                    ax_res=ax_res,
                )
            except Exception as e:
                ax_ts.set_title(f"PLOT ERROR: {e}", color="red")
                print(f"PLOT ERROR: {e}")

            fig.tight_layout()

            if args.save:
                out_path = (
                    save_dir / dataset_name / model_name
                    / f"{csv_path.stem}.png"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"  saved {out_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
