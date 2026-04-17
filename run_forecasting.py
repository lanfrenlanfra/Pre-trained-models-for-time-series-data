from __future__ import annotations

import argparse
import json5
import math
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from termcolor import colored
from statsmodels.tsa.ar_model import AutoReg

from anomaly_detection_forecasting.core import TimeSeriesWrapper
from anomaly_detection_forecasting.models.base import ModelResult
from anomaly_detection_forecasting.models.granite_ttm import GraniteTTMDetector

try:
    import torch as _torch
    from chronos import ChronosPipeline as _ChronosPipeline
except Exception:
    _torch = None
    _ChronosPipeline = None


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


class ARRollingForecaster:
    """
    AR model with rolling out-of-sample forecasting — a fair comparison to GraniteTTM.

    ARDetector (used for anomaly detection) fits the model on the ENTIRE series and
    returns in-sample fittedvalues.  That is data leakage when used as a forecasting
    baseline: the model "saw" the test points during training, so its errors are
    unrealistically low.

    This class mirrors GraniteTTM's rolling-window scheme exactly:
      - slide a context window of `context_length` points,
      - fit AR on that window only,
      - produce an out-of-sample forecast for the next `prediction_length` points,
      - advance by `step` and repeat.
    All predictions are strictly out-of-sample.
    """

    def __init__(self, order: int, context_length: int, prediction_length: int,
                 step: int = None, warmup_points: int = None, **kwargs):
        self.order = order
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.step = step or prediction_length
        self.warmup_points = warmup_points or context_length

    def _forecast_channel(self, values: np.ndarray) -> np.ndarray:
        n = len(values)
        preds = np.full(n, np.nan, dtype=float)

        for start in range(0, n - self.context_length, self.step):
            ctx_end = start + self.context_length
            pred_end = min(ctx_end + self.prediction_length, n)
            horizon = pred_end - ctx_end

            context = values[start:ctx_end]
            fallback = float(np.mean(context))  # sensible default if model fails

            try:
                fit = AutoReg(context, lags=self.order, old_names=False).fit()

                # Stability check: all roots of the AR polynomial must lie
                # *outside* the unit circle for the model to be stationary.
                # Non-stationary models produce exponentially diverging forecasts.
                # suppress the divide-by-zero RuntimeWarning that statsmodels
                # emits when a root is exactly zero (reciprocal of 0).
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    roots = fit.roots
                if len(roots) > 0 and not np.all(np.abs(roots) > 1.0 + 1e-6):
                    preds[ctx_end:pred_end] = fallback
                    continue

                fc = fit.predict(start=len(context), end=len(context) + horizon - 1)

                # Hard clip: predictions must stay within 10× the observed range
                # of the context window to catch any residual numerical blow-up.
                ctx_min, ctx_max = float(np.min(context)), float(np.max(context))
                ctx_span = max(ctx_max - ctx_min, 1e-6)
                fc = np.clip(fc, ctx_min - 10 * ctx_span, ctx_max + 10 * ctx_span)

                preds[ctx_end:pred_end] = fc[:horizon]

            except Exception:
                preds[ctx_end:pred_end] = fallback

        # Fill the warmup prefix (no forecast available) with actual values
        first_valid_idx = np.where(~np.isnan(preds))[0]
        if len(first_valid_idx):
            preds[: first_valid_idx[0]] = values[: first_valid_idx[0]]
        else:
            preds[:] = values
        return preds

    def __call__(self, time_series: TimeSeriesWrapper) -> ModelResult:
        df = time_series.time_series_pd
        cols = list(df.columns)

        forecasts = [
            self._forecast_channel(df[col].to_numpy(dtype=float))
            for col in cols
        ]
        forecast_matrix = np.vstack(forecasts)  # [C, T]
        n = forecast_matrix.shape[1]

        return ModelResult(
            anomaly_scores=np.zeros(n, dtype=float),
            is_anomaly=np.zeros(n, dtype=bool),
            expected_value=forecast_matrix,
            expected_bounds=None,
        )


class ChronosRollingForecaster:
    """
    Chronos (amazon/chronos-t5-*) rolling out-of-sample forecaster.

    Mirrors GraniteTTMDetector's rolling-window scheme for a fair comparison:
      - slide a context window of `context_length` points,
      - produce a probabilistic forecast for the next `prediction_length` points
        and use the sample mean as the point prediction,
      - advance by `step` and repeat.
    All predictions are strictly out-of-sample.
    """

    def __init__(self, hf_model_path: str = "amazon/chronos-t5-small",
                 context_length: int = 512, prediction_length: int = 64,
                 num_samples: int = 20, step: int = None,
                 warmup_points: int = None, device: str = "cpu",
                 batch_size: int = 8, **kwargs):
        self.hf_model_path = hf_model_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.step = step or prediction_length
        self.warmup_points = warmup_points or context_length
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if _ChronosPipeline is None:
            raise ImportError(
                "chronos-forecasting is not installed. Install it with:\n"
                'pip install "chronos-forecasting>=1.3.0"'
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch_dtype.*")
            self._pipeline = _ChronosPipeline.from_pretrained(
                self.hf_model_path,
                device_map=self.device,
                dtype=_torch.float32,
            )
        return self._pipeline

    def _forecast_channel(self, values: np.ndarray,
                          channel_label: str = "?", file_label: str = "?") -> np.ndarray:
        n = len(values)
        preds = np.full(n, np.nan, dtype=float)
        pipeline = self._load_pipeline()

        # Collect all window start positions up front
        starts = list(range(0, n - self.context_length, self.step))
        n_windows = len(starts)
        n_batches = math.ceil(n_windows / self.batch_size) if starts else 0

        tqdm.write(
            f"  [chronos] {file_label} | ch={channel_label} | "
            f"n={n}, windows={n_windows}, batches={n_batches} "
            f"(batch_size={self.batch_size}, step={self.step})"
        )

        t_channel = time.perf_counter()
        for batch_i in range(n_batches):
            batch_starts = starts[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]

            # Build batch tensor [B, context_length]
            contexts = np.stack([
                values[s : s + self.context_length].astype(np.float32)
                for s in batch_starts
            ])  # [B, ctx_len]
            batch_tensor = _torch.tensor(contexts, dtype=_torch.float32)

            t0 = time.perf_counter()
            with _torch.no_grad():
                # samples shape: [B, num_samples, prediction_length]
                samples = pipeline.predict(
                    batch_tensor,
                    self.prediction_length,
                    num_samples=self.num_samples,
                )
            elapsed = time.perf_counter() - t0

            remaining = (n_batches - batch_i - 1) * elapsed
            eta = f"  ETA ~{remaining/60:.1f}min" if remaining > 0 else ""
            tqdm.write(
                f"  [chronos]   batch {batch_i + 1}/{n_batches} "
                f"({len(batch_starts)} windows) → {elapsed:.1f}s{eta}"
            )

            # samples: [B, num_samples, pred_len] → mean over samples → [B, pred_len]
            forecast_means = samples.cpu().numpy().mean(axis=1)  # [B, pred_len]

            for local_i, start in enumerate(batch_starts):
                ctx_end = start + self.context_length
                pred_end = min(ctx_end + self.prediction_length, n)
                horizon = pred_end - ctx_end
                preds[ctx_end:pred_end] = forecast_means[local_i, :horizon]

        tqdm.write(
            f"  [chronos] {file_label} | ch={channel_label} done "
            f"in {time.perf_counter() - t_channel:.1f}s"
        )

        # Fill warmup prefix with actual values.
        # NaN gaps between windows (when step > prediction_length) are left as-is:
        # evaluate_file will compute metrics only on actually-predicted points.
        first_valid_idx = np.where(~np.isnan(preds))[0]
        if len(first_valid_idx):
            preds[: first_valid_idx[0]] = values[: first_valid_idx[0]]
        else:
            preds[:] = values

        return preds

    def __call__(self, time_series: TimeSeriesWrapper,
                 file_label: str = "?") -> ModelResult:
        df = time_series.time_series_pd
        cols = list(df.columns)

        forecasts = [
            self._forecast_channel(df[col].to_numpy(dtype=float),
                                   channel_label=col, file_label=file_label)
            for col in cols
        ]
        forecast_matrix = np.vstack(forecasts)  # [C, T]
        n = forecast_matrix.shape[1]

        return ModelResult(
            anomaly_scores=np.zeros(n, dtype=float),
            is_anomaly=np.zeros(n, dtype=bool),
            expected_value=forecast_matrix,
            expected_bounds=None,
        )


def build_detector(model_name: str, model_params: Dict):
    if model_name == "granite_ttm":
        return GraniteTTMDetector(**model_params)
    if model_name == "autoreg":
        return ARRollingForecaster(
            order=model_params.get("order", 20),
            context_length=model_params["context_length"],
            prediction_length=model_params["prediction_length"],
            step=model_params.get("step"),
            warmup_points=model_params.get("warmup_points"),
        )
    if model_name == "chronos":
        return ChronosRollingForecaster(
            hf_model_path=model_params.get("hf_model_path", "amazon/chronos-t5-small"),
            context_length=model_params["context_length"],
            prediction_length=model_params["prediction_length"],
            num_samples=model_params.get("num_samples", 20),
            step=model_params.get("step"),
            warmup_points=model_params.get("warmup_points"),
            device=model_params.get("device", "cpu"),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_forecast_array(forecast):
    forecast = np.asarray(forecast, dtype=float)
    if forecast.ndim == 1:
        return forecast[None, :]
    if forecast.ndim == 2:
        return forecast
    raise ValueError(f"Unexpected forecast shape: {forecast.shape}")


def value_to_color_higher_better(val: float) -> str:
    """Color for metrics where higher is better (e.g. R2), normalized 0-1."""
    try:
        v = float(val)
    except Exception:
        return str(val)
    v = min(max(v, 0.0), 1.0)
    if v >= 0.85:
        return colored(f"{val:.4g}", "green", attrs=["bold"])
    elif v >= 0.5:
        return colored(f"{val:.4g}", "yellow")
    elif v >= 0.2:
        return colored(f"{val:.4g}", "magenta")
    else:
        return colored(f"{val:.4g}", "red", attrs=["bold"])


def value_to_color_lower_better(val: float, col_min: float, col_max: float) -> str:
    """Color for metrics where lower is better (MAE, RMSE, MAPE, SMAPE).
    Normalizes within column so the best (lowest) value is green."""
    try:
        v = float(val)
    except Exception:
        return str(val)
    span = col_max - col_min
    if span == 0:
        norm = 1.0  # only one value — treat as best
    else:
        # inverted: low raw value → high norm → green
        norm = 1.0 - (v - col_min) / span
    norm = min(max(norm, 0.0), 1.0)
    if norm >= 0.85:
        return colored(f"{val:.4g}", "green", attrs=["bold"])
    elif norm >= 0.5:
        return colored(f"{val:.4g}", "yellow")
    elif norm >= 0.2:
        return colored(f"{val:.4g}", "magenta")
    else:
        return colored(f"{val:.4g}", "red", attrs=["bold"])


def print_colored_table(df: pd.DataFrame, title: str, higher_is_better: bool = False) -> None:
    print(f"\n=== {title} ===")
    all_rows_str = [[str(val) for val in row] for row in df.values]
    idx_width = max(len(str(idx)) for idx in df.index)
    col_widths = []
    for col_idx, col_label in enumerate(df.columns):
        max_data = max(len(str(row[col_idx])) for row in all_rows_str)
        col_widths.append(max(max_data, len(str(col_label)), 6))

    # Pre-compute per-column min/max for lower-is-better coloring
    col_stats = {}
    if not higher_is_better:
        for col_idx, col_label in enumerate(df.columns):
            vals = []
            for row in df.values:
                try:
                    vals.append(float(row[col_idx]))
                except Exception:
                    pass
            col_stats[col_idx] = (min(vals) if vals else 0, max(vals) if vals else 1)

    # Header
    hdr = " " * (idx_width + 2)
    for col_label, width in zip(df.columns, col_widths):
        hdr += f"{str(col_label):<{width}}  "
    print(hdr)

    # Rows
    for idx, row, row_str in zip(df.index, df.values, all_rows_str):
        line = f"{str(idx):<{idx_width}}  "
        for col_idx, (val, val_str, width) in enumerate(zip(row, row_str, col_widths)):
            try:
                fval = float(val)
                if higher_is_better:
                    cval = value_to_color_higher_better(fval)
                else:
                    cmin, cmax = col_stats[col_idx]
                    cval = value_to_color_lower_better(fval, cmin, cmax)
            except Exception:
                cval = val_str
            pad = width - len(val_str)
            line += cval + " " * pad + "  "
        print(line)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")[["mae", "rmse", "mape", "smape", "r2"]]
        .mean()
        .reset_index()
    )


def iter_csv_files(dataset_root: Path):
    for p in sorted(dataset_root.rglob("*.csv")):
        yield p


def read_ts(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    value_cols = [c for c in df.columns if c.startswith("value_")]
    if not value_cols:
        raise ValueError(f"{csv_path}: no value_* columns")
    return df[["timestamp", *value_cols]].copy()

def evaluate_file(csv_path: Path, detector, model_name: str, model_params: Dict) -> List[Dict]:
    df = read_ts(csv_path)
    value_cols = [c for c in df.columns if c.startswith("value_")]
    ts_df = df[value_cols].copy()
    ts_df.index = pd.to_datetime(df["timestamp"])

    time_series = TimeSeriesWrapper(ts_df)

    if isinstance(detector, ChronosRollingForecaster):
        result = detector(time_series, file_label=csv_path.name)
    else:
        result = detector(time_series)

    forecast = normalize_forecast_array(result.expected_value)

    rows: List[Dict] = []
    warmup_points = int(model_params.get("warmup_points") or model_params["context_length"])

    for i, col in enumerate(value_cols):
        y_true = ts_df[col].to_numpy(dtype=float)[warmup_points:]
        y_pred = forecast[i, warmup_points:] if forecast.shape[0] > 1 else forecast[0, warmup_points:]

        # Drop positions where the model produced no prediction (NaN gaps).
        # This occurs when step > prediction_length (e.g. chronos with step=512,
        # pred=64). Metrics are computed only on actually-predicted points so
        # that gaps don't distort scores via fake fill values.
        valid = ~np.isnan(y_pred)
        y_true_eval = y_true[valid]
        y_pred_eval = y_pred[valid]

        rows.append({
            "model": model_name,
            "csv_path": str(csv_path),
            "series": col,
            "mae": mae(y_true_eval, y_pred_eval),
            "rmse": rmse(y_true_eval, y_pred_eval),
            "mape": mape(y_true_eval, y_pred_eval),
            "smape": smape(y_true_eval, y_pred_eval),
            "r2": r2(y_true_eval, y_pred_eval),
            "n_eval_points": int(valid.sum()),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Forecasting Benchmark")
    parser.add_argument("--datasets_root", type=str, default="data")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated dataset names, e.g. AIOPS,NAB,TODS,UCR,WSD,Yahoo"
    )
    parser.add_argument("--f_time_series_metrics_csv", type=str, default="forecasting_per_series.csv")
    parser.add_argument(
        "--f_output_csv",
        type=str,
        default=None,
        help="Path to save aggregated summary (dataset × model), "
             "e.g. forecasting_summary.csv",
    )
    parser.add_argument(
        "--models",
        type=argparse.FileType("r"),
        required=True,
        help="Path to JSON5 file with model configurations (e.g. models.json5)",
    )
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    try:
        configurations = json5.load(args.models)
    except Exception as e:
        raise SystemExit(f"Could not parse models JSON5 file: {e}")

    # Build one detector per model using its forecasting_model_params section
    model_names = list(configurations.keys())
    per_model_params: Dict[str, Dict] = {}
    detectors: Dict = {}

    for model_name, config in configurations.items():
        if "forecasting_model_params" not in config:
            raise SystemExit(
                f"Model '{model_name}' in models JSON5 has no 'forecasting_model_params' section. "
                "Add a 'forecasting_model_params' block for each model you want to forecast with."
            )
        mp = config["forecasting_model_params"]
        per_model_params[model_name] = mp
        detectors[model_name] = build_detector(model_name, mp)

    # Pre-load model pipelines so the download/init happens before tqdm bars start
    for model_name, detector in detectors.items():
        if hasattr(detector, "_load_pipeline"):
            print(f"Loading {model_name} model...", flush=True)
            detector._load_pipeline()
            # Warmup: trigger JIT compilation / first-inference overhead now,
            # before tqdm bars appear, so the first real file doesn't stall at 0%.
            if _torch is not None:
                _ctx_len = getattr(detector, "context_length", 512)
                _pred_len = getattr(detector, "prediction_length", 64)
                _n_samples = getattr(detector, "num_samples", 20)
                _dummy = _torch.zeros(1, _ctx_len, dtype=_torch.float32)
                print(f"  Warming up {model_name} (ctx={_ctx_len}, pred={_pred_len}, samples={_n_samples})...", flush=True)
                t0 = time.perf_counter()
                with _torch.no_grad():
                    detector._pipeline.predict(_dummy, _pred_len, num_samples=_n_samples)
                print(f"  Warmup done in {time.perf_counter() - t0:.1f}s", flush=True)
            print(f"{model_name} ready.", flush=True)
        elif hasattr(detector, "_load_model"):
            print(f"Loading {model_name} model...", flush=True)
            detector._load_model()
            print(f"{model_name} ready.", flush=True)

    all_rows = []

    for dataset_name in dataset_names:
        dataset_root = Path(args.datasets_root) / dataset_name

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

        for model_name, detector in detectors.items():
            mp = per_model_params[model_name]
            csv_files = list(iter_csv_files(dataset_root))
            pbar = tqdm(
                csv_files,
                desc=f"{dataset_name} | {model_name}",
            )
            for csv_path in pbar:
                rows = evaluate_file(csv_path, detector, model_name, mp)

                for row in rows:
                    row["dataset"] = dataset_name

                all_rows.extend(rows)
            pbar.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(args.f_time_series_metrics_csv, index=False)

    # --- Summary CSV: one row per (dataset, model), averaged over all series ---
    summary = (
        df.groupby(["dataset", "model"])[["mae", "rmse", "mape", "smape", "r2"]]
        .mean()
        .round(6)
        .reset_index()
    )
    # Reorder rows: dataset order from CLI, then model order from CLI
    summary["_d"] = summary["dataset"].apply(
        lambda x: dataset_names.index(x) if x in dataset_names else len(dataset_names)
    )
    summary["_m"] = summary["model"].apply(
        lambda x: model_names.index(x) if x in model_names else len(model_names)
    )
    summary = summary.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"]).reset_index(drop=True)

    if args.f_output_csv:
        summary.to_csv(args.f_output_csv, index=False)
        print(f"Saved summary metrics to:    {args.f_output_csv}")

    model_order = model_names  # preserve CLI order

    # Print one colored pivot table per metric (rows=models, columns=datasets)
    for metric, higher_is_better in [
        ("mae",   False),
        ("rmse",  False),
        ("mape",  False),
        ("smape", False),
        ("r2",    True),
    ]:
        pivot = (
            df.groupby(["model", "dataset"])[metric]
            .mean()
            .reset_index()
            .pivot(index="model", columns="dataset", values=metric)
        )
        pivot = pivot.reindex(index=model_order, columns=dataset_names)
        pivot = pivot.round(4)
        print_colored_table(pivot, title=metric.upper(), higher_is_better=higher_is_better)

    print(f"\nSaved per-series metrics to: {args.f_time_series_metrics_csv}")


if __name__ == "__main__":
    main()
