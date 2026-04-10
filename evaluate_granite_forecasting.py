from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from anomaly_detection.core import TimeSeriesWrapper
from anomaly_detection.models.ar import ARDetector
from anomaly_detection.models.granite_ttm import GraniteTTMDetector


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


def build_detector(model_name: str, model_params: Dict):
    if model_name == "granite_ttm":
        return GraniteTTMDetector(**model_params)
    if model_name == "autoreg":
        ar_params = {
            "model_name": "Autoregressive",
            "order": model_params.get("order", 20),
            "threshold": model_params.get("threshold", 3.0),
        }
        return ARDetector(**ar_params)
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_forecast_array(forecast):
    forecast = np.asarray(forecast, dtype=float)
    if forecast.ndim == 1:
        return forecast[None, :]
    if forecast.ndim == 2:
        return forecast
    raise ValueError(f"Unexpected forecast shape: {forecast.shape}")


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

    # detector = GraniteTTMDetector(**model_params)
    result = detector(time_series)

    forecast = normalize_forecast_array(result.expected_value)

    rows: List[Dict] = []
    warmup_points = int(model_params.get("warmup_points") or model_params["context_length"])

    for i, col in enumerate(value_cols):
        y_true = ts_df[col].to_numpy(dtype=float)[warmup_points:]
        y_pred = forecast[i, warmup_points:] if forecast.shape[0] > 1 else forecast[0, warmup_points:]

        rows.append({
            "model": model_name,
            "csv_path": str(csv_path),
            "series": col,
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "smape": smape(y_true, y_pred),
            "r2": r2(y_true, y_pred),
            "n_eval_points": int(len(y_true)),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name, e.g. Yahoo")
    parser.add_argument("--output_csv", type=str, default="granite_forecasting_metrics.csv")
    parser.add_argument("--hf_model_path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--warmup_points", type=int, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default="granite_ttm,autoreg",
        help="Comma-separated models to evaluate: granite_ttm, autoreg",
    )
    parser.add_argument("--order", type=int, default=20, help="AR order for autoreg baseline")
    args = parser.parse_args()

    dataset_root = Path(args.datasets_root) / args.dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    model_params = {
        "hf_model_path": args.hf_model_path,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "threshold": args.threshold,
        "device": args.device,
        "step": args.step,
        "warmup_points": args.warmup_points,
        "order": args.order,
    }

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    all_rows = []
    for model_name in model_names:
        detector = build_detector(model_name, model_params)
        for csv_path in iter_csv_files(dataset_root):
            all_rows.extend(evaluate_file(csv_path, detector, model_name, model_params))

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False)

    summary = summarize_metrics(df)
    print("\n=== Forecasting summary by model ===")
    print(summary.to_string(index=False))
    print(f"\nSaved per-series metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
