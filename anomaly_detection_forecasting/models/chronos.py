from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult

try:
    import torch
    from chronos import ChronosPipeline
except Exception:  # pragma: no cover
    torch = None
    ChronosPipeline = None


@dataclass
class ChronosForecastResult:
    forecast: np.ndarray
    residuals: np.ndarray
    scores: np.ndarray


class ChronosDetector(BaseDetector):
    """
    Anomaly detector based on Amazon Chronos (chronos-t5-*).

    Idea:
    1. Use Chronos as a probabilistic forecaster on rolling windows.
    2. Use the mean of forecast samples as the point prediction.
    3. Compute anomaly scores as normalized forecast residuals.
    4. Mark points as anomalous when score > threshold.

    Notes:
    - Works with univariate and multivariate series (per-channel).
    - Unsupervised residual-based detector — no labels required.
    - Chronos is slower than AR but is a proper TS foundation model.
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_name": "Chronos",
            "hf_model_path": "amazon/chronos-t5-small",
            "context_length": 512,
            "prediction_length": 64,
            "num_samples": 20,
            "threshold": 3.0,
            "step": None,          # defaults to prediction_length
            "device": "cpu",
            "per_channel": True,
            "warmup_points": None,  # defaults to context_length
            "min_std": 1e-6,
            "use_absolute_error": True,
            "batch_size": 8,       # number of windows per predict() call
            "max_series_length": 10_000,  # skip series longer than this (0 = no limit)
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["threshold"] < 0:
            raise ValueError("threshold must be >= 0")
        if params["context_length"] <= 0:
            raise ValueError("context_length must be > 0")
        if params["prediction_length"] <= 0:
            raise ValueError("prediction_length must be > 0")
        if params["device"] not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, mps, auto")

    def _resolve_device(self) -> str:
        requested = self.params["device"]
        if requested != "auto":
            return requested
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_pipeline(self):
        if hasattr(self, "_pipeline") and self._pipeline is not None:
            return self._pipeline

        if ChronosPipeline is None:
            raise ImportError(
                "chronos-forecasting is not installed. Install it with:\n"
                'pip install "chronos-forecasting>=1.3.0"'
            )

        device = self._resolve_device()
        dtype = torch.float32

        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", message=".*torch_dtype.*")
            pipeline = ChronosPipeline.from_pretrained(
                self.params["hf_model_path"],
                device_map=device,
                dtype=dtype,
            )

        self._pipeline = pipeline
        return self._pipeline

    def _forecast_channel(self, values: np.ndarray,
                          channel_label: str = "?", file_label: str = "?") -> ChronosForecastResult:
        context_length  = int(self.params["context_length"])
        prediction_length = int(self.params["prediction_length"])
        step            = int(self.params["step"] or prediction_length)
        warmup_points   = int(self.params["warmup_points"] or context_length)
        num_samples     = int(self.params["num_samples"])
        batch_size      = int(self.params.get("batch_size", 8))
        min_std         = float(self.params["min_std"])
        use_absolute_error = bool(self.params["use_absolute_error"])

        n = len(values)
        scores = np.zeros(n, dtype=float)
        preds  = np.full(n, np.nan, dtype=float)

        if n <= context_length:
            return ChronosForecastResult(
                forecast=np.nan_to_num(preds, nan=float(np.mean(values) if n else 0.0)),
                residuals=np.zeros(n, dtype=float),
                scores=np.zeros(n, dtype=float),
            )

        max_len = int(self.params.get("max_series_length", 0))
        if max_len > 0 and n > max_len:
            tqdm.write(
                f"  [chronos-ad] SKIP {file_label} | ch={channel_label} | "
                f"n={n} > max_series_length={max_len} — returning neutral scores"
            )
            preds[:] = values  # neutral: predict actual (zero residual)
            return ChronosForecastResult(
                forecast=preds,
                residuals=np.zeros(n, dtype=float),
                scores=np.zeros(n, dtype=float),
            )

        pipeline = self._load_pipeline()

        # Collect all window starts up-front for batching
        starts    = list(range(0, n - context_length, step))
        n_windows = len(starts)
        n_batches = math.ceil(n_windows / batch_size) if starts else 0

        tqdm.write(
            f"  [chronos-ad] {file_label} | ch={channel_label} | "
            f"n={n}, windows={n_windows}, batches={n_batches} "
            f"(batch_size={batch_size}, step={step})"
        )

        # ── Pass 1: batched predict, store raw errors per window ──────────────
        window_data: List[tuple] = []  # (ctx_end, pred_end, raw_err)

        t_start = time.perf_counter()
        for batch_i in range(n_batches):
            batch_starts = starts[batch_i * batch_size : (batch_i + 1) * batch_size]

            # Build batch tensor [B, context_length]
            contexts = np.stack([
                values[s : s + context_length].astype(np.float32)
                for s in batch_starts
            ])
            batch_tensor = torch.tensor(contexts, dtype=torch.float32)

            t0 = time.perf_counter()
            with torch.no_grad():
                # samples shape: [B, num_samples, prediction_length]
                samples = pipeline.predict(
                    batch_tensor,
                    prediction_length,
                    num_samples=num_samples,
                )
            elapsed = time.perf_counter() - t0

            remaining = (n_batches - batch_i - 1) * elapsed
            eta = f"  ETA ~{remaining/60:.1f}min" if remaining > 0 else ""
            tqdm.write(
                f"  [chronos-ad]   batch {batch_i + 1}/{n_batches} "
                f"({len(batch_starts)} windows) → {elapsed:.1f}s{eta}"
            )

            # samples: [B, num_samples, pred_len] → mean → [B, pred_len]
            forecast_means = samples.cpu().numpy().mean(axis=1)  # [B, pred_len]

            for local_i, start in enumerate(batch_starts):
                ctx_end  = start + context_length
                pred_end = min(ctx_end + prediction_length, n)
                horizon  = pred_end - ctx_end

                forecast = forecast_means[local_i, :horizon]
                preds[ctx_end:pred_end] = forecast

                actual   = values[ctx_end:pred_end]
                residual = actual - forecast
                raw_err  = np.abs(residual) if use_absolute_error else residual ** 2
                window_data.append((ctx_end, pred_end, raw_err))

        tqdm.write(
            f"  [chronos-ad] {file_label} | ch={channel_label} done "
            f"in {time.perf_counter() - t_start:.1f}s"
        )

        # ── Pass 2: global trimmed-mean scale, then score ─────────────────────
        if window_data:
            all_errs = np.concatenate([r for _, _, r in window_data])

            if len(all_errs) >= 10:
                trim_n = max(1, int(0.10 * len(all_errs)))
                scale = float(np.mean(np.sort(all_errs)[:-trim_n]))
            else:
                scale = float(np.mean(all_errs))

            if not np.isfinite(scale) or scale < min_std:
                scale = min_std

            for ctx_end, pred_end, raw_err in window_data:
                scores[ctx_end:pred_end] = raw_err / scale

        # Fill untouched prefix with actual values
        first_valid = np.where(~np.isnan(preds))[0]
        if len(first_valid):
            first_valid = int(first_valid[0])
            preds[:first_valid] = values[:first_valid]
        else:
            preds[:] = values

        residuals = values - preds
        if warmup_points is not None:
            scores[:warmup_points] = 0.0

        return ChronosForecastResult(forecast=preds, residuals=residuals, scores=scores)

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        values = time_series.time_series_pd["value_0"].to_numpy(dtype=float)
        label = getattr(time_series, "label", "?")
        result = self._forecast_channel(values, channel_label="value_0", file_label=label)

        threshold = float(self.params["threshold"])
        expected = result.forecast
        residual_std = max(float(np.std(result.residuals[np.isfinite(result.residuals)])), self.params["min_std"])
        expected_bounds = np.column_stack(
            [expected - threshold * residual_std, expected + threshold * residual_std]
        )

        return ModelResult(
            anomaly_scores=result.scores.astype(float),
            is_anomaly=(result.scores > threshold),
            expected_value=expected.astype(float),
            expected_bounds=expected_bounds.astype(float),
        )

    def _detect_multivariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        df = time_series.time_series_pd
        cols = list(df.columns)

        per_channel_forecasts = []
        per_channel_scores = []

        label = getattr(time_series, "label", "?")
        for col in cols:
            result = self._forecast_channel(df[col].to_numpy(dtype=float),
                                            channel_label=col, file_label=label)
            per_channel_forecasts.append(result.forecast.astype(float))
            per_channel_scores.append(result.scores.astype(float))

        forecast_matrix = np.vstack(per_channel_forecasts)  # [C, T]
        score_matrix = np.vstack(per_channel_scores)        # [C, T]

        # max aggregation: effective for sparse anomalies in one channel
        aggregated_scores = np.max(score_matrix, axis=0)
        threshold = float(self.params["threshold"])

        return ModelResult(
            anomaly_scores=aggregated_scores.astype(float),
            is_anomaly=(aggregated_scores > threshold),
            expected_value=forecast_matrix.astype(float),
            expected_bounds=None,
        )
