from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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

        pipeline = ChronosPipeline.from_pretrained(
            self.params["hf_model_path"],
            device_map=device,
            torch_dtype=dtype,
        )

        self._pipeline = pipeline
        return self._pipeline

    def _forecast_channel(self, values: np.ndarray) -> ChronosForecastResult:
        context_length = int(self.params["context_length"])
        prediction_length = int(self.params["prediction_length"])
        step = int(self.params["step"] or prediction_length)
        warmup_points = int(self.params["warmup_points"] or context_length)
        num_samples = int(self.params["num_samples"])
        min_std = float(self.params["min_std"])
        use_absolute_error = bool(self.params["use_absolute_error"])

        n = len(values)
        scores = np.zeros(n, dtype=float)
        preds = np.full(n, np.nan, dtype=float)

        if n <= context_length:
            return ChronosForecastResult(
                forecast=np.nan_to_num(preds, nan=float(np.mean(values) if n else 0.0)),
                residuals=np.zeros(n, dtype=float),
                scores=np.zeros(n, dtype=float),
            )

        pipeline = self._load_pipeline()

        # ── Pass 1: run the model, store raw errors for every window ──────────
        window_data: List[tuple] = []  # (ctx_end, pred_end, raw_err array)

        for start in range(0, n - context_length, step):
            ctx_end = start + context_length
            pred_end = min(ctx_end + prediction_length, n)
            horizon = pred_end - ctx_end

            context = values[start:ctx_end].astype(np.float32)

            # Chronos expects a 1-D or 2-D tensor; shape [1, context_length]
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                # samples shape: [num_samples, 1, prediction_length]
                samples = pipeline.predict(
                    context_tensor,
                    prediction_length,
                    num_samples=num_samples,
                )

            # samples → [num_samples, prediction_length]
            samples_np = samples.squeeze(1).cpu().numpy()   # [S, pred_len]
            forecast_mean = samples_np.mean(axis=0)         # [pred_len]

            forecast = forecast_mean[:horizon]
            preds[ctx_end:pred_end] = forecast

            actual = values[ctx_end:pred_end]
            residual = actual - forecast
            raw_err = np.abs(residual) if use_absolute_error else residual ** 2

            window_data.append((ctx_end, pred_end, raw_err))

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
        result = self._forecast_channel(values)

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

        for col in cols:
            result = self._forecast_channel(df[col].to_numpy(dtype=float))
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
