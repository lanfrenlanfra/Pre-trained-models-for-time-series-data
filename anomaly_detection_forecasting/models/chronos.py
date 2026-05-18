from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult

try:
    import torch
    from chronos import BaseChronosPipeline as ChronosPipeline
    try:
        from chronos import ChronosBoltPipeline as _ChronosBoltPipeline
    except ImportError:
        _ChronosBoltPipeline = None
except Exception:
    torch = None
    ChronosPipeline = None
    _ChronosBoltPipeline = None


@dataclass
class ChronosForecastResult:
    forecast: np.ndarray
    residuals: np.ndarray
    scores: np.ndarray


class ChronosDetector(BaseDetector):
    """
    Anomaly detector based on Amazon Chronos (chronos-bolt-* or chronos-t5-*).

    Idea:
    1. Use Chronos as a probabilistic forecaster on rolling windows.
    2. Use the median (Bolt) or mean of samples (T5) as the point prediction.
    3. Compute anomaly scores as normalized forecast residuals.
    4. Mark points as anomalous when score > threshold.

    Notes:
    - Works with univariate and multivariate series (per-channel).
    - Unsupervised residual-based detector — no labels required.
    - Pipeline is cached at the class level so weights are loaded only once
      per benchmark run, regardless of how many series are evaluated.
    """

    _pipeline_cache: dict = {}

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_name": "Chronos",
            "hf_model_path": "amazon/chronos-t5-small",
            "context_length": 512,
            "prediction_length": 64,
            "num_samples": 20,
            "threshold": None,
            "step": None,
            "device": "cpu",
            "per_channel": True,
            "warmup_points": None,
            "min_std": 1e-6,
            "use_absolute_error": True,
            "batch_size": 8,
            "max_series_count": 0,
            "clean_context": True,
            "clean_context_mad_threshold": 3.0,
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params.get("threshold") is not None and params["threshold"] < 0:
            raise ValueError("threshold must be >= 0 (or None to disable)")
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
        cache_key = (self.params["hf_model_path"], self._resolve_device())
        if cache_key in ChronosDetector._pipeline_cache:
            return ChronosDetector._pipeline_cache[cache_key]

        if ChronosPipeline is None:
            raise ImportError(
                "chronos-forecasting is not installed. Install it with:\n"
                'pip install "chronos-forecasting>=1.4.0"'
            )

        device = self._resolve_device()
        dtype = torch.bfloat16

        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", message=".*torch_dtype.*")
            pipeline = ChronosPipeline.from_pretrained(
                self.params["hf_model_path"],
                device_map=device,
                dtype=dtype,
            )

        ChronosDetector._pipeline_cache[cache_key] = pipeline
        return pipeline

    def _forecast_channel(self, values: np.ndarray,
                          channel_label: str = "?", file_label: str = "?") -> ChronosForecastResult:
        context_length = int(self.params["context_length"])
        prediction_length = int(self.params["prediction_length"])
        step = int(self.params["step"] or prediction_length)
        warmup_points = int(self.params["warmup_points"] or context_length)
        num_samples = int(self.params["num_samples"])
        batch_size = int(self.params.get("batch_size", 8))
        min_std = float(self.params["min_std"])
        use_absolute_error = bool(self.params["use_absolute_error"])

        n = len(values)
        scores = np.full(n, np.nan, dtype=float)
        preds = np.full(n, np.nan, dtype=float)

        if n <= context_length:
            return ChronosForecastResult(
                forecast=np.nan_to_num(preds, nan=float(np.mean(values) if n else 0.0)),
                residuals=np.zeros(n, dtype=float),
                scores=scores,
            )

        max_len = int(self.params.get("max_series_length", 0))
        if max_len > 0 and n > max_len:
            preds[:] = values
            return ChronosForecastResult(
                forecast=preds,
                residuals=np.zeros(n, dtype=float),
                scores=scores,
            )

        do_clean = bool(self.params.get("clean_context", True))
        mad_thr  = float(self.params.get("clean_context_mad_threshold", 3.0))

        pipeline = self._load_pipeline()

        is_bolt = (_ChronosBoltPipeline is not None
                   and isinstance(pipeline, _ChronosBoltPipeline))

        series_std = float(np.std(values[np.isfinite(values)])) if n else 0.0
        sigma_floor = max(min_std, 1e-3 * series_std)
        starts  = list(range(0, n - context_length, step))
        last_start = n - context_length
        if last_start > 0 and (not starts or starts[-1] != last_start):
            starts.append(last_start)
        n_windows = len(starts)
        n_batches = math.ceil(n_windows / batch_size) if starts else 0

        window_data: List[tuple] = []

        QUANTILE_TO_SIGMA = 2.5631

        for batch_i in range(n_batches):
            batch_starts = starts[batch_i * batch_size : (batch_i + 1) * batch_size]

            contexts = np.stack([
                (
                    self.clean_context(values[s : s + context_length], mad_threshold=mad_thr)
                    if do_clean else values[s : s + context_length]
                ).astype(np.float32)
                for s in batch_starts
            ])
            batch_tensor = torch.tensor(contexts, dtype=torch.float32)

            with torch.no_grad():
                if is_bolt:
                    output = pipeline.predict(batch_tensor, prediction_length)
                    out_np = output.cpu().numpy()
                    q_low  = out_np[:, 0, :]
                    q_med  = out_np[:, 4, :]
                    q_high = out_np[:, 8, :]
                    forecast_means = q_med
                    local_sigmas = np.maximum(
                        (q_high - q_low) / QUANTILE_TO_SIGMA,
                        sigma_floor,
                    )
                else:
                    output = pipeline.predict(
                        batch_tensor,
                        prediction_length,
                        num_samples=num_samples,
                    )
                    forecast_means = output.cpu().numpy().mean(axis=1)  # [B, pred_len]
                    local_sigmas = None

            for local_i, start in enumerate(batch_starts):
                ctx_end = start + context_length
                pred_end = min(ctx_end + prediction_length, n)
                horizon = pred_end - ctx_end

                forecast = forecast_means[local_i, :horizon]
                preds[ctx_end:pred_end] = forecast

                actual = values[ctx_end:pred_end]
                residual = actual - forecast
                raw_err = np.abs(residual) if use_absolute_error else residual ** 2

                if local_sigmas is not None:
                    local_sigma = local_sigmas[local_i, :horizon]
                    window_data.append((ctx_end, pred_end, raw_err, local_sigma))
                else:
                    window_data.append((ctx_end, pred_end, raw_err, None))

        if window_data:
            if is_bolt:
                for ctx_end, pred_end, raw_err, local_sigma in window_data:
                    scores[ctx_end:pred_end] = raw_err / local_sigma
            else:
                all_errs = np.concatenate([r for _, _, r, _ in window_data])

                if len(all_errs) >= 10:
                    trim_n = max(1, int(0.10 * len(all_errs)))
                    scale = float(np.mean(np.sort(all_errs)[:-trim_n]))
                else:
                    scale = float(np.mean(all_errs))

                if not np.isfinite(scale) or scale < min_std:
                    scale = min_std

                for ctx_end, pred_end, raw_err, _ in window_data:
                    scores[ctx_end:pred_end] = raw_err / scale

        first_valid = np.where(~np.isnan(preds))[0]
        if len(first_valid):
            first_valid = int(first_valid[0])
            preds[:first_valid] = values[:first_valid]
        else:
            preds[:] = values

        residuals = values - preds
        if warmup_points is not None and warmup_points > 0:
            scores[:warmup_points] = np.nan

        return ChronosForecastResult(forecast=preds, residuals=residuals, scores=scores)

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        values = time_series.time_series_pd["value_0"].to_numpy(dtype=float)
        label = getattr(time_series, "label", "?")
        result = self._forecast_channel(values, channel_label="value_0", file_label=label)

        expected = result.forecast
        residual_std = max(float(np.std(result.residuals[np.isfinite(result.residuals)])), self.params["min_std"])
        is_anomaly, expected_bounds = self.threshold_outputs(
            result.scores, expected=expected, residual_std=residual_std,
        )

        return ModelResult(
            anomaly_scores=result.scores.astype(float),
            is_anomaly=is_anomaly,
            expected_value=expected.astype(float),
            expected_bounds=expected_bounds,
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

        forecast_matrix = np.vstack(per_channel_forecasts)
        score_matrix = np.vstack(per_channel_scores)

        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", message="All-NaN slice encountered")
            aggregated_scores = np.nanmax(score_matrix, axis=0)
        is_anomaly, _ = self.threshold_outputs(aggregated_scores)

        return ModelResult(
            anomaly_scores=aggregated_scores.astype(float),
            is_anomaly=is_anomaly,
            expected_value=forecast_matrix.astype(float),
            expected_bounds=None,
        )
