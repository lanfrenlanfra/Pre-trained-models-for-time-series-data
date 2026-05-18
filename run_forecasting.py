from __future__ import annotations

import argparse
import json5
import math
import os
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from termcolor import colored
from statsmodels.tsa.ar_model import AutoReg

from anomaly_detection_forecasting.core import TimeSeriesWrapper
from anomaly_detection_forecasting.models.base import ModelResult
from anomaly_detection_forecasting.models.granite_ttm import GraniteTTMDetector

try:
    import torch as _torch
    from chronos import BaseChronosPipeline as _ChronosPipeline  # auto-routes to ChronosPipeline (T5) or ChronosBoltPipeline
    from chronos import ChronosBoltPipeline as _ChronosBoltPipeline
except Exception:
    _torch = None
    _ChronosPipeline = None
    _ChronosBoltPipeline = None

try:
    import timesfm as _timesfm
except Exception:
    _timesfm = None

try:
    from transformers import PatchTSTForPrediction as _PatchTSTForPrediction
except Exception:
    _PatchTSTForPrediction = None

def _squeeze_forecast(forecast_tensor) -> np.ndarray:
    """Accept [B, 1, pred_len] or [B, pred_len] and return numpy [B, pred_len]."""
    arr = forecast_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.squeeze(1)
    return arr

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
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error = Σ|e| / Σ|y| × 100.
    Aggregates MAPE properly; robust when individual y values are near 0."""
    total = float(np.sum(np.abs(y_true)))
    if total < 1e-8:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100.0)

def mase(y_true, y_pred, seasonality: int = 1):
    """Mean Absolute Scaled Error.
    Scale = MAE of the naive seasonal baseline (lag-s random walk) on y_true.
    MASE < 1 means the model beats the naive baseline; scale-free and stable
    even when y contains zeros — critical for error-rate / throughput series."""
    n = len(y_true)
    if n <= seasonality:
        return float("nan")
    naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
    scale = float(np.mean(naive_errors))
    if scale < 1e-8:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def max_ae(y_true, y_pred):
    """Max Absolute Error — worst single-point deviation.
    Essential for SLA compliance monitoring: a 99th-percentile tail spike
    that RMSE smooths out may still breach an SLA threshold."""
    return float(np.max(np.abs(y_true - y_pred)))

def bias_metric(y_true, y_pred):
    """Mean Signed Error = mean(ŷ - y).
    Positive → model systematically over-forecasts (capacity over-provisioning risk).
    Negative → model systematically under-forecasts (capacity under-provisioning risk)."""
    return float(np.mean(y_pred - y_true))

def nrmse(y_true, y_pred):
    """Normalized RMSE = RMSE / (max(y) - min(y)) × 100.
    Makes RMSE comparable across different metric scales (e.g., CPU % vs. bytes/s)."""
    r = float(np.max(y_true) - np.min(y_true))
    if r < 1e-8:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / r * 100.0)

_ALL_METRICS = ["mae", "rmse", "mape", "smape", "r2", "wape", "mase", "max_ae", "bias", "nrmse"]
_METRIC_DIRECTION = {
    "mae": False, "rmse": False, "mape": False, "smape": False,
    "r2": True,
    "wape": False, "mase": False, "max_ae": False,
    "bias": None,   # zero is best
    "nrmse": False,
}


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

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    roots = fit.roots
                if len(roots) > 0 and not np.all(np.abs(roots) > 1.0 + 1e-6):
                    preds[ctx_end:pred_end] = fallback
                    continue

                fc = fit.predict(start=len(context), end=len(context) + horizon - 1)

                ctx_min, ctx_max = float(np.min(context)), float(np.max(context))
                ctx_span = max(ctx_max - ctx_min, 1e-6)
                fc = np.clip(fc, ctx_min - 10 * ctx_span, ctx_max + 10 * ctx_span)

                preds[ctx_end:pred_end] = fc[:horizon]

            except Exception:
                preds[ctx_end:pred_end] = fallback

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
                dtype=_torch.bfloat16,
            )
        return self._pipeline

    def _forecast_channel(self, values: np.ndarray,
                          channel_label: str = "?", file_label: str = "?") -> np.ndarray:
        n = len(values)
        preds = np.full(n, np.nan, dtype=float)
        pipeline = self._load_pipeline()

        starts = list(range(0, n - self.context_length, self.step))
        n_batches = math.ceil(len(starts) / self.batch_size) if starts else 0

        for batch_i in range(n_batches):
            batch_starts = starts[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]

            contexts = np.stack([
                values[s : s + self.context_length].astype(np.float32)
                for s in batch_starts
            ])
            batch_tensor = _torch.tensor(contexts, dtype=_torch.float32)

            with _torch.no_grad():
                if _ChronosBoltPipeline is not None and isinstance(pipeline, _ChronosBoltPipeline):
                    output = pipeline.predict(batch_tensor, self.prediction_length)
                    forecast_means = output.cpu().numpy()[:, 4, :]
                else:
                    output = pipeline.predict(batch_tensor, self.prediction_length, num_samples=self.num_samples)
                    forecast_means = output.cpu().numpy().mean(axis=1)

            for local_i, start in enumerate(batch_starts):
                ctx_end = start + self.context_length
                pred_end = min(ctx_end + self.prediction_length, n)
                horizon = pred_end - ctx_end
                preds[ctx_end:pred_end] = forecast_means[local_i, :horizon]

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
        forecast_matrix = np.vstack(forecasts)
        n = forecast_matrix.shape[1]

        return ModelResult(
            anomaly_scores=np.zeros(n, dtype=float),
            is_anomaly=np.zeros(n, dtype=bool),
            expected_value=forecast_matrix,
            expected_bounds=None,
        )


class TimesFMRollingForecaster:
    """
    Google TimesFM rolling out-of-sample forecaster.
    Mirrors the rolling-window scheme of other models:
      - slide a context window of `context_length` points,
      - produce a point forecast for the next `prediction_length` steps,
      - advance by `step` and repeat.
    All predictions are strictly out-of-sample.
    Reference: https://github.com/google-research/timesfm
    """

    def __init__(
        self,
        hf_model_path: str = "google/timesfm-2.0-500m-pytorch",
        context_length: int = 512,
        prediction_length: int = 96,
        step: int = None,
        warmup_points: int = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_series_count: int = 0,
        **kwargs,
    ):
        self.hf_model_path = hf_model_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.step = step or prediction_length
        self.warmup_points = warmup_points or context_length
        self.device = device
        self.batch_size = batch_size
        self.max_series_count = max_series_count
        self._model = None
        self._series_processed = 0

    _model_cache: dict = {}

    def reset_series_counter(self) -> None:
        self._series_processed = 0

    def _resolve_device(self):
        """Return (timesfm_backend, post_load_torch_device_or_None).
        Upstream timesfm only knows ``cpu`` / ``gpu`` / ``tpu``. For Apple
        Silicon there's no native MPS path, so we load with ``backend="cpu"``
        and then move the underlying PyTorch module onto MPS afterwards. We
        also flip ``tfm.backend = "gpu"`` so timesfm's ``_forecast`` calls
        ``.cpu()`` before ``.numpy()`` (otherwise an MPS tensor crashes
        ``.numpy()``).
        """
        requested = str(self.device).lower()
        has_cuda = bool(_torch and _torch.cuda.is_available())
        has_mps = bool(
            _torch
            and getattr(_torch.backends, "mps", None) is not None
            and _torch.backends.mps.is_available()
        )

        if requested in ("gpu", "cuda") and has_cuda:
            return "gpu", None
        if requested == "mps" and has_mps:
            return "cpu", _torch.device("mps")
        if requested == "auto":
            if has_cuda:
                return "gpu", None
            if has_mps:
                return "cpu", _torch.device("mps")
            return "cpu", None
        return "cpu", None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if _timesfm is None:
            raise ImportError(
                "timesfm is not installed. Install it with:\n"
                "    pip install timesfm"
            )

        backend, post_device = self._resolve_device()
        cache_key = (
            self.hf_model_path,
            backend,
            int(self.batch_size),
            str(post_device) if post_device is not None else "default",
        )
        cached = TimesFMRollingForecaster._model_cache.get(cache_key)
        if cached is not None:
            self._model = cached
            return cached

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tfm = _timesfm.TimesFm(
                hparams=_timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=int(self.batch_size),
                    horizon_len=self.prediction_length,
                ),
                checkpoint=_timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.hf_model_path
                ),
            )

        if post_device is not None and _torch is not None:
            try:
                tfm._device = post_device
                if getattr(tfm, "_model", None) is not None:
                    tfm._model = tfm._model.to(post_device)
                    tfm._model.eval()
                tfm.backend = "gpu"
            except Exception:
                pass

        TimesFMRollingForecaster._model_cache[cache_key] = tfm
        self._model = tfm
        return tfm

    def _forecast_channel(
        self,
        values: np.ndarray,
        channel_label: str = "?",
        file_label: str = "?",
    ) -> np.ndarray:
        ctx = self.context_length
        pred_len = self.prediction_length
        step = self.step
        batch_size = self.batch_size
        n = len(values)

        preds = np.full(n, np.nan, dtype=float)
        model = self._load_model()
        starts = list(range(0, n - ctx, step))
        n_batches = math.ceil(len(starts) / batch_size) if starts else 0

        for batch_i in range(n_batches):
            batch_starts = starts[batch_i * batch_size : (batch_i + 1) * batch_size]
            contexts = [values[s : s + ctx].astype(np.float32) for s in batch_starts]
            freqs = [0] * len(batch_starts)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    point_forecasts, _ = model.forecast(inputs=contexts, freq=freqs)
                for j, start in enumerate(batch_starts):
                    ctx_end = start + ctx
                    pred_end = min(ctx_end + pred_len, n)
                    horizon = pred_end - ctx_end
                    preds[ctx_end:pred_end] = point_forecasts[j, :horizon]
            except Exception:
                for j, start in enumerate(batch_starts):
                    ctx_end = start + ctx
                    pred_end = min(ctx_end + pred_len, n)
                    preds[ctx_end:pred_end] = float(np.mean(contexts[j]))

        first_valid = np.where(~np.isnan(preds))[0]
        if len(first_valid):
            preds[: first_valid[0]] = values[: first_valid[0]]
        else:
            preds[:] = values
        return preds

    def __call__(
        self, time_series: TimeSeriesWrapper, file_label: str = "?"
    ) -> ModelResult:
        df = time_series.time_series_pd
        cols = list(df.columns)
        n = len(df)

        if self.max_series_count > 0 and self._series_processed >= self.max_series_count:
            return ModelResult(
                anomaly_scores = np.zeros(n, dtype=float),
                is_anomaly = np.zeros(n, dtype=bool),
                expected_value = np.full((len(cols), n), np.nan),
                expected_bounds = None,
            )

        forecasts = [
            self._forecast_channel(
                df[col].to_numpy(dtype=float),
                channel_label=col,
                file_label=file_label,
            )
            for col in cols
        ]
        forecast_matrix = np.vstack(forecasts)
        self._series_processed += 1

        return ModelResult(
            anomaly_scores = np.zeros(n, dtype=float),
            is_anomaly = np.zeros(n, dtype=bool),
            expected_value = forecast_matrix,
            expected_bounds = None,
        )


class PatchTSTRollingForecaster:
    """
    PatchTST (ibm-granite/granite-timeseries-patchtst) rolling out-of-sample forecaster.
    Mirrors the rolling-window scheme of other models:
      - slide a context window of `context_length` points,
      - produce a point forecast for the next `prediction_length` steps,
      - advance by `step` and repeat.
    All predictions are strictly out-of-sample.
    Reference: https://huggingface.co/ibm-granite/granite-timeseries-patchtst
    """

    def __init__(
        self,
        hf_model_path: str = "ibm-granite/granite-timeseries-patchtst",
        context_length: int = 512,
        prediction_length: int = 96,
        step: int = None,
        warmup_points: int = None,
        device: str = "auto",
        max_series_count: int = 0,
        **kwargs,
    ):
        self.hf_model_path = hf_model_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.step = step or prediction_length
        self.warmup_points = warmup_points or context_length
        self.device = device
        self.max_series_count = max_series_count
        self._model = None
        self._series_processed = 0

    def reset_series_counter(self) -> None:
        self._series_processed = 0

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if _torch is None:
            return "cpu"
        if _torch.cuda.is_available():
            return "cuda"
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        if self._model is not None:
            return self._model
        if _PatchTSTForPrediction is None:
            raise ImportError(
                "transformers is not installed or is too old. Install it with:\n"
                '    pip install "transformers>=4.40.0"'
            )
        device = self._resolve_device()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = _PatchTSTForPrediction.from_pretrained(self.hf_model_path)
        model = model.to(device)
        model.eval()
        self._model = model
        return self._model

    def _forecast_channel(
        self,
        values: np.ndarray,
        channel_label: str = "?",
        file_label: str = "?",
    ) -> np.ndarray:
        ctx = self.context_length
        pred_len = self.prediction_length
        step = self.step
        n = len(values)

        preds = np.full(n, np.nan, dtype=float)
        model = self._load_model()
        device = self._resolve_device()
        starts = list(range(0, n - ctx, step))

        for i, start in enumerate(starts):
            ctx_end = start + ctx
            pred_end = min(ctx_end + pred_len, n)
            horizon = pred_end - ctx_end

            context = values[start:ctx_end].astype(np.float32)
            fallback = float(np.mean(context))

            try:
                ctx_tensor = _torch.tensor(
                    context, dtype=_torch.float32, device=device
                ).unsqueeze(0)

                with _torch.no_grad():
                    output = model(past_values=ctx_tensor)

                fc = output.prediction_outputs[0, :, 0].cpu().numpy().astype(float)
                if len(fc) >= pred_len:
                    fc = fc[:pred_len]
                else:
                    pad = np.full(pred_len - len(fc), fc[-1] if len(fc) > 0 else fallback)
                    fc = np.concatenate([fc, pad])

                preds[ctx_end:pred_end] = fc[:horizon]

            except Exception:
                preds[ctx_end:pred_end] = fallback

        first_valid = np.where(~np.isnan(preds))[0]
        if len(first_valid):
            preds[: first_valid[0]] = values[: first_valid[0]]
        else:
            preds[:] = values
        return preds

    def __call__(
        self, time_series: TimeSeriesWrapper, file_label: str = "?"
    ) -> ModelResult:
        df = time_series.time_series_pd
        cols = list(df.columns)
        n = len(df)

        if self.max_series_count > 0 and self._series_processed >= self.max_series_count:
            return ModelResult(
                anomaly_scores=np.zeros(n, dtype=float),
                is_anomaly=np.zeros(n, dtype=bool),
                expected_value=np.full((len(cols), n), np.nan),
                expected_bounds=None,
            )

        forecasts = [
            self._forecast_channel(
                df[col].to_numpy(dtype=float),
                channel_label=col,
                file_label=file_label,
            )
            for col in cols
        ]
        forecast_matrix = np.vstack(forecasts)
        self._series_processed += 1

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
            context_length=model_params.get("context_length", 512),
            prediction_length=model_params.get("prediction_length", 96),
            step=model_params.get("step"),
            warmup_points=model_params.get("warmup_points"),
        )
    if model_name == "chronos":
        return ChronosRollingForecaster(
            hf_model_path=model_params.get("hf_model_path", "amazon/chronos-t5-small"),
            context_length=model_params.get("context_length", 512),
            prediction_length=model_params.get("prediction_length", 64),
            num_samples=model_params.get("num_samples", 20),
            step=model_params.get("step"),
            warmup_points=model_params.get("warmup_points"),
            device=model_params.get("device", "cpu"),
            batch_size=model_params.get("batch_size", 8),
        )
    if model_name == "timesfm":
        return TimesFMRollingForecaster(
            hf_model_path=model_params.get("hf_model_path", "google/timesfm-2.0-500m-pytorch"),
            context_length=model_params.get("context_length", 512),
            prediction_length=model_params.get("prediction_length", 96),
            step=model_params.get("step"),
            warmup_points=model_params.get("warmup_points"),
            device=model_params.get("device", "cpu"),
            batch_size=model_params.get("batch_size", 32),
            max_series_count=model_params.get("max_series_count", 0),
        )
    if model_name == "patch_tst":
        return PatchTSTRollingForecaster(
            hf_model_path=model_params.get("hf_model_path", "ibm-granite/granite-timeseries-patchtst"),
            context_length=model_params.get("context_length", 512),
            prediction_length=model_params.get("prediction_length", 96),
            step=model_params.get("step"),
            warmup_points=model_params.get("warmup_points"),
            device=model_params.get("device", "auto"),
            max_series_count=model_params.get("max_series_count", 0),
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
        norm = 1.0
    else:
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

def value_to_color_symmetric(val: float, col_max_abs: float) -> str:
    """Color for metrics where 0 is best (e.g. bias). Colour by distance from zero."""
    try:
        v = float(val)
    except Exception:
        return str(val)
    norm = (1.0 - abs(v) / col_max_abs) if col_max_abs > 0 else 1.0
    norm = min(max(norm, 0.0), 1.0)
    if norm >= 0.85:
        return colored(f"{val:.4g}", "green", attrs=["bold"])
    elif norm >= 0.5:
        return colored(f"{val:.4g}", "yellow")
    elif norm >= 0.2:
        return colored(f"{val:.4g}", "magenta")
    else:
        return colored(f"{val:.4g}", "red", attrs=["bold"])

def print_colored_table(
    df: pd.DataFrame,
    title: str,
    higher_is_better: Optional[bool] = False,
) -> None:
    """
    Print a coloured ASCII table.
    higher_is_better=True  → green = high value (e.g. R²)
    higher_is_better=False → green = low value  (e.g. MAE, RMSE)
    higher_is_better=None  → green = near zero  (e.g. Bias)
    """
    print(f"\n=== {title} ===")
    all_rows_str = [[str(val) for val in row] for row in df.values]
    idx_width = max(len(str(idx)) for idx in df.index)
    col_widths = []
    for col_idx, col_label in enumerate(df.columns):
        max_data = max(len(str(row[col_idx])) for row in all_rows_str)
        col_widths.append(max(max_data, len(str(col_label)), 6))

    col_stats = {}
    for col_idx in range(len(df.columns)):
        vals = []
        for row in df.values:
            try:
                vals.append(float(row[col_idx]))
            except Exception:
                pass
        if vals:
            col_stats[col_idx] = (min(vals), max(vals), max(abs(v) for v in vals))
        else:
            col_stats[col_idx] = (0.0, 1.0, 1.0)

    hdr = " " * (idx_width + 2)
    for col_label, width in zip(df.columns, col_widths):
        hdr += f"{str(col_label):<{width}}  "
    print(hdr)

    for idx, row, row_str in zip(df.index, df.values, all_rows_str):
        line = f"{str(idx):<{idx_width}}  "
        for col_idx, (val, val_str, width) in enumerate(zip(row, row_str, col_widths)):
            try:
                fval = float(val)
                cmin, cmax, cabs = col_stats[col_idx]
                if higher_is_better is True:
                    cval = value_to_color_higher_better(fval)
                elif higher_is_better is None:
                    cval = value_to_color_symmetric(fval, cabs)
                else:
                    cval = value_to_color_lower_better(fval, cmin, cmax)
            except Exception:
                cval = val_str
            pad = width - len(val_str)
            line += cval + " " * pad + "  "
        print(line)

def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")[_ALL_METRICS]
        .mean()
        .reset_index()
    )

def print_time_table(df: pd.DataFrame, title: str) -> None:
    """Print a plain (uncoloured) table of minute durations.
    ``print_colored_table`` normalises values for colouring, which is useless
    for wall-clock minutes. This printer just formats each cell as
    ``{value:.2f}`` minutes (or ``-`` for NaN/missing) and shows totals as
    an extra column — matching ``run_anomaly_detection.py`` behaviour.
    """
    print(f"\n=== {title} ===")

    def fmt(v):
        try:
            f = float(v)
        except Exception:
            return "-"
        if not np.isfinite(f):
            return "-"
        return f"{f:.2f}"

    cols = [str(c) for c in df.columns]
    idx_w = max((len(str(i)) for i in df.index), default=5)
    cell_strs = [[fmt(v) for v in row] for row in df.values]
    col_w = [
        max(len(cols[i]), *(len(row[i]) for row in cell_strs), 5)
        for i in range(len(cols))
    ]

    header = " " * (idx_w + 2)
    for label, w in zip(cols, col_w):
        header += f"{label:>{w}}  "
    print(header)

    for idx_label, row in zip(df.index, cell_strs):
        line = f"{str(idx_label):<{idx_w}}  "
        for val, w in zip(row, col_w):
            line += f"{val:>{w}}  "
        print(line)

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

def evaluate_file(
    csv_path: Path,
    detector,
    model_name: str,
    model_params: Dict,
    dataset_name: str = "",
    plot_root: Optional[Path] = None,
) -> List[Dict]:
    """
    Run one forecaster on one CSV file and return per-series metric rows.
    Parameters
    csv_path: path to the dataset CSV
    detector: forecaster instance
    model_name: string label used in output rows
    model_params: dict from models.json5 forecasting_model_params section
    dataset_name: dataset folder name (stored in output rows for grouping)
    plot_root: if set, save a per-series two-panel plot to
    ``<plot_root>/<dataset_name>/<model_name>/<csv_stem>.png``
    """
    df = read_ts(csv_path)
    value_cols = [c for c in df.columns if c.startswith("value_")]
    ts_df = df[value_cols].copy()
    ts_df.index = pd.to_datetime(df["timestamp"])
    is_anomaly = (
        df["is_anomaly"].to_numpy(dtype=int).astype(bool)
        if "is_anomaly" in df.columns else None
    )

    time_series = TimeSeriesWrapper(ts_df)

    if isinstance(detector, (ChronosRollingForecaster,
                             TimesFMRollingForecaster, PatchTSTRollingForecaster)):
        result = detector(time_series, file_label=csv_path.name)
    else:
        result = detector(time_series)

    forecast = normalize_forecast_array(result.expected_value)

    rows: List[Dict] = []
    warmup_points = int(
        model_params.get("warmup_points")
        or model_params.get("context_length")
        or 512
    )

    for i, col in enumerate(value_cols):
        y_true_full = ts_df[col].to_numpy(dtype=float)
        y_pred_full = forecast[i, :] if forecast.shape[0] > 1 else forecast[0, :]
        y_true = y_true_full[warmup_points:]
        y_pred = y_pred_full[warmup_points:]
        valid = ~np.isnan(y_pred)
        y_true_eval = y_true[valid]
        y_pred_eval = y_pred[valid]

        if y_true_eval.size == 0:
            row: Dict = {
                "model": model_name,
                "csv_path": str(csv_path),
                "series": col,
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "smape": float("nan"),
                "r2": float("nan"),
                "wape": float("nan"),
                "mase": float("nan"),
                "max_ae": float("nan"),
                "bias": float("nan"),
                "nrmse": float("nan"),
                "n_eval_points": 0,
            }
            rows.append(row)
            continue

        row = {
            "model": model_name,
            "csv_path": str(csv_path),
            "series": col,
            "mae": mae(y_true_eval, y_pred_eval),
            "rmse": rmse(y_true_eval, y_pred_eval),
            "mape": mape(y_true_eval, y_pred_eval),
            "smape": smape(y_true_eval, y_pred_eval),
            "r2": r2(y_true_eval, y_pred_eval),
            "wape": wape(y_true_eval, y_pred_eval),
            "mase": mase(y_true_eval, y_pred_eval),
            "max_ae": max_ae(y_true_eval, y_pred_eval),
            "bias": bias_metric(y_true_eval, y_pred_eval),
            "nrmse": nrmse(y_true_eval, y_pred_eval),
            "n_eval_points": int(valid.sum()),
        }
        rows.append(row)

    if plot_root is not None:
        try:
            _save_forecast_plot(
                times = ts_df.index,
                y_true = ts_df[value_cols[0]].to_numpy(dtype=float),
                y_pred = forecast[0, :],
                is_anomaly = is_anomaly,
                warmup_points = warmup_points,
                model_name = model_name,
                csv_path = csv_path,
                dataset_name = dataset_name,
                plot_root = plot_root,
            )
        except Exception as exc:
            tqdm.write(f"[plot] failed for {csv_path.name}: {exc}")

    return rows

def _save_forecast_plot(
    times: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_anomaly,
    warmup_points: int,
    model_name: str,
    csv_path: Path,
    dataset_name: str,
    plot_root: Path,
) -> None:
    """Render the same two-panel layout as plot_forecasting.py and save to PNG.
    Imported lazily so users who don't pass ``--plot_dir`` don't pay the
    matplotlib import cost.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_forecasting import plot_one

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

    fig, (ax_ts, ax_res) = plt.subplots(
        nrows=2, ncols=1, figsize=(18, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    plot_one(
        times = times,
        y_true = y_true,
        y_pred = y_pred,
        is_anomaly = is_anomaly,
        warmup_points = warmup_points,
        model_name = model_name,
        label = f"{dataset_name}/{csv_path.name}",
        ax_ts = ax_ts,
        ax_res = ax_res,
    )
    fig.tight_layout()

    out_path = plot_root / dataset_name / model_name / f"{csv_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

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
        default="forecasting_summary.csv",
        help="Path to save aggregated summary (dataset × model). "
             "Defaults to forecasting_summary.csv; pass an empty string to skip.",
    )
    parser.add_argument(
        "--models",
        type=argparse.FileType("r"),
        required=True,
        help="Path to JSON5 file with model configurations (e.g. models.json5)",
    )
    parser.add_argument(
        "--plot_dir", type=str, default=None,
        help="Root directory for per-series plots; saves to "
             "<plot_dir>/<dataset>/<model>/<csv_stem>.png "
             "(default: plots/forecasting). Pass --no_plot to disable.",
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Disable per-series plot generation.",
    )
    args = parser.parse_args()

    plot_root: Optional[Path] = None
    if not args.no_plot:
        plot_root = Path(args.plot_dir) if args.plot_dir else Path("plots") / "forecasting"

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    try:
        configurations = json5.load(args.models)
    except Exception as e:
        raise SystemExit(f"Could not parse models JSON5 file: {e}")

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

    for model_name, detector in detectors.items():
        if hasattr(detector, "_load_pipeline"):
            print(f"Loading {model_name} model...", flush=True)
            detector._load_pipeline()

            if _torch is not None:
                _ctx_len = getattr(detector, "context_length", 512)
                _pred_len = getattr(detector, "prediction_length", 64)
                _n_samples = getattr(detector, "num_samples", 20)
                _dummy = _torch.zeros(1, _ctx_len, dtype=_torch.float32)
                print(f"Warming up {model_name} (ctx={_ctx_len}, pred={_pred_len}, samples={_n_samples})...", flush=True)
                t0 = time.perf_counter()
                with _torch.no_grad():
                    if _ChronosBoltPipeline is not None and isinstance(detector._pipeline, _ChronosBoltPipeline):
                        detector._pipeline.predict(_dummy, _pred_len)
                    else:
                        detector._pipeline.predict(_dummy, _pred_len, num_samples=_n_samples)
                print(f"Warmup done in {time.perf_counter() - t0:.1f}s", flush=True)
            print(f"{model_name} ready.", flush=True)
        elif hasattr(detector, "_load_model"):
            print(f"Loading {model_name} model...", flush=True)
            detector._load_model()
            print(f"{model_name} ready.", flush=True)

    all_rows = []
    time_records: List[Dict] = []

    summary_columns = ["dataset", "model", *_ALL_METRICS, "processing_time_min"]
    if args.f_output_csv:
        with open(args.f_output_csv, "w") as f:
            f.write(",".join(summary_columns) + "\n")

    for dataset_name in dataset_names:
        dataset_root = Path(args.datasets_root) / dataset_name

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

        for model_name, detector in detectors.items():
            mp = per_model_params[model_name]
            if hasattr(detector, "reset_series_counter"):
                detector.reset_series_counter()
            csv_files = list(iter_csv_files(dataset_root))
            pbar = tqdm(
                csv_files,
                desc=f"{dataset_name} | {model_name}",
            )
            pair_rows: List[Dict] = []
            _t_start = time.perf_counter()
            for csv_path in pbar:
                rows = evaluate_file(
                    csv_path,
                    detector,
                    model_name,
                    mp,
                    dataset_name=dataset_name,
                    plot_root=plot_root,
                )
                for row in rows:
                    row["dataset"] = dataset_name
                pair_rows.extend(rows)
                all_rows.extend(rows)
            pbar.close()
            processing_time_min = (time.perf_counter() - _t_start) / 60.0
            time_records.append({
                "dataset": dataset_name,
                "model": model_name,
                "processing_time_min": processing_time_min,
            })

            if args.f_output_csv and pair_rows:
                pair_df = pd.DataFrame(pair_rows)
                metric_means = pair_df[_ALL_METRICS].mean().round(6)
                cells = [dataset_name, model_name]
                for m in _ALL_METRICS:
                    v = metric_means.get(m, float("nan"))
                    cells.append("" if pd.isna(v) else f"{float(v):.6g}")
                cells.append(f"{processing_time_min:.4f}")
                with open(args.f_output_csv, "a") as f:
                    f.write(",".join(cells) + "\n")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.f_time_series_metrics_csv, index=False)

    time_df = pd.DataFrame(time_records)

    if args.f_output_csv:
        print(f"Saved summary metrics to: {args.f_output_csv}")

    model_order = model_names

    for metric, direction in [
        ("mae", False),
        ("rmse", False),
        ("mape", False),
        ("smape", False),
        ("r2", True),
        ("wape", False),
        ("mase", False),
        ("max_ae", False),
        ("bias", None),
        ("nrmse", False),
    ]:
        pivot = (
            df.groupby(["model", "dataset"])[metric]
            .mean()
            .reset_index()
            .pivot(index="model", columns="dataset", values=metric)
        )
        pivot = pivot.reindex(index=model_order, columns=dataset_names)
        pivot = pivot.round(4)
        label = {
            "mae": "MAE  (lower is better)",
            "rmse": "RMSE  (lower is better)",
            "mape": "MAPE %  (lower is better)",
            "smape": "SMAPE %  (lower is better)",
            "r2": "R²  (higher is better)",
            "wape": "WAPE %  (lower is better) — robust to near-zero values",
            "mase": "MASE  (lower is better; <1 = beats naive baseline)",
            "max_ae": "MaxAE  (lower is better) — worst single-point error / SLA relevance",
            "bias": "Bias  (zero is best) — mean signed error: + = over-forecast",
            "nrmse": "NRMSE %  (lower is better) — range-normalized RMSE",
        }.get(metric, metric.upper())
        print_colored_table(pivot, title=label, higher_is_better=direction)

    if not time_df.empty:
        time_pivot = time_df.pivot(index="model", columns="dataset", values="processing_time_min")
        time_pivot = time_pivot.reindex(index=model_order, columns=dataset_names)
        time_pivot["TOTAL"] = time_pivot.sum(axis=1, skipna=True)
        print_time_table(time_pivot, title="PROCESSING_TIME_MIN")

    print(f"\nSaved per-series metrics to: {args.f_time_series_metrics_csv}")
    if plot_root is not None:
        print(f"Saved per-series plots to: {plot_root.resolve()}")
    else:
        print("To visualise results run: uv run plot_forecasting.py --save --no_show")


if __name__ == "__main__":
    main()
