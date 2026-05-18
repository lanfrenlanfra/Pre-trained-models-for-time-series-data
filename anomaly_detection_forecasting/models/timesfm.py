from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult

try:
    import timesfm as _timesfm
except Exception:  # pragma: no cover
    _timesfm = None

try:
    import torch as _torch
except Exception:
    _torch = None


class TimesFMDetector(BaseDetector):
    """
    Anomaly detector based on Google TimesFM (timesfm-1.0-200m-pytorch).

    Scoring strategy — mirrors GraniteTTMDetector exactly:
      Pass 1: slide a context window, generate a forecast for the next
              `prediction_length` steps, record absolute residuals.
      Pass 2: compute one global robust scale = trimmed_mean(|residuals|)
              (top-10 % removed so genuine anomalies don't inflate the scale).
              score[t] = |residual[t]| / scale.

    For normally distributed residuals the score denominator equals ~mean(|e|),
    so P(score > 3.0) ≈ 1–2 % of normal points — matching GraniteTTM's
    calibration and giving reasonable precision without further threshold tuning.

    Reference: https://github.com/google-research/timesfm
    """

    _model_cache: dict = {}

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_name": "TimesFM",
            "hf_model_path": "google/timesfm-1.0-200m-pytorch",
            "context_length": 512,
            "prediction_length": 96,
            "threshold": None,
            "device": "auto",
            "step": 96,
            "warmup_points": 512,
            "min_std": 1e-6,
            "batch_size": 64,
            "max_series_count": 0,
            "clean_context": True,
            "clean_context_mad_threshold": 3.0,
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params.get("threshold") is not None and params["threshold"] < 0:
            raise ValueError("threshold must be >= 0 (or None to disable)")

    def _resolve_device(self) -> Tuple[str, Any]:
        """Decide which (timesfm_backend, post-load torch device) to use.

        Upstream timesfm only knows ``cpu`` / ``gpu`` / ``tpu`` (with ``gpu``
        meaning CUDA). On Apple Silicon there's no native MPS path, so we
        load the model with backend="cpu" and then manually move the
        underlying PyTorch module to MPS. We also flip ``tfm.backend`` to
        ``"gpu"`` so that timesfm's ``_forecast`` brings results back to
        CPU with ``.cpu()`` before calling ``.numpy()`` (otherwise the
        MPS tensor would crash on numpy conversion).

        Returns:
            (timesfm_backend, target_torch_device_or_None)
            ``target_torch_device_or_None`` is set when we want to
            post-load .to(...) the model onto a non-CUDA device (MPS).
        """
        requested = str(self.params.get("device", "cpu")).lower()

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
        backend, post_device = self._resolve_device()
        batch_size = int(self.params.get("batch_size", 64))

        cache_key = (
            self.params["hf_model_path"],
            backend,
            batch_size,
            str(post_device) if post_device is not None else "default",
        )
        if cache_key in TimesFMDetector._model_cache:
            return TimesFMDetector._model_cache[cache_key]

        if _timesfm is None:
            raise ImportError(
                "timesfm is not installed. Install it with:\n"
                "    pip install timesfm"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tfm = _timesfm.TimesFm(
                hparams=_timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=batch_size,
                    horizon_len=int(self.params["prediction_length"]),
                ),
                checkpoint=_timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.params["hf_model_path"]
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

        TimesFMDetector._model_cache[cache_key] = tfm
        return tfm

    def _forecast_channel(
        self,
        values: np.ndarray,
        channel_label: str = "?",
        file_label: str = "?",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (preds [T], scores [T])."""
        ctx = int(self.params["context_length"])
        pred_len = int(self.params["prediction_length"])
        step = int(self.params.get("step", pred_len))
        warmup = int(self.params.get("warmup_points", ctx))
        min_std = float(self.params["min_std"])
        batch_size = int(self.params.get("batch_size", 64))
        n = len(values)

        preds = np.full(n, np.nan, dtype=float)
        model = self._load_model()
        do_clean = bool(self.params.get("clean_context", True))
        mad_thr = float(self.params.get("clean_context_mad_threshold", 3.0))
        starts = list(range(0, n - ctx, step))
        last_start = n - ctx
        if last_start > 0 and (not starts or starts[-1] != last_start):
            starts.append(last_start)

        contexts: List[np.ndarray] = []
        targets:  List[Tuple[int, int, int]] = []
        for start in starts:
            ctx_end = start + ctx
            pred_end = min(ctx_end + pred_len, n)
            horizon = pred_end - ctx_end
            raw_ctx = values[start:ctx_end]
            clean_ctx = (
                self.clean_context(raw_ctx, mad_threshold=mad_thr)
                if do_clean else raw_ctx.copy()
            )
            contexts.append(clean_ctx.astype(np.float32))
            targets.append((ctx_end, pred_end, horizon))

        window_data: List[Tuple[int, int, np.ndarray]] = []

        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i + batch_size]
            batch_tgt = targets[i:i + batch_size]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    point_forecast, _ = model.forecast(
                        inputs=batch_ctx,
                        freq=[0] * len(batch_ctx),
                    )
                for j, (ctx_end, pred_end, horizon) in enumerate(batch_tgt):
                    fc = point_forecast[j, :horizon].astype(float)
                    preds[ctx_end:pred_end] = fc
                    actual = values[ctx_end:pred_end].astype(float)
                    abs_err = np.abs(actual - fc)
                    window_data.append((ctx_end, pred_end, abs_err))

            except Exception:
                for j, (ctx_end, pred_end, _h) in enumerate(batch_tgt):
                    preds[ctx_end:pred_end] = float(np.mean(batch_ctx[j]))

        first_valid = np.where(~np.isnan(preds))[0]
        if len(first_valid):
            preds[: first_valid[0]] = values[: first_valid[0]]
        else:
            preds[:] = values

        scores = np.full(n, np.nan, dtype=float)

        if window_data:
            all_errs = np.concatenate([e for _, _, e in window_data])
            if len(all_errs) >= 10:
                trim_n = max(1, int(0.10 * len(all_errs)))
                scale = float(np.mean(np.sort(all_errs)[:-trim_n]))
            else:
                scale = float(np.mean(all_errs))

            if not np.isfinite(scale) or scale < min_std:
                scale = min_std

            for ctx_end, pred_end, abs_err in window_data:
                scores[ctx_end:pred_end] = abs_err / scale

        if warmup and warmup > 0:
            scores[:warmup] = np.nan

        return preds, scores

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        values = time_series.time_series_pd["value_0"].to_numpy(dtype=float)
        label = getattr(time_series, "label", "?")
        preds, scores = self._forecast_channel(
            values, channel_label="value_0", file_label=label
        )

        res_std = max(
            float(np.nanstd(values - preds)),
            float(self.params["min_std"]),
        )
        is_anomaly, exp_bounds = self.threshold_outputs(
            scores, expected=preds, residual_std=res_std,
        )

        return ModelResult(
            anomaly_scores = scores.astype(float),
            is_anomaly = is_anomaly,
            expected_value = preds.astype(float),
            expected_bounds = exp_bounds,
        )

    def _detect_multivariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        df = time_series.time_series_pd
        label = getattr(time_series, "label", "?")

        ch_preds, ch_scores = [], []
        for col in df.columns:
            preds, scores = self._forecast_channel(
                df[col].to_numpy(dtype=float),
                channel_label=col,
                file_label=label,
            )
            ch_preds.append(preds)
            ch_scores.append(scores)

        forecast_matrix = np.vstack(ch_preds)
        score_matrix = np.vstack(ch_scores)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            agg_scores = np.nanmax(score_matrix, axis=0)
        is_anomaly, _ = self.threshold_outputs(agg_scores)

        return ModelResult(
            anomaly_scores=agg_scores.astype(float),
            is_anomaly=is_anomaly,
            expected_value=forecast_matrix.astype(float),
            expected_bounds=None,
        )
