from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult

try:
    import torch
    from tsfm_public.toolkit.get_model import get_model
except Exception:  # pragma: no cover
    torch = None
    get_model = None


@dataclass
class GraniteForecastResult:
    forecast: np.ndarray
    residuals: np.ndarray
    scores: np.ndarray


class GraniteTTMDetector(BaseDetector):
    """
    Anomaly detector based on Granite TinyTimeMixer (TTM).

    Idea:
    1. Use Granite TTM as a forecaster on rolling windows.
    2. Compute anomaly scores as normalized forecast residuals.
    3. Mark points as anomalous when score > threshold.

    Notes:
    - Works with univariate and multivariate series.
    - Does not train labels; this is an unsupervised residual-based detector.
    - For CPU-only setups this is slower than AR, but closer to how a TS foundation
      model is commonly adapted for anomaly detection.
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_name": "GraniteTTM",
            "hf_model_path": "ibm-granite/granite-timeseries-ttm-r2",
            "context_length": 512,
            "prediction_length": 96,
            "threshold": 3.0,
            "step": None,  # defaults to prediction_length
            "device": "cpu",
            "per_channel": True,
            "warmup_points": None,  # defaults to context_length
            "residual_stat_window": None,  # defaults to 10 * prediction_length
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

    def _load_model(self):
        if hasattr(self, "_model") and self._model is not None:
            return self._model

        if get_model is None:
            raise ImportError(
                "granite-tsfm is not installed. Install it with:\n"
                'pip install "granite-tsfm>=0.3.5" "torch>=2.2" "transformers>=4.40"'
            )

        model = get_model(
            model_path=self.params["hf_model_path"],
            context_length=self.params["context_length"],
            prediction_length=self.params["prediction_length"],
            freq_prefix_tuning=False,
            freq=None,
            prefer_l1_loss=False,
            prefer_longer_context=True,
        )

        device = self._resolve_device()
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()

        self._model = model
        return self._model

    # def _load_model(self):
    #     if get_model is None:
    #         raise ImportError(
    #             "granite-tsfm is not installed. Install it with:\n"
    #             'pip install "granite-tsfm>=0.3.5" "torch>=2.2" "transformers>=4.40"'
    #         )
    #     return get_model(
    #         model_path=self.params["hf_model_path"],
    #         context_length=self.params["context_length"],
    #         prediction_length=self.params["prediction_length"],
    #         freq_prefix_tuning=False,
    #         freq=None,
    #         prefer_l1_loss=False,
    #         prefer_longer_context=True,
    #     )

    def _extract_forecast_array(self, output):
        """
        Convert Granite/TSFM model output to a numpy forecast array.

        Expected final shape:
        - [pred_len] for univariate
        - or [pred_len, channels]
        """
        candidate = output

        # Common HF/TSFM structured outputs
        for attr in [
            "prediction_outputs",
            "predictions",
            "prediction",
            "forecast",
            "outputs",
            "logits",
        ]:
            if hasattr(candidate, attr):
                candidate = getattr(candidate, attr)
                break

        # Some outputs behave like dicts
        if isinstance(candidate, dict):
            for key in [
                "prediction_outputs",
                "predictions",
                "prediction",
                "forecast",
                "outputs",
                "logits",
            ]:
                if key in candidate:
                    candidate = candidate[key]
                    break

        # Torch tensor -> numpy
        if hasattr(candidate, "detach"):
            candidate = candidate.detach().cpu().numpy()

        # Final numpy conversion
        candidate = np.asarray(candidate)

        # Expected Granite shapes are usually:
        # [B, pred_len, C] or [pred_len, C] or [pred_len]
        if candidate.ndim == 3:
            candidate = candidate[0]  # [pred_len, C]
        elif candidate.ndim == 2:
            pass  # already [pred_len, C] or [B, pred_len]
        elif candidate.ndim == 1:
            return candidate.astype(float)
        else:
            raise ValueError(f"Unexpected Granite output shape/type: {type(output)} / shape={candidate.shape}")

        # Univariate: [pred_len, 1] -> [pred_len]
        if candidate.ndim == 2 and candidate.shape[-1] == 1:
            candidate = candidate[:, 0]

        return candidate.astype(float)

    def _forecast_channel(self, values: np.ndarray) -> GraniteForecastResult:
        context_length = int(self.params["context_length"])
        prediction_length = int(self.params["prediction_length"])
        step = int(self.params["step"] or prediction_length)
        warmup_points = int(self.params["warmup_points"] or context_length)
        residual_stat_window = int(
            self.params["residual_stat_window"] or max(10 * prediction_length, prediction_length)
        )
        min_std = float(self.params["min_std"])
        use_absolute_error = bool(self.params["use_absolute_error"])

        n = len(values)
        scores = np.zeros(n, dtype=float)
        preds = np.full(n, np.nan, dtype=float)

        if n <= context_length:
            return GraniteForecastResult(
                forecast=np.nan_to_num(preds, nan=float(np.mean(values) if n else 0.0)),
                residuals=np.zeros(n, dtype=float),
                scores=np.zeros(n, dtype=float),
            )

        model = self._load_model()
        device = self._resolve_device()
        # if hasattr(model, "to"):
        #     model = model.to(device)
        # if hasattr(model, "eval"):
        #     model.eval()

        # ── Pass 1: run the model, store raw errors for every window ──────────
        # We do NOT score yet – we need a global scale first.
        # This mirrors how AR computes its scale: fit on the whole series, then
        # derive a single σ that is consistent across the entire signal.
        window_data: List[tuple] = []   # (ctx_end, pred_end, raw_err array)

        with torch.no_grad():
            for start in range(0, n - context_length, step):
                ctx_end = start + context_length
                pred_end = min(ctx_end + prediction_length, n)

                context = values[start:ctx_end].astype(np.float32)
                batch = torch.tensor(context[None, :, None], dtype=torch.float32, device=device)

                output = model(batch)
                forecast = self._extract_forecast_array(output)

                if forecast.ndim == 2:
                    forecast = forecast[:, 0]
                elif forecast.ndim == 1:
                    pass
                else:
                    raise ValueError(f"Unexpected Granite output shape after extraction: {forecast.shape}")

                horizon = pred_end - ctx_end
                forecast = forecast[:horizon]
                preds[ctx_end:pred_end] = forecast

                actual = values[ctx_end:pred_end]
                residual = actual - forecast
                raw_err = np.abs(residual) if use_absolute_error else residual ** 2

                window_data.append((ctx_end, pred_end, raw_err))

        # ── Pass 2: compute one global robust scale, then score ───────────────
        # Concatenate all per-window errors collected above.
        if window_data:
            all_errs = np.concatenate([r for _, _, r in window_data])

            # Trimmed mean: drop the top 10 % of errors (likely anomalies) so
            # the scale is not inflated by the very points we want to detect.
            # This gives a stable estimate of the model's *typical* error on
            # this specific series – analogous to AR's global residual RMS.
            if len(all_errs) >= 10:
                trim_n = max(1, int(0.10 * len(all_errs)))
                scale = float(np.mean(np.sort(all_errs)[:-trim_n]))
            else:
                scale = float(np.mean(all_errs))

            if not np.isfinite(scale) or scale < min_std:
                scale = min_std

            for ctx_end, pred_end, raw_err in window_data:
                scores[ctx_end:pred_end] = raw_err / scale

        # Fill untouched prefix using first observed valid forecast residual scale = 0
        first_valid = np.where(~np.isnan(preds))[0]
        if len(first_valid):
            first_valid = int(first_valid[0])
            preds[:first_valid] = values[:first_valid]
        else:
            preds[:] = values

        residuals = values - preds
        if warmup_points is not None:
            scores[:warmup_points] = 0.0

        return GraniteForecastResult(forecast=preds, residuals=residuals, scores=scores)

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

        # Aggregate multivariate anomaly score.
        # max works better for sparse anomalies in one channel.
        aggregated_scores = np.max(score_matrix, axis=0)
        threshold = float(self.params["threshold"])

        return ModelResult(
            anomaly_scores=aggregated_scores.astype(float),
            is_anomaly=(aggregated_scores > threshold),
            expected_value=forecast_matrix.astype(float),
            expected_bounds=None,
        )
