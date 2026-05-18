from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult

try:
    import torch
    from transformers import PatchTSTForPrediction
except Exception:
    torch = None
    PatchTSTForPrediction = None


@dataclass
class PatchTSTForecastResult:
    forecast: np.ndarray
    residuals: np.ndarray
    scores: np.ndarray


class PatchTSTDetector(BaseDetector):
    """
    Anomaly detector based on PatchTST (ibm-granite/granite-timeseries-patchtst).

    Idea:
    1. Use PatchTST as a forecaster on rolling windows.
    2. Use the model's point prediction output as the forecast.
    3. Compute anomaly scores as normalized forecast residuals (two-pass).
    4. Mark points as anomalous when score > threshold.

    Notes:
    - Works with univariate and multivariate series (per-channel).
    - Unsupervised residual-based detector — no labels required.
    - Model weights are cached at the class level (keyed by model path + device)
      so the same weights are reused across all series in a benchmark run.
    - The ibm-granite PatchTST variant is loaded via the HuggingFace transformers
      library; `transformers>=4.40` is required.

    Reference:
        https://huggingface.co/ibm-granite/granite-timeseries-patchtst
        https://arxiv.org/abs/2211.14730
    """

    _model_cache: dict = {}

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_name": "PatchTST",
            "hf_model_path": "ibm-granite/granite-timeseries-patchtst",
            "context_length": 512,
            "prediction_length": 96,
            "threshold": None,
            "step": None,
            "device": "auto",
            "warmup_points": None,
            "min_std": 1e-6,
            "use_absolute_error": True,
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

    def _model_cache_key(self) -> tuple:
        return (
            self.params["hf_model_path"],
            self._resolve_device(),
        )

    def _load_model(self):
        cache_key = self._model_cache_key()
        if cache_key in PatchTSTDetector._model_cache:
            return PatchTSTDetector._model_cache[cache_key]

        if PatchTSTForPrediction is None:
            raise ImportError(
                "transformers is not installed or is too old. Install it with:\n"
                '    pip install "transformers>=4.40.0"'
            )

        device = self._resolve_device()

        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore")
            model = PatchTSTForPrediction.from_pretrained(
                self.params["hf_model_path"],
            )

        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()

        num_input_channels = int(getattr(model.config, "num_input_channels", 1))
        PatchTSTDetector._model_cache[cache_key] = (model, num_input_channels)
        return model, num_input_channels

    def _run_forecast(
        self,
        model,
        context: np.ndarray,
        pred_len: int,
        device: str,
        num_input_channels: int = 1,
    ) -> np.ndarray:
        """
        Run PatchTST on a single context window.

        Returns a point forecast as a 1-D numpy array of shape [pred_len].

        The ibm-granite PatchTST model has a fixed prediction_length baked into
        its config (typically 96).  We call model() to get prediction_outputs
        of shape [batch, pred_len, num_input_channels] and take channel 0.

        ``num_input_channels`` is the number of channels the checkpoint was
        trained with (7 for granite-patchtst).  For univariate input we
        replicate the series across all channels.  PatchTST in the granite
        variant is channel-independent, so all output channels are identical
        for a single-source input — taking channel 0 is equivalent to any
        other channel and recovers the univariate forecast.
        """
        ctx_arr = context.astype(np.float32)
        if num_input_channels > 1:
            ctx_arr = np.tile(ctx_arr[:, None], (1, num_input_channels))
        else:
            ctx_arr = ctx_arr[:, None]

        ctx_tensor = torch.tensor(
            ctx_arr, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            output = model(past_values=ctx_tensor)

        fc_tensor = output.prediction_outputs
        fc = fc_tensor[0, :, 0].cpu().numpy().astype(float)

        if len(fc) >= pred_len:
            return fc[:pred_len]
        else:
            pad = np.full(pred_len - len(fc), fc[-1] if len(fc) > 0 else 0.0, dtype=float)
            return np.concatenate([fc, pad])

    def _forecast_channel(
        self,
        values: np.ndarray,
        channel_label: str = "?",
        file_label: str = "?",
    ) -> PatchTSTForecastResult:
        context_length = int(self.params["context_length"])
        prediction_length = int(self.params["prediction_length"])
        step = int(self.params["step"] or prediction_length)
        warmup_points = int(self.params["warmup_points"] or context_length)
        min_std = float(self.params["min_std"])
        use_absolute_error = bool(self.params["use_absolute_error"])

        n = len(values)
        scores = np.full(n, np.nan, dtype=float)
        preds = np.full(n, np.nan, dtype=float)

        if n <= context_length:
            return PatchTSTForecastResult(
                forecast=np.nan_to_num(preds, nan=float(np.mean(values) if n else 0.0)),
                residuals=np.zeros(n, dtype=float),
                scores=scores,
            )

        model, num_input_channels = self._load_model()
        device = self._resolve_device()
        do_clean = bool(self.params.get("clean_context", True))
        mad_thr  = float(self.params.get("clean_context_mad_threshold", 3.0))

        starts = list(range(0, n - context_length, step))
        last_start = n - context_length
        if last_start > 0 and (not starts or starts[-1] != last_start):
            starts.append(last_start)

        window_data: List[Tuple[int, int, np.ndarray]] = []

        for i, start in enumerate(starts):
            ctx_end = start + context_length
            pred_end = min(ctx_end + prediction_length, n)
            horizon = pred_end - ctx_end

            raw_ctx = values[start:ctx_end]
            context = (
                self.clean_context(raw_ctx, mad_threshold=mad_thr)
                if do_clean else raw_ctx.copy()
            ).astype(np.float32)
            fallback = float(np.mean(context))

            try:
                forecast = self._run_forecast(
                    model=model,
                    context=context,
                    pred_len=prediction_length,
                    device=device,
                    num_input_channels=num_input_channels,
                )
                fc = forecast[:horizon]
                preds[ctx_end:pred_end] = fc

                actual = values[ctx_end:pred_end].astype(float)
                residual = actual - fc
                raw_err = np.abs(residual) if use_absolute_error else residual ** 2
                window_data.append((ctx_end, pred_end, raw_err))

            except Exception as exc:
                warnings.warn(
                    f"[PatchTST] forecast window {start} failed "
                    f"({exc.__class__.__name__}: {exc!s}); using context-mean fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                preds[ctx_end:pred_end] = fallback

        if window_data:
            all_errs = np.concatenate([e for _, _, e in window_data])

            if len(all_errs) >= 10:
                trim_n = max(1, int(0.10 * len(all_errs)))
                scale = float(np.mean(np.sort(all_errs)[:-trim_n]))
            else:
                scale = float(np.mean(all_errs))

            if not np.isfinite(scale) or scale < min_std:
                scale = min_std

            for ctx_end, pred_end, raw_err in window_data:
                scores[ctx_end:pred_end] = raw_err / scale

        first_valid = np.where(~np.isnan(preds))[0]
        if len(first_valid):
            preds[: first_valid[0]] = values[: first_valid[0]]
        else:
            preds[:] = values

        residuals = values - preds

        if warmup_points is not None and warmup_points > 0:
            scores[:warmup_points] = np.nan

        return PatchTSTForecastResult(forecast=preds, residuals=residuals, scores=scores)

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        values = time_series.time_series_pd["value_0"].to_numpy(dtype=float)
        label = getattr(time_series, "label", "?")
        result = self._forecast_channel(values, channel_label="value_0", file_label=label)

        expected = result.forecast
        residual_std = max(
            float(np.nanstd(result.residuals[np.isfinite(result.residuals)])),
            float(self.params["min_std"]),
        )
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
        label = getattr(time_series, "label", "?")

        per_channel_forecasts = []
        per_channel_scores = []

        for col in cols:
            result = self._forecast_channel(
                df[col].to_numpy(dtype=float),
                channel_label=col,
                file_label=label,
            )
            per_channel_forecasts.append(result.forecast.astype(float))
            per_channel_scores.append(result.scores.astype(float))

        forecast_matrix = np.vstack(per_channel_forecasts)
        score_matrix = np.vstack(per_channel_scores)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            aggregated_scores = np.nanmax(score_matrix, axis=0)
        is_anomaly, _ = self.threshold_outputs(aggregated_scores)

        return ModelResult(
            anomaly_scores=aggregated_scores.astype(float),
            is_anomaly=is_anomaly,
            expected_value=forecast_matrix.astype(float),
            expected_bounds=None,
        )
