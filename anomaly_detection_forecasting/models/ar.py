from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from typing import Dict, Any
from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult


class ARDetector(BaseDetector):
    """
    AR (AutoRegressive) anomaly detection model.

    Implements autoregressive modeling for time series anomaly detection using
    residual analysis with z-score thresholding.
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {"order": 20, "threshold": None, "stable": True, "stable_sensitivity": 1.0}

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["order"] <= 0:
            raise ValueError("Autoregression order must be > 0")
        if params.get("threshold") is not None and params["threshold"] < 0:
            raise ValueError("threshold must be >= 0 (or None to disable)")

    def _check_insufficient_samples(self, time_series: TimeSeriesWrapper, order: int) -> bool:
        """
        Check if there are insufficient samples for AR model fitting.

        Args:
            time_series: Time series data
            order: AR model order

        Returns:
            True if samples are insufficient, False otherwise
        """
        n_samples = time_series.time_series_pd.shape[0]
        min_samples_required = order * 2 + 2
        return n_samples < min_samples_required

    def _get_fallback_result_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Return fallback result with zero scores for univariate time series with insufficient samples.

        Args:
            time_series: Time series data

        Returns:
            ModelResult with zero anomaly scores
        """
        n_samples = time_series.time_series_pd.shape[0]
        z_scores = np.zeros(n_samples)
        expected_value = np.array(time_series.time_series_pd["value_0"])
        is_anomaly, expected_bounds = self.threshold_outputs(
            z_scores, expected=expected_value, residual_std=1.0,
        )
        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=is_anomaly,
            expected_value=expected_value,
            expected_bounds=expected_bounds,
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        order = self.params["order"]
        values = time_series.values
        n = len(values)

        if self._check_insufficient_samples(time_series, order):
            return self._get_fallback_result_univariate(time_series)

        train_n = min(512, max(order * 4, n // 2))
        if train_n >= n - order:
            train_n = order * 4

        model = AutoReg(values[:train_n], lags=order)
        model_fit = model.fit()
        params = model_fit.params

        expected = np.empty(n, dtype=float)
        expected[:order] = values[:order]
        expected[order:train_n] = np.asarray(model_fit.fittedvalues)

        for t in range(train_n, n):
            ctx = values[t - order : t][::-1]
            expected[t] = params[0] + np.dot(params[1:], ctx)

        oos_residuals = values[train_n:] - expected[train_n:]
        residual_std = max(self.calculate_std(oos_residuals), 1e-6)

        z_scores = np.full(n, np.nan, dtype=float)
        z_scores[train_n:] = np.abs(oos_residuals) / residual_std

        is_anomaly, expected_bounds = self.threshold_outputs(
            np.nan_to_num(z_scores, nan=0.0),
            expected=expected,
            residual_std=residual_std,
        )

        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=is_anomaly,
            expected_value=expected,
            expected_bounds=expected_bounds,
        )
