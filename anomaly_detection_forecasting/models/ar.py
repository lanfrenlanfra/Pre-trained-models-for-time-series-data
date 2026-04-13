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
        return {"order": 20, "threshold": 3.0, "stable": True, "stable_sensitivity": 1.0}

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["order"] <= 0:
            raise ValueError("Autoregression order must be > 0")
        if params["threshold"] < 0:
            raise ValueError("threshold must be not less than 0")

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
        expected_bounds = np.column_stack(
            (
                expected_value - self.params["threshold"],
                expected_value + self.params["threshold"],
            )
        )
        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=(z_scores > self.params["threshold"]),
            expected_value=expected_value,
            expected_bounds=expected_bounds,
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        order = self.params["order"]

        if self._check_insufficient_samples(time_series, order):
            return self._get_fallback_result_univariate(time_series)

        model = AutoReg(time_series.time_series_pd["value_0"], lags=order)
        model_fit = model.fit()

        expected = np.concatenate((time_series.values[:order], model_fit.fittedvalues))
        residuals = time_series.values - expected
        residual_std = self.calculate_std(residuals)

        residuals = expected - time_series.values
        z_scores = np.abs(residuals / residual_std)
        expected_bounds = np.column_stack(
            (
                expected - residual_std * self.params["threshold"],
                expected + residual_std * self.params["threshold"],
            )
        )

        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=(z_scores > self.params["threshold"]),
            expected_value=expected,
            expected_bounds=expected_bounds,
        )
