from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator
from typing import Dict, Any

import numpy as np
from scipy import stats
from statsmodels.robust.scale import qn_scale
from ..core import TimeSeriesWrapper


class ModelResult(BaseModel):
    """
    Class for storing the result of an anomaly detection.

    This class is used to store the result of an anomaly detection,
    including the anomaly scores.
    """

    anomaly_scores: Any
    is_anomaly: Any
    expected_value: Any = None
    expected_bounds: Any = None

    @field_validator("anomaly_scores", "is_anomaly")
    @classmethod
    def check_anomaly_scores_numpy_array(cls, v, info):
        if not isinstance(v, np.ndarray):
            raise TypeError(f"{info.field_name} must be a numpy.ndarray")
        if v.ndim != 1:
            raise ValueError(f"{info.field_name} must be a 1D array, but got {v.ndim}D array with shape {v.shape}")
        return v

    @field_validator("expected_value", "expected_bounds")
    @classmethod
    def check_expected_value_numpy_array(cls, v, info):
        if not isinstance(v, np.ndarray) and v is not None:
            raise TypeError(f"{info.field_name} must be a numpy.ndarray or None")
        return v


class BaseDetector(ABC):
    """
    Base class for anomaly detection models.

    This abstract class defines the interface that all anomaly detection
    models must implement.
    """

    def __init__(self, **kwargs):
        """
        Initialize the detector with model-specific parameters.

        Args:
            **kwargs: Model-specific parameters
        """
        self.params = {**self.get_default_params(), **kwargs}
        if "std_type" not in self.params:
            self.params["std_type"] = "default"
        self.validate_params(self.params)

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        Returns:
            Dictionary of default parameter values
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate the provided parameters.

        Args:
            params: Dictionary of parameters to validate

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    def _detect_multivariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Detect anomalies in multivariate time series.

        Args:
            time_series: Multivariate time series data

        Returns:
            ModelResult object containing detected anomalies and anomaly scores

        Raises:
            NotImplementedError: If the detector does not support multivariate time series
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support multivariate time series. "
            f"Received {time_series.n_series} series."
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Detect anomalies in univariate time series.

        Args:
            time_series: Univariate time series data

        Returns:
            ModelResult object containing detected anomalies and anomaly scores

        Raises:
            NotImplementedError: If the detector does not implement univariate detection
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement univariate detection.")

    def __call__(self, time_series: TimeSeriesWrapper) -> ModelResult:
        if time_series.is_multivariate:
            return self._detect_multivariate(time_series)
        else:
            return self._detect_univariate(time_series)

    def threshold_outputs(
        self,
        scores: np.ndarray,
        expected: "np.ndarray | None" = None,
        residual_std: "float | None" = None,
    ) -> tuple:
        """Compute ``(is_anomaly, expected_bounds)`` from anomaly scores using
        the optional ``threshold`` param.

        Background: precision/recall/F1 reported in the benchmark are computed
        directly from the score column using CV- and EVT-calibrated thresholds
        (see ``AnomalyDetectionBenchmark._calculate_single_metrics``). The
        per-detector fixed ``threshold`` value is therefore not used for any
        metric and has been retired as a required configuration key.

        Behaviour:
        - If ``self.params["threshold"]`` is ``None`` (the new default for all
          detectors), returns ``(zeros[bool], None)`` — placeholder values that
          satisfy ``ModelResult`` validation without leaking a hand-picked
          threshold into the pipeline.
        - If ``threshold`` is set to a number (legacy / debugging), the old
          behaviour is preserved: ``is_anomaly = scores > threshold`` and
          ``expected_bounds = expected ± threshold * residual_std`` (when both
          ``expected`` and ``residual_std`` are provided).
        """
        threshold_param = self.params.get("threshold")
        if threshold_param is None:
            return np.zeros(len(scores), dtype=bool), None

        threshold = float(threshold_param)
        is_anomaly = (scores > threshold)

        if expected is not None and residual_std is not None:
            bounds = np.column_stack(
                [
                    expected - threshold * residual_std,
                    expected + threshold * residual_std,
                ]
            ).astype(float)
            return is_anomaly, bounds

        return is_anomaly, None

    @staticmethod
    def clean_context(values: np.ndarray, mad_threshold: float = 3.0) -> np.ndarray:
        """Replace outliers in a context window with a rolling-median baseline.

        Problem this solves
        -------------------
        Forecasting-based detectors use a sliding context window to predict the
        next ``prediction_length`` steps.  When the context already contains an
        anomaly (e.g. the series has a level-shift or spike inside the window),
        the model *adapts* to the anomalous level and predicts it as the new
        normal — so actual values in the prediction zone produce low residuals
        even when they remain anomalous.  This is the dominant reason why neural
        networks can be outperformed by in-sample AR, which never has this
        "context contamination" problem.

        Solution
        --------
        Before passing a context to the model, detect and replace outlier values
        using a rolling median + MAD (Median Absolute Deviation) filter.  The
        cleaned context makes the model forecast the *normal* baseline, so
        actual anomalous values produce high residuals regardless of whether the
        anomaly started inside or outside the current prediction window.

        Parameters
        ----------
        values        : 1-D context window values (length = context_length)
        mad_threshold : outlier cutoff in MAD units (default 3.0 ≈ 99.7 % for
                        Gaussian; lower ⇒ more aggressive cleaning)

        Returns
        -------
        cleaned : copy of ``values`` with outlier positions replaced by the
                  rolling-median baseline; unchanged if MAD ≈ 0 (flat signal).
        """
        n = len(values)
        if n < 5:
            return values.copy()

        from scipy.signal import medfilt
        w = max(5, min(51, n // 10))
        w = w if w % 2 == 1 else w + 1
        baseline = medfilt(values.astype(float), kernel_size=w)

        diff = np.abs(values - baseline)
        mad  = float(np.median(diff))
        if mad < 1e-8:
            return values.copy()   # flat / near-constant signal — nothing to do

        cleaned = values.copy()
        outliers = diff > mad_threshold * mad * 1.4826  # 1.4826 = 1/Φ^{-1}(0.75)
        cleaned[outliers] = baseline[outliers]
        return cleaned

    def calculate_std(self, residual: np.array) -> float:
        """
        Calculate the standard deviation of the residuals.

        Args:
            residual: Array of residuals

        Returns:
            Standard deviation of residuals
        """
        if self.params["std_type"] == "default":
            return np.sqrt(np.mean(residual**2))
        elif self.params["std_type"] == "mad":
            return np.median(np.abs(residual)) / stats.norm.ppf(0.75)
        elif self.params["std_type"] == "iqr":
            return np.subtract(*np.percentile(residual, [75, 25])) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
        elif self.params["std_type"] == "qn_scale":
            return qn_scale(residual)
        else:
            raise ValueError(f"Unknown std_type: {self.params['std_type']}")
