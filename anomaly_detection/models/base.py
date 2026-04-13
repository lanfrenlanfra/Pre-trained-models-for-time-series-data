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
