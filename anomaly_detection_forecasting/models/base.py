from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator
from typing import Dict, Any

import numpy as np
from scipy import stats
from statsmodels.robust.scale import qn_scale
from ..core import TimeSeriesWrapper

class ModelResult(BaseModel):

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

    def __init__(self, **kwargs):
        self.params = {**self.get_default_params(), **kwargs}
        if "std_type" not in self.params:
            self.params["std_type"] = "default"
        self.validate_params(self.params)

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        pass

    def validate_params(self, params: Dict[str, Any]) -> None:
        pass

    def _detect_multivariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support multivariate time series. "
            f"Received {time_series.n_series} series."
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
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
            return values.copy()                                                

        cleaned = values.copy()
        outliers = diff > mad_threshold * mad * 1.4826                           
        cleaned[outliers] = baseline[outliers]
        return cleaned

    def calculate_std(self, residual: np.array) -> float:
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
