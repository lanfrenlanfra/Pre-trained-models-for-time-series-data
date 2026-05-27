from datetime import datetime
from typing import Union, Dict, Optional, Any

import pandas as pd
import numpy as np

from . import TimeSeriesWrapper
from ..models import (
    ARDetector,
    ChronosDetector,
    GraniteTTMDetector,
    PatchTSTDetector,
    TimesFMDetector,
    ModelResult,
)

DEFAULT_CONFIGURATION = {
    "detection_model_params": {"model_name": "Autoregressive", "order": 20, "stable": False},
}

class DetectionResult(ModelResult):
    metadata: Dict[str, Any]

class AnomalyDetectionSystem:

    AVAILABLE_MODELS = {
        "Autoregressive": ARDetector,
        "Chronos": ChronosDetector,
        "GraniteTTM": GraniteTTMDetector,
        "PatchTST": PatchTSTDetector,
        "TimesFM": TimesFMDetector,
    }

    def __init__(
        self,
        transforms_params: Optional[Dict] = None,
        detection_model_params: Dict = DEFAULT_CONFIGURATION["detection_model_params"],
    ):
        if detection_model_params is None:
            raise ValueError("detection_model_params is required")

        self.transforms_params = transforms_params.copy() if transforms_params is not None else {}
        self.detection_model_params = detection_model_params.copy() if detection_model_params is not None else {}

        self.original_model_params = {
            "detection_model_params": detection_model_params,
            "transforms_params": transforms_params,
        }

        if "model_name" not in self.detection_model_params:
            raise ValueError("You should specify model_name in detection_model_params")

        self.model_name = self.detection_model_params.pop("model_name")
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {self.model_name}. Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )

    def detect(
            self,
            time_series: Union[
                TimeSeriesWrapper,
                pd.DataFrame,
                tuple[list[datetime], list[float]],
                tuple[list[datetime], list[list[float]]],
                list[tuple[list[datetime], list[float]]]
            ]
    ) -> DetectionResult:
        if not isinstance(time_series, TimeSeriesWrapper):
            time_series = TimeSeriesWrapper(time_series)

        if self.transforms_params:
            processed_ts = AnomalyDetectionSystem._apply_transforms(time_series, **self.transforms_params)
        else:
            processed_ts = time_series

        detection_result = AnomalyDetectionSystem._detect_anomalies(
            processed_ts, self.model_name, self.detection_model_params
        )

        n_samples = processed_ts.time_series_pd.shape[0]
        if len(detection_result.anomaly_scores) != n_samples:
            raise ValueError(
                f"anomaly_scores length ({len(detection_result.anomaly_scores)}) "
                f"must match processed time series length ({n_samples})"
            )

        scores = detection_result.anomaly_scores

        anomaly_scores = np.interp(
            time_series.original_time_series.index.astype(int),
            processed_ts.time_series_pd.index.astype(int),
            scores,
        )
        expected_value, expected_bounds = self._interpolate_expected_values(detection_result, time_series, processed_ts)

        threshold = self.detection_model_params.get("threshold")
        if threshold is None:
            is_anomaly = np.zeros(len(anomaly_scores), dtype=bool)
        else:
            is_anomaly = (anomaly_scores > float(threshold))

        result = DetectionResult(
            anomaly_scores=anomaly_scores,
            is_anomaly=is_anomaly,
            expected_value=expected_value,
            expected_bounds=expected_bounds,
            metadata={},
        )

        return result

    def _interpolate_expected_values(
        self,
        detection_result: ModelResult,
        time_series: TimeSeriesWrapper,
        processed_ts: TimeSeriesWrapper,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._validate_expected():
            return None, None

        expected_value = None
        if detection_result.expected_value is not None:
            ev = np.asarray(detection_result.expected_value)
            if ev.ndim == 1:
                ev = ev.astype(float, copy=False)
                expected_value = np.interp(
                    time_series.original_time_series.index.astype(int),
                    processed_ts.time_series_pd.index.astype(int),
                    ev,
                )
            elif ev.ndim == 2:

                expected_value = ev

        expected_bounds = None
        if detection_result.expected_bounds is not None:
            eb = np.asarray(detection_result.expected_bounds)
            if eb.ndim == 2 and eb.shape[1] == 2:
                eb = eb.astype(float, copy=False)
                expected_bounds = np.column_stack(
                    [
                        np.interp(
                            time_series.original_time_series.index.astype(int),
                            processed_ts.time_series_pd.index.astype(int),
                            eb[:, i],
                        )
                        for i in range(2)
                    ]
                )

        return expected_value, expected_bounds

    def _validate_expected(self):
        if "apply_stl_decomposition" in self.transforms_params and self.transforms_params["apply_stl_decomposition"]:
            return False
        if "apply_spectral_residual" in self.transforms_params and self.transforms_params["apply_spectral_residual"]:
            return False
        return True

    @staticmethod
    def _apply_transforms(time_series: TimeSeriesWrapper, **kwargs):
        time_series = time_series.copy()

        return time_series.apply_transforms(**kwargs)

    @staticmethod
    def _detect_anomalies(
        time_series: TimeSeriesWrapper,
        model_name: str,
        detection_model_params: Dict,
    ) -> ModelResult:
        detector = AnomalyDetectionSystem.AVAILABLE_MODELS[model_name](**detection_model_params)

        return detector(time_series)
