from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class BaseLogger(ABC):
    """
    Base class for benchmark loggers.

    This abstract class defines the interface that all benchmark loggers must implement.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def log_single_series_metrics(self, series_name: str, metrics: Dict, anomalies: pd.DataFrame, *args, **kwargs):
        """Log metrics and artifacts for a single time series.

        Args:
            series_name: Name of the series (e.g., "series_001")
            metrics: Dictionary containing metrics for this series
            result: Dictionary containing anomalies DataFrame and metadata
        """
        pass
