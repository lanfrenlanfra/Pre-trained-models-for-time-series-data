from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd

class BaseLogger(ABC):

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def log_single_series_metrics(self, series_name: str, metrics: Dict, anomalies: pd.DataFrame, *args, **kwargs):
        pass
