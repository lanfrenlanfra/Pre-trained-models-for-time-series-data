import os
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from anomaly_detection_forecasting import AnomalyDetectionSystem
from merlion.evaluate.anomaly import TSADMetric
from merlion.utils import TimeSeries
from src.metrics import get_auc_pr, get_f1_best, get_pointwise_f1_pa
from tqdm import tqdm

from .dataset import Dataset
from .loggers import BaseLogger
from .utils import generate_detection_windows


@dataclass
class ProcessWorkerArgs:
    idx: int
    item: Dict[str, Any]
    detector_configs: Dict
    alert_window: timedelta
    history_window: timedelta
    all_at_once: bool
    auto_threshold: bool


class AnomalyDetectionBenchmark:
    def __init__(self, detector_configs: Dict, logger: Optional[BaseLogger] = None):
        self.detector_configs = detector_configs
        self.logger = logger
        self.results = []
        self.metrics = {}
        self.dataset = None

    def run(
        self,
        dataset: Dataset,
        alert_window: timedelta = timedelta(days=1),
        history_window: timedelta = timedelta(days=14),
        all_at_once: bool = False,
        auto_threshold=False,
    ) -> Dict:
        self.dataset = dataset
        self.results = []
        self.metrics = {}

        pbar = tqdm(
            total=len(dataset),
            desc=f"Processing time series with {self.detector_configs['detection_model_params']['model_name']}",
        )
        for idx, item in enumerate(dataset):
            args = ProcessWorkerArgs(
                idx=idx,
                item=item,
                detector_configs=self.detector_configs,
                alert_window=alert_window,
                history_window=history_window,
                all_at_once=all_at_once,
                auto_threshold=auto_threshold,
            )
            result_data = self._process_single_item_worker(args)

            if result_data is None:
                pbar.update(1)
                continue

            idx, result, metrics, item = result_data

            series_name = f"series_{idx:03d}"
            self.metrics[series_name] = metrics

            if result is not None:
                # Skipped series have result=None — don't add to results list
                self.results.append(result)

            if self.logger and result is not None:
                logger_args = dict(
                    series_name=series_name,
                    metrics=metrics,
                    anomalies=result,
                    csv_path=item["csv_path"],
                    overall_metrics=self.metrics.copy(),
                    step_id=idx,
                )
                self.logger.log_single_series_metrics(**logger_args)

            pbar.update(1)

        pbar.close()
        return self.get_stats(as_dict=True)

    @staticmethod
    def _process_single_item_worker(args: ProcessWorkerArgs):
        ad_configs = {k: v for k, v in args.detector_configs.items()
                      if k != "forecasting_model_params"}

        time_series_df = args.item["time_series"]
        n_obs = len(time_series_df)
        ts_start = pd.to_datetime(time_series_df["timestamp"].min())
        ts_end   = pd.to_datetime(time_series_df["timestamp"].max())

        # ── Early-exit for Chronos max_series_length ────────────────────────
        # When Chronos skips a series it returns all-zero scores, which would
        # pollute summary metrics.  Detect the skip here and record NaN metrics
        # so the series is excluded from the mean in get_stats().
        det_params = args.detector_configs.get("detection_model_params", {})
        model_name_cfg = det_params.get("model_name", "")
        max_series_length = int(det_params.get("max_series_length", 0))
        if model_name_cfg == "Chronos" and max_series_length > 0 and n_obs > max_series_length:
            metrics = {
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "f1_best": np.nan,
                "f1_pointwise_pa_best": np.nan,
                "auc_pr": np.nan,
                "best_threshold": np.nan,
                "skipped": True,
                "csv_path": args.item["csv_path"],
                "processing_time": 0.0,
                "time_length": (ts_end - ts_start).total_seconds(),
                "n_observations": n_obs,
            }
            return (args.idx, None, metrics, args.item)
        # ────────────────────────────────────────────────────────────────────

        detector = AnomalyDetectionSystem(**ad_configs)

        start_time = pd.Timestamp.now()
        result = AnomalyDetectionBenchmark.process_time_series(
            time_series_df,
            args.alert_window,
            args.history_window,
            args.all_at_once,
            args.auto_threshold,
            detector,
        )
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()

        if result is None:
            warnings.warn(f"Skipping time series {args.idx} due to history window bigger than whole series")
            return None

        # Calculate and store metrics for this series
        metrics = AnomalyDetectionBenchmark._calculate_single_metrics(result)
        metrics["csv_path"] = args.item["csv_path"]
        metrics["processing_time"] = processing_time
        metrics["time_length"] = (ts_end - ts_start).total_seconds()
        metrics["n_observations"] = n_obs

        return (args.idx, result, metrics, args.item)

    @staticmethod
    def process_time_series(
        time_series: pd.DataFrame,
        alert_window: timedelta,
        history_window: timedelta,
        all_at_once: bool,
        auto_threshold: bool,
        detector: AnomalyDetectionSystem,
    ) -> pd.DataFrame:
        # Find all value columns (value_0, value_1, ..., value_N)
        value_cols = [col for col in time_series.columns if col.startswith('value_')]
        if not value_cols:
            raise ValueError("No value columns found in time series. Expected columns starting with 'value_'")

        # Create DataFrame with all value columns for multivariate detection
        values_df = time_series[value_cols].copy()
        values_df.index = pd.to_datetime(time_series["timestamp"])

        ground_truth_df = pd.DataFrame(
            {"value": time_series["is_anomaly"].values},
            index=pd.to_datetime(time_series["timestamp"]),
        )

        # Create anomalies DataFrame
        # For visualization compatibility, store first column as "value"
        # The detector will use all columns from values_df for multivariate detection
        first_value_col = value_cols[0]
        anomalies = pd.DataFrame(
            {
                "value": values_df[first_value_col].values,
                "ground_truth": ground_truth_df["value"],
            },
            index=ground_truth_df.index,
        )

        anomalies_percentage = anomalies["ground_truth"].sum() / len(anomalies) * 100

        if all_at_once:
            detection_result = detector.detect(values_df)
            anomalies["predicted"] = detection_result.is_anomaly
            anomalies["score"] = detection_result.anomaly_scores
            return anomalies

        if values_df.index[-1] - values_df.index[0] < history_window:
            return None

        # Process windows from right to left
        predictions_anomaly, predictions_score = [], []

        time = values_df.index
        for start, alert_end, history_end in generate_detection_windows(time, alert_window, history_window):
            window_data = values_df.iloc[start:history_end]
            window_detection_result = detector.detect(window_data)
            n_alert_events = history_end - alert_end
            predictions_anomaly.append(window_detection_result.is_anomaly[-n_alert_events:])
            predictions_score.append(window_detection_result.anomaly_scores[-n_alert_events:])

        # Combine predictions and trim unused values
        anomalies = anomalies.iloc[-sum([len(i) for i in predictions_anomaly]) :]
        anomalies["predicted"] = np.concatenate(predictions_anomaly[::-1])
        anomalies["score"] = np.concatenate(predictions_score[::-1])

        return anomalies

    @staticmethod
    def _calculate_single_metrics(anomalies: pd.DataFrame) -> Dict:
        ground_truth, predicted, score = (
            anomalies["ground_truth"],
            anomalies["predicted"],
            anomalies["score"],
        )
        ground_truth_ts = TimeSeries.from_pd(anomalies[["ground_truth"]].rename(columns={"ground_truth": "value"}))
        predicted_ts = TimeSeries.from_pd(anomalies[["predicted"]].rename(columns={"predicted": "value"}))

        if ground_truth.sum() == 0 and predicted.sum() == 0:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "f1_best": 1.0,
                "f1_pointwise_pa_best": 1.0,
                "auc_pr": 0.0,
                "best_threshold": 100.0,
            }

        f1_best, threshold = get_f1_best(ground_truth, score)

        # TSADMetric enum is broken in Python 3.13: functools.partial becomes a
        # descriptor, so TSADMetric.X returns the partial directly (not an enum
        # member). In Python < 3.13 the partial lives at TSADMetric.X.value.
        # getattr(m, 'value', m) covers both cases cleanly.
        def _eval(metric):
            fn = getattr(metric, 'value', metric)
            return fn(ground_truth=ground_truth_ts, predict=predicted_ts)

        return {
            "precision": _eval(TSADMetric.Precision),
            "recall":    _eval(TSADMetric.Recall),
            "f1":        _eval(TSADMetric.F1),
            "f1_best": f1_best,
            "f1_pointwise_pa_best": get_pointwise_f1_pa(ground_truth.values, predicted.values),
            "auc_pr": get_auc_pr(ground_truth, score),
            "best_threshold": threshold,
        }

    def get_stats(self, as_dict: bool = False) -> pd.DataFrame:
        if len(self.metrics) == 0:
            raise ValueError("No results available. Run benchmark first.")

        # Convert metrics dict to DataFrame for easier calculation
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient="index")

        # Exclude series that were skipped (e.g. Chronos max_series_length).
        # Their metric columns are NaN; filtering them explicitly makes the
        # intent clear and prevents bool 'skipped' from entering the mean.
        if "skipped" in metrics_df.columns:
            n_skipped = int(metrics_df["skipped"].eq(True).sum())
            if n_skipped:
                tqdm.write(
                    f"  [benchmark] {n_skipped}/{len(metrics_df)} series were skipped "
                    f"— excluded from summary metrics"
                )
            active_df = metrics_df[metrics_df["skipped"] != True].copy()
        else:
            active_df = metrics_df

        if len(active_df) == 0:
            metric_cols = ["precision", "recall", "f1", "f1_best",
                           "f1_pointwise_pa_best", "auc_pr", "best_threshold"]
            result = {c: np.nan for c in metric_cols}
            return result if as_dict else pd.DataFrame([result])

        # Return mean of all numeric columns except csv_path, rounded to 3 decimal places
        numeric_cols = active_df.select_dtypes(include=[np.number]).columns
        stats_df = pd.DataFrame([active_df[numeric_cols].mean()]).round(3)
        return stats_df.iloc[0].to_dict() if as_dict else stats_df
