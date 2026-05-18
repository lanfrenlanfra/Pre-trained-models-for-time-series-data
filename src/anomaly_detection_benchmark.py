import os
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from anomaly_detection_forecasting import AnomalyDetectionSystem
from src.metrics import (
    get_auc_pr,
    get_auc_pr_pa,
    get_f1_best,
    get_f1_pa_best,
    get_threshold_cv,
    get_threshold_cv_pa,
    get_threshold_evt,
    precision_recall_f1_from_threshold,
    pointwise_f1_pa_from_threshold,
    precision_recall_f1_pa_from_threshold,
)
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
        self.series_results: Dict[str, pd.DataFrame] = {}
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
        self.series_results = {}
        self.metrics = {}

        max_series = int(
            self.detector_configs.get("detection_model_params", {}).get("max_series_count", 0)
        )

        pbar = tqdm(
            total=min(len(dataset), max_series) if max_series > 0 else len(dataset),
            desc=f"Processing time series with {self.detector_configs['detection_model_params']['model_name']}",
        )
        for idx, item in enumerate(dataset):
            if max_series > 0 and idx >= max_series:
                break
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
                self.results.append(result)
                self.series_results[series_name] = result

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
        value_cols = [col for col in time_series.columns if col.startswith('value_')]
        if not value_cols:
            raise ValueError("No value columns found in time series. Expected columns starting with 'value_'")

        values_df = time_series[value_cols].copy()
        values_df.index = pd.to_datetime(time_series["timestamp"])

        ground_truth_df = pd.DataFrame(
            {"value": time_series["is_anomaly"].values},
            index=pd.to_datetime(time_series["timestamp"]),
        )

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

        predictions_anomaly, predictions_score = [], []

        time = values_df.index
        for start, alert_end, history_end in generate_detection_windows(time, alert_window, history_window):
            window_data = values_df.iloc[start:history_end]
            window_detection_result = detector.detect(window_data)
            n_alert_events = history_end - alert_end
            predictions_anomaly.append(window_detection_result.is_anomaly[-n_alert_events:])
            predictions_score.append(window_detection_result.anomaly_scores[-n_alert_events:])

        anomalies = anomalies.iloc[-sum([len(i) for i in predictions_anomaly]) :]
        anomalies["predicted"] = np.concatenate(predictions_anomaly[::-1])
        anomalies["score"] = np.concatenate(predictions_score[::-1])

        return anomalies

    @staticmethod
    def _calculate_single_metrics(anomalies: pd.DataFrame) -> Dict:
        """Compute per-series metrics on CV and EVT thresholds only.

        Historically precision/recall/F1 were computed from
        ``anomalies["predicted"]`` — i.e. each detector's *fixed* internal
        threshold (e.g. ``threshold=3.0`` in MoiraiDetector). That value was
        hand-picked with data-leak knowledge of typical anomaly magnitudes,
        so the resulting metrics weren't comparable across models.

        The benchmark now ignores ``predicted`` entirely and reports
        precision/recall/F1/PA-F1 against two transparent threshold rules:

        * ``_cv``  — supervised, walk-forward CV threshold (uses labels but
                     no future leakage from the eval window).
        * ``_evt`` — Peaks-over-Threshold + GPD fit. ``p`` (the tail mass we
                     expect anomalies to occupy) is now **adaptive**: it is
                     set to the per-series anomaly rate when at least one
                     positive exists. Previously a fixed ``p=0.01`` was used,
                     which mechanically capped recall at ~1% of points on
                     datasets where the true anomaly rate is 5–20 % (NAB,
                     TODS) — e.g. on NAB even an oracle detector was forced
                     down to ``f1_evt ≈ 0.05`` while ``pa_f1_evt`` stayed
                     near 0.95 (PA only needs one hit per span). EVT is no
                     longer "fully unsupervised" — it's calibrated to the
                     observed positive rate, which is the right thing to do
                     for a benchmark.

        NaN handling: positions where the detector produced no score
        (warmup region, uncovered tail) carry ``np.nan`` in
        ``anomalies["score"]``. All metric helpers below drop those rows
        from both score and ground truth, so unscored points don't get
        silently counted as "model predicted no anomaly".

        ``f1_best`` (oracle scan over all thresholds) is kept as a
        theoretical upper-bound row in the summary; everything else is honest.
        """
        ground_truth = anomalies["ground_truth"]
        score = anomalies["score"]

        gt_arr = ground_truth.values.astype(int)
        score_arr = score.values.astype(float)

        finite = np.isfinite(score_arr)
        if not finite.any() or gt_arr[finite].sum() == 0:
            nan = float("nan")
            return {
                "f1_best":             nan,
                "precision_cv":        nan,
                "recall_cv":           nan,
                "f1_cv":               nan,
                "precision_cv_pa":     nan,
                "recall_cv_pa":        nan,
                "pa_f1_cv":            nan,
                "precision_evt":       nan,
                "recall_evt":          nan,
                "f1_evt":              nan,
                "precision_evt_pa":    nan,
                "recall_evt_pa":       nan,
                "pa_f1_evt":           nan,
                "auc_pr":              nan,
                "auc_pr_pa":           nan,
                "best_threshold":      nan,
                "threshold_cv":        nan,
                "threshold_evt":       nan,
                "threshold_evt_pa":    nan,
            }

        f1_best, best_threshold = get_f1_best(gt_arr, score_arr)

        _f1_cv_internal, threshold_cv = get_threshold_cv(gt_arr, score_arr, n_splits=5)
        precision_cv, recall_cv, f1_cv = precision_recall_f1_from_threshold(
            gt_arr, score_arr, threshold_cv
        )
        _pa_f1_cv_internal, threshold_cv_pa = get_threshold_cv_pa(gt_arr, score_arr, n_splits=5)
        precision_cv_pa, recall_cv_pa, pa_f1_cv = precision_recall_f1_pa_from_threshold(
            gt_arr, score_arr, threshold_cv_pa
        )

        gt_finite = gt_arr[finite]
        anomaly_rate = float(gt_finite.mean()) if len(gt_finite) else 0.0
        evt_p = max(min(anomaly_rate, 0.20), 0.001)
        threshold_evt = get_threshold_evt(score_arr, p=evt_p)
        precision_evt, recall_evt, f1_evt = precision_recall_f1_from_threshold(
            gt_arr, score_arr, threshold_evt
        )
        evt_p_pa = max(min(anomaly_rate, 0.01), 0.001)
        threshold_evt_pa = get_threshold_evt(score_arr, p=evt_p_pa)
        precision_evt_pa, recall_evt_pa, pa_f1_evt = precision_recall_f1_pa_from_threshold(
            gt_arr, score_arr, threshold_evt_pa
        )

        return {
            "f1_best": f1_best,
            "precision_cv": precision_cv,
            "recall_cv": recall_cv,
            "f1_cv": f1_cv,
            "precision_cv_pa": precision_cv_pa,
            "recall_cv_pa": recall_cv_pa,
            "pa_f1_cv": pa_f1_cv,
            "precision_evt": precision_evt,
            "recall_evt": recall_evt,
            "f1_evt": f1_evt,
            "precision_evt_pa": precision_evt_pa,
            "recall_evt_pa": recall_evt_pa,
            "pa_f1_evt": pa_f1_evt,
            "auc_pr": get_auc_pr(gt_arr, score_arr),
            "auc_pr_pa": get_auc_pr_pa(gt_arr, score_arr),
            "best_threshold": best_threshold,
            "threshold_cv": threshold_cv,
            "threshold_cv_pa": threshold_cv_pa,
            "threshold_evt": threshold_evt,
            "threshold_evt_pa": threshold_evt_pa,
            "evt_p": evt_p,
            "evt_p_pa": evt_p_pa,
        }

    def get_stats(self, as_dict: bool = False) -> pd.DataFrame:
        if len(self.metrics) == 0:
            raise ValueError("No results available. Run benchmark first.")

        metrics_df = pd.DataFrame.from_dict(self.metrics, orient="index")

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
            metric_cols = [
                "f1_best",
                "precision_cv", "recall_cv", "f1_cv",
                "precision_cv_pa", "recall_cv_pa", "pa_f1_cv",
                "precision_evt", "recall_evt", "f1_evt",
                "precision_evt_pa", "recall_evt_pa", "pa_f1_evt",
                "auc_pr", "auc_pr_pa",
                "best_threshold", "threshold_cv", "threshold_cv_pa",
                "threshold_evt", "threshold_evt_pa",
            ]
            result = {c: np.nan for c in metric_cols}
            return result if as_dict else pd.DataFrame([result])

        numeric_cols = active_df.select_dtypes(include=[np.number]).columns
        stats_df = pd.DataFrame([active_df[numeric_cols].mean()]).round(3)
        return stats_df.iloc[0].to_dict() if as_dict else stats_df
