from datetime import timedelta
from typing import Dict
import os
import sys

import pandas as pd
from src.grapher import plot_time_series

from .base_logger import BaseLogger
from termcolor import colored


class InlineLogger(BaseLogger):
    def __init__(self, backend: str = 'plotly', clear_screen: bool = True, **kwargs):
        self.backend = backend
        self.params = kwargs
        self.clear_screen_opt = clear_screen

    def clear_screen(self):
        # Cross-platform clear screen
        if sys.platform.startswith('win'):
            os.system('cls')
        else:
            os.system('clear')

    def log_single_series_metrics(
        self, series_name: str, metrics: Dict, anomalies: pd.DataFrame, csv_path: str, **kwargs
    ):
        """Log metrics and artifacts for a single time series to stdout."""

        if self.clear_screen_opt:
            self.clear_screen()

        print("\n" + colored(f"=== Results for: {series_name} ({csv_path}) ===", 'cyan', attrs=['bold']))

        print(colored('➤ Anomalies Summary:', 'yellow', attrs=['bold']))
        summary = anomalies[['predicted', 'ground_truth']].sum()
        print(
            f"  Ground truth: {colored(summary['ground_truth'], 'green', attrs=['bold'])} | "
            f"Predicted: {colored(summary['predicted'], 'red', attrs=['bold'])}"
        )

        if self.backend is not None:
            print(colored('\n➤ Plotting time series (interactive)...', 'blue'))
            plot_time_series(
                timestamp=anomalies.index,
                value=anomalies['value'],
                labeling_gt=anomalies['ground_truth'],
                labeling_predicted=anomalies['predicted'],
                title=f"{series_name} ({csv_path})",
                backend=self.backend,
            )

        print(colored('\n➤ Metrics:', 'magenta', attrs=['bold']))
        padding = max(len(str(metric)) for metric in metrics)
        for metric, value in metrics.items():
            metric_name = metric + ":"
            if metric == 'time_length':
                td = timedelta(seconds=value)
                val_str = colored(str(td), 'white', attrs=['bold'])
            else:
                val_str = colored(str(value), 'white')
            print(f"  {colored(metric_name.ljust(padding+2), 'green')}{val_str}")

        print(colored('=' * 60, 'cyan'))
