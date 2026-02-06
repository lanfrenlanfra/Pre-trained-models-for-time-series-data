import argparse
import json5
import logging
import pandas as pd

from src.anomaly_detection_benchmark import AnomalyDetectionBenchmark
from src.dataset import Dataset
from src.loggers import InlineLogger, UnderdeepLogger, MLflowLogger

from termcolor import colored


def value_to_color(val):
    """
    Convert value to colored text for terminal output.
    """
    try:
        v = float(val)
    except Exception:
        return str(val)

    v = min(max(v, 0.0), 1.0)

    if v >= 0.85:
        color = "green"
        attrs = ["bold"]
    elif v >= 0.5:
        color = "yellow"
        attrs = []
    elif v >= 0.2:
        color = "magenta"
        attrs = []
    else:
        color = "red"
        attrs = ["bold"]

    return colored(f"{v:.3f}", color, attrs=attrs)


def print_colored_table(df, title):
    print(f"\n=== {title} ===")
    # Build the colored string table
    colored_rows = []
    for idx, row in enumerate(df.itertuples(index=False, name=None)):
        colored_row = [value_to_color(val) for val in row]
        colored_rows.append(colored_row)
    # Compute widths (based on non-colored string representation)
    all_rows = [[str(val) for val in row] for row in df.values]
    col_widths = []
    # Max width for index column
    idx_width = max(len(str(idx)) for idx in df.index)
    # Max width for each data column
    for col_idx in range(len(df.columns)):
        col_label = str(df.columns[col_idx])
        max_data = max(len(str(row[col_idx])) for row in all_rows)
        col_widths.append(max(max_data, len(col_label), 6))
    # Print header
    hdr = " " * (idx_width + 2)
    for col_label, width in zip(df.columns, col_widths):
        hdr += f"{col_label:<{width}}  "
    print(hdr)
    # Print rows
    for idx, row, colored_row in zip(df.index, all_rows, colored_rows):
        line = f"{str(idx):<{idx_width}}  "
        for val, cval, width in zip(row, colored_row, col_widths):
            disp_len = len(str(val))
            pad = width - disp_len
            line += cval + " " * pad + "  "
        print(line)


def setup_logging():
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.disabled = True

    cmdstan_logger = logging.getLogger("cmdstan")
    cmdstan_logger.disabled = True


def main():
    parser = argparse.ArgumentParser(description='Run Anomaly Detection Benchmark')
    parser.add_argument('--datasets', type=str, required=True, help='Comma-separated list of dataset names')
    parser.add_argument(
        '--models', type=argparse.FileType('r'), required=True, help='Path to JSON file with model configurations.'
    )
    parser.add_argument(
        '--logger',
        type=str,
        default='inline',
        choices=['inline', 'underdeep', 'mlflow'],  # you have inline only. Actually, try implementing WandB logger
        help='Logger to use (default: inline)',
    )
    parser.add_argument('--windowed', dest='all_at_once', action='store_false', help='Process series in windows')
    parser.set_defaults(all_at_once=True)

    parser.add_argument('--output_md', type=str, help='Save results to Markdown file with YFM tables')
    parser.add_argument(
        '--no_auto_threshold', action='store_false', dest='auto_threshold', help='Disable auto threshold selection.'
    )

    parser.add_argument('--output_csv', type=str, help='Save results to CSV file')
    parser.add_argument(
        '--time_series_metrics_csv', type=str, help='Save results for each time series to CSV file with'
    )

    args = parser.parse_args()

    datasets = [name.strip() for name in args.datasets.split(',') if name.strip()]

    try:
        configurations = json5.load(args.models)
    except Exception as e:
        parser.error(f"Could not parse models JSON file: {e}")

    if not isinstance(configurations, dict) or not all(
        isinstance(item, dict) and len(item) in [2, 3] for item in configurations.values()
    ):
        parser.error(
            "The models JSON should be a dictionary with model names as keys and configurations as values: {'model_name': config, ...}"
        )

    stats = []  # Collect all results
    time_series_metrics = []  # Time series metrics for each config

    for dataset_name in datasets:
        dataset = Dataset(f'data/{dataset_name}/')
        for config_name, configuration in configurations.items():
            if args.logger == 'inline':
                logger = InlineLogger(backend=None)
            elif args.logger == 'mlflow':
                logger = MLflowLogger(
                    experiment_name=dataset_name.lower().replace('/', '-'),
                    run_name=config_name,
                    detector_config=configuration,
                )
            else:
                logger = UnderdeepLogger(
                    project_code="test-kek",
                    experiment_code=dataset_name.lower().replace('/', '-'),
                    run_name=config_name,
                    detector_config=configuration,
                )
            benchmark = AnomalyDetectionBenchmark(
                detector_configs=configuration,
                logger=logger,
            )
            metrics = benchmark.run(dataset, all_at_once=args.all_at_once, auto_threshold=args.auto_threshold)
            time_series_metrics_df = pd.DataFrame.from_dict(benchmark.metrics, orient="index")
            time_series_metrics_df['config_name'] = config_name
            time_series_metrics.append(time_series_metrics_df)

            stats.append(
                {
                    "dataset": dataset_name,
                    "model": config_name,
                    "f1_best": metrics.get("f1_best"),
                    "f1_pointwise_pa_best": metrics.get("f1_pointwise_pa_best"),
                    "f1": metrics.get("f1"),
                    "recall": metrics.get("recall"),
                    "precision": metrics.get("precision"),
                    "auc_pr": metrics.get("auc_pr"),
                }
            )

            # Append results to CSV if requested
            if args.output_csv:
                with open(args.output_csv, 'a') as f:
                    f.write(
                        f"{dataset_name},{config_name},{metrics.get('f1_best', '')},"
                        f"{metrics.get('f1_pointwise_pa_best', '')},{metrics.get('f1', '')},"
                        f"{metrics.get('recall', '')},{metrics.get('precision', '')},{metrics.get('auc_pr', '')}\n"
                    )

    df = pd.DataFrame(stats)
    index = configurations.keys()

    for metric in ("f1_best", "f1_pointwise_pa_best", "f1", "recall", "precision", "auc_pr"):
        pivot = df.pivot(index="model", columns="dataset", values=metric)
        pivot = pivot.reindex(index=index, columns=datasets)
        print_colored_table(pivot, title=metric.upper())

    if args.time_series_metrics_csv:
        pd.concat(time_series_metrics, axis=0).to_csv(args.time_series_metrics_csv, index=False)


if __name__ == "__main__":
    setup_logging()
    main()
