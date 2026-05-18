import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json5
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning)

from src.anomaly_detection_benchmark import AnomalyDetectionBenchmark
from src.dataset import Dataset
from src.loggers import InlineLogger

from termcolor import colored

def _find_spans(flags, times):
    flags = np.asarray(flags, dtype=bool)
    diffs = np.diff(np.concatenate([[False], flags, [False]]).astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    for s, e in zip(starts, ends):
        yield times[s], times[min(e, len(times) - 1)]

def save_series_plot(
    anomalies: pd.DataFrame,
    threshold: float,
    model_name: str,
    csv_path: str,
    save_path: Path,
    threshold_label: str = "oracle",
):
    """Draw and save a two-panel anomaly detection plot for one series.
    ``predicted`` is always recomputed from ``score`` and ``threshold`` so
    that oracle / CV / EVT plots are all self-consistent.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    times = anomalies.index
    v = anomalies["value"].to_numpy(dtype=float)
    gt = anomalies["ground_truth"].to_numpy(dtype=int)
    score = anomalies["score"].to_numpy(dtype=float)
    pred = (score > threshold).astype(int)

    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    title = (
        f"{model_name}  [{threshold_label}]  |  {csv_path}  |  "
        f"n={len(v)}  gt={int(gt.sum())}  pred={int(pred.sum())}  "
        f"TP={tp}  FP={fp}  FN={fn}  "
        f"P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}  threshold={threshold:.2g}"
    )

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.grid": True,
        "grid.color": "#E5E7EB", "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, (ax_ts, ax_sc) = plt.subplots(
        nrows=2, ncols=1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("white")

    fmt = ticker.FuncFormatter(lambda x, _: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.4g}")

    first_gt = True
    for t0, t1 in _find_spans(gt, times):
        ax_ts.axvspan(t0, t1, color="#FEF9C3", alpha=0.7, zorder=1,
                      label="ground truth anomaly" if first_gt else "_nolegend_")
        first_gt = False

    ax_ts.plot(times, v, color="#3B82F6", linewidth=0.8, zorder=4, label="signal")

    scatter_kw = dict(zorder=6, s=35, linewidths=0.4, edgecolors="white")
    tp_mask = (gt == 1) & (pred == 1)
    fn_mask = (gt == 1) & (pred == 0)
    fp_mask = (gt == 0) & (pred == 1)
    if tp_mask.any():
        ax_ts.scatter(times[tp_mask], v[tp_mask], color="#16A34A", label=f"TP ({tp_mask.sum()})", **scatter_kw)
    if fn_mask.any():
        ax_ts.scatter(times[fn_mask], v[fn_mask], color="#F97316", label=f"FN ({fn_mask.sum()})", **scatter_kw)
    if fp_mask.any():
        ax_ts.scatter(times[fp_mask], v[fp_mask], color="#DC2626", label=f"FP ({fp_mask.sum()})", **scatter_kw)

    ax_ts.set_ylim(bottom=0)
    ax_ts.set_ylabel("value", fontsize=8)
    ax_ts.set_xlabel("time", fontsize=8)
    ax_ts.yaxis.set_major_formatter(fmt)
    ax_ts.tick_params(axis="x", labelsize=7, rotation=15)
    ax_ts.tick_params(axis="y", labelsize=7)
    ax_ts.set_facecolor("white")
    ax_ts.set_title(title, fontsize=7, pad=5)
    ax_ts.legend(fontsize=6, loc="upper right", ncol=3, framealpha=0.85)

    for t0, t1 in _find_spans(gt, times):
        ax_sc.axvspan(t0, t1, color="#FEF9C3", alpha=0.5, zorder=1)

    ax_sc.plot(times, score, color="#7C3AED", linewidth=0.7, zorder=3, label="anomaly score")
    ax_sc.axhline(threshold, color="#DC2626", linewidth=1.1, linestyle="--",
                  zorder=5, label=f"threshold = {threshold:.2g}")
    ax_sc.fill_between(times, score, threshold, where=(score > threshold),
                       color="#7C3AED", alpha=0.2, zorder=2, label="score > threshold")

    sc_max = max(float(np.nanmax(score)) if np.isfinite(score).any() else threshold, threshold * 1.1)
    ax_sc.set_ylim(bottom=0, top=sc_max * 1.15)
    ax_sc.set_ylabel("anomaly score", fontsize=8)
    ax_sc.set_xlabel("time", fontsize=8)
    ax_sc.yaxis.set_major_formatter(fmt)
    ax_sc.tick_params(axis="x", labelsize=7, rotation=15)
    ax_sc.tick_params(axis="y", labelsize=7)
    ax_sc.set_facecolor("white")
    ax_sc.legend(fontsize=6, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

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
    colored_rows = []
    for idx, row in enumerate(df.itertuples(index=False, name=None)):
        colored_row = [value_to_color(val) for val in row]
        colored_rows.append(colored_row)
    all_rows = [[str(val) for val in row] for row in df.values]
    col_widths = []
    idx_width = max(len(str(idx)) for idx in df.index)
    for col_idx in range(len(df.columns)):
        col_label = str(df.columns[col_idx])
        max_data = max(len(str(row[col_idx])) for row in all_rows)
        col_widths.append(max(max_data, len(col_label), 6))
    hdr = " " * (idx_width + 2)
    for col_label, width in zip(df.columns, col_widths):
        hdr += f"{col_label:<{width}}  "
    print(hdr)
    for idx, row, colored_row in zip(df.index, all_rows, colored_rows):
        line = f"{str(idx):<{idx_width}}  "
        for val, cval, width in zip(row, colored_row, col_widths):
            disp_len = len(str(val))
            pad = width - disp_len
            line += cval + " " * pad + "  "
        print(line)


def print_time_table(df, title):
    """Print a plain (uncoloured) table of minute durations.
    ``print_colored_table`` clamps values to [0, 1] which makes any time > 1
    show as the same "green/bold" — useless for durations. This printer just
    formats each cell as ``{value:.2f}`` minutes (or ``-`` for NaN/missing)
    and shows totals as a normal column.
    """
    print(f"\n=== {title} ===")

    def fmt(v):
        try:
            f = float(v)
        except Exception:
            return "-"
        if not np.isfinite(f):
            return "-"
        return f"{f:.2f}"

    cols = [str(c) for c in df.columns]
    idx_w = max((len(str(i)) for i in df.index), default=5)
    cell_strs = [[fmt(v) for v in row] for row in df.values]
    col_w = [
        max(len(cols[i]), *(len(row[i]) for row in cell_strs), 5)
        for i in range(len(cols))
    ]

    header = " " * (idx_w + 2)
    for label, w in zip(cols, col_w):
        header += f"{label:>{w}}  "
    print(header)

    for idx_label, row in zip(df.index, cell_strs):
        line = f"{str(idx_label):<{idx_w}}  "
        for val, w in zip(row, col_w):
            line += f"{val:>{w}}  "
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
        choices=['inline', 'underdeep', 'mlflow'],
        help='Logger to use (default: inline)',
    )
    parser.add_argument('--windowed', dest='all_at_once', action='store_false', help='Process series in windows')
    parser.set_defaults(all_at_once=True)

    parser.add_argument('--output_md', type=str, help='Save results to Markdown file with YFM tables')
    parser.add_argument(
        '--no_auto_threshold', action='store_false', dest='auto_threshold', help='Disable auto threshold selection.'
    )

    parser.add_argument('--ad_output_csv', type=str, help='Save results to CSV file')
    parser.add_argument(
        '--ad_time_series_metrics_csv', type=str, help='Save results for each time series to CSV file with'
    )

    parser.add_argument(
        '--plot_dir', type=str, default=None,
        help='Root directory for plots; saves to <plot_dir>/<dataset>/<model>/series_XXX.png '
             '(default: plots/anomaly_detection)'
    )
    parser.add_argument(
        '--models_filter', type=str, default=None,
        help='Comma-separated list of model keys from the JSON file to run (e.g. "timesfm,moirai"). '
             'If omitted, all models in the file are run.'
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

    if args.models_filter:
        _SMART_QUOTES = '«»“”‘’"\'`'
        normalized_filter = args.models_filter
        for ch in _SMART_QUOTES:
            normalized_filter = normalized_filter.replace(ch, '')

        allowed = {k.strip() for k in normalized_filter.split(',') if k.strip()}
        available_keys = list(configurations.keys())
        configurations = {k: v for k, v in configurations.items() if k in allowed}
        if not configurations:
            parser.error(
                f"--models_filter '{args.models_filter}' matched no keys in the JSON file "
                f"(normalized to {sorted(allowed)!r}). "
                f"Available keys: {available_keys}"
            )

    stats = []
    time_series_metrics = []

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
            _t_start = time.perf_counter()
            metrics = benchmark.run(dataset, all_at_once=args.all_at_once, auto_threshold=args.auto_threshold)
            processing_time_min = (time.perf_counter() - _t_start) / 60.0

            plot_root = Path(args.plot_dir) if args.plot_dir else Path("plots") / "anomaly_detection"
            for series_name, anomalies_df in benchmark.series_results.items():
                series_metrics = benchmark.metrics.get(series_name, {})
                csv_path = series_metrics.get("csv_path", series_name)
                plot_base = plot_root / dataset_name / config_name

                threshold_variants = [
                    ("oracle", float(series_metrics.get("best_threshold", 3.0))),
                    ("cv", float(series_metrics.get("threshold_cv", 3.0))),
                    ("evt", float(series_metrics.get("threshold_evt", 3.0))),
                ]
                for label, thr in threshold_variants:
                    plot_path = plot_base / f"{series_name}_{label}.png"
                    try:
                        save_series_plot(
                            anomalies=anomalies_df,
                            threshold=thr,
                            model_name=config_name,
                            csv_path=csv_path,
                            save_path=plot_path,
                            threshold_label=label,
                        )
                    except Exception as e:
                        print(f"  [plot] failed for {series_name} ({label}): {e}")

            time_series_metrics_df = pd.DataFrame.from_dict(benchmark.metrics, orient="index")
            time_series_metrics_df['config_name'] = config_name
            time_series_metrics.append(time_series_metrics_df)

            stats.append(
                {
                    "dataset": dataset_name,
                    "model": config_name,
                    "f1_best": metrics.get("f1_best"),
                    "precision_cv": metrics.get("precision_cv"),
                    "recall_cv": metrics.get("recall_cv"),
                    "f1_cv": metrics.get("f1_cv"),
                    "precision_cv_pa": metrics.get("precision_cv_pa"),
                    "recall_cv_pa": metrics.get("recall_cv_pa"),
                    "pa_f1_cv": metrics.get("pa_f1_cv"),
                    "precision_evt": metrics.get("precision_evt"),
                    "recall_evt": metrics.get("recall_evt"),
                    "f1_evt": metrics.get("f1_evt"),
                    "precision_evt_pa": metrics.get("precision_evt_pa"),
                    "recall_evt_pa": metrics.get("recall_evt_pa"),
                    "pa_f1_evt": metrics.get("pa_f1_evt"),
                    "auc_pr": metrics.get("auc_pr"),
                    "auc_pr_pa": metrics.get("auc_pr_pa"),
                    "processing_time_min": processing_time_min,
                }
            )

            if args.ad_output_csv:
                header = (
                    "dataset,model,f1_best,"
                    "precision_cv,recall_cv,f1_cv,precision_cv_pa,recall_cv_pa,pa_f1_cv,"
                    "precision_evt,recall_evt,f1_evt,precision_evt_pa,recall_evt_pa,pa_f1_evt,"
                    "auc_pr,auc_pr_pa,processing_time_min\n"
                )
                if not os.path.exists(args.ad_output_csv):
                    with open(args.ad_output_csv, "w") as f:
                        f.write(header)
                with open(args.ad_output_csv, "a") as f:
                    f.write(
                        f"{dataset_name},{config_name},{metrics.get('f1_best', '')},"
                        f"{metrics.get('precision_cv', '')},{metrics.get('recall_cv', '')},"
                        f"{metrics.get('f1_cv', '')},{metrics.get('precision_cv_pa', '')},"
                        f"{metrics.get('recall_cv_pa', '')},{metrics.get('pa_f1_cv', '')},"
                        f"{metrics.get('precision_evt', '')},{metrics.get('recall_evt', '')},"
                        f"{metrics.get('f1_evt', '')},{metrics.get('precision_evt_pa', '')},"
                        f"{metrics.get('recall_evt_pa', '')},{metrics.get('pa_f1_evt', '')},"
                        f"{metrics.get('auc_pr', '')},{metrics.get('auc_pr_pa', '')},"
                        f"{processing_time_min:.4f}\n"
                    )

    df = pd.DataFrame(stats)
    index = configurations.keys()

    for metric in (
        "f1_best",
        "precision_cv", "recall_cv", "f1_cv",
        "precision_cv_pa", "recall_cv_pa", "pa_f1_cv",
        "precision_evt", "recall_evt", "f1_evt",
        "precision_evt_pa", "recall_evt_pa", "pa_f1_evt",
        "auc_pr",
    ):
        pivot = df.pivot(index="model", columns="dataset", values=metric)
        pivot = pivot.reindex(index=index, columns=datasets)
        print_colored_table(pivot, title=metric.upper())

    time_pivot = df.pivot(index="model", columns="dataset", values="processing_time_min")
    time_pivot = time_pivot.reindex(index=index, columns=datasets)
    time_pivot["TOTAL"] = time_pivot.sum(axis=1, skipna=True)
    print_time_table(time_pivot, title="PROCESSING_TIME_MIN")

    if args.ad_time_series_metrics_csv:
        df_new = pd.concat(time_series_metrics, axis=0)
        write_header = not os.path.exists(args.ad_time_series_metrics_csv)
        df_new.to_csv(args.ad_time_series_metrics_csv, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    setup_logging()
    main()
