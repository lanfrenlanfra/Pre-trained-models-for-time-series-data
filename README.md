**English** | [Русский](README.ru.md)

# Pre-trained Models for Time Series Data

A comparison of pre-trained time series models for **anomaly detection** and **forecasting**. Compared models: AR (autoregression), GraniteTTM, Chronos, TimesFM, PatchTST.   

---

## Environment setup

The environment is created with [`uv`](https://docs.astral.sh/uv/)

```bash
uv sync
source .venv/bin/activate
```

---

## Data

Datasets come from the [TimeSeriesBench](https://github.com/CSTCloudOps/datasets) project   
Dataset descriptions: <https://thedatumorg.github.io/TSB-AD/>   
Ready-made data archive: <https://disk.yandex.ru/d/xVd33nmvuR3NTw>   
Each CSV contains the columns: `timestamp` (unix ms), `value_0` [, `value_1`, ...], `is_anomaly`

---

## Project structure

```
.
├── run_anomaly_detection.py - Run anomaly detection
├── run_forecasting.py - Run forecasting
│
├── plot_anomaly_detection.py - Visualize anomaly detection results
├── plot_forecasting.py - Visualize forecasting results
├── plot_time_series.py - Visualize raw series from the dataset
│
├── models.json5 - Model configurations
│
├── src/ - Anomaly detection infrastructure
│   ├── anomaly_detection_benchmark.py - Runs anomaly detection over the whole dataset and collects metrics
│   ├── dataset.py - Iterator over the dataset's CSV files
│   ├── metrics.py - Metrics, CV/EVT thresholds
│   ├── grapher.py - Series plotting
│   ├── loggers/
│   │   ├── base_logger.py - Abstract base class for benchmark loggers
│   │   └── inline_logger.py - Prints each series' metrics to the terminal as they are processed
│   └── utils/
│       └── utils.py - Splitting a series into sliding windows
│
├── anomaly_detection_forecasting/
│   ├── core/
│   │   ├── time_series.py - Converts a time series to a unified format before passing it to a model
│   │   └── system.py - Chains all pipeline steps into a single call
│   └── models/
│       ├── base.py - Defines the common model output format
│       ├── ar.py - AutoRegressive
│       ├── chronos.py - Chronos
│       ├── granite_ttm.py - GraniteTTM
│       ├── patch_tst.py - PatchTST
│       └── timesfm.py - TimesFM
│
├── scripts/ - Helper scripts
│   ├── dataset_stats.py - Dataset statistics (sizes, % of anomalies)
│   └── plot_dataset_samples.py - Example series fragments
│
├── data/ - Datasets, created at runtime, not committed to the repository
├── plots/ - Generated plots, created at runtime, not committed to the repository
├── figures/ - Illustrations for the report
│
├── anomaly_detection_summary.csv - Aggregated anomaly detection metrics
├── anomaly_detection_per_series.csv - Per-series metrics (anomaly detection)
├── forecasting_summary.csv - Aggregated forecasting metrics
├── forecasting_per_series.csv - Per-series metrics (forecasting)
│
├── pyproject.toml
└── uv.lock
```

---

## models.json5 — configuration format

The file describes all models. The top-level key is an arbitrary model name, used as a label in the output and in CSV files.

```json5
{
    "autoreg": {
        // Series preprocessing parameters applied before detection
        "transforms_params": {
            "apply_normalization": false,
            "apply_moving_average": false,
            "moving_average_n_steps": 1
        },
        // Anomaly detector parameters (for run_anomaly_detection.py)
        "detection_model_params": {
            "model_name": "Autoregressive",
            "order": 20,
            ...
        },
        // Forecasting parameters (for run_forecasting.py)
        "forecasting_model_params": {
            "context_length": 512, // context window length
            "prediction_length": 96, // forecast horizon
            "step": 96, // sliding window step
            "warmup_points": 512, // points at the start of the series excluded from metrics
            ...
        }
    },
    "granite_ttm": { ... },
    "chronos":     { ... },
    "timesfm":     { ... },
    "patch_tst":   { ... }
}
```
Common parameters for neural network models:

| Parameter | Description                                                  |
|---|--------------------------------------------------------------|
| `hf_model_path` | HuggingFace repo id or local path to a checkpoint            |
| `context_length` | Number of points in the model's context                      |
| `prediction_length` | Forecast horizon per call                                    |
| `step` | Window shift step (usually = `prediction_length`)            |
| `warmup_points` | Points excluded from metric evaluation                       |
| `device` | `cpu`, `cuda`, `mps`, `auto`                                 |
| `batch_size` | Batch size during inference                                  |
| `max_series_count` | Series limit (0 = no limit)                                  |

---

## Running

### Anomaly detection

```bash
uv run run_anomaly_detection.py \
  --datasets "AIOPS,TODS,WSD,Yahoo" \
  --models models.json5 \
  --ad_output_csv anomaly_detection_summary.csv \
  --ad_time_series_metrics_csv anomaly_detection_per_series.csv
```

Additional flags:

| Flag | Description |
|---|---|
| `--models_filter "chronos,autoreg"` | Run only the listed models from the JSON5 file |
| `--windowed` | Sliding mode (history window + alert window) instead of all-at-once |
| `--plot_dir plots/anomaly_detection` | Save PNG plots to disk |
| `--ad_output_csv FILE` | CSV with aggregated metrics (dataset × model) |
| `--ad_time_series_metrics_csv FILE` | CSV with per-series metrics |

### Forecasting

```bash
uv run run_forecasting.py \
  --datasets "AIOPS,TODS,WSD,Yahoo" \
  --models models.json5 \
  --f_output_csv forecasting_summary.csv \
  --f_time_series_metrics_csv forecasting_per_series.csv
```

Additional flags:

| Flag | Description |
|---|---|
| `--plot_dir plots/forecasting` | Save PNG plots to disk |
| `--no_plot` | Disable plot generation |
| `--datasets_root data` | Root folder of the datasets |

---

## Terminal output format

After each dataset and each model finishes, the script prints colored `model × dataset` tables for every metric. Values are colored on a green-to-red scale: green = best value in the column, red = worst.

---

## Plot generation

Plots can be generated during the benchmark (`--plot_dir`) or separately afterwards:

```bash
# Forecasting plots
uv run plot_forecasting.py --save --no_show

# Anomaly detection plots
uv run plot_anomaly_detection.py --save --no_show

# Raw series from the dataset
uv run plot_time_series.py --save --no_show
```

All plots are saved to `plots/<task>/<dataset>/<model>/`.

### Helper scripts

```bash
# Dataset statistics
uv run scripts/dataset_stats.py

# Example series fragments (figures/dataset_examples.png)
uv run scripts/plot_dataset_samples.py
```
