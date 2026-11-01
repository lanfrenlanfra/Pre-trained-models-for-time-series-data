**English** | [Русский](README.ru.md)

# Scripts for the "Data" section of the thesis

## 1. Building example series fragments

Picks one representative file from each dataset
(AIOPS, TODS, WSD, Yahoo), shows a continuous fragment around the
first anomaly, marks the labeled anomalies in red, and
saves the result as a single PNG.

```bash
# from the repository root
python scripts/plot_dataset_samples.py \
    --data-root data \
    --out-dir figures \
    --copy-to "path/to/Pre-trained-models-for-time-series-data/graphics"
```

Output:

- `figures/dataset_examples.png` — a local copy in the repository;
- `Pre-trained-models-for-time-series-data/graphics/dataset_examples.png` — a copy in the folder with
  `coursework.tex`, which `pdflatex` picks up automatically
  via `\graphicspath{{graphics/}}`, so
  `\includegraphics{dataset_examples.png}` works without
  any extra setup.

To specify particular example files:

```bash
python scripts/plot_dataset_samples.py \
    --files "AIOPS=data/AIOPS/05f10d3a-239c-3bef-9bdc-a2feeb0037aa.csv,TODS=data/TODS/0.csv"
```

## 2. Recomputing dataset statistics (for the text and Table 2.1)

For each dataset, prints the number of series, the minimum,
median and maximum length, the sampling step, the total number
of points, the global share of anomalies, and the number of series that
contain at least one labeled anomaly. It also prints
a breakdown by Yahoo sub-collections (A3, A4, real\_, synthetic\_).

```bash
python scripts/dataset_stats.py --data-root data
```

If the data changes (files added to or removed from
`data/`), just rerun this script and copy the
new numbers into Table 2.1 in `coursework.tex`. The current values in
the thesis correspond to the following data snapshot:

| Dataset | Number of series | Length (median) | Step   | Total points | Anomalies, % |
|---------|-----------------:|----------------:|--------|-------------:|-------------:|
| AIOPS   |               29 |         263,590 | 1 min  |       6.15 M |         2.27 |
| TODS    |               15 |          10,000 | 5 min  |       0.15 M |         5.85 |
| WSD     |              210 |          36,471 | 1 min  |       7.66 M |         1.60 |
| Yahoo   |              367 |           1,680 | 1 hour |       0.57 M |         0.68 |

## Dependencies

```bash
pip install pandas numpy matplotlib
```

(All three packages are already installed in the project's `.venv` virtual environment.)
