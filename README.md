Окружение создается утилитой [`uv`](https://docs.astral.sh/uv/) командой:

```bash
uv sync
source .venv/bin/activate
```

## Данные

Датасеты взяты из проекта TimeSeriesBench <https://github.com/CSTCloudOps/datasets>
Описания датасетов <https://thedatumorg.github.io/TSB-AD/>.
Готовый архив с данными <https://disk.yandex.ru/d/xVd33nmvuR3NTw>

## Запуск экспериментов

Обнаружение аномалий:

```bash
uv run run_anomaly_detection.py \
  --datasets "AIOPS, TODS, WSD, Yahoo" \
  --models models.json5 \
  --ad_output_csv anomaly_detection_summary.csv \
  --ad_time_series_metrics_csv anomaly_detection_per_series.csv
```

Прогнозирование:

```bash
uv run run_forecasting.py \
  --datasets "AIOPS, TODS, WSD, Yahoo" \
  --models models.json5 \
  --f_output_csv forecasting_summary.csv \
  --f_time_series_metrics_csv forecasting_per_series.csv
```

## Генерация графиков

Per-series графики прогнозов и срабатываний детекторов генерируются отдельно
скриптами и складываются в локальную папку `plots/`
