- `anomaly_detection/` - скрипты для системы детекции аномалий
- `benchmark/` - скрипты для бенчмарка 
- Датасеты: https://disk.yandex.ru/d/xVd33nmvuR3NTw - их надо положить в `/benchmark/data`

Запустить детекцию аномалий:
```bash
uv sync
uv run run_anomaly_detection.py \
  --datasets "AIOPS, NAB, TODS, UCR, WSD, Yahoo" \
  --models models.json5 \
  --ad_output_csv anomaly_detection_summary.csv \
  --ad_time_series_metrics_csv anomaly_detection_per_series.csv
```

Запустить прогнозирование:
```bash
uv run run_forecasting.py \
  --datasets "AIOPS, NAB, TODS, UCR, WSD, Yahoo" \
  --models models.json5 \
  --f_output_csv forecasting_summary.csv  \
  --f_time_series_metrics_csv forecasting_per_series.csv
```