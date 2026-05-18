# Pre-trained models for time series data

Сравнение статистического бенчмарка AR(p) и четырёх предобученных stateless-моделей
(Granite TTM, Chronos, TimesFM, PatchTST) на задачах краткосрочного прогнозирования
и обнаружения аномалий по четырём датасетам мониторинга IT-инфраструктуры
(AIOPS, TODS, WSD, Yahoo).

## Установка окружения

Зависимости фиксированы в `pyproject.toml` и `uv.lock`. Окружение создаётся
утилитой [`uv`](https://docs.astral.sh/uv/) одной командой:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` сам создаёт локальную папку `.venv/` с правильной версией Python и
ставит туда все пакеты ровно тех версий, что указаны в `uv.lock` — это даёт
побитовую воспроизводимость окружения. Папка `.venv/` в репозиторий не
коммитится (она в `.gitignore`).

## Данные

Исходные датасеты взяты из проекта TimeSeriesBench
(<https://github.com/CSTCloudOps/datasets>); описания самих датасетов —
<https://thedatumorg.github.io/TSB-AD/>.

Готовый архив с данными в нужной структуре каталогов лежит на Яндекс.Диске:
<https://disk.yandex.ru/d/xVd33nmvuR3NTw> (папка `/benchmark/data`).
Разархивировать в корень репозитория так, чтобы получилась структура
`data/AIOPS/`, `data/TODS/`, `data/WSD/`, `data/Yahoo/`. Сама папка `data/`
не коммитится из-за объёма (~360 МБ).

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
скриптами и складываются в локальную папку `plots/`, которая в репозиторий
не коммитится из-за объёма (~3 ГБ):

```bash
uv run plot_time_series.py          # примеры рядов с разметкой аномалий
uv run plot_forecasting.py          # графики прогнозов по моделям
uv run plot_anomaly_detection.py    # срабатывания детекторов на рядах
```
