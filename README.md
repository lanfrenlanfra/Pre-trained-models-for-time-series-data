# Pre-trained Models for Time Series Data

Сравнение предобученных моделей временных рядов для задач **обнаружения аномалий** и **прогнозирования**. Сравниваются: AR (авторегрессия), GraniteTTM, Chronos, TimesFM, PatchTST.

---

## Установка окружения

Окружение создается утилитой [`uv`](https://docs.astral.sh/uv/)

```bash
uv sync
source .venv/bin/activate
```

---

## Данные

Датасеты из проекта [TimeSeriesBench](https://github.com/CSTCloudOps/datasets)   
Описания датасетов: <https://thedatumorg.github.io/TSB-AD/>   
Готовый архив с данными: <https://disk.yandex.ru/d/xVd33nmvuR3NTw>   
Каждый CSV содержит колонки: `timestamp` (unix ms), `value_0` [, `value_1`, ...], `is_anomaly`

---

## Структура проекта

```
.
├── run_anomaly_detection.py - Запуск детекции аномалий
├── run_forecasting.py - Запуск прогнозирования
│
├── plot_anomaly_detection.py - Визуализация результатов детекции аномалий
├── plot_forecasting.py - Визуализация результатов прогнозирования
├── plot_time_series.py - Визуализация сырых рядов из датасета
│
├── models.json5 - Конфигурации моделей
│
├── src/ - Инфраструктура детекции аномалий
│   ├── anomaly_detection_benchmark.py - Запускает детекцию аномалий по всему датасету и собирает метрики
│   ├── dataset.py - Итератор по CSV-файлам датасета
│   ├── metrics.py - Метрики, CV/EVT-пороги
│   ├── grapher.py - Отрисовка рядов
│   ├── loggers/
│   │   ├── base_logger.py - Абстрактный базовый класс для логгеров бенча
│   │   └── inline_logger.py - Печатает метрики каждой серии в терминал по мере обработки
│   └── utils/
│       └── utils.py - Нарезка ряда на скользящие окна
│
├── anomaly_detection_forecasting/
│   ├── core/
│   │   ├── time_series.py - Приводит временной ряд к единому формату перед тем, как отдать его в модель
│   │   └── system.py - Склеивает все шаги пайплайна в один вызов
│   └── models/
│       ├── base.py - Задает единый формат результата моделей
│       ├── ar.py - AutoRegressive
│       ├── chronos.py - Chronos
│       ├── granite_ttm.py - GraniteTTM
│       ├── patch_tst.py - PatchTST
│       └── timesfm.py - TimesFM
│
├── scripts/ - Вспомогательные скрипты
│   ├── dataset_stats.py - Статистика по датасетам (размеры, % аномалий)
│   └── plot_dataset_samples.py - Примеры фрагментов рядов
│
├── data/ - Датасеты, создаются при запуске, в репозиторий не коммитятся
├── plots/ - Сгенерированные графики, создаются при запуске, в репозиторий не коммитятся
├── figures/ - Иллюстрации для отчета
│
├── anomaly_detection_summary.csv - Агрегированные метрики аномалий
├── anomaly_detection_per_series.csv - Метрики по каждой серии (аномалии)
├── forecasting_summary.csv - Агрегированные метрики прогнозирования
├── forecasting_per_series.csv - Метрики по каждой серии (прогноз)
│
├── pyproject.toml
└── uv.lock
```

---

## models.json5 — формат конфигурации

Файл описывает все модели. Ключ верхнего уровня — произвольное имя модели, которое используется как метка в выводе и CSV.

```json5
{
    "autoreg": {
        // Параметры предобработки ряда перед детекцией
        "transforms_params": {
            "apply_normalization": false,
            "apply_moving_average": false,
            "moving_average_n_steps": 1
        },
        // Параметры детектора аномалий (для run_anomaly_detection.py)
        "detection_model_params": {
            "model_name": "Autoregressive",
            "order": 20,
            ...
        },
        // Параметры прогнозирования (для run_forecasting.py)
        "forecasting_model_params": {
            "context_length": 512, // длина контекстного окна
            "prediction_length": 96, // горизонт прогноза
            "step": 96, // шаг скользящего окна
            "warmup_points": 512, // точки в начале ряда, исключаемые из метрик
            ...
        }
    },
    "granite_ttm": { ... },
    "chronos":     { ... },
    "timesfm":     { ... },
    "patch_tst":   { ... }
}
```
Общие параметры для нейросетевых моделей:

| Параметр | Описание                                           |
|---|----------------------------------------------------|
| `hf_model_path` | HuggingFace repo id или локальный путь к чекпоинту |
| `context_length` | Количество точек в контексте модели                |
| `prediction_length` | Горизонт прогноза за один вызов                    |
| `step` | Шаг сдвига окна (обычно = `prediction_length`)     |
| `warmup_points` | Точки, исключаемые из оценки метрик                |
| `device` | `cpu`, `cuda`, `mps`, `auto`                       |
| `batch_size` | Размер батча при инференсе                         |
| `max_series_count` | Лимит серий (0 = без лимита)                       |

---

## Запуск

### Обнаружение аномалий

```bash
uv run run_anomaly_detection.py \
  --datasets "AIOPS,TODS,WSD,Yahoo" \
  --models models.json5 \
  --ad_output_csv anomaly_detection_summary.csv \
  --ad_time_series_metrics_csv anomaly_detection_per_series.csv
```

Дополнительные флаги:

| Флаг | Описание |
|---|---|
| `--models_filter "chronos,autoreg"` | Запустить только указанные модели из JSON5 |
| `--windowed` | Скользящий режим (окно истории + окно alert) вместо all-at-once |
| `--plot_dir plots/anomaly_detection` | Сохранять PNG-графики на диск |
| `--ad_output_csv FILE` | CSV с агрегированными метриками (dataset × model) |
| `--ad_time_series_metrics_csv FILE` | CSV с метриками по каждой серии |

### Прогнозирование

```bash
uv run run_forecasting.py \
  --datasets "AIOPS,TODS,WSD,Yahoo" \
  --models models.json5 \
  --f_output_csv forecasting_summary.csv \
  --f_time_series_metrics_csv forecasting_per_series.csv
```

Дополнительные флаги:

| Флаг | Описание |
|---|---|
| `--plot_dir plots/forecasting` | Сохранять PNG-графики на диск |
| `--no_plot` | Отключить генерацию графиков |
| `--datasets_root data` | Корневая папка датасетов |

---

## Формат вывода в терминале

После завершения каждого датасета и каждой модели скрипт печатает цветные таблицы вида `модель × датасет` для каждой метрики. Значения окрашиваются по шкале зеленый-красный. Зеленый = лучшее значение в столбце, красный = худшее.

---

## Генерация графиков

Графики можно генерировать как в процессе бенча (`--plot_dir`), так и отдельно после:

```bash
# Графики прогнозирования
uv run plot_forecasting.py --save --no_show

# Графики детекции аномалий
uv run plot_anomaly_detection.py --save --no_show

# Сырые ряды из датасета
uv run plot_time_series.py --save --no_show
```

Все графики сохраняются в `plots/<task>/<dataset>/<model>/`.

### Вспомогательные скрипты

```bash
# Статистика по датасетам
uv run scripts/dataset_stats.py

# Примеры фрагментов рядов (figures/dataset_examples.png)
uv run scripts/plot_dataset_samples.py
```
