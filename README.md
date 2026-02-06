- `anomaly_detection/` - скрипты для системы детекции аномалий
- `benchmark/` - скрипты для бенчмарка 
- Датасеты: https://disk.yandex.ru/d/xVd33nmvuR3NTw - их надо положить в `/benchmark/data`

Запустить бенчмарк можно командой:
```bash
uv sync
uv run run_benchmark.py \
  --datasets "AIOPS, NAB, TODS, UCR, WSD, Yahoo" \
  --models models.json5
```