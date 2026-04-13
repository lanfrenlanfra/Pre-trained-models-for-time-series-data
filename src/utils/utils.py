import pandas as pd
from datetime import timedelta


def generate_detection_windows(time: pd.DatetimeIndex, alert_window: timedelta, history_window: timedelta):
    start_time = time[-1] - history_window
    history_end = len(time)
    alert_end = len(time)
    start = len(time) - 1
    while start_time >= time[0]:
        while start > 0 and time[start - 1] > start_time:
            start -= 1
        while alert_end > 0 and time[alert_end - 1] > start_time + history_window - alert_window:
            alert_end -= 1
        while history_end > 0 and time[history_end - 1] > start_time + history_window:
            history_end -= 1
        yield start, alert_end, history_end
        start_time -= alert_window
