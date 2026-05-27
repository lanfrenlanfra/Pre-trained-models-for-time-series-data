from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

def plot_time_series(timestamp, value, labeling_gt, labeling_predicted, title='', backend='plotly'):
    if backend == 'matplotlib':
        plot_time_series_matplotlib(timestamp, value, labeling_gt, labeling_predicted, title)
    elif backend == 'plotly':
        fig = plot_time_series_plotly(timestamp, value, labeling_gt, labeling_predicted, title)
        fig.show()
    elif backend == 'None':
        pass
    else:
        RuntimeError(f'There is no backend {backend}')

def plot_time_series_matplotlib(
    timestamp: Iterable,
    value: Iterable,
    labeling_gt: Iterable,
    labeling_predicted: Iterable,
    scores: Optional[Iterable] = None,
    threshold: float = None,
    title: str = '',
    save_path: str = '',
    show: bool = True,
):
    if scores is not None:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(14, 8), dpi=80,
            gridspec_kw={'height_ratios': [3, 1]}
        )
        fig.suptitle(title)
    else:
        fig, ax1 = plt.subplots(figsize=(14, 6), dpi=80)
        ax1.set_title(title)

    sns.lineplot(x=timestamp, y=value, ax=ax1, label='time series')
    ax1.set_ylabel('value')

    ground_truth_df = pd.DataFrame({'value': labeling_gt}, index=timestamp)

    labeling_gt = np.array(labeling_gt)
    change_points = np.where(np.diff(labeling_gt))[0]
    if labeling_gt[0]:
        change_points = np.insert(change_points, 0, -1)
    if labeling_gt[-1]:
        change_points = np.append(change_points, len(labeling_gt) - 1)

    for i in range(0, len(change_points), 2):
        if i + 1 >= len(change_points):
            break

        start_idx = change_points[i] + 1
        end_idx = change_points[i + 1]

        ax1.axvline(
            x=timestamp[start_idx], color="red", linestyle="--", alpha=0.5,
        )
        ax1.axvline(x=timestamp[end_idx], color="red", linestyle="--", alpha=0.5)
        ax1.axvspan(
            timestamp[start_idx], timestamp[end_idx], color="red", alpha=0.2
        )

    detected_df = pd.DataFrame({'value': value, 'predicted': labeling_predicted}, index=timestamp)
    detected_df = detected_df.loc[detected_df['predicted'] == 1, 'value']
    sns.scatterplot(x=detected_df.index, y=detected_df, color='red', label='detected anomalies', zorder=10, ax=ax1)
    if not detected_df.empty:
        ax1.legend(loc='upper left')

    if scores is not None:
        sns.lineplot(x=ground_truth_df.index, y=scores, color='red', label='scores', ax=ax2)
        if threshold is not None:
            ax2.axhline(threshold, ls='--', color='black', label='threshold')
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax * 2)
        ax2.legend(loc='upper right')
        ax2.set_ylabel('score')
        ax2.tick_params(axis='x', rotation=75)
    else:
        ax1.tick_params(axis='x', rotation=75)

    fig.tight_layout(rect=[0, 0, 1, 0.97] if scores is not None else None)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    if show:
        plt.show()

    plt.close(fig)
    plt.cla()
    plt.clf()

def plot_time_series_plotly(timestamp, value, labeling_gt, labeling_predicted=None, title='') -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=timestamp, y=value, mode='lines', name='Time Series', line=dict(color='blue')))

    y_min = min(value)
    y_max = max(value)
    y_range = y_max - y_min
    fig.update_layout(yaxis=dict(range=[y_min - 0.1 * y_range, y_max + 0.1 * y_range]))

    fig = add_anomaly_segments(fig=fig, timestamps=timestamp, is_anomaly=labeling_gt)

    if labeling_predicted is not None:
        predicted_points = np.where(labeling_predicted)[0]
        if len(predicted_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=np.array(timestamp)[predicted_points],
                    y=np.array(value)[predicted_points],
                    mode='markers',
                    name='Predicted Anomalies',
                    marker=dict(color='red', size=8, symbol='circle'),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        height=600,
        width=1600,
        hovermode='x unified',
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )

    return fig

def add_anomaly_segments(fig: go.Figure, timestamps: Iterable, is_anomaly: Iterable):
    timestamps = np.array(timestamps)
    is_anomaly = np.array(is_anomaly)

    change_points = np.where(np.diff(is_anomaly))[0]

    if is_anomaly[0]:
        change_points = np.insert(change_points, 0, -1)

    if is_anomaly[-1]:
        change_points = np.append(change_points, len(is_anomaly) - 1)

    for i in range(0, len(change_points), 2):
        if i + 1 >= len(change_points):
            break

        start_idx = change_points[i] + 1
        end_idx = change_points[i + 1]

        fig.add_shape(
            type="line",
            x0=timestamps[start_idx],
            y0=0,
            x1=timestamps[start_idx],
            y1=1,
            yref="paper",
            line=dict(color="red", width=1, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=timestamps[end_idx],
            y0=0,
            x1=timestamps[end_idx],
            y1=1,
            yref="paper",
            line=dict(color="red", width=1, dash="dash"),
        )

        fig.add_shape(
            type="rect",
            x0=timestamps[start_idx],
            y0=0,
            x1=timestamps[end_idx],
            y1=1,
            yref="paper",
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
        )

    return fig
