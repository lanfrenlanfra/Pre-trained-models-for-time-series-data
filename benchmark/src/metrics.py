import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def get_auc_pr_pa(y_true, y_score):
    """Get AUC-PR point adjusted score.

    Args:
        y_true: array-like of true values
        y_score: array-like of predicted values

    Returns:
        AUC-PR score
    """
    if sum(y_true) == 0:
        return 0.0
    y_true_compressed, y_score_compressed = compress_point_adjusted(y_true, y_score)
    return average_precision_score(y_true_compressed, y_score_compressed, average='micro')


def get_auc_pr(y_true, y_score):
    """Get AUC-PR score.

    Args:
        y_true: array-like of true values
        y_score: array-like of predicted values

    Returns:
        AUC-PR score
    """
    if sum(y_true) == 0:
        return 0.0
    return average_precision_score(y_true, y_score, average='micro')


def compress_point_adjusted(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    diffs = np.diff(np.concatenate([[0], y_true, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    compressed_true = []
    compressed_score = []
    last_end = 0

    for start, end in zip(starts, ends):
        compressed_true.extend(y_true[last_end:start])
        compressed_score.extend(y_score[last_end:start])
        if end > start:
            compressed_true.append(y_true[start])
            compressed_score.append(np.max(y_score[start:end]))
        last_end = end
    compressed_true.extend(y_true[last_end:])
    compressed_score.extend(y_score[last_end:])

    return np.array(compressed_true), np.array(compressed_score)


def get_f1_best(y_true, y_score):
    """Get F1 score for best threshold.

    Args:
        y_true: array-like of true values
        y_score: array-like of predicted values

    Returns:
        F1 score for best threshold and best threshold
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if sum(y_true) == 0:
        return 1.0, 100.0
    y_true_compressed, y_score_compressed = compress_point_adjusted(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true_compressed, y_score_compressed)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    thresholds = np.concatenate(
        [thresholds, [max(y_score) + 1e-6]]
    )  # thresholds from precision_recall_curve are missing highest value

    highest_threshold = thresholds[np.argmax(f1)]
    negative_class_scores = y_score[y_true == 0]
    lowest_threshold = negative_class_scores[negative_class_scores < highest_threshold].max() + 1e-6

    return (
        np.max(f1),
        lowest_threshold,
    )


def get_pointwise_f1_pa(y_true, y_pred):
    """Compute point-adjusted pointwise F1 score for binary predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")

    adjusted_pred = y_pred.copy()

    diffs = np.diff(np.concatenate([[0], y_true, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for start, end in zip(starts, ends):
        if np.any(y_pred[start:end]):
            adjusted_pred[start:end] = 1

    true_positive = np.logical_and(y_true == 1, adjusted_pred == 1).sum()
    false_positive = np.logical_and(y_true == 0, adjusted_pred == 1).sum()
    false_negative = np.logical_and(y_true == 1, adjusted_pred == 0).sum()

    if true_positive == 0 and false_positive == 0 and false_negative == 0:
        return 1.0

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)

    return 2 * precision * recall / (precision + recall + 1e-8)
