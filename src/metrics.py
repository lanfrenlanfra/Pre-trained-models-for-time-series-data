import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.stats import genpareto

def get_auc_pr_pa(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score)
    if not finite.any():
        return float("nan")
    y_true = y_true[finite]
    y_score = y_score[finite]
    if y_true.sum() == 0:
        return float("nan")

    inflated = y_score.copy()
    diffs = np.diff(np.concatenate([[0], y_true, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    for s, e in zip(starts, ends):
        inflated[s:e] = float(np.max(y_score[s:e]))

    return average_precision_score(y_true, inflated, average='micro')

def get_auc_pr(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score)
    if not finite.any():
        return float("nan")
    y_true = y_true[finite]
    y_score = y_score[finite]
    if y_true.sum() == 0:
        return float("nan")
    return average_precision_score(y_true, y_score, average='micro')

def get_f1_best(ground_truth, score):
    ground_truth = np.asarray(ground_truth).astype(int)
    score = np.asarray(score, dtype=float)

    finite = np.isfinite(score)
    ground_truth = ground_truth[finite]
    score = score[finite]

    n = len(score)
    n_pos = int(ground_truth.sum())

    if n == 0:
        return 0.0, 0.0
    if n_pos == 0:
        return 0.0, float(np.max(score)) + 1e-6
    if n_pos == n:
        return 1.0, float(np.min(score)) - 1e-6

    order = np.argsort(-score, kind="mergesort")
    sc_sorted = score[order]
    gt_sorted = ground_truth[order].astype(np.int64)

    cum_tp = np.cumsum(gt_sorted)
    indices = np.arange(1, n + 1, dtype=np.int64)
    cum_fp = indices - cum_tp
    cum_fn = n_pos - cum_tp

    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    recall = cum_tp / (cum_tp + cum_fn + 1e-8)
    f1_array = 2 * precision * recall / (precision + recall + 1e-8)

    boundary = np.empty(n, dtype=bool)
    boundary[:-1] = sc_sorted[:-1] != sc_sorted[1:]
    boundary[-1] = True
    f1_array = np.where(boundary, f1_array, -np.inf)

    best_idx = int(np.argmax(f1_array))
    best_f1  = float(f1_array[best_idx])

    if best_idx == n - 1:
        best_threshold = float(sc_sorted[-1] - 1e-6)
    else:
        hi = float(sc_sorted[best_idx])
        lo = float(sc_sorted[best_idx + 1])
        best_threshold = 0.5 * (hi + lo) if hi != lo else hi - 1e-6

    return best_f1, best_threshold

def precision_recall_f1_from_threshold(
    ground_truth: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    ground_truth = np.asarray(ground_truth).astype(int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    ground_truth = ground_truth[finite]
    score = score[finite]
    predicted = (score > threshold).astype(int)
    tp = int(((predicted == 1) & (ground_truth == 1)).sum())
    fp = int(((predicted == 1) & (ground_truth == 0)).sum())
    fn = int(((predicted == 0) & (ground_truth == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)

def _f1_from_threshold(ground_truth: np.ndarray, score: np.ndarray, threshold: float) -> float:
    _p, _r, f1 = precision_recall_f1_from_threshold(ground_truth, score, threshold)
    return f1

def pointwise_f1_pa_from_threshold(
    ground_truth: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> float:
    score = np.asarray(score, dtype=float)
    ground_truth = np.asarray(ground_truth).astype(int)
    finite = np.isfinite(score)
    score = score[finite]
    ground_truth = ground_truth[finite]
    predicted = (score > threshold).astype(int)
    return get_pointwise_f1_pa(ground_truth, predicted)

def precision_recall_f1_pa_from_threshold(
    ground_truth: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    score = np.asarray(score, dtype=float)
    ground_truth = np.asarray(ground_truth).astype(int)
    finite = np.isfinite(score)
    score = score[finite]
    ground_truth = ground_truth[finite]
    predicted = (score > threshold).astype(int)

    adjusted_pred = predicted.copy()
    diffs = np.diff(np.concatenate([[0], ground_truth, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    for s, e in zip(starts, ends):
        if np.any(predicted[s:e]):
            adjusted_pred[s:e] = 1

    tp = int(((ground_truth == 1) & (adjusted_pred == 1)).sum())
    fp = int(((ground_truth == 0) & (adjusted_pred == 1)).sum())
    fn = int(((ground_truth == 1) & (adjusted_pred == 0)).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)

def get_f1_pa_best(ground_truth, score):
    ground_truth = np.asarray(ground_truth).astype(int)
    score = np.asarray(score, dtype=float)

    finite = np.isfinite(score)
    ground_truth = ground_truth[finite]
    score = score[finite]

    n = len(score)
    n_pos = int(ground_truth.sum())

    if n == 0 or n_pos == 0:
        return 0.0, (float(np.max(score)) + 1e-6 if n > 0 else 0.0)
    if n_pos == n:
        return 1.0, float(np.min(score)) - 1e-6

    diffs = np.diff(np.concatenate([[0], ground_truth, [0]]))
    span_starts = np.where(diffs == 1)[0]
    span_ends = np.where(diffs == -1)[0]
    n_spans = len(span_starts)

    span_id = np.full(n, -1, dtype=np.int32)
    span_len = np.zeros(n_spans, dtype=np.int64)
    for i, (s, e) in enumerate(zip(span_starts, span_ends)):
        span_id[s:e] = i
        span_len[i]  = e - s

    order = np.argsort(-score, kind="mergesort")
    sc_sorted = score[order]
    sid_sorted = span_id[order]

    tp_pa = np.int64(0)
    fp = np.int64(0)
    fn_pa = np.int64(n_pos)
    span_hit = np.zeros(n_spans, dtype=bool)

    best_f1 = 0.0
    best_thr = float(sc_sorted[-1]) - 1e-6

    for k in range(n):
        sid = int(sid_sorted[k])
        if sid >= 0:
            if not span_hit[sid]:
                span_hit[sid] = True
                L = int(span_len[sid])
                tp_pa += L
                fn_pa -= L
        else:
            fp += 1

        if k < n - 1 and sc_sorted[k] == sc_sorted[k + 1]:
            continue

        hi = float(sc_sorted[k])
        lo = float(sc_sorted[k + 1]) if k < n - 1 else hi - 2e-6
        thr = 0.5 * (hi + lo) if hi != lo else hi - 1e-6

        prec = tp_pa / (tp_pa + fp + 1e-8)
        rec = tp_pa / (tp_pa + fn_pa + 1e-8)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return float(best_f1), float(best_thr)

def get_threshold_cv_pa(ground_truth, score, n_splits: int = 5):
    ground_truth = np.asarray(ground_truth).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(score)

    fold_size = n // (n_splits + 1)
    if fold_size < 2:
        finite = np.isfinite(score)
        if not finite.any():
            return 0.0, 0.0
        f1, thr = get_f1_pa_best(ground_truth[finite], score[finite])
        return f1, thr

    thresholds = []
    for k in range(1, n_splits + 1):
        val_start = k * fold_size
        val_end = min((k + 1) * fold_size, n)

        val_gt = ground_truth[val_start:val_end]
        val_score = score[val_start:val_end]

        finite = np.isfinite(val_score)
        if not finite.any():
            continue
        val_gt = val_gt[finite]
        val_score = val_score[finite]

        if val_gt.sum() == 0:
            continue

        _, fold_thr = get_f1_pa_best(val_gt, val_score)
        thresholds.append(fold_thr)

    finite_all = np.isfinite(score)
    if not thresholds:
        if not finite_all.any():
            return 0.0, 0.0
        f1, thr = get_f1_pa_best(ground_truth[finite_all], score[finite_all])
        return f1, thr

    cv_threshold_pa = float(np.median(thresholds))
    _, _, pa_f1 = precision_recall_f1_pa_from_threshold(
        ground_truth[finite_all], score[finite_all], cv_threshold_pa
    )
    return pa_f1, cv_threshold_pa

def get_threshold_cv(ground_truth, score, n_splits: int = 5):
    ground_truth = np.asarray(ground_truth).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(score)

    fold_size = n // (n_splits + 1)
    if fold_size < 2:
        finite = np.isfinite(score)
        if not finite.any():
            return 0.0, 0.0
        f1, thr = get_f1_best(ground_truth[finite], score[finite])
        return f1, thr

    thresholds = []
    for k in range(1, n_splits + 1):
        val_start = k * fold_size
        val_end = min((k + 1) * fold_size, n)

        val_gt = ground_truth[val_start:val_end]
        val_score = score[val_start:val_end]

        finite = np.isfinite(val_score)
        if not finite.any():
            continue
        val_gt = val_gt[finite]
        val_score = val_score[finite]

        if val_gt.sum() == 0:
            continue

        _, fold_thr = get_f1_best(val_gt, val_score)
        thresholds.append(fold_thr)

    finite_all = np.isfinite(score)
    if not thresholds:
        if not finite_all.any():
            return 0.0, 0.0
        f1, thr = get_f1_best(ground_truth[finite_all], score[finite_all])
        return f1, thr

    cv_threshold = float(np.median(thresholds))
    f1_cv = _f1_from_threshold(
        ground_truth[finite_all], score[finite_all], cv_threshold
    )
    return f1_cv, cv_threshold

def get_threshold_evt(score, p: float = 0.01, init_percentile: float = 90.0):
    score = np.asarray(score, dtype=float)
    score = score[np.isfinite(score)]

    if len(score) == 0:
        return 0.0

    u = float(np.percentile(score, init_percentile))
    exceedances = score[score > u] - u

    if len(exceedances) < 10:
        return float(np.percentile(score, 100.0 * (1.0 - p)))

    try:
        c, _loc, scale = genpareto.fit(exceedances, floc=0)
        tail_prob = 1.0 - init_percentile / 100.0
        target_survival = p / tail_prob

        if target_survival >= 1.0:
            return float(u)

        x = genpareto.ppf(1.0 - target_survival, c, loc=0, scale=scale)
        return float(u + x)
    except Exception:
        return float(np.percentile(score, 100.0 * (1.0 - p)))

def get_pointwise_f1_pa(y_true, y_pred):
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
