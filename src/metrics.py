import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.stats import genpareto


def get_auc_pr_pa(y_true, y_score):
    """Point-adjusted AUC-PR — threshold-free analogue of ``pa_f1``.

    Definition (consistent with the point-wise PA precision/recall used
    elsewhere in the benchmark):

    For each ground-truth anomaly span, replace the score of every point
    inside the span with the **maximum** score inside that span. Then
    compute regular AUC-PR on the resulting (true, inflated_score) pairs.

    Why this construction is exactly the PA AUC-PR. As the threshold τ
    sweeps from high to low:

      * A span "wakes up" (becomes entirely predicted under PA) exactly
        when τ crosses the span's max-score from above. After the wake-up
        all L points of the span flip simultaneously to TP_pa — adding L
        to the TP_pa count.
      * A non-anomaly point becomes a false positive when τ crosses its
        own score.

    Inflating each span's scores to the span max is the simplest way to
    make ``average_precision_score`` reproduce that monotone behaviour:
    every point in a span shares the same effective score, so they all
    cross the threshold together. The resulting AUC-PR equals the integral
    of PA-precision-PA-recall over all thresholds. This matches the
    ``precision_pa`` / ``recall_pa`` numbers we report alongside
    pa_f1_cv / pa_f1_evt — both use point counts weighted by span length
    rather than the "compress each span to one event" alternative.

    Returns NaN for anomaly-free series (consistent with how plain
    ``get_auc_pr`` treats them — exclude from the dataset mean rather than
    fold in a meaningless value). NaN positions in ``y_score`` are dropped
    before the inflation step.
    """
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
    """Get AUC-PR score.

    Args:
        y_true: array-like of true values
        y_score: array-like of predicted values

    Returns:
        AUC-PR score, or NaN for anomaly-free series (consistent with f1_best
        special-case: those series are excluded from AUC-PR averaging rather
        than pulling the mean down with a semantically meaningless value).

    NaN-score handling: positions where the model produced no score (warmup,
    uncovered tail) are dropped from both ``y_true`` and ``y_score`` before
    AUC-PR is computed. This matches CV/EVT semantics: unscored positions are
    not silently counted as "low-score negatives".
    """
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

def get_f1_best(ground_truth, score):
    """Best-F1 oracle scan: pick the threshold that maximises F1 on this
    series (cheats by looking at labels — used as an upper-bound reference
    against the honest CV / EVT thresholds).

    Vectorised O(N log N) implementation (one sort + one cumsum). Replaces
    the previous Python-loop version which was O(N · #unique_scores) ≈ O(N²)
    for continuous scores and made ``_calculate_single_metrics`` minutes-slow
    on long series (AIOPS / WSD have up to ~300k points). Returns the same
    maximum F1 — both enumerate the same set of distinct prediction sets
    ``(score > θ)`` — the only difference is that the threshold returned may
    sit at the midpoint between adjacent distinct scores rather than at one
    of the score values itself; both reproduce identical TP/FP/FN counts.
    """
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

# def get_f1_best(y_true, y_score):
#     """Get F1 score for best threshold.
#
#     Args:
#         y_true: array-like of true values
#         y_score: array-like of predicted values
#
#     Returns:
#         F1 score for best threshold and best threshold
#     """
#     y_true = np.asarray(y_true)
#     y_score = np.asarray(y_score)
#     if sum(y_true) == 0:
#         return 1.0, 100.0
#     y_true_compressed, y_score_compressed = compress_point_adjusted(y_true, y_score)
#     precision, recall, thresholds = precision_recall_curve(y_true_compressed, y_score_compressed)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)
#
#     thresholds = np.concatenate(
#         [thresholds, [max(y_score) + 1e-6]]
#     )
#
#     highest_threshold = thresholds[np.argmax(f1)]
#     negative_class_scores = y_score[y_true == 0]
#     lowest_threshold = negative_class_scores[negative_class_scores < highest_threshold].max() + 1e-6
#
#     return (
#         np.max(f1),
#         lowest_threshold,
#     )


def precision_recall_f1_from_threshold(
    ground_truth: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Return (precision, recall, F1) for ``score > threshold`` predictions.

    Mirrors the precision/recall/F1 accounting used in oracle threshold search
    so every metric reported in the benchmark stays consistent across the
    different threshold strategies (oracle / CV / EVT).

    NaN positions in ``score`` are dropped before binarisation. This matches
    the convention used elsewhere: "no score" ≠ "scored as no-anomaly".
    """
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
    """Helper: compute F1 given pre-computed scores and a threshold."""
    _p, _r, f1 = precision_recall_f1_from_threshold(ground_truth, score, threshold)
    return f1


def pointwise_f1_pa_from_threshold(
    ground_truth: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> float:
    """Point-adjusted F1 computed from a ``score > threshold`` binarisation.

    Wraps ``get_pointwise_f1_pa`` so PA F1 can be reported against any
    threshold (CV / EVT / oracle) rather than only against the detector's
    hard-coded fixed threshold.

    NaN positions in ``score`` are treated as "not scored" — they are
    dropped together with their matching ground-truth labels before the
    PA accounting runs.
    """
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
    """Return point-adjusted ``(precision_pa, recall_pa, f1_pa)``.

    The PA protocol (Xu et al. 2018) inflates predictions: for every
    ground-truth anomaly span, if *any* predicted point inside the span is
    flagged as positive, the entire span is treated as predicted-positive.
    Precision and recall are then computed on those inflated predictions.

    Why this exists alongside the plain F1 helper: the codebase used to
    surface only ``pa_f1`` (computed inside ``get_pointwise_f1_pa``) while
    the underlying ``precision_pa`` and ``recall_pa`` were thrown away.
    That made the reported PA column asymmetric with the non-PA columns
    (``precision_cv`` / ``recall_cv`` / ``f1_cv``) and hid which side of
    the trade-off was hurting any given series — high pa_f1 with low
    precision_pa means "predicts a lot of garbage but accidentally hits
    every span", which is a very different failure from "few predictions,
    most concentrated in spans".

    NaN positions in ``score`` are dropped before binarisation, matching
    the convention used by the non-PA helpers.
    """
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
    """Oracle best PA-F1: find the threshold maximising point-adjusted F1.

    Like get_f1_best but optimises PA-F1 rather than point-wise F1.

    Key insight: for PA metrics the optimal threshold is *much higher* than for
    point-wise F1.  PA gives full span credit as soon as *any* point inside the
    span is predicted positive, so it strongly prefers high-precision / sparse
    predictions — a regime where point-wise optimal thresholds systematically
    over-predict (high recall, low precision ⇒ low pa_f1).

    O(N log N) incremental algorithm: sort by score descending, track the first
    hit in each anomaly span, accumulate tp_pa / fp / fn_pa incrementally.
    """
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
    """Walk-forward CV threshold selection optimised for PA-F1.

    Mirrors get_threshold_cv but selects each fold's threshold via
    get_f1_pa_best (optimises PA-F1) instead of get_f1_best (optimises
    point-wise F1).

    Why a separate function matters: PA-F1 prefers high-precision / sparse
    predictions (a single correctly flagged point inside a span gives full
    span credit), so its optimal threshold is systematically *higher* than
    the point-wise optimal threshold.  Using the point-wise CV threshold for
    PA metrics (the original bug) under-thresholds predictions, inflating FP
    and tanking precision_cv_pa — exactly the pattern visible in NAB where
    recall_cv_pa ≈ 1.0 but precision_cv_pa ≈ 0.18 ⇒ pa_f1_cv ≈ 0.30, while
    pa_f1_evt (which uses a strict p ≤ 0.01 tail mass) reaches ≈ 0.96.

    Returns
    pa_f1_cv_internal : float (PA-F1 evaluated on the full scored region)
    threshold_cv_pa : float (median of per-fold PA-optimal thresholds)
    """
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
    """Walk-forward CV threshold selection (supervised, no future leakage).

    Splits the series into (n_splits + 1) equal temporal chunks.
    For each fold k ∈ [1, n_splits], treats chunk k as a labeled validation
    window and finds the best F1 threshold on it independently.
    The final threshold is the **median** across folds that had any positives.

    Why median, not mean (previous version used ``np.mean``):
    on non-stationary series (NAB, AIOPS) the per-fold score scale can vary by
    an order of magnitude between folds. A single outlier fold then pulls the
    averaged threshold far away from the optimum on the rest of the series,
    so ``f1_cv`` drops well below what each individual fold's threshold would
    have produced. The median is robust to a single bad fold while staying
    identical to the mean on stationary scores.

    NaN handling: positions where ``score`` is NaN (e.g. warmup / uncovered
    tail in foundation-model detectors) are dropped from CV fitting and
    excluded from final F1 accounting — they don't pretend to be "predicted
    no anomaly".

    Returns
    f1_cv: F1 on the scored positions of the series using the CV-selected threshold.
    threshold_cv: the median of per-fold oracle thresholds.
    """
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
    """Extreme Value Theory threshold via Peaks-over-Threshold (POT).

    Fits a Generalized Pareto Distribution to score exceedances above the
    ``init_percentile``-th percentile and returns the quantile at tail
    probability ``p`` (i.e. we expect ~p fraction of points to be anomalies).

    No labels required — purely unsupervised.

    Returns threshold : float
    """
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
