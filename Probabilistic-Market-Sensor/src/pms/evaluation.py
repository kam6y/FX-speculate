"""Evaluation metrics for direction and range predictions.

This module implements evaluation functions for assessing model performance
according to the requirements in PMS-REQ-USDJPY-v5.0 sections 14.2 and 14.3.

Direction Evaluation (14.2):
- Balanced Accuracy
- Macro-F1
- Per-class Recall (especially Up/Down)
- Confusion Matrix

Range Evaluation (14.3):
- Pinball Loss (q=0.10, 0.50, 0.90)
- Coverage (P(y <= q10), P(y <= q90))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .labels import HORIZONS

# Constants
DIRECTION_CLASSES: List[str] = ["up", "down", "neutral", "choppy"]
DEFAULT_PINBALL_QUANTILES = [0.10, 0.50, 0.90]

# Pass criteria from PMS-REQ-USDJPY-v5.0 section 14.2
CRITERIA_PRIMARY_BALANCED_ACCURACY = 0.40
CRITERIA_PRIMARY_MACRO_F1 = 0.35
CRITERIA_PRIMARY_RECALL = 0.30

CRITERIA_SECONDARY_BALANCED_ACCURACY = 0.38
CRITERIA_SECONDARY_MACRO_F1 = 0.33
CRITERIA_SECONDARY_RECALL = 0.28

# Coverage tolerance from section 14.3
COVERAGE_Q10_TARGET = 0.10
COVERAGE_Q90_TARGET = 0.90
COVERAGE_TOLERANCE = 0.03


class EvaluationError(ValueError):
    """Exception raised for evaluation errors."""

    pass


# =============================================================================
# Direction Evaluation Metrics
# =============================================================================


@dataclass(frozen=True)
class DirectionMetrics:
    """Direction prediction metrics for a single horizon.

    Attributes:
        horizon_min: Horizon in minutes
        balanced_accuracy: Balanced accuracy (average of per-class recalls)
        macro_f1: Macro-averaged F1 score
        up_recall: Recall for 'up' class
        down_recall: Recall for 'down' class
        neutral_recall: Recall for 'neutral' class
        choppy_recall: Recall for 'choppy' class
        up_precision: Precision for 'up' class
        down_precision: Precision for 'down' class
        neutral_precision: Precision for 'neutral' class
        choppy_precision: Precision for 'choppy' class
        confusion_matrix: 4x4 confusion matrix (dict of dicts)
        total_samples: Total number of samples
    """

    horizon_min: int
    balanced_accuracy: float
    macro_f1: float
    up_recall: float
    down_recall: float
    neutral_recall: float
    choppy_recall: float
    up_precision: float
    down_precision: float
    neutral_precision: float
    choppy_precision: float
    confusion_matrix: Dict[str, Dict[str, int]]
    total_samples: int


@dataclass(frozen=True)
class DirectionEvaluationResult:
    """Complete direction evaluation result.

    Attributes:
        metrics_by_horizon: Dict mapping horizon to DirectionMetrics
        baseline_a_metrics: Baseline-A metrics by horizon (optional)
        baseline_b_metrics: Baseline-B metrics by horizon (optional)
    """

    metrics_by_horizon: Dict[int, DirectionMetrics]
    baseline_a_metrics: Dict[int, DirectionMetrics] = field(default_factory=dict)
    baseline_b_metrics: Dict[int, DirectionMetrics] = field(default_factory=dict)


def compute_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: List[str] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix for multi-class classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names (default: DIRECTION_CLASSES)

    Returns:
        Nested dict: confusion_matrix[true_label][pred_label] = count

    Raises:
        EvaluationError: If y_true and y_pred have different lengths or unknown labels
    """
    if classes is None:
        classes = DIRECTION_CLASSES

    if len(y_true) != len(y_pred):
        raise EvaluationError("y_true and y_pred must have same length")

    matrix = {c: {p: 0 for p in classes} for c in classes}

    for true, pred in zip(y_true, y_pred):
        if true not in classes:
            raise EvaluationError(f"unknown true label: {true}")
        if pred not in classes:
            raise EvaluationError(f"unknown predicted label: {pred}")
        matrix[true][pred] += 1

    return matrix


def compute_precision_recall_f1(
    confusion_matrix: Dict[str, Dict[str, int]],
    class_name: str,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for a single class.

    Args:
        confusion_matrix: Confusion matrix (true -> pred -> count)
        class_name: Target class name

    Returns:
        Tuple of (precision, recall, f1)
    """
    # True positives
    tp = confusion_matrix[class_name][class_name]

    # False positives (predicted as class but actually other)
    fp = sum(
        confusion_matrix[other][class_name]
        for other in confusion_matrix
        if other != class_name
    )

    # False negatives (actually class but predicted as other)
    fn = sum(
        confusion_matrix[class_name][other]
        for other in confusion_matrix[class_name]
        if other != class_name
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def compute_direction_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    horizon: int,
) -> DirectionMetrics:
    """
    Compute all direction metrics for a single horizon.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        horizon: Horizon in minutes

    Returns:
        DirectionMetrics with all computed metrics
    """
    cm = compute_confusion_matrix(y_true, y_pred)

    # Compute per-class metrics
    precisions = {}
    recalls = {}
    f1_scores = []

    for class_name in DIRECTION_CLASSES:
        precision, recall, f1 = compute_precision_recall_f1(cm, class_name)
        precisions[class_name] = precision
        recalls[class_name] = recall
        f1_scores.append(f1)

    # Balanced accuracy = average of recalls
    balanced_accuracy = sum(recalls.values()) / len(recalls)

    # Macro F1 = average of F1 scores
    macro_f1 = sum(f1_scores) / len(f1_scores)

    return DirectionMetrics(
        horizon_min=horizon,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        up_recall=recalls["up"],
        down_recall=recalls["down"],
        neutral_recall=recalls["neutral"],
        choppy_recall=recalls["choppy"],
        up_precision=precisions["up"],
        down_precision=precisions["down"],
        neutral_precision=precisions["neutral"],
        choppy_precision=precisions["choppy"],
        confusion_matrix=cm,
        total_samples=len(y_true),
    )


def evaluate_direction_predictions(
    y_true_by_horizon: Dict[int, Sequence[str]],
    y_pred_by_horizon: Dict[int, Sequence[str]],
    baseline_a_pred_by_horizon: Dict[int, Sequence[str]] = None,
    baseline_b_pred_by_horizon: Dict[int, Sequence[str]] = None,
) -> DirectionEvaluationResult:
    """
    Evaluate direction predictions for all horizons.

    Args:
        y_true_by_horizon: Ground truth labels by horizon
        y_pred_by_horizon: Model predictions by horizon
        baseline_a_pred_by_horizon: Baseline-A predictions (optional)
        baseline_b_pred_by_horizon: Baseline-B predictions (optional)

    Returns:
        DirectionEvaluationResult with all metrics
    """
    metrics = {}
    baseline_a_metrics = {}
    baseline_b_metrics = {}

    for horizon in y_true_by_horizon:
        y_true = y_true_by_horizon[horizon]

        if horizon not in y_pred_by_horizon:
            raise EvaluationError(f"missing predictions for horizon {horizon}")
        y_pred = y_pred_by_horizon[horizon]
        if not y_pred:
            raise EvaluationError(f"predictions for horizon {horizon} must not be empty")
        metrics[horizon] = compute_direction_metrics(y_true, y_pred, horizon)

        if baseline_a_pred_by_horizon is not None:
            if horizon not in baseline_a_pred_by_horizon:
                raise EvaluationError(
                    f"baseline_a predictions missing for horizon {horizon}"
                )
            baseline_a_pred = baseline_a_pred_by_horizon[horizon]
            if not baseline_a_pred:
                raise EvaluationError(
                    f"baseline_a predictions for horizon {horizon} must not be empty"
                )
            baseline_a_metrics[horizon] = compute_direction_metrics(
                y_true, baseline_a_pred, horizon
            )

        if baseline_b_pred_by_horizon is not None:
            if horizon not in baseline_b_pred_by_horizon:
                raise EvaluationError(
                    f"baseline_b predictions missing for horizon {horizon}"
                )
            baseline_b_pred = baseline_b_pred_by_horizon[horizon]
            if not baseline_b_pred:
                raise EvaluationError(
                    f"baseline_b predictions for horizon {horizon} must not be empty"
                )
            baseline_b_metrics[horizon] = compute_direction_metrics(
                y_true, baseline_b_pred, horizon
            )

    return DirectionEvaluationResult(
        metrics_by_horizon=metrics,
        baseline_a_metrics=baseline_a_metrics,
        baseline_b_metrics=baseline_b_metrics,
    )


# =============================================================================
# Range Evaluation Metrics
# =============================================================================


@dataclass(frozen=True)
class RangeMetrics:
    """Range prediction metrics for a single horizon.

    Attributes:
        horizon_min: Horizon in minutes
        pinball_loss_q10: Pinball loss for q=0.10
        pinball_loss_q50: Pinball loss for q=0.50 (equivalent to MAE)
        pinball_loss_q90: Pinball loss for q=0.90
        pinball_loss_avg: Average pinball loss across quantiles
        coverage_q10: P(y <= q10) - should be ~0.10
        coverage_q90: P(y <= q90) - should be ~0.90
        total_samples: Total number of samples
    """

    horizon_min: int
    pinball_loss_q10: float
    pinball_loss_q50: float
    pinball_loss_q90: float
    pinball_loss_avg: float
    coverage_q10: float
    coverage_q90: float
    total_samples: int


@dataclass(frozen=True)
class RangeEvaluationResult:
    """Complete range evaluation result.

    Attributes:
        metrics_by_horizon: Dict mapping horizon to RangeMetrics
        baseline_metrics: Baseline metrics by horizon
    """

    metrics_by_horizon: Dict[int, RangeMetrics]
    baseline_metrics: Dict[int, RangeMetrics] = field(default_factory=dict)


def pinball_loss(
    y_true: float,
    y_pred: float,
    quantile: float,
) -> float:
    """
    Compute pinball loss (quantile loss) for a single prediction.

    Pinball loss formula:
    - If y_true >= y_pred: quantile * (y_true - y_pred)
    - If y_true < y_pred: (1 - quantile) * (y_pred - y_true)

    Args:
        y_true: Actual value
        y_pred: Predicted quantile value
        quantile: Target quantile (0.0 to 1.0)

    Returns:
        Pinball loss value (non-negative)
    """
    error = y_true - y_pred
    if error >= 0:
        return quantile * error
    else:
        return (1 - quantile) * (-error)


def compute_pinball_loss_batch(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    quantile: float,
) -> float:
    """
    Compute average pinball loss for a batch.

    Args:
        y_true: Actual values
        y_pred: Predicted quantile values
        quantile: Target quantile

    Returns:
        Average pinball loss

    Raises:
        EvaluationError: If sequences have different lengths or are empty
    """
    if len(y_true) != len(y_pred):
        raise EvaluationError("y_true and y_pred must have same length")
    if not y_true:
        raise EvaluationError("sequences must not be empty")

    total_loss = sum(pinball_loss(t, p, quantile) for t, p in zip(y_true, y_pred))
    return total_loss / len(y_true)


def compute_coverage(
    y_true: Sequence[float],
    quantile_pred: Sequence[float],
) -> float:
    """
    Compute coverage: P(y_true <= quantile_pred).

    Args:
        y_true: Actual values
        quantile_pred: Predicted quantile values

    Returns:
        Coverage proportion (0.0 to 1.0)

    Raises:
        EvaluationError: If sequences have different lengths or are empty
    """
    if len(y_true) != len(quantile_pred):
        raise EvaluationError("y_true and quantile_pred must have same length")
    if not y_true:
        raise EvaluationError("sequences must not be empty")

    covered = sum(1 for t, p in zip(y_true, quantile_pred) if t <= p)
    return covered / len(y_true)


def compute_range_metrics(
    y_true: Sequence[float],
    q10_pred: Sequence[float],
    q50_pred: Sequence[float],
    q90_pred: Sequence[float],
    horizon: int,
) -> RangeMetrics:
    """
    Compute all range metrics for a single horizon.

    Args:
        y_true: Actual close prices at t+H
        q10_pred: Predicted q10 values
        q50_pred: Predicted q50 (median) values
        q90_pred: Predicted q90 values
        horizon: Horizon in minutes

    Returns:
        RangeMetrics with all computed metrics
    """
    pl_q10 = compute_pinball_loss_batch(y_true, q10_pred, 0.10)
    pl_q50 = compute_pinball_loss_batch(y_true, q50_pred, 0.50)
    pl_q90 = compute_pinball_loss_batch(y_true, q90_pred, 0.90)
    pl_avg = (pl_q10 + pl_q50 + pl_q90) / 3.0

    cov_q10 = compute_coverage(y_true, q10_pred)
    cov_q90 = compute_coverage(y_true, q90_pred)

    return RangeMetrics(
        horizon_min=horizon,
        pinball_loss_q10=pl_q10,
        pinball_loss_q50=pl_q50,
        pinball_loss_q90=pl_q90,
        pinball_loss_avg=pl_avg,
        coverage_q10=cov_q10,
        coverage_q90=cov_q90,
        total_samples=len(y_true),
    )


def evaluate_range_predictions(
    y_true_by_horizon: Dict[int, Sequence[float]],
    q10_by_horizon: Dict[int, Sequence[float]],
    q50_by_horizon: Dict[int, Sequence[float]],
    q90_by_horizon: Dict[int, Sequence[float]],
    baseline_q10_by_horizon: Dict[int, Sequence[float]] = None,
    baseline_q50_by_horizon: Dict[int, Sequence[float]] = None,
    baseline_q90_by_horizon: Dict[int, Sequence[float]] = None,
) -> RangeEvaluationResult:
    """
    Evaluate range predictions for all horizons.

    Args:
        y_true_by_horizon: Actual values by horizon
        q10_by_horizon: Model q10 predictions by horizon
        q50_by_horizon: Model q50 predictions by horizon
        q90_by_horizon: Model q90 predictions by horizon
        baseline_q10_by_horizon: Baseline q10 predictions (optional)
        baseline_q50_by_horizon: Baseline q50 predictions (optional)
        baseline_q90_by_horizon: Baseline q90 predictions (optional)

    Returns:
        RangeEvaluationResult with all metrics
    """
    metrics = {}
    baseline_metrics = {}
    baseline_provided = any(
        b is not None
        for b in (baseline_q10_by_horizon, baseline_q50_by_horizon, baseline_q90_by_horizon)
    )
    if baseline_provided and (
        baseline_q10_by_horizon is None
        or baseline_q50_by_horizon is None
        or baseline_q90_by_horizon is None
    ):
        raise EvaluationError("baseline q10/q50/q90 must be provided together")

    for horizon in y_true_by_horizon:
        y_true = y_true_by_horizon[horizon]

        if horizon not in q10_by_horizon:
            raise EvaluationError(f"missing q10 predictions for horizon {horizon}")
        if horizon not in q50_by_horizon:
            raise EvaluationError(f"missing q50 predictions for horizon {horizon}")
        if horizon not in q90_by_horizon:
            raise EvaluationError(f"missing q90 predictions for horizon {horizon}")

        q10_pred = q10_by_horizon[horizon]
        q50_pred = q50_by_horizon[horizon]
        q90_pred = q90_by_horizon[horizon]

        if not q10_pred:
            raise EvaluationError(f"q10 predictions for horizon {horizon} must not be empty")
        if not q50_pred:
            raise EvaluationError(f"q50 predictions for horizon {horizon} must not be empty")
        if not q90_pred:
            raise EvaluationError(f"q90 predictions for horizon {horizon} must not be empty")

        metrics[horizon] = compute_range_metrics(
            y_true,
            q10_pred,
            q50_pred,
            q90_pred,
            horizon,
        )

        if baseline_provided:
            if horizon not in baseline_q10_by_horizon:
                raise EvaluationError(
                    f"baseline q10 predictions missing for horizon {horizon}"
                )
            if horizon not in baseline_q50_by_horizon:
                raise EvaluationError(
                    f"baseline q50 predictions missing for horizon {horizon}"
                )
            if horizon not in baseline_q90_by_horizon:
                raise EvaluationError(
                    f"baseline q90 predictions missing for horizon {horizon}"
                )

            baseline_q10 = baseline_q10_by_horizon[horizon]
            baseline_q50 = baseline_q50_by_horizon[horizon]
            baseline_q90 = baseline_q90_by_horizon[horizon]

            if not baseline_q10:
                raise EvaluationError(
                    f"baseline q10 predictions for horizon {horizon} must not be empty"
                )
            if not baseline_q50:
                raise EvaluationError(
                    f"baseline q50 predictions for horizon {horizon} must not be empty"
                )
            if not baseline_q90:
                raise EvaluationError(
                    f"baseline q90 predictions for horizon {horizon} must not be empty"
                )

            baseline_metrics[horizon] = compute_range_metrics(
                y_true,
                baseline_q10,
                baseline_q50,
                baseline_q90,
                horizon,
            )

    return RangeEvaluationResult(
        metrics_by_horizon=metrics,
        baseline_metrics=baseline_metrics,
    )


# =============================================================================
# Pass Criteria Checking
# =============================================================================


def check_direction_criteria(
    metrics: DirectionMetrics,
    is_primary: bool = False,
    baseline_a_metrics: DirectionMetrics | None = None,
    baseline_b_metrics: DirectionMetrics | None = None,
) -> Dict[str, bool]:
    """
    Check if direction metrics meet pass criteria.

    Primary (30min): Balanced Acc >= 0.40, Macro-F1 >= 0.35, Up/Down Recall >= 0.30
    Secondary (10/60min): Balanced Acc >= 0.38, Macro-F1 >= 0.33, Up/Down Recall >= 0.28
    Baseline comparison: exceed the higher baseline metric when provided.

    Args:
        metrics: Direction metrics to check
        is_primary: True for 30-minute (primary criterion)
        baseline_a_metrics: Baseline-A metrics (optional)
        baseline_b_metrics: Baseline-B metrics (optional)

    Returns:
        Dict of criterion name to pass/fail status
    """
    if is_primary:
        ba_threshold = CRITERIA_PRIMARY_BALANCED_ACCURACY
        f1_threshold = CRITERIA_PRIMARY_MACRO_F1
        recall_threshold = CRITERIA_PRIMARY_RECALL
    else:
        ba_threshold = CRITERIA_SECONDARY_BALANCED_ACCURACY
        f1_threshold = CRITERIA_SECONDARY_MACRO_F1
        recall_threshold = CRITERIA_SECONDARY_RECALL

    criteria = {
        f"balanced_accuracy_{metrics.horizon_min}m": (
            metrics.balanced_accuracy >= ba_threshold
        ),
        f"macro_f1_{metrics.horizon_min}m": metrics.macro_f1 >= f1_threshold,
        f"up_recall_{metrics.horizon_min}m": metrics.up_recall >= recall_threshold,
        f"down_recall_{metrics.horizon_min}m": metrics.down_recall >= recall_threshold,
    }

    def _baseline_max(a: float | None, b: float | None) -> float | None:
        values = [v for v in (a, b) if v is not None]
        return max(values) if values else None

    if baseline_a_metrics is not None or baseline_b_metrics is not None:
        baseline_balanced_accuracy = _baseline_max(
            baseline_a_metrics.balanced_accuracy if baseline_a_metrics else None,
            baseline_b_metrics.balanced_accuracy if baseline_b_metrics else None,
        )
        baseline_macro_f1 = _baseline_max(
            baseline_a_metrics.macro_f1 if baseline_a_metrics else None,
            baseline_b_metrics.macro_f1 if baseline_b_metrics else None,
        )
        baseline_up_recall = _baseline_max(
            baseline_a_metrics.up_recall if baseline_a_metrics else None,
            baseline_b_metrics.up_recall if baseline_b_metrics else None,
        )
        baseline_down_recall = _baseline_max(
            baseline_a_metrics.down_recall if baseline_a_metrics else None,
            baseline_b_metrics.down_recall if baseline_b_metrics else None,
        )

        if baseline_balanced_accuracy is not None:
            criteria[f"baseline_balanced_accuracy_{metrics.horizon_min}m"] = (
                metrics.balanced_accuracy > baseline_balanced_accuracy
            )
        if baseline_macro_f1 is not None:
            criteria[f"baseline_macro_f1_{metrics.horizon_min}m"] = (
                metrics.macro_f1 > baseline_macro_f1
            )
        if baseline_up_recall is not None:
            criteria[f"baseline_up_recall_{metrics.horizon_min}m"] = (
                metrics.up_recall > baseline_up_recall
            )
        if baseline_down_recall is not None:
            criteria[f"baseline_down_recall_{metrics.horizon_min}m"] = (
                metrics.down_recall > baseline_down_recall
            )

    return criteria


def check_range_criteria(
    metrics: RangeMetrics,
    baseline_metrics: RangeMetrics | None = None,
) -> Dict[str, bool]:
    """
    Check if range metrics meet pass criteria.

    - Pinball loss average must beat baseline (if provided)
    - Coverage: P(y <= q10) = 0.10 +/- 0.03
    - Coverage: P(y <= q90) = 0.90 +/- 0.03

    Args:
        metrics: Model range metrics
        baseline_metrics: Baseline range metrics (optional)

    Returns:
        Dict of criterion name to pass/fail status
    """
    horizon = metrics.horizon_min

    criteria = {
        f"coverage_q10_{horizon}m": (
            COVERAGE_Q10_TARGET - COVERAGE_TOLERANCE
            <= metrics.coverage_q10
            <= COVERAGE_Q10_TARGET + COVERAGE_TOLERANCE
        ),
        f"coverage_q90_{horizon}m": (
            COVERAGE_Q90_TARGET - COVERAGE_TOLERANCE
            <= metrics.coverage_q90
            <= COVERAGE_Q90_TARGET + COVERAGE_TOLERANCE
        ),
    }

    if baseline_metrics is not None:
        criteria[f"pinball_beats_baseline_{horizon}m"] = (
            metrics.pinball_loss_avg < baseline_metrics.pinball_loss_avg
        )

    return criteria


# =============================================================================
# Report Formatting
# =============================================================================


def format_direction_report(
    results: DirectionEvaluationResult,
    horizons: List[int] = None,
) -> str:
    """
    Format direction evaluation results as human-readable report.

    Args:
        results: Direction evaluation results
        horizons: Horizons to include (default: all available)

    Returns:
        Multi-line string report
    """
    if horizons is None:
        horizons = sorted(
            set(results.baseline_a_metrics.keys())
            | set(results.baseline_b_metrics.keys())
            | set(results.metrics_by_horizon.keys())
        )

    lines = [
        "=" * 70,
        "Direction Evaluation Report",
        "=" * 70,
        "",
    ]

    for horizon in horizons:
        is_primary = horizon == 30
        label = "PRIMARY" if is_primary else "Secondary"

        lines.append(f"--- Horizon {horizon}m ({label}) ---")

        # Baseline-A
        if horizon in results.baseline_a_metrics:
            ba = results.baseline_a_metrics[horizon]
            lines.extend(
                [
                    "  Baseline-A (always neutral):",
                    f"    Balanced Accuracy: {ba.balanced_accuracy:.4f}",
                    f"    Macro-F1: {ba.macro_f1:.4f}",
                    f"    Up Recall: {ba.up_recall:.4f}",
                    f"    Down Recall: {ba.down_recall:.4f}",
                ]
            )

        # Baseline-B
        if horizon in results.baseline_b_metrics:
            bb = results.baseline_b_metrics[horizon]
            lines.extend(
                [
                    "  Baseline-B (1min return sign):",
                    f"    Balanced Accuracy: {bb.balanced_accuracy:.4f}",
                    f"    Macro-F1: {bb.macro_f1:.4f}",
                    f"    Up Recall: {bb.up_recall:.4f}",
                    f"    Down Recall: {bb.down_recall:.4f}",
                ]
            )

        # Best baseline comparison
        if horizon in results.baseline_a_metrics and horizon in results.baseline_b_metrics:
            ba = results.baseline_a_metrics[horizon]
            bb = results.baseline_b_metrics[horizon]
            best = "B" if bb.balanced_accuracy > ba.balanced_accuracy else "A"
            best_acc = max(ba.balanced_accuracy, bb.balanced_accuracy)
            lines.append(f"  -> Best Baseline: {best} (Balanced Acc: {best_acc:.4f})")

        # Model metrics (if present)
        if horizon in results.metrics_by_horizon:
            m = results.metrics_by_horizon[horizon]
            lines.extend(
                [
                    "  Model:",
                    f"    Balanced Accuracy: {m.balanced_accuracy:.4f}",
                    f"    Macro-F1: {m.macro_f1:.4f}",
                    f"    Up Recall: {m.up_recall:.4f}",
                    f"    Down Recall: {m.down_recall:.4f}",
                ]
            )

        lines.append("")

    return "\n".join(lines)


def format_range_report(
    results: RangeEvaluationResult,
    horizons: List[int] = None,
) -> str:
    """
    Format range evaluation results as human-readable report.

    Args:
        results: Range evaluation results
        horizons: Horizons to include (default: all available)

    Returns:
        Multi-line string report
    """
    if horizons is None:
        horizons = sorted(
            set(results.baseline_metrics.keys()) | set(results.metrics_by_horizon.keys())
        )

    lines = [
        "=" * 70,
        "Range Evaluation Report",
        "=" * 70,
        "",
    ]

    for horizon in horizons:
        lines.append(f"--- Horizon {horizon}m ---")

        # Baseline
        if horizon in results.baseline_metrics:
            b = results.baseline_metrics[horizon]
            lines.extend(
                [
                    "  Baseline (fixed quantile):",
                    f"    Pinball Loss (avg): {b.pinball_loss_avg:.6f}",
                    f"    Pinball Loss q10: {b.pinball_loss_q10:.6f}",
                    f"    Pinball Loss q50: {b.pinball_loss_q50:.6f}",
                    f"    Pinball Loss q90: {b.pinball_loss_q90:.6f}",
                    f"    Coverage q10: {b.coverage_q10:.4f} (target: 0.10 +/- 0.03)",
                    f"    Coverage q90: {b.coverage_q90:.4f} (target: 0.90 +/- 0.03)",
                ]
            )

        # Model metrics (if present)
        if horizon in results.metrics_by_horizon:
            m = results.metrics_by_horizon[horizon]
            lines.extend(
                [
                    "  Model:",
                    f"    Pinball Loss (avg): {m.pinball_loss_avg:.6f}",
                    f"    Coverage q10: {m.coverage_q10:.4f}",
                    f"    Coverage q90: {m.coverage_q90:.4f}",
                ]
            )

        lines.append("")

    return "\n".join(lines)


def format_confusion_matrix(
    cm: Dict[str, Dict[str, int]],
    classes: List[str] = None,
) -> str:
    """
    Format confusion matrix as a readable string table.

    Args:
        cm: Confusion matrix (true -> pred -> count)
        classes: Class names in order (default: DIRECTION_CLASSES)

    Returns:
        Formatted confusion matrix string
    """
    if classes is None:
        classes = DIRECTION_CLASSES

    # Header
    col_width = 10
    lines = ["Confusion Matrix (rows=true, cols=predicted):", ""]

    header = " " * col_width + "".join(c[:col_width].rjust(col_width) for c in classes)
    lines.append(header)

    # Rows
    for true_cls in classes:
        row = true_cls[:col_width].ljust(col_width)
        for pred_cls in classes:
            count = cm[true_cls][pred_cls]
            row += str(count).rjust(col_width)
        lines.append(row)

    return "\n".join(lines)


