"""Tests for evaluation metrics module."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.evaluation import (  # noqa: E402
    DIRECTION_CLASSES,
    CRITERIA_PRIMARY_BALANCED_ACCURACY,
    CRITERIA_PRIMARY_MACRO_F1,
    CRITERIA_PRIMARY_RECALL,
    CRITERIA_SECONDARY_BALANCED_ACCURACY,
    CRITERIA_SECONDARY_MACRO_F1,
    CRITERIA_SECONDARY_RECALL,
    COVERAGE_Q10_TARGET,
    COVERAGE_Q90_TARGET,
    COVERAGE_TOLERANCE,
    DirectionMetrics,
    RangeMetrics,
    EvaluationError,
    compute_confusion_matrix,
    compute_precision_recall_f1,
    compute_direction_metrics,
    evaluate_direction_predictions,
    pinball_loss,
    compute_pinball_loss_batch,
    compute_coverage,
    compute_range_metrics,
    evaluate_range_predictions,
    check_direction_criteria,
    check_range_criteria,
    format_direction_report,
    format_range_report,
    format_confusion_matrix,
)


class ConfusionMatrixTests(unittest.TestCase):
    """Tests for confusion matrix computation."""

    def test_perfect_predictions(self) -> None:
        """Test confusion matrix with perfect predictions."""
        y_true = ["up", "down", "neutral", "choppy"]
        y_pred = ["up", "down", "neutral", "choppy"]

        cm = compute_confusion_matrix(y_true, y_pred)

        # Diagonal should be 1, off-diagonal should be 0
        for cls in DIRECTION_CLASSES:
            self.assertEqual(cm[cls][cls], 1)
            for other in DIRECTION_CLASSES:
                if other != cls:
                    self.assertEqual(cm[cls][other], 0)

    def test_all_wrong_predictions(self) -> None:
        """Test confusion matrix with all wrong predictions."""
        y_true = ["up", "up", "up"]
        y_pred = ["down", "down", "down"]

        cm = compute_confusion_matrix(y_true, y_pred)

        self.assertEqual(cm["up"]["down"], 3)
        self.assertEqual(cm["up"]["up"], 0)

    def test_mixed_predictions(self) -> None:
        """Test confusion matrix with mixed predictions."""
        y_true = ["up", "up", "down", "down", "neutral", "choppy"]
        y_pred = ["up", "down", "down", "neutral", "neutral", "choppy"]

        cm = compute_confusion_matrix(y_true, y_pred)

        self.assertEqual(cm["up"]["up"], 1)
        self.assertEqual(cm["up"]["down"], 1)
        self.assertEqual(cm["down"]["down"], 1)
        self.assertEqual(cm["down"]["neutral"], 1)
        self.assertEqual(cm["neutral"]["neutral"], 1)
        self.assertEqual(cm["choppy"]["choppy"], 1)

    def test_length_mismatch_raises(self) -> None:
        """Test that mismatched lengths raise error."""
        with self.assertRaises(EvaluationError) as context:
            compute_confusion_matrix(["up"], ["up", "down"])
        self.assertIn("same length", str(context.exception))

    def test_empty_sequences(self) -> None:
        """Test confusion matrix with empty sequences."""
        cm = compute_confusion_matrix([], [])

        # All cells should be 0
        for true_cls in DIRECTION_CLASSES:
            for pred_cls in DIRECTION_CLASSES:
                self.assertEqual(cm[true_cls][pred_cls], 0)

    def test_unknown_label_raises(self) -> None:
        """Test that unknown labels raise error."""
        with self.assertRaises(EvaluationError):
            compute_confusion_matrix(["up", "unknown"], ["up", "up"])
        with self.assertRaises(EvaluationError):
            compute_confusion_matrix(["up", "up"], ["up", "unknown"])


class PrecisionRecallF1Tests(unittest.TestCase):
    """Tests for precision, recall, F1 computation."""

    def test_perfect_single_class(self) -> None:
        """Test metrics for perfect predictions of a single class."""
        cm = {c: {p: 0 for p in DIRECTION_CLASSES} for c in DIRECTION_CLASSES}
        cm["up"]["up"] = 10  # 10 correct up predictions

        precision, recall, f1 = compute_precision_recall_f1(cm, "up")

        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(f1, 1.0)

    def test_no_true_positives(self) -> None:
        """Test metrics when there are no true positives."""
        cm = {c: {p: 0 for p in DIRECTION_CLASSES} for c in DIRECTION_CLASSES}
        cm["up"]["down"] = 10  # All up samples predicted as down
        cm["down"]["up"] = 5  # Some down samples predicted as up

        precision, recall, f1 = compute_precision_recall_f1(cm, "up")

        # TP=0, FP=5, FN=10
        self.assertEqual(precision, 0.0)  # 0 / (0 + 5)
        self.assertEqual(recall, 0.0)  # 0 / (0 + 10)
        self.assertEqual(f1, 0.0)

    def test_typical_case(self) -> None:
        """Test metrics for typical predictions."""
        cm = {c: {p: 0 for p in DIRECTION_CLASSES} for c in DIRECTION_CLASSES}
        cm["up"]["up"] = 7  # TP
        cm["up"]["down"] = 3  # FN (up predicted as down)
        cm["down"]["up"] = 2  # FP (down predicted as up)

        precision, recall, f1 = compute_precision_recall_f1(cm, "up")

        # TP=7, FP=2, FN=3
        expected_precision = 7 / (7 + 2)  # 0.778
        expected_recall = 7 / (7 + 3)  # 0.7
        expected_f1 = 2 * expected_precision * expected_recall / (
            expected_precision + expected_recall
        )

        self.assertAlmostEqual(precision, expected_precision, places=4)
        self.assertAlmostEqual(recall, expected_recall, places=4)
        self.assertAlmostEqual(f1, expected_f1, places=4)


class DirectionMetricsTests(unittest.TestCase):
    """Tests for direction metrics computation."""

    def test_perfect_balanced_accuracy(self) -> None:
        """Test balanced accuracy = 1.0 for perfect predictions."""
        y_true = ["up"] * 10 + ["down"] * 10 + ["neutral"] * 10 + ["choppy"] * 10
        y_pred = y_true.copy()

        metrics = compute_direction_metrics(y_true, y_pred, horizon=30)

        self.assertAlmostEqual(metrics.balanced_accuracy, 1.0, places=6)
        self.assertAlmostEqual(metrics.macro_f1, 1.0, places=6)
        self.assertAlmostEqual(metrics.up_recall, 1.0, places=6)
        self.assertAlmostEqual(metrics.down_recall, 1.0, places=6)

    def test_always_neutral_baseline(self) -> None:
        """Test metrics for always-neutral predictions (Baseline-A)."""
        y_true = ["up"] * 10 + ["down"] * 10 + ["neutral"] * 10 + ["choppy"] * 10
        y_pred = ["neutral"] * 40

        metrics = compute_direction_metrics(y_true, y_pred, horizon=30)

        # Only neutral has non-zero recall
        self.assertEqual(metrics.neutral_recall, 1.0)
        self.assertEqual(metrics.up_recall, 0.0)
        self.assertEqual(metrics.down_recall, 0.0)
        self.assertEqual(metrics.choppy_recall, 0.0)

        # Balanced accuracy = average of recalls = 0.25
        self.assertAlmostEqual(metrics.balanced_accuracy, 0.25, places=6)

    def test_imbalanced_classes(self) -> None:
        """Test metrics with imbalanced class distribution."""
        # Many neutral, few up/down/choppy
        y_true = ["neutral"] * 100 + ["up"] * 5 + ["down"] * 5 + ["choppy"] * 5
        y_pred = ["neutral"] * 115  # Always predict neutral

        metrics = compute_direction_metrics(y_true, y_pred, horizon=30)

        # Balanced accuracy should still be 0.25 (only neutral recall is 1.0)
        self.assertAlmostEqual(metrics.balanced_accuracy, 0.25, places=6)

    def test_total_samples_count(self) -> None:
        """Test that total samples is correctly counted."""
        y_true = ["up"] * 50 + ["down"] * 30
        y_pred = ["up"] * 50 + ["down"] * 30

        metrics = compute_direction_metrics(y_true, y_pred, horizon=30)

        self.assertEqual(metrics.total_samples, 80)

    def test_horizon_preserved(self) -> None:
        """Test that horizon is preserved in result."""
        y_true = ["up", "down", "neutral", "choppy"]
        y_pred = y_true.copy()

        for horizon in [10, 30, 60]:
            metrics = compute_direction_metrics(y_true, y_pred, horizon=horizon)
            self.assertEqual(metrics.horizon_min, horizon)


class PinballLossTests(unittest.TestCase):
    """Tests for pinball loss computation."""

    def test_pinball_loss_exact_prediction(self) -> None:
        """Test pinball loss when prediction equals actual."""
        loss = pinball_loss(y_true=100.0, y_pred=100.0, quantile=0.5)
        self.assertEqual(loss, 0.0)

    def test_pinball_loss_underestimate_q50(self) -> None:
        """Test pinball loss when prediction underestimates (q=0.5)."""
        # y_true > y_pred: loss = quantile * (y_true - y_pred)
        loss = pinball_loss(y_true=110.0, y_pred=100.0, quantile=0.5)
        self.assertAlmostEqual(loss, 0.5 * 10.0, places=6)

    def test_pinball_loss_overestimate_q50(self) -> None:
        """Test pinball loss when prediction overestimates (q=0.5)."""
        # y_true < y_pred: loss = (1 - quantile) * (y_pred - y_true)
        loss = pinball_loss(y_true=100.0, y_pred=110.0, quantile=0.5)
        self.assertAlmostEqual(loss, 0.5 * 10.0, places=6)

    def test_pinball_loss_q90_asymmetry(self) -> None:
        """Test pinball loss asymmetry for q=0.90."""
        # Underestimating at q=0.90 is heavily penalized
        under_loss = pinball_loss(y_true=110.0, y_pred=100.0, quantile=0.9)
        over_loss = pinball_loss(y_true=100.0, y_pred=110.0, quantile=0.9)

        # Underestimate: 0.9 * 10 = 9.0
        self.assertAlmostEqual(under_loss, 9.0, places=6)
        # Overestimate: 0.1 * 10 = 1.0
        self.assertAlmostEqual(over_loss, 1.0, places=6)

    def test_pinball_loss_q10_asymmetry(self) -> None:
        """Test pinball loss asymmetry for q=0.10."""
        # Overestimating at q=0.10 is heavily penalized
        under_loss = pinball_loss(y_true=110.0, y_pred=100.0, quantile=0.1)
        over_loss = pinball_loss(y_true=100.0, y_pred=110.0, quantile=0.1)

        # Underestimate: 0.1 * 10 = 1.0
        self.assertAlmostEqual(under_loss, 1.0, places=6)
        # Overestimate: 0.9 * 10 = 9.0
        self.assertAlmostEqual(over_loss, 9.0, places=6)

    def test_pinball_loss_batch(self) -> None:
        """Test batch pinball loss computation."""
        y_true = [100.0, 110.0, 90.0]
        y_pred = [100.0, 100.0, 100.0]  # All predict 100

        loss = compute_pinball_loss_batch(y_true, y_pred, quantile=0.5)

        # Individual losses: 0, 5, 5
        expected = (0 + 0.5 * 10 + 0.5 * 10) / 3
        self.assertAlmostEqual(loss, expected, places=6)

    def test_pinball_loss_batch_empty_raises(self) -> None:
        """Test that empty sequences raise error."""
        with self.assertRaises(EvaluationError):
            compute_pinball_loss_batch([], [], quantile=0.5)

    def test_pinball_loss_batch_length_mismatch_raises(self) -> None:
        """Test that mismatched lengths raise error."""
        with self.assertRaises(EvaluationError):
            compute_pinball_loss_batch([100.0], [100.0, 110.0], quantile=0.5)


class CoverageTests(unittest.TestCase):
    """Tests for coverage computation."""

    def test_perfect_coverage(self) -> None:
        """Test coverage when all values below prediction."""
        y_true = [100.0, 101.0, 102.0]
        q_pred = [150.0, 150.0, 150.0]  # All predictions above true

        coverage = compute_coverage(y_true, q_pred)
        self.assertEqual(coverage, 1.0)

    def test_zero_coverage(self) -> None:
        """Test coverage when all values above prediction."""
        y_true = [100.0, 101.0, 102.0]
        q_pred = [50.0, 50.0, 50.0]  # All predictions below true

        coverage = compute_coverage(y_true, q_pred)
        self.assertEqual(coverage, 0.0)

    def test_partial_coverage(self) -> None:
        """Test partial coverage."""
        y_true = [100.0, 101.0, 102.0, 103.0]
        q_pred = [101.5, 101.5, 101.5, 101.5]  # 2 below, 2 above

        coverage = compute_coverage(y_true, q_pred)
        self.assertEqual(coverage, 0.5)

    def test_boundary_coverage(self) -> None:
        """Test coverage at boundary (y_true == q_pred)."""
        y_true = [100.0, 100.0, 100.0]
        q_pred = [100.0, 100.0, 100.0]

        coverage = compute_coverage(y_true, q_pred)
        self.assertEqual(coverage, 1.0)  # y <= q is satisfied when equal

    def test_coverage_empty_raises(self) -> None:
        """Test that empty sequences raise error."""
        with self.assertRaises(EvaluationError):
            compute_coverage([], [])


class RangeMetricsTests(unittest.TestCase):
    """Tests for range metrics computation."""

    def test_compute_range_metrics(self) -> None:
        """Test complete range metrics computation."""
        y_true = [100.0, 110.0, 90.0, 105.0, 95.0]
        q10_pred = [95.0, 95.0, 95.0, 95.0, 95.0]
        q50_pred = [100.0, 100.0, 100.0, 100.0, 100.0]
        q90_pred = [105.0, 105.0, 105.0, 105.0, 105.0]

        metrics = compute_range_metrics(y_true, q10_pred, q50_pred, q90_pred, horizon=30)

        self.assertEqual(metrics.horizon_min, 30)
        self.assertEqual(metrics.total_samples, 5)

        # Verify metrics are computed
        self.assertIsInstance(metrics.pinball_loss_q10, float)
        self.assertIsInstance(metrics.pinball_loss_q50, float)
        self.assertIsInstance(metrics.pinball_loss_q90, float)
        self.assertIsInstance(metrics.pinball_loss_avg, float)
        self.assertIsInstance(metrics.coverage_q10, float)
        self.assertIsInstance(metrics.coverage_q90, float)

        # Verify average is correct
        expected_avg = (
            metrics.pinball_loss_q10 + metrics.pinball_loss_q50 + metrics.pinball_loss_q90
        ) / 3
        self.assertAlmostEqual(metrics.pinball_loss_avg, expected_avg, places=6)


class CriteriaCheckTests(unittest.TestCase):
    """Tests for criteria checking functions."""

    def test_direction_criteria_primary_pass(self) -> None:
        """Test direction criteria that pass primary thresholds."""
        cm = {c: {p: 0 for p in DIRECTION_CLASSES} for c in DIRECTION_CLASSES}
        for c in DIRECTION_CLASSES:
            cm[c][c] = 10

        metrics = DirectionMetrics(
            horizon_min=30,
            balanced_accuracy=0.45,
            macro_f1=0.40,
            up_recall=0.35,
            down_recall=0.35,
            neutral_recall=0.50,
            choppy_recall=0.45,
            up_precision=0.45,
            down_precision=0.45,
            neutral_precision=0.50,
            choppy_precision=0.45,
            confusion_matrix=cm,
            total_samples=40,
        )

        criteria = check_direction_criteria(metrics, is_primary=True)

        self.assertTrue(criteria["balanced_accuracy_30m"])
        self.assertTrue(criteria["macro_f1_30m"])
        self.assertTrue(criteria["up_recall_30m"])
        self.assertTrue(criteria["down_recall_30m"])

    def test_direction_criteria_primary_fail(self) -> None:
        """Test direction criteria that fail primary thresholds."""
        cm = {c: {p: 0 for p in DIRECTION_CLASSES} for c in DIRECTION_CLASSES}

        metrics = DirectionMetrics(
            horizon_min=30,
            balanced_accuracy=0.30,  # Below 0.40
            macro_f1=0.25,  # Below 0.35
            up_recall=0.20,  # Below 0.30
            down_recall=0.20,  # Below 0.30
            neutral_recall=0.40,
            choppy_recall=0.30,
            up_precision=0.25,
            down_precision=0.25,
            neutral_precision=0.40,
            choppy_precision=0.30,
            confusion_matrix=cm,
            total_samples=40,
        )

        criteria = check_direction_criteria(metrics, is_primary=True)

        self.assertFalse(criteria["balanced_accuracy_30m"])
        self.assertFalse(criteria["macro_f1_30m"])
        self.assertFalse(criteria["up_recall_30m"])
        self.assertFalse(criteria["down_recall_30m"])

    def test_direction_criteria_secondary(self) -> None:
        """Test direction criteria for secondary horizons."""
        cm = {c: {p: 0 for p in DIRECTION_CLASSES} for c in DIRECTION_CLASSES}

        # Pass secondary but fail primary thresholds
        metrics = DirectionMetrics(
            horizon_min=10,
            balanced_accuracy=0.39,  # >= 0.38 but < 0.40
            macro_f1=0.34,  # >= 0.33 but < 0.35
            up_recall=0.29,  # >= 0.28 but < 0.30
            down_recall=0.29,
            neutral_recall=0.50,
            choppy_recall=0.40,
            up_precision=0.30,
            down_precision=0.30,
            neutral_precision=0.50,
            choppy_precision=0.40,
            confusion_matrix=cm,
            total_samples=40,
        )

        # Should pass secondary criteria
        secondary = check_direction_criteria(metrics, is_primary=False)
        self.assertTrue(secondary["balanced_accuracy_10m"])
        self.assertTrue(secondary["macro_f1_10m"])
        self.assertTrue(secondary["up_recall_10m"])
        self.assertTrue(secondary["down_recall_10m"])

        # Should fail primary criteria
        primary = check_direction_criteria(metrics, is_primary=True)
        self.assertFalse(primary["balanced_accuracy_10m"])
        self.assertFalse(primary["macro_f1_10m"])
        self.assertFalse(primary["up_recall_10m"])
        self.assertFalse(primary["down_recall_10m"])

    def test_range_criteria_pass(self) -> None:
        """Test range criteria that pass."""
        metrics = RangeMetrics(
            horizon_min=30,
            pinball_loss_q10=0.05,
            pinball_loss_q50=0.10,
            pinball_loss_q90=0.05,
            pinball_loss_avg=0.0667,
            coverage_q10=0.10,  # Exactly target
            coverage_q90=0.90,  # Exactly target
            total_samples=1000,
        )

        baseline = RangeMetrics(
            horizon_min=30,
            pinball_loss_q10=0.10,
            pinball_loss_q50=0.15,
            pinball_loss_q90=0.10,
            pinball_loss_avg=0.1167,  # Higher than model
            coverage_q10=0.10,
            coverage_q90=0.90,
            total_samples=1000,
        )

        criteria = check_range_criteria(metrics, baseline)

        self.assertTrue(criteria["pinball_beats_baseline_30m"])
        self.assertTrue(criteria["coverage_q10_30m"])
        self.assertTrue(criteria["coverage_q90_30m"])

    def test_range_criteria_coverage_tolerance(self) -> None:
        """Test range criteria coverage tolerance (+/- 0.03)."""
        # At lower bound of tolerance
        metrics_low = RangeMetrics(
            horizon_min=30,
            pinball_loss_q10=0.05,
            pinball_loss_q50=0.10,
            pinball_loss_q90=0.05,
            pinball_loss_avg=0.0667,
            coverage_q10=0.07,  # 0.10 - 0.03
            coverage_q90=0.87,  # 0.90 - 0.03
            total_samples=1000,
        )

        criteria_low = check_range_criteria(metrics_low)
        self.assertTrue(criteria_low["coverage_q10_30m"])
        self.assertTrue(criteria_low["coverage_q90_30m"])

        # At upper bound of tolerance
        metrics_high = RangeMetrics(
            horizon_min=30,
            pinball_loss_q10=0.05,
            pinball_loss_q50=0.10,
            pinball_loss_q90=0.05,
            pinball_loss_avg=0.0667,
            coverage_q10=0.13,  # 0.10 + 0.03
            coverage_q90=0.93,  # 0.90 + 0.03
            total_samples=1000,
        )

        criteria_high = check_range_criteria(metrics_high)
        self.assertTrue(criteria_high["coverage_q10_30m"])
        self.assertTrue(criteria_high["coverage_q90_30m"])

        # Outside tolerance
        metrics_outside = RangeMetrics(
            horizon_min=30,
            pinball_loss_q10=0.05,
            pinball_loss_q50=0.10,
            pinball_loss_q90=0.05,
            pinball_loss_avg=0.0667,
            coverage_q10=0.05,  # Below 0.07
            coverage_q90=0.95,  # Above 0.93
            total_samples=1000,
        )

        criteria_outside = check_range_criteria(metrics_outside)
        self.assertFalse(criteria_outside["coverage_q10_30m"])
        self.assertFalse(criteria_outside["coverage_q90_30m"])


class EvaluationResultTests(unittest.TestCase):
    """Tests for evaluation result functions."""

    def test_evaluate_direction_predictions(self) -> None:
        """Test direction prediction evaluation."""
        y_true = {30: ["up"] * 10 + ["down"] * 10 + ["neutral"] * 10 + ["choppy"] * 10}
        y_pred = {30: ["neutral"] * 40}  # Baseline-A like
        baseline_a = {30: ["neutral"] * 40}
        baseline_b = {30: ["up"] * 20 + ["down"] * 20}

        result = evaluate_direction_predictions(
            y_true, y_pred, baseline_a, baseline_b
        )

        self.assertIn(30, result.metrics_by_horizon)
        self.assertIn(30, result.baseline_a_metrics)
        self.assertIn(30, result.baseline_b_metrics)

    def test_evaluate_direction_predictions_missing_horizon_raises(self) -> None:
        """Test missing predictions raise error."""
        y_true = {30: ["up", "down"]}
        y_pred = {}

        with self.assertRaises(EvaluationError):
            evaluate_direction_predictions(y_true, y_pred)

    def test_evaluate_direction_predictions_empty_predictions_raises(self) -> None:
        """Test empty predictions raise error."""
        y_true = {30: ["up", "down"]}
        y_pred = {30: []}

        with self.assertRaises(EvaluationError):
            evaluate_direction_predictions(y_true, y_pred)

    def test_evaluate_direction_predictions_missing_baseline_horizon_raises(
        self,
    ) -> None:
        """Test missing baseline horizon raises error."""
        y_true = {30: ["up", "down"]}
        y_pred = {30: ["up", "down"]}

        with self.assertRaises(EvaluationError):
            evaluate_direction_predictions(y_true, y_pred, baseline_a_pred_by_horizon={})

    def test_evaluate_range_predictions(self) -> None:
        """Test range prediction evaluation."""
        y_true = {30: [100.0, 110.0, 90.0]}
        q10 = {30: [95.0, 95.0, 95.0]}
        q50 = {30: [100.0, 100.0, 100.0]}
        q90 = {30: [105.0, 105.0, 105.0]}

        result = evaluate_range_predictions(y_true, q10, q50, q90)

        self.assertIn(30, result.metrics_by_horizon)

    def test_evaluate_range_predictions_missing_quantile_raises(self) -> None:
        """Test missing quantile predictions raise error."""
        y_true = {30: [100.0, 110.0]}
        q10 = {30: [95.0, 95.0]}
        q50 = {}
        q90 = {30: [105.0, 105.0]}

        with self.assertRaises(EvaluationError):
            evaluate_range_predictions(y_true, q10, q50, q90)

    def test_evaluate_range_predictions_empty_quantile_raises(self) -> None:
        """Test empty quantile predictions raise error."""
        y_true = {30: [100.0, 110.0]}
        q10 = {30: []}
        q50 = {30: [100.0, 100.0]}
        q90 = {30: [105.0, 105.0]}

        with self.assertRaises(EvaluationError):
            evaluate_range_predictions(y_true, q10, q50, q90)

    def test_evaluate_range_predictions_partial_baseline_raises(self) -> None:
        """Test that partial baseline inputs raise error."""
        y_true = {30: [100.0, 110.0]}
        q10 = {30: [95.0, 95.0]}
        q50 = {30: [100.0, 100.0]}
        q90 = {30: [105.0, 105.0]}

        with self.assertRaises(EvaluationError):
            evaluate_range_predictions(
                y_true,
                q10,
                q50,
                q90,
                baseline_q10_by_horizon={30: [95.0, 95.0]},
                baseline_q50_by_horizon=None,
                baseline_q90_by_horizon={30: [105.0, 105.0]},
            )

    def test_evaluate_range_predictions_missing_baseline_horizon_raises(self) -> None:
        """Test missing baseline horizon raises error."""
        y_true = {30: [100.0, 110.0]}
        q10 = {30: [95.0, 95.0]}
        q50 = {30: [100.0, 100.0]}
        q90 = {30: [105.0, 105.0]}

        with self.assertRaises(EvaluationError):
            evaluate_range_predictions(
                y_true,
                q10,
                q50,
                q90,
                baseline_q10_by_horizon={},
                baseline_q50_by_horizon={},
                baseline_q90_by_horizon={},
            )


class ReportFormattingTests(unittest.TestCase):
    """Tests for report formatting functions."""

    def test_format_direction_report(self) -> None:
        """Test direction report formatting."""
        from pms.evaluation import DirectionEvaluationResult

        cm = {c: {p: 0 for p in DIRECTION_CLASSES} for c in DIRECTION_CLASSES}
        for c in DIRECTION_CLASSES:
            cm[c][c] = 10

        metrics = DirectionMetrics(
            horizon_min=30,
            balanced_accuracy=0.45,
            macro_f1=0.40,
            up_recall=0.35,
            down_recall=0.35,
            neutral_recall=0.50,
            choppy_recall=0.45,
            up_precision=0.45,
            down_precision=0.45,
            neutral_precision=0.50,
            choppy_precision=0.45,
            confusion_matrix=cm,
            total_samples=40,
        )

        result = DirectionEvaluationResult(
            metrics_by_horizon={},
            baseline_a_metrics={30: metrics},
            baseline_b_metrics={30: metrics},
        )

        report = format_direction_report(result)

        self.assertIn("Direction Evaluation Report", report)
        self.assertIn("30m", report)
        self.assertIn("PRIMARY", report)
        self.assertIn("Baseline-A", report)
        self.assertIn("Baseline-B", report)

    def test_format_range_report(self) -> None:
        """Test range report formatting."""
        from pms.evaluation import RangeEvaluationResult

        metrics = RangeMetrics(
            horizon_min=30,
            pinball_loss_q10=0.05,
            pinball_loss_q50=0.10,
            pinball_loss_q90=0.05,
            pinball_loss_avg=0.0667,
            coverage_q10=0.10,
            coverage_q90=0.90,
            total_samples=1000,
        )

        result = RangeEvaluationResult(
            metrics_by_horizon={},
            baseline_metrics={30: metrics},
        )

        report = format_range_report(result)

        self.assertIn("Range Evaluation Report", report)
        self.assertIn("30m", report)
        self.assertIn("Pinball Loss", report)
        self.assertIn("Coverage", report)

    def test_format_confusion_matrix(self) -> None:
        """Test confusion matrix formatting."""
        cm = {c: {p: 0 for p in DIRECTION_CLASSES} for c in DIRECTION_CLASSES}
        cm["up"]["up"] = 10
        cm["up"]["down"] = 2
        cm["down"]["down"] = 8

        formatted = format_confusion_matrix(cm)

        self.assertIn("Confusion Matrix", formatted)
        self.assertIn("up", formatted)
        self.assertIn("down", formatted)
        self.assertIn("neutral", formatted)
        self.assertIn("choppy", formatted)


class ConstantsTests(unittest.TestCase):
    """Tests for module constants."""

    def test_direction_classes(self) -> None:
        """Test DIRECTION_CLASSES constant."""
        self.assertEqual(DIRECTION_CLASSES, ["up", "down", "neutral", "choppy"])

    def test_criteria_thresholds(self) -> None:
        """Test criteria threshold constants."""
        # Primary thresholds
        self.assertEqual(CRITERIA_PRIMARY_BALANCED_ACCURACY, 0.40)
        self.assertEqual(CRITERIA_PRIMARY_MACRO_F1, 0.35)
        self.assertEqual(CRITERIA_PRIMARY_RECALL, 0.30)

        # Secondary thresholds
        self.assertEqual(CRITERIA_SECONDARY_BALANCED_ACCURACY, 0.38)
        self.assertEqual(CRITERIA_SECONDARY_MACRO_F1, 0.33)
        self.assertEqual(CRITERIA_SECONDARY_RECALL, 0.28)

    def test_coverage_constants(self) -> None:
        """Test coverage target constants."""
        self.assertEqual(COVERAGE_Q10_TARGET, 0.10)
        self.assertEqual(COVERAGE_Q90_TARGET, 0.90)
        self.assertEqual(COVERAGE_TOLERANCE, 0.03)


if __name__ == "__main__":
    unittest.main()
