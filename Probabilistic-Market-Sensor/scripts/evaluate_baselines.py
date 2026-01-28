"""Script to evaluate baseline models and generate reports.

This script evaluates Baseline-A (always neutral) and Baseline-B (1-min return sign)
for direction prediction, and the fixed-quantile baseline for range prediction.

Usage:
    python scripts/evaluate_baselines.py [--output-dir PATH] [--step N] [--limit N]

Output:
    - baseline_evaluation_report.json: Machine-readable evaluation results
    - baseline_evaluation_report.md: Human-readable evaluation report
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.baselines import (  # noqa: E402
    RETURN_THRESHOLD_PCT,
    compute_return_1,
    predict_baseline_a,
    predict_baseline_b,
    train_range_baseline,
    predict_range_baseline,
    TrainedRangeBaseline,
)
from pms.evaluation import (  # noqa: E402
    DIRECTION_CLASSES,
    compute_direction_metrics,
    compute_range_metrics,
    format_direction_report,
    format_range_report,
    format_confusion_matrix,
    DirectionMetrics,
    RangeMetrics,
    DirectionEvaluationResult,
    RangeEvaluationResult,
)
from pms.labels import HORIZONS, generate_labels, BarrierConfig, LabelError  # noqa: E402
from pms.validation import OhlcvBar  # noqa: E402

# Data paths
DATA_DIR = ROOT.parent / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUT_DIR = DATA_DIR / "evaluation"

ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def load_bars(filepath: Path, limit: int | None = None) -> List[OhlcvBar]:
    """Load OHLCV bars from JSONL file."""
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    bars = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            record = json.loads(line)
            bar = OhlcvBar(
                timestamp_utc=datetime.strptime(
                    record["timestamp_utc"], ISO8601_FORMAT
                ).replace(tzinfo=timezone.utc),
                open=float(record["open"]),
                high=float(record["high"]),
                low=float(record["low"]),
                close=float(record["close"]),
                volume=float(record["volume"]),
            )
            bars.append(bar)
    return bars


def find_split_files(splits_dir: Path) -> Dict[str, Path]:
    """Find train/val/test split files."""
    candidates: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for filepath in splits_dir.glob("*.jsonl"):
        if "train" in filepath.name:
            candidates["train"].append(filepath)
        elif "val" in filepath.name:
            candidates["val"].append(filepath)
        elif "test" in filepath.name:
            candidates["test"].append(filepath)

    files: Dict[str, Path] = {}
    for split_name, paths in candidates.items():
        if not paths:
            continue
        if len(paths) > 1:
            names = ", ".join(sorted(p.name for p in paths))
            raise ValueError(f"multiple {split_name} split files found: {names}")
        files[split_name] = paths[0]

    return files


def summarize_period(bars: List[OhlcvBar]) -> Dict[str, str | int]:
    """Summarize data period with start/end timestamps and bar count."""
    if not bars:
        return {}
    return {
        "start": bars[0].timestamp_utc.strftime(ISO8601_FORMAT),
        "end": bars[-1].timestamp_utc.strftime(ISO8601_FORMAT),
        "bars": len(bars),
    }


def evaluate_direction_baselines(
    bars: List[OhlcvBar],
    horizons: List[int] = HORIZONS,
    config: BarrierConfig | None = None,
    step: int = 1,
) -> Tuple[
    Dict[int, DirectionMetrics],
    Dict[int, DirectionMetrics],
    Dict[int, Dict[str, int]],
]:
    """
    Evaluate direction baselines on data.

    Args:
        bars: OHLCV bars to evaluate on
        horizons: List of horizons to evaluate
        config: Barrier configuration for label generation
        step: Step size for evaluation (use larger for faster evaluation)

    Returns:
        Tuple of (baseline_a_metrics, baseline_b_metrics, label_distributions)

    Raises:
        ValueError: If atr_window < 2 (Baseline-B requires two bars)
    """
    if config is None:
        config = BarrierConfig(horizons=horizons)
    if config.atr_window < 2:
        raise ValueError("atr_window must be >= 2 for Baseline-B evaluation")

    max_horizon = max(horizons)
    min_required = config.atr_window + max_horizon

    # Collect predictions and labels
    baseline_a_preds = {h: [] for h in horizons}
    baseline_b_preds = {h: [] for h in horizons}
    true_labels = {h: [] for h in horizons}

    # Evaluate at each point
    evaluated = 0
    skipped = 0

    for i in range(0, len(bars) - min_required + 1, step):
        window = bars[i : i + min_required]

        try:
            result = generate_labels(window, horizons=horizons, config=config)
        except LabelError:
            skipped += 1
            continue

        # Get true labels
        for label in result.labels:
            true_labels[label.horizon_min].append(label.label_class)

        # Baseline-A: always neutral
        for h in horizons:
            baseline_a_preds[h].append("neutral")

        # Baseline-B: 1-min return sign
        # Use the reference bar (t) and previous bar (t-1)
        t = len(window) - max_horizon - 1
        if t >= 1:
            return_1 = compute_return_1(window[t - 1].close, window[t].close)
            abs_return = abs(return_1)

            if abs_return < RETURN_THRESHOLD_PCT:
                pred = "neutral"
            elif return_1 >= 0:
                pred = "up"
            else:
                pred = "down"

            for h in horizons:
                baseline_b_preds[h].append(pred)

        evaluated += 1

    print(f"  Evaluated {evaluated} samples, skipped {skipped}")

    # Compute label distributions
    label_distributions = {}
    for h in horizons:
        label_distributions[h] = {
            "up": 0, "down": 0, "neutral": 0, "choppy": 0
        }
        for lbl in true_labels[h]:
            label_distributions[h][lbl] += 1

    # Print label distributions
    for h in horizons:
        dist = label_distributions[h]
        total = sum(dist.values())
        print(f"  Horizon {h}m label distribution:")
        for cls in ["up", "down", "neutral", "choppy"]:
            pct = dist[cls] / total * 100 if total > 0 else 0
            print(f"    {cls}: {dist[cls]} ({pct:.1f}%)")

    # Compute metrics
    baseline_a_metrics = {}
    baseline_b_metrics = {}

    for h in horizons:
        baseline_a_metrics[h] = compute_direction_metrics(
            true_labels[h], baseline_a_preds[h], h
        )
        baseline_b_metrics[h] = compute_direction_metrics(
            true_labels[h], baseline_b_preds[h], h
        )

    return baseline_a_metrics, baseline_b_metrics, label_distributions


def evaluate_range_baseline(
    train_bars: List[OhlcvBar],
    test_bars: List[OhlcvBar],
    horizons: List[int] = HORIZONS,
    step: int = 1,
) -> Dict[int, RangeMetrics]:
    """
    Evaluate range baseline on test data.

    Args:
        train_bars: Training bars for fitting baseline quantiles
        test_bars: Test bars for evaluation
        horizons: List of horizons to evaluate
        step: Step size for evaluation

    Returns:
        Dictionary of RangeMetrics by horizon
    """
    # Train range baselines
    print("  Training range baselines...")
    trained_baselines = {}
    for h in horizons:
        trained_baselines[h] = train_range_baseline(train_bars, h)
        print(
            f"    Horizon {h}m: q10={trained_baselines[h].return_q10:.6f}, "
            f"q50={trained_baselines[h].return_q50:.6f}, "
            f"q90={trained_baselines[h].return_q90:.6f}"
        )

    # Evaluate on test data
    print("  Evaluating on test data...")
    y_true = {h: [] for h in horizons}
    q10_pred = {h: [] for h in horizons}
    q50_pred = {h: [] for h in horizons}
    q90_pred = {h: [] for h in horizons}

    evaluated = 0
    max_horizon = max(horizons)
    for i in range(0, len(test_bars) - max_horizon, step):
        current_close = test_bars[i].close

        for h in horizons:
            future_idx = i + h
            if future_idx < len(test_bars):
                # True future close
                y_true[h].append(test_bars[future_idx].close)

                # Baseline prediction
                pred = predict_range_baseline(current_close, trained_baselines[h])
                q10_pred[h].append(pred.q10)
                q50_pred[h].append(pred.q50)
                q90_pred[h].append(pred.q90)

        evaluated += 1

    print(f"  Evaluated {evaluated} samples")

    # Compute metrics
    metrics = {}
    for h in horizons:
        if y_true[h]:
            metrics[h] = compute_range_metrics(
                y_true[h], q10_pred[h], q50_pred[h], q90_pred[h], h
            )

    return metrics


def generate_json_report(
    direction_a: Dict[int, DirectionMetrics],
    direction_b: Dict[int, DirectionMetrics],
    range_metrics: Dict[int, RangeMetrics],
    range_baselines: Dict[int, TrainedRangeBaseline],
    horizons: List[int],
    evaluation_step: int,
    data_periods: Dict[str, Dict[str, str | int]] | None = None,
    range_train_split: str | None = None,
) -> dict:
    """Generate JSON report data."""
    report = {
        "report_version": "1.2",
        "generated_at": datetime.now(timezone.utc).strftime(ISO8601_FORMAT),
        "evaluation_step": evaluation_step,
        "horizons": horizons,
        "direction_evaluation": {},
        "range_evaluation": {},
        "summary": {},
    }
    if data_periods:
        report["data_periods"] = data_periods
    if range_train_split:
        report["range_baseline_train_split"] = range_train_split

    # Direction evaluation
    for h in horizons:
        report["direction_evaluation"][str(h)] = {
            "baseline_a": {
                "balanced_accuracy": direction_a[h].balanced_accuracy,
                "macro_f1": direction_a[h].macro_f1,
                "up_recall": direction_a[h].up_recall,
                "down_recall": direction_a[h].down_recall,
                "neutral_recall": direction_a[h].neutral_recall,
                "choppy_recall": direction_a[h].choppy_recall,
                "confusion_matrix": direction_a[h].confusion_matrix,
                "total_samples": direction_a[h].total_samples,
            },
            "baseline_b": {
                "balanced_accuracy": direction_b[h].balanced_accuracy,
                "macro_f1": direction_b[h].macro_f1,
                "up_recall": direction_b[h].up_recall,
                "down_recall": direction_b[h].down_recall,
                "neutral_recall": direction_b[h].neutral_recall,
                "choppy_recall": direction_b[h].choppy_recall,
                "confusion_matrix": direction_b[h].confusion_matrix,
                "total_samples": direction_b[h].total_samples,
            },
            "best_baseline": (
                "B"
                if direction_b[h].balanced_accuracy > direction_a[h].balanced_accuracy
                else "A"
            ),
            "best_balanced_accuracy": max(
                direction_a[h].balanced_accuracy, direction_b[h].balanced_accuracy
            ),
        }

    # Range evaluation
    for h in horizons:
        if h in range_metrics and h in range_baselines:
            report["range_evaluation"][str(h)] = {
                "trained_quantiles": {
                    "return_q10": range_baselines[h].return_q10,
                    "return_q50": range_baselines[h].return_q50,
                    "return_q90": range_baselines[h].return_q90,
                },
                "metrics": {
                    "pinball_loss_q10": range_metrics[h].pinball_loss_q10,
                    "pinball_loss_q50": range_metrics[h].pinball_loss_q50,
                    "pinball_loss_q90": range_metrics[h].pinball_loss_q90,
                    "pinball_loss_avg": range_metrics[h].pinball_loss_avg,
                    "coverage_q10": range_metrics[h].coverage_q10,
                    "coverage_q90": range_metrics[h].coverage_q90,
                    "total_samples": range_metrics[h].total_samples,
                },
            }

    # Summary
    report["summary"] = {
        "direction_baselines_established": len(direction_a) == len(horizons),
        "range_baselines_established": len(range_metrics) == len(horizons),
    }

    return report


def generate_markdown_report(
    direction_a: Dict[int, DirectionMetrics],
    direction_b: Dict[int, DirectionMetrics],
    range_metrics: Dict[int, RangeMetrics],
    range_baselines: Dict[int, TrainedRangeBaseline],
    horizons: List[int],
    label_distributions: Dict[int, Dict[str, int]] | None = None,
    data_periods: Dict[str, Dict[str, str | int]] | None = None,
    range_train_split: str | None = None,
    evaluation_step: int | None = None,
) -> str:
    """Generate Markdown report."""
    lines = [
        "# Baseline Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime(ISO8601_FORMAT)}",
        "",
    ]

    if evaluation_step is not None:
        lines.extend(
            [
                f"Evaluation step: {evaluation_step}",
                "",
            ]
        )

    if data_periods:
        lines.extend(
            [
                "## Data Periods",
                "",
            ]
        )
        for key in ("train", "val", "test"):
            info = data_periods.get(key)
            if not info:
                continue
            lines.append(
                f"- {key.capitalize()}: {info['start']} to {info['end']} "
                f"(bars: {info['bars']})"
            )
        lines.append("")

    lines.extend(
        [
        "## Direction Baselines",
        "",
        ]
    )

    for h in horizons:
        is_primary = h == 30
        label = "PRIMARY" if is_primary else "Secondary"

        lines.append(f"### {h}-minute ({label})")
        lines.append("")

        # Show label distribution if available
        if label_distributions and h in label_distributions:
            dist = label_distributions[h]
            total = sum(dist.values())
            lines.append("**Label Distribution:**")
            for cls in ["up", "down", "neutral", "choppy"]:
                count = dist.get(cls, 0)
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"- {cls}: {count} ({pct:.1f}%)")
            lines.append("")

            # Warning if any class is missing
            missing = [cls for cls in ["up", "down", "neutral", "choppy"] if dist.get(cls, 0) == 0]
            if missing:
                lines.append(
                    f"**Note:** Classes {missing} have no samples. "
                    "Recall/F1 for these classes are reported as 0.0."
                )
                lines.append("")

        lines.append("| Metric | Baseline-A | Baseline-B | Better |")
        lines.append("|--------|------------|------------|--------|")

        ba = direction_a[h]
        bb = direction_b[h]

        def better(a: float, b: float) -> str:
            return "B" if b > a else "A" if a > b else "="

        lines.append(
            f"| Balanced Accuracy | {ba.balanced_accuracy:.4f} | "
            f"{bb.balanced_accuracy:.4f} | {better(ba.balanced_accuracy, bb.balanced_accuracy)} |"
        )
        lines.append(
            f"| Macro-F1 | {ba.macro_f1:.4f} | "
            f"{bb.macro_f1:.4f} | {better(ba.macro_f1, bb.macro_f1)} |"
        )
        lines.append(
            f"| Up Recall | {ba.up_recall:.4f} | "
            f"{bb.up_recall:.4f} | {better(ba.up_recall, bb.up_recall)} |"
        )
        lines.append(
            f"| Down Recall | {ba.down_recall:.4f} | "
            f"{bb.down_recall:.4f} | {better(ba.down_recall, bb.down_recall)} |"
        )
        lines.append("")

        # Best baseline for this horizon
        best = "B" if bb.balanced_accuracy > ba.balanced_accuracy else "A"
        best_acc = max(ba.balanced_accuracy, bb.balanced_accuracy)
        lines.append(f"**Best Baseline: {best}** (Balanced Accuracy: {best_acc:.4f})")
        lines.append("")

        lines.append("**Confusion Matrix (Baseline-A)**")
        lines.append("")
        lines.append("```text")
        lines.append(format_confusion_matrix(ba.confusion_matrix))
        lines.append("```")
        lines.append("")

        lines.append("**Confusion Matrix (Baseline-B)**")
        lines.append("")
        lines.append("```text")
        lines.append(format_confusion_matrix(bb.confusion_matrix))
        lines.append("```")
        lines.append("")

    lines.append("## Range Baselines")
    lines.append("")
    if range_train_split:
        lines.append(f"- Range baseline training split: {range_train_split}")
        lines.append("")
    lines.append(
        "Note: Coverage values are diagnostic for baseline (PASS/FAIL applies to model)."
    )
    lines.append("")

    for h in horizons:
        if h in range_metrics and h in range_baselines:
            lines.append(f"### {h}-minute")
            lines.append("")

            rb = range_baselines[h]
            lines.append(
                f"- Training Return Quantiles: q10={rb.return_q10*100:.4f}%, "
                f"q50={rb.return_q50*100:.4f}%, q90={rb.return_q90*100:.4f}%"
            )

            rm = range_metrics[h]
            lines.append(f"- Pinball Loss (avg): {rm.pinball_loss_avg:.6f}")
            lines.append(f"  - q10: {rm.pinball_loss_q10:.6f}")
            lines.append(f"  - q50: {rm.pinball_loss_q50:.6f}")
            lines.append(f"  - q90: {rm.pinball_loss_q90:.6f}")

            lines.append(
                f"- Coverage q10: {rm.coverage_q10:.4f} (target: 0.10 +/- 0.03)"
            )
            lines.append(
                f"- Coverage q90: {rm.coverage_q90:.4f} (target: 0.90 +/- 0.03)"
            )
            lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("### Direction Baselines (Model Must Exceed)")
    lines.append("")
    for h in horizons:
        ba = direction_a[h]
        bb = direction_b[h]
        best = "B" if bb.balanced_accuracy > ba.balanced_accuracy else "A"
        best_acc = max(ba.balanced_accuracy, bb.balanced_accuracy)
        label = "PRIMARY" if h == 30 else "Secondary"
        lines.append(f"- **{h}m ({label})**: Best = Baseline-{best}, Balanced Accuracy = {best_acc:.4f}")
    lines.append("")

    lines.append("### Range Baselines (Model Must Beat)")
    lines.append("")
    for h in horizons:
        if h in range_metrics:
            rm = range_metrics[h]
            lines.append(f"- **{h}m**: Pinball Loss (avg) = {rm.pinball_loss_avg:.6f}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for evaluation (larger = faster, default 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of bars to load (for testing)",
    )
    parser.add_argument(
        "--range-train-split",
        type=str,
        choices=["train", "train+val"],
        default="train",
        help="Split(s) used to fit range baseline quantiles",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Baseline Evaluation")
    print("=" * 70)
    print()

    # Find split files
    print("Finding split files...")
    split_files = find_split_files(SPLITS_DIR)

    if "train" not in split_files:
        print(f"Error: No train file found in {SPLITS_DIR}")
        sys.exit(1)
    if "test" not in split_files:
        print(f"Error: No test file found in {SPLITS_DIR}")
        sys.exit(1)

    print(f"  Train: {split_files['train'].name}")
    if "val" in split_files:
        print(f"  Val: {split_files['val'].name}")
    print(f"  Test: {split_files['test'].name}")
    print()

    # Load data
    print("Loading data...")
    train_bars = load_bars(split_files["train"], limit=args.limit)
    val_bars: List[OhlcvBar] = []
    if "val" in split_files:
        val_bars = load_bars(split_files["val"], limit=args.limit)
    test_bars = load_bars(split_files["test"], limit=args.limit)
    print(f"  Train: {len(train_bars)} bars")
    if val_bars:
        print(f"  Val: {len(val_bars)} bars")
    print(f"  Test: {len(test_bars)} bars")
    print()

    data_periods = {
        "train": summarize_period(train_bars),
        "test": summarize_period(test_bars),
    }
    if val_bars:
        data_periods["val"] = summarize_period(val_bars)

    range_train_bars = train_bars
    if args.range_train_split == "train+val":
        if not val_bars:
            print("Error: --range-train-split train+val requires a val split file.")
            sys.exit(1)
        range_train_bars = train_bars + val_bars

    horizons = HORIZONS

    # Evaluate direction baselines on test set
    print("Evaluating direction baselines on test set...")
    baseline_a_metrics, baseline_b_metrics, label_distributions = evaluate_direction_baselines(
        test_bars,
        horizons=horizons,
        step=args.step,
    )
    print()

    # Evaluate range baseline
    print("Evaluating range baseline...")
    range_metrics = evaluate_range_baseline(
        range_train_bars,
        test_bars,
        horizons=horizons,
        step=args.step,
    )

    # Train range baselines for report
    range_baselines = {}
    for h in horizons:
        range_baselines[h] = train_range_baseline(range_train_bars, h)
    print()

    # Generate reports
    print("Generating reports...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_report = generate_json_report(
        baseline_a_metrics,
        baseline_b_metrics,
        range_metrics,
        range_baselines,
        horizons,
        evaluation_step=args.step,
        data_periods=data_periods,
        range_train_split=args.range_train_split,
    )
    # Add label distributions to JSON report
    json_report["label_distributions"] = {
        str(h): dist for h, dist in label_distributions.items()
    }
    json_path = args.output_dir / "baseline_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    print(f"  JSON report: {json_path}")

    # Markdown report
    md_report = generate_markdown_report(
        baseline_a_metrics,
        baseline_b_metrics,
        range_metrics,
        range_baselines,
        horizons,
        label_distributions,
        data_periods=data_periods,
        range_train_split=args.range_train_split,
        evaluation_step=args.step,
    )
    md_path = args.output_dir / "baseline_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"  Markdown report: {md_path}")
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print("Direction Baselines (on test set):")
    for h in horizons:
        ba = baseline_a_metrics[h]
        bb = baseline_b_metrics[h]
        best = "B" if bb.balanced_accuracy > ba.balanced_accuracy else "A"
        best_acc = max(ba.balanced_accuracy, bb.balanced_accuracy)
        label = "PRIMARY" if h == 30 else "Secondary"
        print(f"  {h}m ({label}): Best = Baseline-{best}, Balanced Accuracy = {best_acc:.4f}")

    print()
    print("Range Baselines:")
    for h in horizons:
        if h in range_metrics:
            rm = range_metrics[h]
            print(f"  {h}m: Pinball Loss (avg) = {rm.pinball_loss_avg:.6f}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
