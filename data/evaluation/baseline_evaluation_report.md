# Baseline Evaluation Report

Generated: 2026-01-28T07:12:18Z

Evaluation step: 1

## Data Periods

- Train: 2024-01-01T22:00:00Z to 2025-07-21T20:37:00Z (bars: 573874)
- Val: 2025-07-21T22:38:00Z to 2025-10-28T19:58:00Z (bars: 101162)
- Test: 2025-10-28T21:59:00Z to 2026-01-26T20:59:00Z (bars: 88321)

## Direction Baselines

### 10-minute (Secondary)

**Label Distribution:**
- up: 19248 (21.8%)
- down: 17987 (20.4%)
- neutral: 145 (0.2%)
- choppy: 50862 (57.6%)

| Metric | Baseline-A | Baseline-B | Better |
|--------|------------|------------|--------|
| Balanced Accuracy | 0.2500 | 0.2619 | B |
| Macro-F1 | 0.0008 | 0.0298 | B |
| Up Recall | 0.0000 | 0.0336 | B |
| Down Recall | 0.0000 | 0.0347 | B |

**Best Baseline: B** (Balanced Accuracy: 0.2619)

**Confusion Matrix (Baseline-A)**

```text
Confusion Matrix (rows=true, cols=predicted):

                  up      down   neutral    choppy
up                 0         0     19248         0
down               0         0     17987         0
neutral            0         0       145         0
choppy             0         0     50862         0
```

**Confusion Matrix (Baseline-B)**

```text
Confusion Matrix (rows=true, cols=predicted):

                  up      down   neutral    choppy
up               647       778     17823         0
down             723       625     16639         0
neutral            1         2       142         0
choppy          1881      2022     46959         0
```

### 30-minute (PRIMARY)

**Label Distribution:**
- up: 11056 (12.5%)
- down: 10181 (11.5%)
- neutral: 6 (0.0%)
- choppy: 66999 (75.9%)

| Metric | Baseline-A | Baseline-B | Better |
|--------|------------|------------|--------|
| Balanced Accuracy | 0.2500 | 0.2265 | A |
| Macro-F1 | 0.0000 | 0.0277 | B |
| Up Recall | 0.0000 | 0.0353 | B |
| Down Recall | 0.0000 | 0.0374 | B |

**Best Baseline: A** (Balanced Accuracy: 0.2500)

**Confusion Matrix (Baseline-A)**

```text
Confusion Matrix (rows=true, cols=predicted):

                  up      down   neutral    choppy
up                 0         0     11056         0
down               0         0     10181         0
neutral            0         0         6         0
choppy             0         0     66999         0
```

**Confusion Matrix (Baseline-B)**

```text
Confusion Matrix (rows=true, cols=predicted):

                  up      down   neutral    choppy
up               390       475     10191         0
down             425       381      9375         0
neutral            1         0         5         0
choppy          2436      2571     61992         0
```

### 60-minute (Secondary)

**Label Distribution:**
- up: 7730 (8.8%)
- down: 6852 (7.8%)
- neutral: 0 (0.0%)
- choppy: 73660 (83.5%)

**Note:** Classes ['neutral'] have no samples. Recall/F1 for these classes are reported as 0.0.

| Metric | Baseline-A | Baseline-B | Better |
|--------|------------|------------|--------|
| Balanced Accuracy | 0.0000 | 0.0190 | B |
| Macro-F1 | 0.0000 | 0.0260 | B |
| Up Recall | 0.0000 | 0.0376 | B |
| Down Recall | 0.0000 | 0.0384 | B |

**Best Baseline: B** (Balanced Accuracy: 0.0190)

**Confusion Matrix (Baseline-A)**

```text
Confusion Matrix (rows=true, cols=predicted):

                  up      down   neutral    choppy
up                 0         0      7730         0
down               0         0      6852         0
neutral            0         0         0         0
choppy             0         0     73660         0
```

**Confusion Matrix (Baseline-B)**

```text
Confusion Matrix (rows=true, cols=predicted):

                  up      down   neutral    choppy
up               291       359      7080         0
down             291       263      6298         0
neutral            0         0         0         0
choppy          2670      2805     68185         0
```

## Range Baselines

- Range baseline training split: train

Note: Coverage values are diagnostic for baseline (PASS/FAIL applies to model).

### 10-minute

- Training Return Quantiles: q10=-0.0533%, q50=0.0006%, q90=0.0530%
- Pinball Loss (avg): 0.013868
  - q10: 0.011145
  - q50: 0.019902
  - q90: 0.010557
- Coverage q10: 0.0556 (target: 0.10 +/- 0.03)
- Coverage q90: 0.9453 (target: 0.90 +/- 0.03)

### 30-minute

- Training Return Quantiles: q10=-0.0930%, q50=0.0013%, q90=0.0929%
- Pinball Loss (avg): 0.023994
  - q10: 0.019451
  - q50: 0.034603
  - q90: 0.017929
- Coverage q10: 0.0553 (target: 0.10 +/- 0.03)
- Coverage q90: 0.9484 (target: 0.90 +/- 0.03)

### 60-minute

- Training Return Quantiles: q10=-0.1328%, q50=0.0020%, q90=0.1329%
- Pinball Loss (avg): 0.033994
  - q10: 0.027709
  - q50: 0.049189
  - q90: 0.025085
- Coverage q10: 0.0556 (target: 0.10 +/- 0.03)
- Coverage q90: 0.9474 (target: 0.90 +/- 0.03)

## Summary

### Direction Baselines (Model Must Exceed)

- **10m (Secondary)**: Best = Baseline-B, Balanced Accuracy = 0.2619
- **30m (PRIMARY)**: Best = Baseline-A, Balanced Accuracy = 0.2500
- **60m (Secondary)**: Best = Baseline-B, Balanced Accuracy = 0.0190

### Range Baselines (Model Must Beat)

- **10m**: Pinball Loss (avg) = 0.013868
- **30m**: Pinball Loss (avg) = 0.023994
- **60m**: Pinball Loss (avg) = 0.033994
