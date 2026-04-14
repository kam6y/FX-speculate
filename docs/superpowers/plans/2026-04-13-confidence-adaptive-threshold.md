# 信頼度推定 + 適応的閾値 実装プラン

> **Status (2026-04-14):** 本プランのうち **AdaptiveThreshold（ボラティリティスケーリング + ABSTAIN）は撤去** された。tune セット (124サンプル) では有意な改善が得られず、コミット `b481f13 refactor: remove adaptive threshold, keep confidence estimation only` で ConfidenceEstimator のみを残す構成にリファクタ済み。以降のタスクのうち `model/adaptive_threshold.py` および関連する `evaluate.py` / `predict.py` の AdaptiveThreshold 呼び出しは **実装されていない**。歴史的経緯として保管しているが、新規参照時は実装コード側（`model/confidence.py`, `scripts/evaluate.py`, `scripts/predict.py`）を優先すること。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TFTの予測出力に信頼度推定と適応的閾値を後段として追加し、方向精度と予測の実用性を向上させる。

**Architecture:** TFTパイプラインは変更せず、推論後の後段処理として信頼度推定（アンサンブル一致度・分位点スプレッド・シグナル強度の合成スコア）と適応的閾値（ATRベースのボラティリティスケーリング＋棄権ゾーン）を追加する。パラメータはtuneセットで最適化し、testセットで検証する。

**Tech Stack:** Python, NumPy, pytest, Streamlit, SQLite

---

## ファイル構成

| ファイル | 責務 |
|---------|------|
| `config.py` | 信頼度・適応閾値のデフォルト定数 |
| `model/confidence.py` | ConfidenceEstimator: 3シグナルの算出と合成 |
| `model/adaptive_threshold.py` | AdaptiveThreshold: ボラティリティスケーリングと棄権判定 |
| `scripts/evaluate.py` | パラメータ最適化（tune set）と統合評価（test set） |
| `scripts/predict.py` | 推論時の信頼度・適応閾値適用、DB保存 |
| `dashboard/app.py` | 信頼度表示パネル追加 |
| `tests/test_confidence.py` | ConfidenceEstimator のユニットテスト |
| `tests/test_adaptive_threshold.py` | AdaptiveThreshold のユニットテスト |

---

### Task 1: config.py に定数を追加

**Files:**
- Modify: `config.py:76-89`

- [ ] **Step 1: 信頼度・適応閾値の定数を追加**

`config.py` の末尾（`DATA_SPLIT_RATIOS` の後）に以下を追加:

```python
# --- Confidence Estimation ---
CONFIDENCE_WEIGHTS = (0.4, 0.3, 0.3)  # (agreement, spread, strength) tune で上書き
SIGNAL_CLIP_MAX = 2.0
CONFIDENCE_COVERAGE_TARGET = 0.70

# --- Adaptive Threshold ---
VOL_CLAMP_MIN = 0.7
VOL_CLAMP_MAX = 1.5
DAMPING_HORIZONS = [1, 2, 3]
DAMPING_FACTOR = 0.5
ABSTAIN_MARGIN = 0.05
```

- [ ] **Step 2: テストが壊れていないことを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/ -v --tb=short -q`
Expected: 既存テストが全て PASS

- [ ] **Step 3: コミット**

```bash
git add config.py
git commit -m "feat: add confidence and adaptive threshold config constants"
```

---

### Task 2: AdaptiveThreshold モジュール (TDD)

**Files:**
- Create: `tests/test_adaptive_threshold.py`
- Create: `model/adaptive_threshold.py`

- [ ] **Step 1: テストを書く**

`tests/test_adaptive_threshold.py`:

```python
"""model/adaptive_threshold.py のテスト。"""

import pytest
from model.adaptive_threshold import AdaptiveThreshold


class TestComputeScaler:
    def test_normal_vol(self):
        at = AdaptiveThreshold({"horizon_1": 0.003})
        assert at.compute_scaler(0.005, 0.005) == pytest.approx(1.0)

    def test_high_vol_clamped(self):
        at = AdaptiveThreshold({"horizon_1": 0.003})
        # raw = 0.010 / 0.005 = 2.0 → clamped to 1.5
        assert at.compute_scaler(0.010, 0.005) == pytest.approx(1.5)

    def test_low_vol_clamped(self):
        at = AdaptiveThreshold({"horizon_1": 0.003})
        # raw = 0.002 / 0.005 = 0.4 → clamped to 0.7
        assert at.compute_scaler(0.002, 0.005) == pytest.approx(0.7)

    def test_moderate_high_vol(self):
        at = AdaptiveThreshold({"horizon_1": 0.003})
        # raw = 0.006 / 0.005 = 1.2 → within clamp range
        assert at.compute_scaler(0.006, 0.005) == pytest.approx(1.2)


class TestGetThreshold:
    def test_h1_damped(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            damping_factor=0.5,
            damping_horizons=[1, 2, 3],
        )
        # raw_scaler = 0.0075 / 0.005 = 1.5, damped = 1 + 0.5*(1.5-1) = 1.25
        result = at.get_threshold(1, current_atr=0.0075, median_atr=0.005)
        assert result == pytest.approx(0.003 * 1.25)

    def test_h5_not_damped(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_5": 0.003},
            damping_factor=0.5,
            damping_horizons=[1, 2, 3],
        )
        # raw_scaler = 0.0075 / 0.005 = 1.5, no damping for H5
        result = at.get_threshold(5, current_atr=0.0075, median_atr=0.005)
        assert result == pytest.approx(0.003 * 1.5)

    def test_normal_vol_no_change(self):
        at = AdaptiveThreshold(base_thresholds={"horizon_1": 0.003})
        result = at.get_threshold(1, current_atr=0.005, median_atr=0.005)
        assert result == pytest.approx(0.003)


class TestClassify:
    def test_up(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            abstain_margin=0.05,
        )
        assert at.classify(0.005, 1, 0.005, 0.005) == "UP"

    def test_down(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            abstain_margin=0.05,
        )
        assert at.classify(0.001, 1, 0.005, 0.005) == "DOWN"

    def test_abstain_near_threshold(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            abstain_margin=0.10,
        )
        # threshold=0.003, zone=0.003*0.10=0.0003
        # signal=0.00305 → |0.00305-0.003|=0.00005 < 0.0003 → ABSTAIN
        assert at.classify(0.00305, 1, 0.005, 0.005) == "ABSTAIN"

    def test_just_outside_abstain_zone(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            abstain_margin=0.01,
        )
        # threshold=0.003, zone=0.003*0.01=0.00003
        # signal=0.0031 → |0.0031-0.003|=0.0001 > 0.00003 → UP
        assert at.classify(0.0031, 1, 0.005, 0.005) == "UP"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/test_adaptive_threshold.py -v`
Expected: FAIL (ImportError: model.adaptive_threshold)

- [ ] **Step 3: 実装を書く**

`model/adaptive_threshold.py`:

```python
"""適応的閾値: ボラティリティに応じて方向判定閾値を動的調整する。"""

from config import (
    VOL_CLAMP_MIN,
    VOL_CLAMP_MAX,
    DAMPING_HORIZONS,
    DAMPING_FACTOR,
    ABSTAIN_MARGIN,
)


class AdaptiveThreshold:
    """ATR ベースのボラティリティスケーリングと棄権ゾーンによる方向判定。"""

    def __init__(
        self,
        base_thresholds: dict[str, float],
        vol_clamp_min: float = VOL_CLAMP_MIN,
        vol_clamp_max: float = VOL_CLAMP_MAX,
        damping_horizons: list[int] | None = None,
        damping_factor: float = DAMPING_FACTOR,
        abstain_margin: float = ABSTAIN_MARGIN,
    ):
        self.base_thresholds = base_thresholds
        self.vol_clamp_min = vol_clamp_min
        self.vol_clamp_max = vol_clamp_max
        self.damping_horizons = damping_horizons if damping_horizons is not None else DAMPING_HORIZONS
        self.damping_factor = damping_factor
        self.abstain_margin = abstain_margin

    def compute_scaler(self, current_atr: float, median_atr: float) -> float:
        """ATR 比率からボラティリティスケーラーを算出する。"""
        if median_atr <= 0:
            return 1.0
        raw = current_atr / median_atr
        return max(self.vol_clamp_min, min(self.vol_clamp_max, raw))

    def get_threshold(
        self, horizon: int, current_atr: float, median_atr: float,
    ) -> float:
        """ホライゾン別の適応的閾値を返す。"""
        scaler = self.compute_scaler(current_atr, median_atr)
        if horizon in self.damping_horizons:
            scaler = 1.0 + self.damping_factor * (scaler - 1.0)
        base = self.base_thresholds[f"horizon_{horizon}"]
        return base * scaler

    def classify(
        self,
        direction_signal: float,
        horizon: int,
        current_atr: float,
        median_atr: float,
    ) -> str:
        """方向を判定する。閾値近傍は ABSTAIN を返す。"""
        threshold = self.get_threshold(horizon, current_atr, median_atr)
        abstain_zone = abs(threshold) * self.abstain_margin
        if abs(direction_signal - threshold) < abstain_zone:
            return "ABSTAIN"
        return "UP" if direction_signal > threshold else "DOWN"
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/test_adaptive_threshold.py -v`
Expected: 全て PASS

- [ ] **Step 5: コミット**

```bash
git add model/adaptive_threshold.py tests/test_adaptive_threshold.py
git commit -m "feat: add AdaptiveThreshold module with volatility scaling and abstain zone"
```

---

### Task 3: ConfidenceEstimator モジュール (TDD)

**Files:**
- Create: `tests/test_confidence.py`
- Create: `model/confidence.py`

- [ ] **Step 1: テストを書く**

`tests/test_confidence.py`:

```python
"""model/confidence.py のテスト。"""

import numpy as np
import pytest
from model.confidence import ConfidenceEstimator


class TestEnsembleAgreement:
    def test_full_agreement_up(self):
        ce = ConfidenceEstimator()
        signals = [0.004, 0.005, 0.003, 0.006, 0.0035]
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(1.0)

    def test_full_agreement_down(self):
        ce = ConfidenceEstimator()
        signals = [0.001, 0.002, 0.0015, 0.0005, 0.0025]
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(1.0)

    def test_split_3_2(self):
        ce = ConfidenceEstimator()
        signals = [0.004, 0.005, 0.002, 0.001, 0.006]
        # 3 UP, 2 DOWN → 3/5 = 0.6
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(0.6)

    def test_split_4_1(self):
        ce = ConfidenceEstimator()
        signals = [0.004, 0.005, 0.002, 0.006, 0.0035]
        # 4 UP, 1 DOWN → 4/5 = 0.8
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(0.8)


class TestSpreadScore:
    def test_no_calibration_returns_half(self):
        ce = ConfidenceEstimator()
        assert ce.spread_score(0.01, -0.01) == pytest.approx(0.5)

    def test_smallest_spread_high_score(self):
        percentiles = np.array([0.005, 0.010, 0.015, 0.020])
        ce = ConfidenceEstimator(spread_percentiles=percentiles)
        # spread=0.003 < min(percentiles) → rank=0 → score=1.0
        assert ce.spread_score(0.002, -0.001) == pytest.approx(1.0)

    def test_largest_spread_low_score(self):
        percentiles = np.array([0.005, 0.010, 0.015, 0.020])
        ce = ConfidenceEstimator(spread_percentiles=percentiles)
        # spread=0.030 > max → rank=1.0 → score=0.0
        assert ce.spread_score(0.020, -0.010) == pytest.approx(0.0)

    def test_mid_spread(self):
        percentiles = np.array([0.005, 0.010, 0.015, 0.020])
        ce = ConfidenceEstimator(spread_percentiles=percentiles)
        # spread=0.012 → between index 1 and 2 → rank=2/4=0.5 → score=0.5
        assert ce.spread_score(0.006, -0.006) == pytest.approx(0.5)


class TestSignalStrength:
    def test_at_threshold(self):
        ce = ConfidenceEstimator()
        assert ce.signal_strength(0.003, 0.003) == pytest.approx(0.0)

    def test_far_from_threshold(self):
        ce = ConfidenceEstimator(signal_clip_max=2.0)
        # |0.009 - 0.003| / 0.003 = 2.0 → clipped to 2.0 → normalized = 1.0
        assert ce.signal_strength(0.009, 0.003) == pytest.approx(1.0)

    def test_moderate_distance(self):
        ce = ConfidenceEstimator(signal_clip_max=2.0)
        # |0.006 - 0.003| / 0.003 = 1.0 → 1.0/2.0 = 0.5
        assert ce.signal_strength(0.006, 0.003) == pytest.approx(0.5)


class TestCompositeScore:
    def test_all_high(self):
        ce = ConfidenceEstimator(
            weights=(0.4, 0.3, 0.3),
            signal_clip_max=2.0,
        )
        # agreement=1.0, spread=0.5 (uncalibrated), strength≈1.0
        score = ce.score(
            per_model_signals=[0.005, 0.006, 0.004, 0.007, 0.005],
            threshold=0.003,
            q90=0.01,
            q10=-0.01,
            direction_signal=0.009,
        )
        # 0.4*1.0 + 0.3*0.5 + 0.3*1.0 = 0.85
        assert score == pytest.approx(0.85)

    def test_weights_sum_to_one(self):
        ce = ConfidenceEstimator(weights=(0.5, 0.3, 0.2))
        assert sum(ce.weights) == pytest.approx(1.0)


class TestClassifyLevel:
    def test_high(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.8) == "HIGH"

    def test_medium(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.5) == "MEDIUM"

    def test_low(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.3) == "LOW"

    def test_boundary_high(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.7) == "HIGH"

    def test_boundary_medium(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.4) == "MEDIUM"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/test_confidence.py -v`
Expected: FAIL (ImportError: model.confidence)

- [ ] **Step 3: 実装を書く**

`model/confidence.py`:

```python
"""信頼度推定: アンサンブル一致度・分位点スプレッド・シグナル強度の合成スコア。"""

import numpy as np

from config import CONFIDENCE_WEIGHTS, SIGNAL_CLIP_MAX


class ConfidenceEstimator:
    """予測の信頼度を3つのシグナルの加重平均で推定する。"""

    def __init__(
        self,
        weights: tuple[float, float, float] = CONFIDENCE_WEIGHTS,
        spread_percentiles: np.ndarray | None = None,
        signal_clip_max: float = SIGNAL_CLIP_MAX,
        confidence_boundaries: tuple[float, float] = (0.33, 0.66),
    ):
        self.weights = weights
        self.spread_percentiles = spread_percentiles
        self.signal_clip_max = signal_clip_max
        self.confidence_boundaries = confidence_boundaries

    def ensemble_agreement(
        self, per_model_signals: list[float], threshold: float,
    ) -> float:
        """5モデルの方向一致度を返す (0.6〜1.0)。"""
        n_up = sum(1 for s in per_model_signals if s > threshold)
        n_down = len(per_model_signals) - n_up
        return max(n_up, n_down) / len(per_model_signals)

    def spread_score(self, q90: float, q10: float) -> float:
        """分位点スプレッドを信頼度に変換する (0〜1, 小さいほど高信頼)。"""
        spread = q90 - q10
        if self.spread_percentiles is None:
            return 0.5
        rank = np.searchsorted(self.spread_percentiles, spread) / len(
            self.spread_percentiles
        )
        return 1.0 - float(rank)

    def signal_strength(self, direction_signal: float, threshold: float) -> float:
        """シグナルと閾値の距離を正規化した強度 (0〜1)。"""
        raw = abs(direction_signal - threshold) / max(abs(threshold), 1e-8)
        clipped = min(raw, self.signal_clip_max)
        return clipped / self.signal_clip_max

    def score(
        self,
        per_model_signals: list[float],
        threshold: float,
        q90: float,
        q10: float,
        direction_signal: float,
    ) -> float:
        """合成信頼スコアを返す (0〜1)。"""
        w_agr, w_spr, w_str = self.weights
        agr = self.ensemble_agreement(per_model_signals, threshold)
        spr = self.spread_score(q90, q10)
        stg = self.signal_strength(direction_signal, threshold)
        return w_agr * agr + w_spr * spr + w_str * stg

    def classify_level(self, confidence_score: float) -> str:
        """信頼スコアを HIGH/MEDIUM/LOW に分類する。"""
        low_boundary, high_boundary = self.confidence_boundaries
        if confidence_score >= high_boundary:
            return "HIGH"
        if confidence_score >= low_boundary:
            return "MEDIUM"
        return "LOW"
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/test_confidence.py -v`
Expected: 全て PASS

- [ ] **Step 5: コミット**

```bash
git add model/confidence.py tests/test_confidence.py
git commit -m "feat: add ConfidenceEstimator module with ensemble agreement, spread, and signal strength"
```

---

### Task 4: ensemble_predict を拡張してモデル別シグナルを返す

**Files:**
- Modify: `scripts/evaluate.py:69-101`

- [ ] **Step 1: ensemble_predict に per_model_signals を追加**

`scripts/evaluate.py` の `ensemble_predict` 関数の `result` dict に1行追加する:

```python
def ensemble_predict(
    models: list[TemporalFusionTransformer],
    dataloader,
) -> dict:
    """top-k モデルのアンサンブル予測。

    - median: q50 平均
    - q10: 各モデルの min
    - q90: 各モデルの max
    - direction_signal: 全分位点の加重平均（分布の歪み情報を活用）
    - per_model_signals: モデル別の direction_signal
    """
    all_preds = []
    for model in models:
        preds = model.predict(dataloader, mode="quantiles", return_x=False)
        all_preds.append(preds)

    stacked = torch.stack(all_preds)  # (n_models, batch, horizon, quantiles)
    q_mid = QUANTILES.index(0.5)

    quantile_weights = torch.tensor(QUANTILE_SIGNAL_WEIGHTS, device=stacked.device)
    model_mean = stacked.mean(dim=0)  # (batch, horizon, quantiles)
    direction_signal = (model_mean * quantile_weights).sum(dim=-1)

    per_model_signals = (stacked * quantile_weights).sum(dim=-1)  # (n_models, batch, horizon)

    q10_idx = QUANTILES.index(0.1)
    q90_idx = QUANTILES.index(0.9)

    result = {
        "median": model_mean[..., q_mid],
        "q10": stacked[:, :, :, q10_idx].min(dim=0).values,
        "q90": stacked[:, :, :, q90_idx].max(dim=0).values,
        "direction_signal": direction_signal,
        "per_model_signals": per_model_signals,
    }
    return result
```

変更点は `per_model_signals` の算出行と `result` dict への追加のみ。

- [ ] **Step 2: 既存テストが壊れていないことを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/ -v --tb=short -q`
Expected: 全て PASS

- [ ] **Step 3: コミット**

```bash
git add scripts/evaluate.py
git commit -m "feat: extend ensemble_predict to return per-model direction signals"
```

---

### Task 5: evaluate.py にパラメータ最適化と統合評価を追加

**Files:**
- Modify: `scripts/evaluate.py`

- [ ] **Step 1: import を追加**

`scripts/evaluate.py` の import ブロックに以下を追加:

```python
from model.confidence import ConfidenceEstimator
from model.adaptive_threshold import AdaptiveThreshold
from config import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    TOP_K_CHECKPOINTS,
    QUANTILES,
    QUANTILE_SIGNAL_WEIGHTS,
    CONFIDENCE_COVERAGE_TARGET,
    SIGNAL_CLIP_MAX,
    VOL_CLAMP_MIN,
    VOL_CLAMP_MAX,
    DAMPING_HORIZONS,
    DAMPING_FACTOR,
)
```

- [ ] **Step 2: ATR 抽出ヘルパーを追加**

`find_optimal_threshold` 関数の後に以下を追加:

```python
def extract_sample_atrs(
    df: pd.DataFrame, n_samples: int,
) -> np.ndarray:
    """データセットの各サンプルに対応する ATR 値を抽出する。

    各サンプル i のエンコーダ最終位置は df.iloc[i + ENCODER_LENGTH - 1]。
    """
    atrs = []
    for i in range(n_samples):
        idx = i + ENCODER_LENGTH - 1
        if idx < len(df):
            atrs.append(df.iloc[idx]["atr"])
        else:
            atrs.append(df["atr"].iloc[-1])
    return np.array(atrs)
```

- [ ] **Step 3: 信頼度パラメータ最適化関数を追加**

```python
def optimize_confidence_params(
    preds: dict,
    actuals: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    """tune セットで信頼度の重みを最適化する。

    Args:
        preds: ensemble_predict の出力 (per_model_signals, q10, q90, direction_signal)
        actuals: (n_samples, horizon) の実績値
        thresholds: ホライゾン別の基本閾値

    Returns:
        最適化されたパラメータ dict
    """
    n_samples = preds["direction_signal"].shape[0]

    # スプレッドのパーセンタイル分布を各ホライゾンで算出
    all_spreads = {}
    for h in range(PREDICTION_LENGTH):
        spreads = (preds["q90"][:, h] - preds["q10"][:, h]).numpy()
        all_spreads[h] = np.sort(spreads)

    best_score = -1.0
    best_weights = (0.34, 0.33, 0.33)

    for w1_int in range(11):
        w1 = w1_int * 0.1
        for w2_int in range(11 - w1_int):
            w2 = w2_int * 0.1
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -0.001:
                continue

            total_correct = 0
            total_selected = 0

            for h in range(PREDICTION_LENGTH):
                threshold = thresholds[f"horizon_{h+1}"]
                actuals_h = actuals[:n_samples, h]
                actual_up = actuals_h > 0

                ce = ConfidenceEstimator(
                    weights=(w1, w2, w3),
                    spread_percentiles=all_spreads[h],
                    signal_clip_max=SIGNAL_CLIP_MAX,
                )

                scores = []
                for i in range(n_samples):
                    per_model = preds["per_model_signals"][:, i, h].tolist()
                    s = ce.score(
                        per_model_signals=per_model,
                        threshold=threshold,
                        q90=float(preds["q90"][i, h]),
                        q10=float(preds["q10"][i, h]),
                        direction_signal=float(preds["direction_signal"][i, h]),
                    )
                    scores.append(s)

                scores = np.array(scores)
                cutoff = np.percentile(scores, (1 - CONFIDENCE_COVERAGE_TARGET) * 100)
                selected = scores >= cutoff

                dir_signals = preds["direction_signal"][:, h].numpy()
                pred_up = dir_signals > threshold
                correct = pred_up[selected] == actual_up[selected]

                total_correct += correct.sum()
                total_selected += selected.sum()

            if total_selected > 0:
                acc = total_correct / total_selected
                if acc > best_score:
                    best_score = acc
                    best_weights = (w1, w2, w3)

    # 信頼度スコアの境界値を算出 (tune セットの 33/66 パーセンタイル)
    all_scores = []
    for h in range(PREDICTION_LENGTH):
        threshold = thresholds[f"horizon_{h+1}"]
        ce = ConfidenceEstimator(
            weights=best_weights,
            spread_percentiles=all_spreads[h],
            signal_clip_max=SIGNAL_CLIP_MAX,
        )
        for i in range(n_samples):
            per_model = preds["per_model_signals"][:, i, h].tolist()
            s = ce.score(
                per_model_signals=per_model,
                threshold=threshold,
                q90=float(preds["q90"][i, h]),
                q10=float(preds["q10"][i, h]),
                direction_signal=float(preds["direction_signal"][i, h]),
            )
            all_scores.append(s)

    all_scores = np.array(all_scores)
    low_boundary = float(np.percentile(all_scores, 33))
    high_boundary = float(np.percentile(all_scores, 66))

    # スプレッドパーセンタイルを JSON 用にリスト化
    spread_pcts = {
        f"horizon_{h+1}": all_spreads[h].tolist()
        for h in range(PREDICTION_LENGTH)
    }

    return {
        "weights": list(best_weights),
        "spread_percentiles": spread_pcts,
        "confidence_boundaries": [low_boundary, high_boundary],
        "best_coverage_accuracy": float(best_score),
    }
```

- [ ] **Step 4: 適応閾値パラメータ最適化関数を追加**

```python
def optimize_adaptive_params(
    preds: dict,
    actuals: np.ndarray,
    thresholds: dict[str, float],
    sample_atrs: np.ndarray,
    train_atr_median: float,
) -> dict:
    """tune セットで abstain_margin を最適化する。

    Args:
        preds: ensemble_predict の出力
        actuals: (n_samples, horizon) の実績値
        thresholds: ホライゾン別の基本閾値
        sample_atrs: 各サンプルの ATR 値
        train_atr_median: 学習データの ATR 中央値

    Returns:
        最適化されたパラメータ dict
    """
    n_samples = preds["direction_signal"].shape[0]
    best_score = -np.inf
    best_margin = 0.0

    for margin_int in range(21):  # 0.00 ~ 0.20
        margin = margin_int * 0.01
        at = AdaptiveThreshold(
            base_thresholds=thresholds,
            abstain_margin=margin,
        )

        total_correct = 0
        total_judged = 0
        total_pred_up = 0
        total_actual_up = 0

        for h in range(PREDICTION_LENGTH):
            actuals_h = actuals[:n_samples, h]

            for i in range(n_samples):
                direction = at.classify(
                    float(preds["direction_signal"][i, h]),
                    h + 1,
                    float(sample_atrs[i]),
                    train_atr_median,
                )
                if direction == "ABSTAIN":
                    continue

                actual_up = actuals_h[i] > 0
                pred_up = direction == "UP"
                total_correct += int(pred_up == actual_up)
                total_judged += 1
                total_pred_up += int(pred_up)
                total_actual_up += int(actual_up)

        if total_judged == 0:
            continue

        accuracy = total_correct / total_judged
        pred_up_ratio = total_pred_up / total_judged
        actual_up_ratio = total_actual_up / total_judged
        ratio_gap = abs(pred_up_ratio - actual_up_ratio)
        score = accuracy - 1.0 * ratio_gap

        if score > best_score:
            best_score = score
            best_margin = margin

    return {
        "abstain_margin": best_margin,
        "train_atr_median": train_atr_median,
        "best_score": float(best_score),
    }
```

- [ ] **Step 5: evaluate() 関数を拡張**

`evaluate()` の閾値チューニングブロック（`print("=== 閾値チューニング ===")` 以降）を以下に置き換える:

```python
    # --- 閾値チューニング (tune セットで実施) ---
    print("=== 閾値チューニング ===")
    tune_dataset = training.from_dataset(training, tune_df, stop_randomization=True)
    tune_loader = tune_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0,
    )
    tune_preds = ensemble_predict(models, tune_loader)

    thresholds = {}
    tune_actual_all = torch.stack([
        y for (_, (y, _)) in tune_loader.dataset
    ])
    for h in range(PREDICTION_LENGTH):
        preds_h = tune_preds["direction_signal"][:, h].numpy()
        actuals_h = tune_actual_all[:len(preds_h), h].numpy()
        thresholds[f"horizon_{h+1}"] = float(find_optimal_threshold(preds_h, actuals_h))

    print(f"  Thresholds: {thresholds}")

    threshold_path = ARTIFACT_DIR / "thresholds.json"
    with open(threshold_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    # --- 信頼度パラメータ最適化 ---
    print("=== 信頼度パラメータ最適化 ===")
    n_tune_samples = tune_preds["direction_signal"].shape[0]
    tune_actuals_np = tune_actual_all[:n_tune_samples].numpy()

    confidence_params = optimize_confidence_params(
        tune_preds, tune_actuals_np, thresholds,
    )
    print(f"  Best weights: {confidence_params['weights']}")
    print(f"  Coverage accuracy: {confidence_params['best_coverage_accuracy']:.3f}")

    conf_path = ARTIFACT_DIR / "confidence_params.json"
    with open(conf_path, "w") as f:
        json.dump(confidence_params, f, indent=2)

    # --- 適応閾値パラメータ最適化 ---
    print("=== 適応閾値パラメータ最適化 ===")
    train_atr_median = float(train_df["atr"].median())
    tune_atrs = extract_sample_atrs(tune_df, n_tune_samples)

    adaptive_params = optimize_adaptive_params(
        tune_preds, tune_actuals_np, thresholds, tune_atrs, train_atr_median,
    )
    print(f"  Best abstain_margin: {adaptive_params['abstain_margin']}")
    print(f"  Best score: {adaptive_params['best_score']:.3f}")

    adaptive_path = ARTIFACT_DIR / "adaptive_threshold_params.json"
    with open(adaptive_path, "w") as f:
        json.dump(adaptive_params, f, indent=2)
```

- [ ] **Step 6: テストセット評価ブロックを拡張**

`evaluate()` の `print("=== テストセット評価 ===")` 以降のブロックを以下に置き換える:

```python
    # --- テストセットで評価 ---
    print("=== テストセット評価 ===")
    test_dataset = training.from_dataset(training, test_df, stop_randomization=True)
    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0,
    )
    test_preds = ensemble_predict(models, test_loader)

    report = {"thresholds": thresholds, "horizons": {}}

    actual_all = torch.stack([
        y for (_, (y, _)) in test_loader.dataset
    ])

    n_test_samples = test_preds["direction_signal"].shape[0]
    test_atrs = extract_sample_atrs(test_df, n_test_samples)

    # 信頼度・適応閾値のインスタンス生成
    at = AdaptiveThreshold(
        base_thresholds=thresholds,
        abstain_margin=adaptive_params["abstain_margin"],
    )

    for h in range(PREDICTION_LENGTH):
        median_h = test_preds["median"][:, h].numpy()
        dir_signal_h = test_preds["direction_signal"][:, h].numpy()
        actuals_h = actual_all[:n_test_samples, h].numpy()

        mae = float(np.abs(median_h - actuals_h).mean())
        rmse = float(np.sqrt(((median_h - actuals_h) ** 2).mean()))

        # ベースライン方向精度 (固定閾値)
        threshold = thresholds[f"horizon_{h+1}"]
        preds_thresholded = (dir_signal_h > threshold).astype(float)
        baseline_metrics = compute_direction_metrics(actuals_h, preds_thresholded)

        # 適応閾値 + 信頼度による評価
        spread_pcts = np.array(confidence_params["spread_percentiles"][f"horizon_{h+1}"])
        ce = ConfidenceEstimator(
            weights=tuple(confidence_params["weights"]),
            spread_percentiles=spread_pcts,
            signal_clip_max=SIGNAL_CLIP_MAX,
            confidence_boundaries=tuple(confidence_params["confidence_boundaries"]),
        )

        n_judged = 0
        n_correct = 0
        n_high_conf = 0
        n_high_conf_correct = 0
        n_tradeable = 0
        n_tradeable_correct = 0

        for i in range(n_test_samples):
            direction = at.classify(
                float(dir_signal_h[i]), h + 1,
                float(test_atrs[i]), train_atr_median,
            )
            per_model = test_preds["per_model_signals"][:, i, h].tolist()
            conf_score = ce.score(
                per_model, threshold,
                float(test_preds["q90"][i, h]),
                float(test_preds["q10"][i, h]),
                float(dir_signal_h[i]),
            )
            conf_level = ce.classify_level(conf_score)
            actual_up = actuals_h[i] > 0

            if direction != "ABSTAIN":
                pred_up = direction == "UP"
                n_judged += 1
                n_correct += int(pred_up == actual_up)

            if conf_level == "HIGH":
                n_high_conf += 1
                pred_up_base = dir_signal_h[i] > threshold
                n_high_conf_correct += int(pred_up_base == actual_up)

            should_trade = direction != "ABSTAIN" and conf_level != "LOW"
            if should_trade:
                pred_up = direction == "UP"
                n_tradeable += 1
                n_tradeable_correct += int(pred_up == actual_up)

        report["horizons"][f"horizon_{h+1}"] = {
            "mae": mae,
            "rmse": rmse,
            **baseline_metrics,
            "adaptive_accuracy": float(n_correct / n_judged) if n_judged > 0 else 0.0,
            "adaptive_coverage": float(n_judged / n_test_samples),
            "high_conf_accuracy": float(n_high_conf_correct / n_high_conf) if n_high_conf > 0 else 0.0,
            "high_conf_count": n_high_conf,
            "tradeable_accuracy": float(n_tradeable_correct / n_tradeable) if n_tradeable > 0 else 0.0,
            "tradeable_coverage": float(n_tradeable / n_test_samples),
        }
        print(f"  Horizon {h+1}: MAE={mae:.6f}, RMSE={rmse:.6f}, "
              f"BaselineAcc={baseline_metrics['direction_accuracy']:.3f}, "
              f"AdaptiveAcc={report['horizons'][f'horizon_{h+1}']['adaptive_accuracy']:.3f} "
              f"(coverage={report['horizons'][f'horizon_{h+1}']['adaptive_coverage']:.1%}), "
              f"HighConfAcc={report['horizons'][f'horizon_{h+1}']['high_conf_accuracy']:.3f} "
              f"(n={n_high_conf}), "
              f"TradeableAcc={report['horizons'][f'horizon_{h+1}']['tradeable_accuracy']:.3f} "
              f"(coverage={report['horizons'][f'horizon_{h+1}']['tradeable_coverage']:.1%})")
```

- [ ] **Step 7: テストが壊れていないことを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/ -v --tb=short -q`
Expected: 全て PASS

- [ ] **Step 8: コミット**

```bash
git add scripts/evaluate.py
git commit -m "feat: add confidence and adaptive threshold optimization to evaluation pipeline"
```

---

### Task 6: predict.py を更新

**Files:**
- Modify: `scripts/predict.py`

- [ ] **Step 1: import を追加**

```python
from model.confidence import ConfidenceEstimator
from model.adaptive_threshold import AdaptiveThreshold
```

- [ ] **Step 2: DB マイグレーションに新カラムを追加**

`migrate_db` 関数の `for col, col_type in [...]` リストに追加:

```python
def migrate_db(db_path) -> None:
    """predictions テーブルに新カラムを idempotent に追加する。"""
    with sqlite3.connect(str(db_path)) as conn:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        for col, col_type in [
            ("actual_return", "REAL"),
            ("actual_direction", "TEXT"),
            ("is_correct", "INTEGER"),
            ("confidence_score", "REAL"),
            ("confidence_level", "TEXT"),
            ("should_trade", "INTEGER"),
            ("adaptive_threshold", "REAL"),
        ]:
            if col not in existing:
                conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {col} {col_type}"
                )
```

- [ ] **Step 3: predict_daily のパラメータロードと推論を更新**

`predict_daily()` の「閾値ロード」部分以降を以下に置き換え:

```python
    # 閾値ロード
    threshold_path = ARTIFACT_DIR / "thresholds.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = {f"horizon_{h+1}": 0.0 for h in range(PREDICTION_LENGTH)}

    # 信頼度パラメータロード
    conf_path = ARTIFACT_DIR / "confidence_params.json"
    if conf_path.exists():
        with open(conf_path) as f:
            confidence_params = json.load(f)
    else:
        confidence_params = None

    # 適応閾値パラメータロード
    adaptive_path = ARTIFACT_DIR / "adaptive_threshold_params.json"
    if adaptive_path.exists():
        with open(adaptive_path) as f:
            adaptive_params = json.load(f)
    else:
        adaptive_params = None

    # 現在の ATR を取得
    current_atr = float(features["atr"].iloc[-1])
    train_atr_median = adaptive_params["train_atr_median"] if adaptive_params else current_atr

    # AdaptiveThreshold インスタンス
    at = AdaptiveThreshold(
        base_thresholds=thresholds,
        abstain_margin=adaptive_params["abstain_margin"] if adaptive_params else 0.0,
    )

    # 結果を整形
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        periods=PREDICTION_LENGTH,
    )

    results = []
    for h in range(PREDICTION_LENGTH):
        base_threshold = thresholds.get(f"horizon_{h+1}", 0.0)
        median_val = float(preds["median"][0, h])
        dir_signal = float(preds["direction_signal"][0, h])

        # 適応閾値による方向判定
        direction = at.classify(dir_signal, h + 1, current_atr, train_atr_median)
        adapt_thresh = at.get_threshold(h + 1, current_atr, train_atr_median)

        # 信頼度スコア
        if confidence_params is not None:
            spread_pcts = np.array(
                confidence_params["spread_percentiles"][f"horizon_{h+1}"]
            )
            ce = ConfidenceEstimator(
                weights=tuple(confidence_params["weights"]),
                spread_percentiles=spread_pcts,
                confidence_boundaries=tuple(confidence_params["confidence_boundaries"]),
            )
            per_model = preds["per_model_signals"][:, 0, h].tolist()
            conf_score = ce.score(
                per_model, base_threshold,
                float(preds["q90"][0, h]),
                float(preds["q10"][0, h]),
                dir_signal,
            )
            conf_level = ce.classify_level(conf_score)
        else:
            conf_score = 0.0
            conf_level = "MEDIUM"

        should_trade = direction != "ABSTAIN" and conf_level != "LOW"

        results.append({
            "prediction_date": str(date.today()),
            "target_date": str(future_dates[h].date()),
            "horizon": h + 1,
            "median": median_val,
            "direction_signal": dir_signal,
            "q10": float(preds["q10"][0, h]),
            "q90": float(preds["q90"][0, h]),
            "threshold": base_threshold,
            "direction": direction,
            "confidence_score": round(conf_score, 4),
            "confidence_level": conf_level,
            "should_trade": int(should_trade),
            "adaptive_threshold": round(adapt_thresh, 6),
        })

    results_df = pd.DataFrame(results)
    print("\n=== 予測結果 ===")
    print(results_df.to_string(index=False))

    # SQLite に保存
    PREDICTIONS_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(PREDICTIONS_DB)) as conn:
        results_df.to_sql("predictions", conn, if_exists="append", index=False)
    print(f"\n  Saved to {PREDICTIONS_DB}")
```

- [ ] **Step 4: 既存テストが壊れていないことを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/ -v --tb=short -q`
Expected: 全て PASS

- [ ] **Step 5: コミット**

```bash
git add scripts/predict.py
git commit -m "feat: apply confidence scoring and adaptive thresholds in daily prediction"
```

---

### Task 7: ダッシュボードに信頼度パネルを追加

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: panel_direction_signals を拡張して信頼度を表示**

`dashboard/app.py` の `panel_direction_signals` 関数を以下に置き換え:

```python
def panel_direction_signals(preds: pd.DataFrame) -> None:
    """方向シグナル: ホライゾン別 UP/DOWN/ABSTAIN を信頼度付きで表示。"""
    st.subheader("方向シグナル")
    st.caption(
        "各ホライゾン (1〜5 営業日先) の方向予測。"
        "全分位点の加重平均 (direction signal) が適応閾値を上回れば UP、下回れば DOWN、"
        "閾値近傍なら ABSTAIN と判定する。信頼度 (HIGH/MEDIUM/LOW) はアンサンブル一致度・"
        "予測不確実性・シグナル強度の合成スコア。"
    )
    if preds.empty:
        st.info("データがありません。")
        return

    latest = _get_latest_predictions(preds)
    if latest.empty:
        st.info("データがありません。")
        return

    cols = st.columns(PREDICTION_LENGTH)
    for i, (_, row) in enumerate(latest.head(PREDICTION_LENGTH).iterrows()):
        direction = row.get("direction", "N/A")
        median_val = row.get("median", 0.0)
        dir_signal = row.get("direction_signal", median_val)
        conf_level = row.get("confidence_level", "")
        conf_score = row.get("confidence_score", None)
        should_trade = row.get("should_trade", None)

        if direction == "UP":
            arrow, delta_color = "↑", "normal"
        elif direction == "DOWN":
            arrow, delta_color = "↓", "inverse"
        else:
            arrow, delta_color = "－", "off"

        label = f"{int(row['horizon'])}日後 ({row['target_date'].strftime('%m/%d')})"
        cols[i].metric(
            label=label,
            value=f"{arrow} {direction}",
            delta=f"signal={dir_signal:.5f}",
            delta_color=delta_color,
        )
        if conf_level:
            conf_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf_level, "")
            trade_label = "Trade" if should_trade else "Skip"
            cols[i].caption(f"{conf_color} 信頼度: {conf_level} ({trade_label})")
```

- [ ] **Step 2: panel_confidence_detail パネルを追加**

`panel_accuracy_history` 関数の前に以下を追加:

```python
def panel_confidence_detail(preds: pd.DataFrame) -> None:
    """信頼度詳細: 過去予測の信頼度別的中率。"""
    st.subheader("信頼度別パフォーマンス")
    st.caption(
        "過去の予測を信頼度レベル (HIGH/MEDIUM/LOW) 別に集計した的中率。"
        "HIGH の的中率が全体平均を上回っていれば、信頼度推定が有効に機能している。"
    )

    if preds.empty or "confidence_level" not in preds.columns:
        st.info("信頼度データがありません。最新の evaluate.py → predict.py を実行してください。")
        return

    filled = preds[preds["is_correct"].notna() & preds["confidence_level"].notna()].copy()
    if filled.empty:
        st.info("実績データがまだありません。")
        return

    rows = []
    for level in ["HIGH", "MEDIUM", "LOW"]:
        subset = filled[filled["confidence_level"] == level]
        if subset.empty:
            rows.append({"信頼度": level, "件数": 0, "的中率": "N/A"})
        else:
            acc = subset["is_correct"].mean()
            rows.append({
                "信頼度": level,
                "件数": len(subset),
                "的中率": f"{acc:.1%}",
            })

    # should_trade の成績
    tradeable = filled[filled["should_trade"] == 1]
    if not tradeable.empty:
        trade_acc = tradeable["is_correct"].mean()
        rows.append({
            "信頼度": "Trade対象のみ",
            "件数": len(tradeable),
            "的中率": f"{trade_acc:.1%}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
```

- [ ] **Step 3: main() に信頼度パネルを組み込む**

`main()` の `panel_prediction_tracking(preds_df)` の後に信頼度パネルを追加:

```python
    # --- 予測 vs 実績 ---
    panel_prediction_tracking(preds_df)
    st.divider()

    # --- 信頼度別パフォーマンス ---
    panel_confidence_detail(preds_df)
    st.divider()
```

- [ ] **Step 4: 既存テストが壊れていないことを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/ -v --tb=short -q`
Expected: 全て PASS

- [ ] **Step 5: コミット**

```bash
git add dashboard/app.py
git commit -m "feat: add confidence level display and performance panel to dashboard"
```

---

### Task 8: 統合テスト

**Files:**
- Modify: `tests/test_integration.py` (必要に応じて)

- [ ] **Step 1: 全テストを実行して回帰がないことを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/ -v`
Expected: 全て PASS

- [ ] **Step 2: evaluate.py を実行して最適化結果を確認**

Run: `PYTHONPATH=. uv run python scripts/evaluate.py`

確認事項:
- `artifacts/thresholds.json` が生成される
- `artifacts/confidence_params.json` が生成される（weights, spread_percentiles, confidence_boundaries）
- `artifacts/adaptive_threshold_params.json` が生成される（abstain_margin, train_atr_median）
- テストセットの結果で `AdaptiveAcc`, `HighConfAcc`, `TradeableAcc` が表示される
- ベースラインからの改善が統計的に有意か確認

- [ ] **Step 3: predict.py を実行して推論結果を確認**

Run: `PYTHONPATH=. uv run python scripts/predict.py`

確認事項:
- 出力に `confidence_score`, `confidence_level`, `should_trade`, `adaptive_threshold` が含まれる
- `direction` に UP/DOWN/ABSTAIN のいずれかが出力される
- SQLite に新カラム付きで保存される

- [ ] **Step 4: コミット**

```bash
git add -A
git commit -m "feat: confidence estimation and adaptive threshold system complete"
```
