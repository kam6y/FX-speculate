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
        """モデル間の方向一致度 (多数派比率, 0.5〜1.0)。閾値ぴったりは DOWN に計上。"""
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
