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
