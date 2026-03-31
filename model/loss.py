"""カスタム損失関数。

DirectionAwareQuantileLoss: QuantileLoss にスムーズな方向ペナルティを追加。
tanh スムージングでゼロ付近の勾配不連続を回避する。
"""

import torch
from pytorch_forecasting.metrics import QuantileLoss

from config import DIRECTION_WEIGHT, SMOOTHING_TEMPERATURE, DEAD_ZONE, QUANTILES


class DirectionAwareQuantileLoss(QuantileLoss):
    """QuantileLoss + 方向ペナルティ。"""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        direction_weight: float = DIRECTION_WEIGHT,
        smoothing_temperature: float = SMOOTHING_TEMPERATURE,
        dead_zone: float = DEAD_ZONE,
        **kwargs,
    ):
        super().__init__(quantiles=quantiles or QUANTILES, **kwargs)
        self.direction_weight = direction_weight
        self.smoothing_temperature = smoothing_temperature
        self.dead_zone = dead_zone

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算する。

        Args:
            y_pred: (batch, horizon, n_quantiles)
            target: (batch, horizon)
        Returns:
            (batch, horizon, n_quantiles) の損失テンソル
        """
        ql = super().loss(y_pred, target)

        if self.direction_weight <= 0:
            return ql

        q_mid = y_pred.size(-1) // 2
        pred_median = y_pred[..., q_mid]

        pred_dir = torch.tanh(pred_median / self.smoothing_temperature)
        target_dir = torch.tanh(target / self.smoothing_temperature)

        mismatch = (1 - pred_dir * target_dir) / 2

        dead_zone_mask = (target.abs() < self.dead_zone).float()
        mismatch = mismatch * (1 - dead_zone_mask)

        # 中央値分位点のみにペナルティを適用（他の分位点の意味を保持）
        dir_penalty = torch.zeros_like(ql)
        dir_penalty[..., q_mid] = mismatch * self.direction_weight

        return ql + dir_penalty
