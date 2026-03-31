"""カスタム損失関数。

DirectionAwareQuantileLoss: QuantileLoss に tanh ベースの方向ペナルティを追加。
"""

import torch
from pytorch_forecasting.metrics import QuantileLoss

from config import DIRECTION_WEIGHT, SMOOTHING_TEMPERATURE, DEAD_ZONE, QUANTILES, PREDICTION_LENGTH


class DirectionAwareQuantileLoss(QuantileLoss):
    """QuantileLoss + ホライズン重み付き方向ペナルティ。"""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        direction_weight: float = DIRECTION_WEIGHT,
        smoothing_temperature: float = SMOOTHING_TEMPERATURE,
        dead_zone: float = DEAD_ZONE,
        horizon_weights: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(quantiles=quantiles or QUANTILES, **kwargs)
        self.direction_weight = direction_weight
        self.smoothing_temperature = smoothing_temperature
        self.dead_zone = dead_zone
        if horizon_weights is not None:
            assert len(horizon_weights) == PREDICTION_LENGTH, (
                f"horizon_weights length {len(horizon_weights)} "
                f"must match PREDICTION_LENGTH {PREDICTION_LENGTH}"
            )
        hw = torch.tensor(horizon_weights, dtype=torch.float32) if horizon_weights else torch.empty(0)
        self.register_buffer("_hw", hw)

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

        q_mid = list(self.quantiles).index(0.5)
        pred_median = y_pred[..., q_mid]

        pred_dir = torch.tanh(pred_median / self.smoothing_temperature)
        target_dir = torch.tanh(target / self.smoothing_temperature)

        mismatch = (1 - pred_dir * target_dir) / 2
        dead_zone_mask = (target.abs() < self.dead_zone).float()
        mismatch = mismatch * (1 - dead_zone_mask)

        if self._hw.numel() > 0:
            hw = self._hw[:target.shape[1]]
            mismatch = mismatch * hw.unsqueeze(0)

        dir_penalty = torch.zeros_like(ql)
        dir_penalty[..., q_mid] = mismatch * self.direction_weight

        return ql + dir_penalty
