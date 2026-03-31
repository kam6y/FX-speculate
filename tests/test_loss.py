"""model/loss.py のテスト。"""

import torch
import pytest
from model.loss import DirectionAwareQuantileLoss


class TestDirectionAwareQuantileLoss:
    def setup_method(self):
        self.loss_fn = DirectionAwareQuantileLoss(
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            direction_weight=1.0,
            smoothing_temperature=0.1,
            dead_zone=1e-4,
        )

    def test_output_shape(self):
        y_pred = torch.randn(32, 5, 5)
        target = torch.randn(32, 5)
        loss = self.loss_fn.loss(y_pred, target)
        assert loss.shape == y_pred.shape

    def test_correct_direction_lower_loss(self):
        """方向が正しい予測は、間違った予測より損失が小さい。"""
        target = torch.tensor([[0.01, 0.02, 0.01, 0.01, 0.01]])
        correct = torch.zeros(1, 5, 5)
        correct[:, :, 2] = 0.01
        wrong = torch.zeros(1, 5, 5)
        wrong[:, :, 2] = -0.01
        loss_correct = self.loss_fn.loss(correct, target).mean()
        loss_wrong = self.loss_fn.loss(wrong, target).mean()
        assert loss_correct < loss_wrong

    def test_dead_zone_suppresses_penalty(self):
        """ターゲットがデッドゾーン内なら方向ペナルティは抑制される。"""
        tiny_target = torch.tensor([[1e-5, 1e-5, 1e-5, 1e-5, 1e-5]])
        y_pred = torch.zeros(1, 5, 5)
        y_pred[:, :, 2] = -1e-5
        loss_fn_no_dir = DirectionAwareQuantileLoss(
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            direction_weight=0.0,
        )
        loss_with = self.loss_fn.loss(y_pred, tiny_target).mean()
        loss_without = loss_fn_no_dir.loss(y_pred, tiny_target).mean()
        assert abs(loss_with.item() - loss_without.item()) < 0.01

    def test_gradient_flows(self):
        """勾配が正常に流れることを確認。"""
        y_pred = torch.randn(8, 5, 5, requires_grad=True)
        target = torch.randn(8, 5)
        loss = self.loss_fn.loss(y_pred, target).mean()
        loss.backward()
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
