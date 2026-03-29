"""TFT モデル定義・ロード・アンサンブル予測

DirectionAwareQuantileLoss をここに定義することで、
チェックポイントの pickle 参照を model.DirectionAwareQuantileLoss に統一し、
__main__ パッチを不要にする。
"""

import sys
from pathlib import Path

import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from config import DEVICE, PIN_MEMORY


class DirectionAwareQuantileLoss(QuantileLoss):
    """QuantileLoss に方向ペナルティを加えたカスタム損失。

    median 予測の符号が実績と一致しない場合に追加ペナルティを課す。
    """

    def __init__(self, quantiles=None, direction_weight: float = 1.0, **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)
        self.direction_weight = direction_weight

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ql = super().loss(y_pred, target)
        q_mid = y_pred.size(-1) // 2
        pred_median = y_pred[..., q_mid]
        sign_mismatch = (pred_median * target < 0).float()
        dir_penalty = sign_mismatch * (pred_median - target).abs()
        return ql + self.direction_weight * dir_penalty.unsqueeze(-1)


# チェックポイントが __main__.DirectionAwareQuantileLoss で保存されている場合に備え、
# __main__ にもクラスを登録しておく (既存モデルの互換性)
_main = sys.modules.get("__main__")
if _main and not hasattr(_main, "DirectionAwareQuantileLoss"):
    _main.DirectionAwareQuantileLoss = DirectionAwareQuantileLoss


def load_model(ckpt_path: Path) -> TemporalFusionTransformer:
    """チェックポイントからモデルをロード"""
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.to(DEVICE)
    return model


def find_checkpoints(directory: Path, top_k: int | None = None) -> list[Path]:
    """ディレクトリ内のチェックポイントを val_loss 順で返す"""
    ckpts = sorted(directory.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {directory}")

    def parse_val_loss(p: Path) -> float:
        try:
            return float(p.stem.split("val_loss=")[1].split("-v")[0])
        except (IndexError, ValueError):
            return float("inf")

    ckpts.sort(key=parse_val_loss)
    if top_k:
        ckpts = ckpts[:top_k]
    return ckpts


def ensemble_predict(
    models: list[TemporalFusionTransformer], dl
) -> torch.Tensor:
    """複数モデルの予測を平均してアンサンブル予測を返す"""
    sum_preds = None
    for m in models:
        p = m.predict(dl, mode="quantiles").to(DEVICE)
        if sum_preds is None:
            sum_preds = p.clone()
        else:
            sum_preds += p
        del p
    return sum_preds / len(models)


def ensemble_predict_with_actuals(
    models: list[TemporalFusionTransformer], dl
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """アンサンブル予測 + actuals + pred_time_idx を返す"""
    preds = ensemble_predict(models, dl)
    actuals_list, time_idx_list = [], []
    for x, y in dl:
        actuals_list.append(y[0].to(DEVICE, non_blocking=True))
        if "decoder_time_idx" in x:
            time_idx_list.append(x["decoder_time_idx"][:, 0])
    actuals = torch.cat(actuals_list)
    pred_time_idx = torch.cat(time_idx_list) if time_idx_list else torch.arange(actuals.size(0))
    return preds, actuals, pred_time_idx
