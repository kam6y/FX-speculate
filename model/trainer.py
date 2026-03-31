"""Lightning Trainer 設定と TFT モデル構築。"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config import (
    ARTIFACT_DIR,
    MAX_EPOCHS,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    HIDDEN_SIZE,
    ATTENTION_HEAD_SIZE,
    DROPOUT,
    HIDDEN_CONTINUOUS_SIZE,
    OUTPUT_SIZE,
    TOP_K_CHECKPOINTS,
)
from model.loss import DirectionAwareQuantileLoss

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True


def build_trainer(max_epochs: int = MAX_EPOCHS, fast_dev_run: bool = False) -> pl.Trainer:
    """Lightning Trainer を構築する。"""
    ckpt_dir = ARTIFACT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="tft-epoch={epoch:02d}-val_loss={val_loss:.4f}",
            auto_insert_metric_name=False,
            monitor="val_loss",
            mode="min",
            save_top_k=TOP_K_CHECKPOINTS,
        ),
    ]

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=callbacks,
        gradient_clip_val=0.1,
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
    )


def build_tft(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """TFT モデルを構築する。"""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        output_size=OUTPUT_SIZE,
        loss=DirectionAwareQuantileLoss(),
        reduce_on_plateau_patience=4,
    )
