"""統合テスト: パイプライン全体を小さいデータで通す。"""

import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import prepare_data, split_data, create_datasets
from data.events import compute_event_features
from data.features import (
    compute_technical_features,
    compute_market_returns,
    compute_calendar_features,
)
from model.loss import DirectionAwareQuantileLoss
from model.trainer import build_tft


def _make_synthetic_features(n: int = 500) -> pd.DataFrame:
    """合成 OHLCV データから特徴量を構築して返す。

    1. 500営業日の合成 OHLCV を生成
    2. テクニカル指標・市場リターン・カレンダー特徴量・イベント特徴量を計算
    3. 合成マクロデータ (fred_* + rate_diff) を追加
    4. 全列を concat -> 列重複除去 -> dropna
    """
    np.random.seed(42)
    index = pd.bdate_range("2023-01-02", periods=n, freq="B")

    # 合成 OHLCV (usdjpy ベース)
    close = 140.0 + np.cumsum(np.random.randn(n) * 0.3)
    ohlcv_usdjpy = pd.DataFrame(
        {
            "usdjpy_open": close + np.random.randn(n) * 0.1,
            "usdjpy_high": close + np.abs(np.random.randn(n) * 0.3),
            "usdjpy_low": close - np.abs(np.random.randn(n) * 0.3),
            "usdjpy_close": close,
            "usdjpy_volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=index,
    )

    # テクニカル指標 (Close/High/Low 列を要求)
    ohlcv_renamed = ohlcv_usdjpy.rename(
        columns={
            "usdjpy_open": "Open",
            "usdjpy_high": "High",
            "usdjpy_low": "Low",
            "usdjpy_close": "Close",
            "usdjpy_volume": "Volume",
        }
    )
    technical = compute_technical_features(ohlcv_renamed)

    # 市場リターン (usdjpy_close + 関連市場 close が必要)
    market_df = ohlcv_usdjpy.copy()
    market_df["sp500_close"] = 4000.0 + np.cumsum(np.random.randn(n) * 10)
    market_df["nikkei_close"] = 32000.0 + np.cumsum(np.random.randn(n) * 50)
    market_df["vix_close"] = 20.0 + np.random.randn(n) * 2
    market_df["oil_close"] = 80.0 + np.cumsum(np.random.randn(n) * 0.5)
    market_df["gold_close"] = 1900.0 + np.cumsum(np.random.randn(n) * 3)
    returns = compute_market_returns(market_df)

    # カレンダー特徴量
    calendar = compute_calendar_features(index)

    # イベント特徴量
    events = compute_event_features(index)

    # 合成マクロデータ (定数値)
    macro = pd.DataFrame(
        {
            "fred_us_10y": np.full(n, 4.5),
            "fred_jp_10y": np.full(n, 1.0),
            "fred_ff_rate": np.full(n, 5.25),
            "fred_cpi": np.full(n, 310.0),
            "fred_unemployment": np.full(n, 3.7),
            "fred_gdp": np.full(n, 28000.0),
            "fred_m2": np.full(n, 21000.0),
            "fred_dxy": np.full(n, 104.0),
            "rate_diff": np.full(n, 3.5),  # fred_us_10y - fred_jp_10y
        },
        index=index,
    )

    # 全特徴量を結合 -> 重複列除去 -> dropna
    combined = pd.concat(
        [ohlcv_usdjpy, market_df, technical, returns, calendar, events, macro],
        axis=1,
    )
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.dropna()
    return combined


class TestEndToEndPipeline:
    """パイプライン全体のエンドツーエンドテスト。"""

    def test_dataset_creation(self):
        """prepare_data -> split_data -> create_datasets が非空データセットを返す。"""
        features = _make_synthetic_features(n=500)
        prepped = prepare_data(features)
        train, val, tune, test = split_data(prepped)

        assert len(train) > 0, "train split is empty"
        assert len(val) > 0, "val split is empty"

        training_ds, validation_ds = create_datasets(train, val)

        assert len(training_ds) > 0, "training dataset is empty"
        assert len(validation_ds) > 0, "validation dataset is empty"

    def test_model_forward_pass(self):
        """TFT がフォワードパスを実行でき、出力が 5 分位数を持つ。"""
        features = _make_synthetic_features(n=500)
        prepped = prepare_data(features)
        train, val, _tune, _test = split_data(prepped)

        training_ds, validation_ds = create_datasets(train, val)
        tft = build_tft(training_ds)
        tft.eval()

        # DataLoader からバッチを 1 つ取り出してフォワードパス
        from torch.utils.data import DataLoader

        loader = training_ds.to_dataloader(train=False, batch_size=8, num_workers=0)
        batch_x, batch_y = next(iter(loader))

        with torch.no_grad():
            output = tft(batch_x)

        # output は dict; "prediction" キーが (batch, horizon, n_quantiles)
        prediction = output["prediction"]
        assert prediction.ndim == 3, f"expected 3-dim output, got shape {prediction.shape}"
        from config import OUTPUT_SIZE
        assert prediction.shape[-1] == OUTPUT_SIZE, (
            f"expected {OUTPUT_SIZE} quantiles, got {prediction.shape[-1]}"
        )

    def test_loss_backward_pass(self):
        """損失計算と勾配フローが正常に動作する。"""
        features = _make_synthetic_features(n=500)
        prepped = prepare_data(features)
        train, val, _tune, _test = split_data(prepped)

        training_ds, _validation_ds = create_datasets(train, val)
        tft = build_tft(training_ds)

        from torch.utils.data import DataLoader

        loader = training_ds.to_dataloader(train=True, batch_size=8, num_workers=0)
        batch_x, batch_y = next(iter(loader))

        output = tft(batch_x)
        prediction = output["prediction"]

        # ターゲットを prediction と形状が合うように取り出す
        target = batch_y[0]  # (batch, horizon)

        loss_fn = DirectionAwareQuantileLoss()
        loss = loss_fn.loss(prediction, target).mean()

        assert not torch.isnan(loss), "loss is NaN"
        assert loss.item() >= 0, "loss should be non-negative"

        loss.backward()

        # モデルパラメータに勾配が流れていることを確認
        grad_found = any(
            p.grad is not None for p in tft.parameters() if p.requires_grad
        )
        assert grad_found, "no gradients found in model parameters"
