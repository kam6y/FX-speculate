"""TFT 推論パイプライン

既存チェックポイントをロードし、最新データから明日以降の予測を生成。
train_tft.py のデータパイプラインを再利用。
"""

import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# train_tft.py をインポートするためパスを追加
TFT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TFT_DIR))

from train_tft import (
    CONFIG,
    ARTIFACT_DIR,
    DEVICE,
    PIN_MEMORY,
    DirectionAwareQuantileLoss,
    fetch_market_data,
    add_technical_indicators,
    add_calendar_features,
    add_event_distance_features,
    add_holiday_flags,
    add_news_sentiment_proxy,
    add_market_regime,
    preprocess,
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    UNKNOWN_CATEGORICALS,
    _build_unknown_reals,
)
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import EncoderNormalizer

# チェックポイントが __main__.DirectionAwareQuantileLoss で保存されているため
# sys.modules['__main__'] に登録して unpickle 可能にする
# (uvicorn 経由だと __main__ は uvicorn 自体になるため sys.modules で直接パッチ)
sys.modules["__main__"].DirectionAwareQuantileLoss = DirectionAwareQuantileLoss

warnings.filterwarnings("ignore")


DASHBOARD_MODEL_DIR = Path(__file__).parent / "model"


def find_best_checkpoint() -> Path:
    """ダッシュボード専用モデルディレクトリから最良チェックポイントを返す。

    dashboard/model/ にデプロイ済みモデルのみを参照。
    デプロイされていなければエラー。
    """
    ckpts = list(DASHBOARD_MODEL_DIR.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No deployed model in {DASHBOARD_MODEL_DIR}. "
            "Run 'uv run python train_tft.py --deploy' first."
        )
    ckpts.sort(key=_parse_val_loss)
    return ckpts[0]


def _parse_val_loss(p: Path) -> float:
    try:
        return float(p.stem.split("val_loss=")[1].split("-v")[0])
    except (IndexError, ValueError):
        return float("inf")


def load_model(ckpt_path: Path | None = None) -> TemporalFusionTransformer:
    """チェックポイントからモデルをロード"""
    if ckpt_path is None:
        ckpt_path = find_best_checkpoint()
    # uvicorn 経由でも確実に __main__ にクラスを登録
    main_mod = sys.modules.get("__main__")
    if main_mod and not hasattr(main_mod, "DirectionAwareQuantileLoss"):
        main_mod.DirectionAwareQuantileLoss = DirectionAwareQuantileLoss
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.to(DEVICE)
    return model


def prepare_latest_data() -> pd.DataFrame:
    """最新のマーケットデータを取得して前処理"""
    df = fetch_market_data(CONFIG)
    df = add_technical_indicators(df)
    df = add_calendar_features(df)
    df = add_event_distance_features(df)
    df = add_holiday_flags(df)
    df = add_news_sentiment_proxy(df)
    df = add_market_regime(df)
    df = preprocess(df, CONFIG)
    return df


def create_prediction_dataset(df: pd.DataFrame) -> TimeSeriesDataSet:
    """推論用 TimeSeriesDataSet を構築 (全データをencoderとして使用)"""
    unknown_reals = _build_unknown_reals(df)
    unknown_cats = [c for c in UNKNOWN_CATEGORICALS if c in df.columns]

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="log_return",
        group_ids=["group_id"],
        max_encoder_length=CONFIG["MAX_ENCODER_LENGTH"],
        min_encoder_length=3,
        max_prediction_length=CONFIG["MAX_PREDICTION_LENGTH"],
        static_categoricals=["group_id"],
        time_varying_known_categoricals=KNOWN_CATEGORICALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_categoricals=unknown_cats,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=EncoderNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


def predict_tomorrow() -> dict:
    """明日 (次の営業日) の予測を生成

    Returns:
        dict with keys:
            prediction_date: 予測実行日
            target_date: 予測対象日
            current_price: 最新終値
            horizons: [{horizon, pred_lr, q10, q90, direction, confidence}, ...]
    """
    model = load_model()
    df = prepare_latest_data()

    # 最新日付と価格
    last_date = df.index[-1]
    last_price = float(df["close"].iloc[-1]) if "close" in df.columns else None

    # 最新データの末尾からprediction_lengthだけ予測
    dataset = create_prediction_dataset(df)
    dl = dataset.to_dataloader(
        train=False,
        batch_size=CONFIG["BATCH_SIZE"],
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    with torch.no_grad():
        preds = model.predict(dl, mode="quantiles").to(DEVICE)

    # 最後のサンプル (= 最新時点からの予測)
    last_pred = preds[-1]  # [pred_len, n_quantiles]
    n_q = last_pred.size(1)
    q_mid = n_q // 2

    # 次の営業日を計算
    next_bday = last_date + pd.offsets.BDay(1)

    horizons = []
    for h in range(min(last_pred.size(0), 10)):
        pred_lr = float(last_pred[h, q_mid].item())
        q10 = float(last_pred[h, 0].item())
        q90 = float(last_pred[h, -1].item())
        direction = "UP" if pred_lr > 0 else "DOWN"

        # confidence: 分位点の一致度 (全分位点が同方向なら高い)
        all_same_sign = (q10 > 0 and q90 > 0) or (q10 < 0 and q90 < 0)
        confidence = min(abs(pred_lr) / max(abs(q90 - q10), 1e-8), 1.0)
        if all_same_sign:
            confidence = min(confidence + 0.3, 1.0)

        target_date = last_date + pd.offsets.BDay(h + 1)

        horizons.append({
            "horizon": h + 1,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "pred_log_return": round(pred_lr, 6),
            "q10": round(q10, 6),
            "q90": round(q90, 6),
            "direction": direction,
            "confidence": round(confidence, 4),
        })

    # メモリ解放
    del model, preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "prediction_date": last_date.strftime("%Y-%m-%d"),
        "current_price": round(last_price, 3) if last_price else None,
        "last_date": last_date.strftime("%Y-%m-%d"),
        "horizons": horizons,
    }


def get_backtest_data() -> dict:
    """バックテストデータ (エクイティカーブ用) を取得"""
    model = load_model()
    df = prepare_latest_data()

    n = len(df)
    train_end = int(n * CONFIG["TRAIN_RATIO"])
    val_end = int(n * (CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"]))

    train_cutoff = int(df.iloc[train_end - 1]["time_idx"])
    val_cutoff = int(df.iloc[val_end - 1]["time_idx"])

    unknown_reals = _build_unknown_reals(df)
    unknown_cats = [c for c in UNKNOWN_CATEGORICALS if c in df.columns]

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target="log_return",
        group_ids=["group_id"],
        max_encoder_length=CONFIG["MAX_ENCODER_LENGTH"],
        min_encoder_length=3,
        max_prediction_length=CONFIG["MAX_PREDICTION_LENGTH"],
        static_categoricals=["group_id"],
        time_varying_known_categoricals=KNOWN_CATEGORICALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_categoricals=unknown_cats,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=EncoderNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    test = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=val_cutoff + 1, stop_randomization=True,
    )

    dl = test.to_dataloader(
        train=False, batch_size=CONFIG["BATCH_SIZE"], num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    with torch.no_grad():
        preds = model.predict(dl, mode="quantiles").to(DEVICE)

    actuals_list = []
    time_idx_list = []
    for x, y in dl:
        actuals_list.append(y[0].to(DEVICE, non_blocking=True))
        if "decoder_time_idx" in x:
            time_idx_list.append(x["decoder_time_idx"][:, 0])
    actuals = torch.cat(actuals_list)
    pred_time_idx = torch.cat(time_idx_list) if time_idx_list else torch.arange(actuals.size(0))

    n_q = preds.size(2)
    q_mid = n_q // 2

    # Quantile calibration (tensor-level, all horizons)
    q_lo_t = preds[:, :, 0]
    q_hi_t = preds[:, :, -1]
    q50_t = preds[:, :, q_mid]

    quantile_cal = {
        "q10": round(float((actuals <= q_lo_t).float().mean().item()), 4),
        "q50": round(float((actuals <= q50_t).float().mean().item()), 4),
        "q90": round(float((actuals <= q_hi_t).float().mean().item()), 4),
        "coverage_80": round(float(((actuals >= q_lo_t) & (actuals <= q_hi_t)).float().mean().item()), 4),
    }

    # Multi-horizon direction accuracy
    horizon_acc = []
    for h_end, h_name in [(1, "1D"), (2, "2D"), (3, "3D"), (5, "5D"), (10, "10D")]:
        if h_end <= actuals.size(1):
            actual_cum = actuals[:, :h_end].sum(dim=1)
            pred_cum = q50_t[:, :h_end].sum(dim=1)
            acc = float(((actual_cum > 0) == (pred_cum > 0)).float().mean().item())
            horizon_acc.append({"horizon": h_name, "accuracy": round(acc, 4)})

    # time_idx -> 日付マッピング
    idx_to_date = {int(row["time_idx"]): row.name.strftime("%Y-%m-%d") for _, row in df.iterrows()}

    # 重複 time_idx を集約 (同一日の予測を平均)
    ti_np = pred_time_idx.cpu().numpy()
    pred_1d_raw = preds[:, 0, q_mid].cpu().numpy()
    actual_1d_raw = actuals[:, 0].cpu().numpy()
    q10_1d_raw = preds[:, 0, 0].cpu().numpy()
    q90_1d_raw = preds[:, 0, -1].cpu().numpy()

    unique_ti = np.unique(ti_np)
    unique_ti.sort()

    pred_1d = np.array([pred_1d_raw[ti_np == ti].mean() for ti in unique_ti])
    actual_1d = np.array([actual_1d_raw[ti_np == ti].mean() for ti in unique_ti])
    dates = [idx_to_date.get(int(ti), "") for ti in unique_ti]

    # Confidence calibration
    q10_1d = np.array([q10_1d_raw[ti_np == ti].mean() for ti in unique_ti])
    q90_1d = np.array([q90_1d_raw[ti_np == ti].mean() for ti in unique_ti])
    spread = np.maximum(np.abs(q90_1d - q10_1d), 1e-8)
    conf_scores = np.minimum(np.abs(pred_1d) / spread, 1.0)
    same_sign = ((q10_1d > 0) & (q90_1d > 0)) | ((q10_1d < 0) & (q90_1d < 0))
    conf_scores[same_sign] = np.minimum(conf_scores[same_sign] + 0.3, 1.0)

    correct_dir = (pred_1d > 0) == (actual_1d > 0)
    n_bins = 5
    bin_edges = np.unique(np.percentile(conf_scores, np.linspace(0, 100, n_bins + 1)))
    if len(bin_edges) < 3:
        bin_edges = np.linspace(conf_scores.min(), conf_scores.max() + 1e-9, n_bins + 1)
    n_actual_bins = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(conf_scores, bin_edges) - 1, 0, n_actual_bins - 1)
    conf_calibration = []
    for b in range(n_actual_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt >= 2:
            conf_calibration.append({
                "bin": f"{bin_edges[b]:.3f}-{bin_edges[b+1]:.3f}",
                "mid": round(float((bin_edges[b] + bin_edges[b+1]) / 2), 4),
                "accuracy": round(float(correct_dir[mask].mean()), 4),
                "avg_conf": round(float(conf_scores[mask].mean()), 4),
                "count": cnt,
            })

    cost = (CONFIG["SPREAD_PIPS"] + CONFIG["SLIPPAGE_PIPS"]) * CONFIG["PIP_SIZE"]
    cost_lr = cost / 150.0

    signals = np.zeros_like(pred_1d)
    signals[pred_1d > 0] = 1.0
    signals[pred_1d < 0] = -1.0

    pnl = signals * actual_1d - np.abs(signals) * cost_lr
    cum_pnl = np.cumsum(pnl)

    # Buy & Hold
    bh_pnl = actual_1d - cost_lr
    bh_cum = np.cumsum(bh_pnl)

    # 方向精度
    correct = (pred_1d > 0) == (actual_1d > 0)
    rolling_acc = pd.Series(correct.astype(float)).rolling(60, min_periods=20).mean().values

    # 月次PnL
    df_pnl = pd.DataFrame({"date": pd.to_datetime(dates[:len(pnl)]), "pnl": pnl})
    df_pnl["year"] = df_pnl["date"].dt.year
    df_pnl["month"] = df_pnl["date"].dt.month
    monthly = df_pnl.groupby(["year", "month"])["pnl"].sum().reset_index()
    monthly_data = [
        {"year": int(r["year"]), "month": int(r["month"]), "pnl": round(float(r["pnl"]), 4)}
        for _, r in monthly.iterrows()
    ]

    del model, preds, actuals
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dates": dates,
        "equity_curve": [round(float(v), 6) for v in cum_pnl],
        "buy_hold": [round(float(v), 6) for v in bh_cum],
        "drawdown": [round(float(v), 6) for v in (cum_pnl - np.maximum.accumulate(cum_pnl))],
        "rolling_accuracy": [round(float(v), 4) if not np.isnan(v) else None for v in rolling_acc],
        "monthly_pnl": monthly_data,
        "daily_pnl": [round(float(v), 6) for v in pnl],
        "signals": [int(s) for s in signals],
        "confidence_calibration": conf_calibration,
        "quantile_calibration": quantile_cal,
        "horizon_accuracy": horizon_acc,
    }


def load_saved_metrics() -> dict:
    """保存済みメトリクスを読み込み"""
    metrics_path = ARTIFACT_DIR / "tft_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def check_alerts(metrics: dict) -> list[dict]:
    """Section 7.3: アラート条件チェック"""
    alerts = []

    dir_acc = metrics.get("direction_accuracy") or metrics.get("ensemble_direction_1d", 0)
    if dir_acc < 0.50:
        alerts.append({
            "type": "accuracy_degradation",
            "severity": "warning",
            "message": f"方向精度が50%未満に低下 ({dir_acc:.1%})",
        })

    max_dd = metrics.get("trade_max_drawdown", 0)
    if max_dd < -0.15:
        alerts.append({
            "type": "high_drawdown",
            "severity": "critical",
            "message": f"最大ドローダウンが-15%超 ({max_dd:.1%})",
        })

    return alerts


# ═══════════════════════════════════════════════════════
# バックテストキャッシュ
# ═══════════════════════════════════════════════════════
CACHE_PATH = Path(__file__).parent / "backtest_cache.json"


def get_backtest_data_cached(force: bool = False) -> dict:
    """キャッシュがあればそこから読み込み、なければ計算してキャッシュ"""
    if not force and CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    data = get_backtest_data()

    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)

    return data


# ═══════════════════════════════════════════════════════
# 実績値の自動更新
# ═══════════════════════════════════════════════════════
def update_actuals():
    """未入力の予測対象日に対して実績値を取得・更新する

    yfinance から最新データを取得し、各 target_date の log_return を計算して
    predictions テーブルの actual_log_return / actual_direction / is_correct を埋める。
    """
    import database as db_mod

    pending = db_mod.get_pending_target_dates()
    if not pending:
        return 0

    # 最新の市場データを取得 (直近分だけで十分)
    try:
        import yfinance as yf
        fx = yf.download("JPY=X", start=pending[0], auto_adjust=True, progress=False)
        if fx.empty:
            return 0
        close = fx["Close"].squeeze()
        lr = np.log(close / close.shift(1)).dropna()
    except Exception:
        return 0

    updated = 0
    for target_date in pending:
        try:
            dt = pd.Timestamp(target_date)
            if dt in lr.index:
                actual_lr = float(lr.loc[dt])
                actual_dir = "UP" if actual_lr > 0 else "DOWN"
                db_mod.update_actual(target_date, 1, actual_lr, actual_dir)
                updated += 1
        except (KeyError, ValueError):
            continue

    return updated


# ═══════════════════════════════════════════════════════
# DB蓄積予測からライブエクイティカーブ生成
# ═══════════════════════════════════════════════════════
def compute_live_equity() -> dict:
    """predictions テーブルの実績入り予測からエクイティカーブを計算"""
    import database as db_mod

    rows = db_mod.get_predictions_with_actuals()
    if not rows:
        return {"dates": [], "equity_curve": [], "rolling_accuracy": [],
                "daily_pnl": [], "n_trades": 0, "win_rate": 0, "total_pnl": 0}

    cost_lr = (CONFIG["SPREAD_PIPS"] + CONFIG["SLIPPAGE_PIPS"]) * CONFIG["PIP_SIZE"] / 150.0

    dates = []
    daily_pnl = []
    correct_list = []

    for r in rows:
        signal = 1.0 if r["direction"] == "UP" else -1.0
        pnl = signal * r["actual_log_return"] - abs(signal) * cost_lr
        dates.append(r["target_date"])
        daily_pnl.append(round(pnl, 6))
        correct_list.append(1 if r["is_correct"] == 1 else 0)

    cum_pnl = np.cumsum(daily_pnl).tolist()

    # ローリング精度
    rolling_acc = []
    window = 20
    for i in range(len(correct_list)):
        start = max(0, i - window + 1)
        w = correct_list[start:i + 1]
        rolling_acc.append(round(sum(w) / len(w), 4) if w else 0)

    n_trades = len(rows)
    wins = sum(correct_list)

    return {
        "dates": dates,
        "equity_curve": [round(v, 6) for v in cum_pnl],
        "rolling_accuracy": rolling_acc,
        "daily_pnl": daily_pnl,
        "n_trades": n_trades,
        "win_rate": round(wins / n_trades, 4) if n_trades else 0,
        "total_pnl": round(cum_pnl[-1], 4) if cum_pnl else 0,
    }
