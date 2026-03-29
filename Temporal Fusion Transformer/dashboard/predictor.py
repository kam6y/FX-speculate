"""TFT 推論パイプライン

既存チェックポイントをロードし、最新データから明日以降の予測を生成。
共有モジュール (config, model, data, train) を再利用。
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 親ディレクトリをパスに追加
TFT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TFT_DIR))

from config import CONFIG, DEVICE, PIN_MEMORY
from model import load_model, ensemble_predict, ensemble_predict_with_actuals
from data import prepare_data, create_datasets, build_unknown_reals, tsds_kwargs, UNKNOWN_CATEGORICALS

warnings.filterwarnings("ignore")

DASHBOARD_MODEL_DIR = Path(__file__).parent / "model"
CACHE_PATH = Path(__file__).parent / "backtest_cache.json"


def find_all_checkpoints() -> list[Path]:
    """ダッシュボード専用モデルディレクトリから全チェックポイントを返す"""
    ckpts = sorted(DASHBOARD_MODEL_DIR.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No deployed model in {DASHBOARD_MODEL_DIR}. "
            "Run 'uv run python train.py --deploy' first."
        )
    return ckpts


def load_models():
    """全seedチェックポイントをロード"""
    ckpts = find_all_checkpoints()
    models = [load_model(p) for p in ckpts]
    print(f"Loaded {len(models)} model(s) for ensemble")
    return models


def predict_tomorrow() -> dict:
    """明日 (次の営業日) の予測を生成"""
    models = load_models()
    df = prepare_data(CONFIG)

    last_date = df.index[-1]
    last_price = float(df["close"].iloc[-1]) if "close" in df.columns else None

    unknown_reals = build_unknown_reals(df)
    unknown_cats = [c for c in UNKNOWN_CATEGORICALS if c in df.columns]

    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import EncoderNormalizer

    dataset = TimeSeriesDataSet(
        df,
        **tsds_kwargs(CONFIG, unknown_reals, unknown_cats),
    )
    dl = dataset.to_dataloader(
        train=False, batch_size=CONFIG["BATCH_SIZE"], num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    with torch.no_grad():
        preds = ensemble_predict(models, dl)

    last_pred = preds[-1]
    n_q = last_pred.size(1)
    q_mid = n_q // 2

    horizons = []
    for h in range(min(last_pred.size(0), 10)):
        pred_lr = float(last_pred[h, q_mid].item())
        q10 = float(last_pred[h, 0].item())
        q90 = float(last_pred[h, -1].item())
        direction = "UP" if pred_lr > 0 else "DOWN"

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

    del models, preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "prediction_date": last_date.strftime("%Y-%m-%d"),
        "current_price": round(last_price, 3) if last_price else None,
        "last_date": last_date.strftime("%Y-%m-%d"),
        "horizons": horizons,
    }


def get_backtest_data() -> dict:
    """バックテストデータ — compute_backtest_summary を利用"""
    from train import compute_backtest_summary

    models = load_models()
    df = prepare_data(CONFIG)

    config = dict(CONFIG)
    if "close" in df.columns:
        config["PRICE_REFERENCE"] = float(df["close"].mean())

    _, _, test = create_datasets(df, config)

    dl = test.to_dataloader(
        train=False, batch_size=CONFIG["BATCH_SIZE"], num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    with torch.no_grad():
        preds, actuals, pred_time_idx = ensemble_predict_with_actuals(models, dl)

    result = compute_backtest_summary(preds, actuals, pred_time_idx, df, config)

    del models, preds, actuals
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


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


def check_alerts(metrics: dict) -> list[dict]:
    """アラート条件チェック"""
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


def update_actuals():
    """未入力の予測対象日に対して実績値を取得・更新"""
    import database as db_mod

    pending = db_mod.get_pending_target_dates()
    if not pending:
        return 0

    try:
        import yfinance as yf
        start_dt = (pd.Timestamp(pending[0]) - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        fx = yf.download("JPY=X", start=start_dt, auto_adjust=True, progress=False)
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


def compute_live_equity() -> dict:
    """predictions テーブルの実績入り予測からエクイティカーブを計算"""
    import database as db_mod

    rows = db_mod.get_predictions_with_actuals()
    if not rows:
        return {"dates": [], "equity_curve": [], "rolling_accuracy": [],
                "daily_pnl": [], "n_trades": 0, "win_rate": 0, "total_pnl": 0}

    cost_lr = (CONFIG["SPREAD_PIPS"] + CONFIG["SLIPPAGE_PIPS"]) * CONFIG["PIP_SIZE"] / CONFIG.get("PRICE_REFERENCE", 150.0)

    dates, daily_pnl, correct_list = [], [], []
    for r in rows:
        signal = 1.0 if r["direction"] == "UP" else -1.0
        pnl = signal * r["actual_log_return"] - abs(signal) * cost_lr
        dates.append(r["target_date"])
        daily_pnl.append(round(pnl, 6))
        correct_list.append(1 if r["is_correct"] == 1 else 0)

    cum_pnl = np.cumsum(daily_pnl).tolist()

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
