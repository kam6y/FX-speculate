"""TFT USD/JPY ダッシュボード - FastAPI バックエンド

設計書 Section 7 に基づく運用ダッシュボード。
- 明日の予測表示
- エクイティカーブ / ローリング精度 / 月次リターン (Chart.js)
- モデルメトリクス一覧
- アラート管理

起動:
    cd "Temporal Fusion Transformer/dashboard"
    uv run uvicorn app:app --reload --port 8501
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# train_tft.py の DirectionAwareQuantileLoss を __main__ に登録
# (チェックポイントが __main__ モジュール名で pickle 保存されているため)
TFT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TFT_DIR))
from train_tft import DirectionAwareQuantileLoss
sys.modules["__main__"].DirectionAwareQuantileLoss = DirectionAwareQuantileLoss

import database as db

logger = logging.getLogger("tft-scheduler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="TFT USD/JPY Dashboard")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
def startup():
    # startup 時にも __main__ パッチを再適用 (--reload サブプロセス対策)
    main_mod = sys.modules.get("__main__")
    if main_mod and not hasattr(main_mod, "DirectionAwareQuantileLoss"):
        main_mod.DirectionAwareQuantileLoss = DirectionAwareQuantileLoss
    db.init_db()
    _seed_metrics_if_empty()
    _start_scheduler()


# ═══════════════════════════════════════════════════════
# Section 7.1: 定時実行スケジューラ (APScheduler)
#
#   17:00 JST - データ収集・特徴量更新
#   17:30 JST - TFT推論 → DB保存 → アラートチェック
# ═══════════════════════════════════════════════════════
_scheduler = None

def _start_scheduler():
    """APScheduler を起動して定時ジョブを登録"""
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    _scheduler = BackgroundScheduler(timezone="Asia/Tokyo")

    # 17:30 JST (月〜金): 推論実行
    _scheduler.add_job(
        _scheduled_predict,
        CronTrigger(hour=17, minute=30, day_of_week="mon-fri", timezone="Asia/Tokyo"),
        id="daily_predict",
        name="Daily TFT Prediction (17:30 JST)",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    _scheduler.start()
    logger.info("Scheduler started: daily prediction at 17:30 JST (Mon-Fri)")


def _save_predictions_to_db(result: dict):
    """予測結果をDBに保存"""
    for h in result["horizons"]:
        db.save_prediction(
            prediction_date=result["prediction_date"],
            target_date=h["target_date"],
            horizon=h["horizon"],
            pred_lr=h["pred_log_return"],
            pred_q10=h["q10"],
            pred_q90=h["q90"],
            direction=h["direction"],
            confidence=h["confidence"],
            current_price=result["current_price"] or 0,
        )


def _scheduled_predict():
    """定時推論ジョブ: 実績更新 → predict_tomorrow → DB保存 → アラートチェック"""
    logger.info("=== Scheduled job started ===")
    try:
        from predictor import predict_tomorrow, check_alerts, load_saved_metrics, update_actuals

        # 1. 過去の予測に実績値を埋める
        n_updated = update_actuals()
        if n_updated:
            logger.info(f"Updated {n_updated} predictions with actual returns")

        # 2. 明日の予測を実行
        result = predict_tomorrow()

        # 3. DB保存
        _save_predictions_to_db(result)

        if result["horizons"]:
            h1 = result["horizons"][0]
            logger.info(
                f"Prediction saved: {h1['target_date']} -> {h1['direction']} "
                f"(lr={h1['pred_log_return']:.6f}, conf={h1['confidence']:.2f})"
            )

        # 4. アラートチェック (バックテストデータから算出)
        from predictor import get_backtest_data_cached
        bt = get_backtest_data_cached()
        s = bt.get("summary", {})
        h1d = next((h for h in bt.get("horizon_accuracy", []) if h["horizon"] == "1D"), None)
        metrics = {
            "direction_accuracy": h1d["accuracy"] if h1d else s.get("win_rate", 0),
            "trade_max_drawdown": s.get("max_drawdown", 0),
        }
        alerts = check_alerts(metrics)
        for a in alerts:
            db.create_alert(a["type"], a["severity"], a["message"])
            logger.warning(f"Alert: [{a['severity']}] {a['message']}")

        logger.info("=== Scheduled job completed ===")

    except Exception as e:
        logger.error(f"Scheduled job failed: {e}", exc_info=True)
        db.create_alert("prediction_failure", "critical", f"定時推論失敗: {e}")


@app.on_event("shutdown")
def shutdown():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


def _seed_metrics_if_empty():
    """初回起動時: tft_metrics.json からメトリクスをDBに投入"""
    existing = db.get_latest_metrics()
    if existing:
        return
    try:
        from predictor import load_saved_metrics
        metrics = load_saved_metrics()
        if metrics:
            db.save_metrics(metrics, source="initial_seed")
    except Exception:
        pass


# ─── Pages ───

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ─── API Endpoints ───

@app.get("/api/metrics")
async def api_metrics():
    """モデルメトリクス取得 — バックテストデータから算出"""
    try:
        from predictor import get_backtest_data_cached
        data = get_backtest_data_cached()
        s = data.get("summary", {})
        h1d = next((h for h in data.get("horizon_accuracy", []) if h["horizon"] == "1D"), None)
        metrics = {
            "ensemble_direction_1d": h1d["accuracy"] if h1d else s.get("win_rate"),
            "trade_sharpe_ratio": s.get("sharpe_ratio"),
            "trade_profit_factor": s.get("profit_factor"),
            "trade_total_pnl": s.get("total_pnl"),
            "trade_max_drawdown": s.get("max_drawdown"),
            "trade_win_rate": s.get("win_rate"),
            "trade_n_trades": s.get("n_trades"),
            "mae_1d": s.get("mae_1d"),
            "pred_up_ratio_1d": s.get("pred_up_ratio_1d"),
            "direction_accuracy": h1d["accuracy"] if h1d else s.get("win_rate"),
        }
        return JSONResponse(metrics)
    except Exception as e:
        # フォールバック: DB or tft_metrics.json
        logger.error(f"/api/metrics backtest fallback: {e}", exc_info=True)
        metrics = db.get_latest_metrics()
        if not metrics:
            try:
                from predictor import load_saved_metrics
                metrics = load_saved_metrics()
            except Exception:
                metrics = {}
        return JSONResponse(metrics)


@app.get("/api/predict")
async def api_predict():
    """明日の予測を実行してDBに保存"""
    try:
        from predictor import predict_tomorrow, check_alerts
        result = predict_tomorrow()

        # DBに保存
        _save_predictions_to_db(result)

        # アラートチェック (バックテストデータから算出)
        from predictor import get_backtest_data_cached
        bt = get_backtest_data_cached()
        s = bt.get("summary", {})
        h1d = next((h for h in bt.get("horizon_accuracy", []) if h["horizon"] == "1D"), None)
        alert_metrics = {
            "direction_accuracy": h1d["accuracy"] if h1d else s.get("win_rate", 0),
            "trade_max_drawdown": s.get("max_drawdown", 0),
        }
        alerts = check_alerts(alert_metrics)
        for a in alerts:
            db.create_alert(a["type"], a["severity"], a["message"])

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"/api/predict failed: {e}", exc_info=True)
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=500,
        )


@app.get("/api/predictions/latest")
async def api_latest_prediction():
    """最新の予測結果"""
    pred = db.get_latest_prediction()
    if pred:
        return JSONResponse(pred)
    return JSONResponse({"message": "No predictions yet"}, status_code=404)


@app.get("/api/predictions/history")
async def api_prediction_history():
    """予測履歴"""
    preds = db.get_recent_predictions(limit=60)
    return JSONResponse(preds)


@app.get("/api/backtest")
async def api_backtest():
    """バックテストデータ (キャッシュ優先)"""
    try:
        from predictor import get_backtest_data_cached
        data = get_backtest_data_cached()
        return JSONResponse(data)
    except Exception as e:
        logger.error(f"/api/backtest failed: {e}", exc_info=True)
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=500,
        )


@app.get("/api/backtest/refresh")
async def api_backtest_refresh():
    """バックテストキャッシュを強制再計算"""
    try:
        from predictor import get_backtest_data_cached
        data = get_backtest_data_cached(force=True)
        return JSONResponse(data)
    except Exception as e:
        logger.error(f"/api/backtest/refresh failed: {e}", exc_info=True)
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=500,
        )


@app.get("/api/live-equity")
async def api_live_equity():
    """DB蓄積予測からライブエクイティカーブ"""
    try:
        from predictor import compute_live_equity
        data = compute_live_equity()
        return JSONResponse(data)
    except Exception as e:
        logger.error(f"/api/live-equity failed: {e}", exc_info=True)
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=500,
        )


@app.post("/api/update-actuals")
async def api_update_actuals():
    """実績値を手動更新"""
    try:
        from predictor import update_actuals
        n = update_actuals()
        return JSONResponse({"updated": n})
    except Exception as e:
        logger.error(f"/api/update-actuals failed: {e}", exc_info=True)
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=500,
        )


@app.get("/api/alerts")
async def api_alerts():
    """アクティブアラート"""
    alerts = db.get_active_alerts()
    return JSONResponse(alerts)


@app.post("/api/alerts/{alert_id}/resolve")
async def api_resolve_alert(alert_id: int):
    """アラート解決"""
    db.resolve_alert(alert_id)
    return JSONResponse({"status": "resolved"})


@app.get("/api/rolling-accuracy")
async def api_rolling_accuracy():
    """ローリング方向精度"""
    data = db.get_rolling_accuracy()
    return JSONResponse(data)


@app.get("/api/pnl-history")
async def api_pnl_history():
    """PnL 履歴"""
    data = db.get_pnl_history()
    return JSONResponse(data)


@app.get("/api/scheduler")
async def api_scheduler():
    """スケジューラ状態"""
    if not _scheduler:
        return JSONResponse({"status": "not_started", "jobs": []})

    jobs = []
    for job in _scheduler.get_jobs():
        next_run = job.next_run_time
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": next_run.isoformat() if next_run else None,
            "next_run_display": next_run.strftime("%Y-%m-%d %H:%M JST") if next_run else "---",
        })

    return JSONResponse({
        "status": "running" if _scheduler.running else "stopped",
        "jobs": jobs,
    })
