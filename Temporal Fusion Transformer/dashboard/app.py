"""TFT USD/JPY ダッシュボード - FastAPI バックエンド

起動:
    cd "Temporal Fusion Transformer/dashboard"
    uv run uvicorn app:app --reload --port 8501
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 親ディレクトリをパスに追加 (config, model 等の import 用)
TFT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(TFT_DIR))

# model.py を import するだけで DirectionAwareQuantileLoss が __main__ に登録される
import model as _model_mod  # noqa: F401

import database as db

logger = logging.getLogger("tft-scheduler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="TFT USD/JPY Dashboard")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
def startup():
    db.init_db()
    _start_scheduler()


# ═══════════════════════════════════════════════════════
# スケジューラ (APScheduler)
# ═══════════════════════════════════════════════════════
_scheduler = None


def _start_scheduler():
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    _scheduler = BackgroundScheduler(timezone="Asia/Tokyo")
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


def _extract_alert_metrics(bt: dict) -> dict:
    """バックテストデータから方向精度・最大DDを抽出 (アラートチェック用)"""
    s = bt.get("summary", {})
    h1d = next((h for h in bt.get("horizon_accuracy", []) if h["horizon"] == "1D"), None)
    return {
        "direction_accuracy": h1d["accuracy"] if h1d else s.get("win_rate", 0),
        "trade_max_drawdown": s.get("max_drawdown", 0),
    }


def _save_predictions_to_db(result: dict):
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
    """定時推論ジョブ"""
    logger.info("=== Scheduled job started ===")
    try:
        from predictor import predict_tomorrow, check_alerts, update_actuals, get_backtest_data_cached

        n_updated = update_actuals()
        if n_updated:
            logger.info(f"Updated {n_updated} predictions with actual returns")

        result = predict_tomorrow()
        _save_predictions_to_db(result)

        if result["horizons"]:
            h1 = result["horizons"][0]
            logger.info(f"Prediction saved: {h1['target_date']} -> {h1['direction']} "
                        f"(lr={h1['pred_log_return']:.6f}, conf={h1['confidence']:.2f})")

        bt = get_backtest_data_cached()
        alerts = check_alerts(_extract_alert_metrics(bt))
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


# ─── Pages ───

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ─── API Endpoints ───

@app.get("/api/metrics")
async def api_metrics():
    """モデルメトリクス取得"""
    try:
        from predictor import get_backtest_data_cached
        data = get_backtest_data_cached()
        s = data.get("summary", {})
        am = _extract_alert_metrics(data)
        dir_acc = am["direction_accuracy"]
        return JSONResponse({
            "ensemble_direction_1d": dir_acc,
            "trade_sharpe_ratio": s.get("sharpe_ratio"),
            "trade_profit_factor": s.get("profit_factor"),
            "trade_total_pnl": s.get("total_pnl"),
            "trade_max_drawdown": s.get("max_drawdown"),
            "trade_win_rate": s.get("win_rate"),
            "trade_n_trades": s.get("n_trades"),
            "mae_1d": s.get("mae_1d"),
            "pred_up_ratio_1d": s.get("pred_up_ratio_1d"),
            "direction_accuracy": dir_acc,
        })
    except Exception as e:
        logger.error(f"/api/metrics failed: {e}", exc_info=True)
        return JSONResponse({})


@app.get("/api/predict")
async def api_predict():
    """明日の予測を実行してDBに保存"""
    try:
        from predictor import predict_tomorrow, check_alerts, get_backtest_data_cached
        result = predict_tomorrow()
        _save_predictions_to_db(result)

        bt = get_backtest_data_cached()
        alerts = check_alerts(_extract_alert_metrics(bt))
        for a in alerts:
            db.create_alert(a["type"], a["severity"], a["message"])

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"/api/predict failed: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/predictions/latest")
async def api_latest_prediction():
    pred = db.get_latest_prediction()
    if pred:
        return JSONResponse(pred)
    return JSONResponse({"message": "No predictions yet"}, status_code=404)


@app.get("/api/predictions/history")
async def api_prediction_history():
    return JSONResponse(db.get_recent_predictions(limit=60))


@app.get("/api/backtest")
async def api_backtest():
    try:
        from predictor import get_backtest_data_cached
        return JSONResponse(get_backtest_data_cached())
    except Exception as e:
        logger.error(f"/api/backtest failed: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/backtest/refresh")
async def api_backtest_refresh():
    try:
        from predictor import get_backtest_data_cached
        return JSONResponse(get_backtest_data_cached(force=True))
    except Exception as e:
        logger.error(f"/api/backtest/refresh failed: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/live-equity")
async def api_live_equity():
    try:
        from predictor import compute_live_equity
        return JSONResponse(compute_live_equity())
    except Exception as e:
        logger.error(f"/api/live-equity failed: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.post("/api/update-actuals")
async def api_update_actuals():
    try:
        from predictor import update_actuals
        return JSONResponse({"updated": update_actuals()})
    except Exception as e:
        logger.error(f"/api/update-actuals failed: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/alerts")
async def api_alerts():
    return JSONResponse(db.get_active_alerts())


@app.post("/api/alerts/{alert_id}/resolve")
async def api_resolve_alert(alert_id: int):
    db.resolve_alert(alert_id)
    return JSONResponse({"status": "resolved"})


@app.get("/api/rolling-accuracy")
async def api_rolling_accuracy():
    return JSONResponse(db.get_rolling_accuracy())


@app.get("/api/pnl-history")
async def api_pnl_history():
    return JSONResponse(db.get_pnl_history())


@app.get("/api/scheduler")
async def api_scheduler():
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
    return JSONResponse({"status": "running" if _scheduler.running else "stopped", "jobs": jobs})
