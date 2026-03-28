"""
TFT USD/JPY モデル包括評価スクリプト

保存済みチェックポイントからモデルをロードし、テストデータで詳細評価を行う。
出力:
  - artifacts/eval_report.json   : 全メトリクス
  - artifacts/eval_*.png         : 各種可視化グラフ

実行:
    uv run python evaluate_tft.py                          # best 3 ckpt アンサンブル
    uv run python evaluate_tft.py --ckpt path/to/model.ckpt  # 単一モデル
    uv run python evaluate_tft.py --top-k 5                # top-5 アンサンブル
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

from train_tft import (
    CONFIG,
    ARTIFACT_DIR,
    DEVICE,
    PIN_MEMORY,
    DirectionAwareQuantileLoss,
    prepare_data,
    create_datasets,
    evaluate,
    backtest_trading,
)
from pytorch_forecasting import TemporalFusionTransformer

EVAL_DIR = ARTIFACT_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# チェックポイント選択
# ===================================================================
def find_best_checkpoints(ckpt_dir: Path, top_k: int = 3) -> list[Path]:
    """val_loss が最小の top-k チェックポイントを返す"""
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    def parse_val_loss(p: Path) -> float:
        name = p.stem
        try:
            return float(name.split("val_loss=")[1].split("-v")[0])
        except (IndexError, ValueError):
            return float("inf")

    ckpts.sort(key=parse_val_loss)
    selected = ckpts[:top_k]
    print(f"Selected {len(selected)} checkpoints:")
    for p in selected:
        print(f"  {p.name} (val_loss={parse_val_loss(p):.4f})")
    return selected


def load_model(ckpt_path: Path) -> TemporalFusionTransformer:
    """チェックポイントからモデルをロード"""
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.to(DEVICE)
    return model


# ===================================================================
# アンサンブル予測
# ===================================================================
def ensemble_predict(
    models: list[TemporalFusionTransformer],
    dataset,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """複数モデルの予測を平均してアンサンブル

    Returns: (ens_preds, actuals, encoder_last, pred_time_idx)
        pred_time_idx: 各サンプルの予測開始 time_idx (日付マッピング用)
    """
    dl = dataset.to_dataloader(
        train=False, batch_size=config["BATCH_SIZE"], num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    sum_preds = None
    for model in models:
        preds = model.predict(dl, mode="quantiles").to(DEVICE)
        if sum_preds is None:
            sum_preds = preds.clone()
        else:
            sum_preds += preds
        del preds

    ens_preds = sum_preds / len(models)

    # actuals / encoder_last / prediction time_idx
    actuals_list, enc_last_list, time_idx_list = [], [], []
    for x, y in dl:
        actuals_list.append(y[0].to(DEVICE, non_blocking=True))
        if "encoder_target" in x:
            enc_last_list.append(x["encoder_target"][:, -1].to(DEVICE, non_blocking=True))
        # decoder_time_idx の最初の要素 = 予測開始時点
        if "decoder_time_idx" in x:
            time_idx_list.append(x["decoder_time_idx"][:, 0])
    actuals = torch.cat(actuals_list)
    encoder_last = torch.cat(enc_last_list) if enc_last_list else None
    pred_time_idx = torch.cat(time_idx_list) if time_idx_list else torch.arange(actuals.size(0))

    return ens_preds, actuals, encoder_last, pred_time_idx


# ===================================================================
# メトリクス計算
# ===================================================================
def compute_full_metrics(
    preds: torch.Tensor, actuals: torch.Tensor, config: dict
) -> dict:
    """包括的なメトリクスを計算"""
    q_mid = preds.size(2) // 2
    q50 = preds[:, :, q_mid]
    q_lo = preds[:, :, 0]
    q_hi = preds[:, :, -1]

    diff = q50 - actuals
    mae = diff.abs().mean().item()
    rmse = (diff**2).mean().sqrt().item()

    # 方向精度 (累積リターンベース: 1〜h日の合計で方向判定)
    dir_metrics = {}
    for h_end, h_name in [(1, "1d"), (2, "2d"), (3, "3d"), (5, "5d"), (10, "10d")]:
        if h_end <= actuals.size(1):
            actual_cum = actuals[:, :h_end].sum(dim=1)
            pred_cum = q50[:, :h_end].sum(dim=1)
            acc = ((actual_cum > 0) == (pred_cum > 0)).float().mean().item()
            dir_metrics[f"direction_{h_name}"] = round(acc, 4)

    # 分位点カバレッジ
    coverage = ((actuals >= q_lo) & (actuals <= q_hi)).float().mean().item()

    # ホライズン別 MAE
    horizon_mae = {}
    for h_idx, h_name in [(0, "1d"), (4, "5d"), (9, "10d")]:
        if h_idx < preds.size(1):
            h_mae = (q50[:, h_idx] - actuals[:, h_idx]).abs().mean().item()
            horizon_mae[f"mae_{h_name}"] = round(h_mae, 4)

    # 予測の calibration (分位点別カバレッジ)
    calibration = {}
    quantile_labels = config.get("QUANTILES", [0.1, 0.5, 0.9])
    for qi, q_val in enumerate(quantile_labels):
        below = (actuals <= preds[:, :, qi]).float().mean().item()
        calibration[f"calibration_q{int(q_val*100)}"] = round(below, 4)

    # トレーディングメトリクス
    trade = backtest_trading(preds, actuals, None, config)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "quantile_coverage_80": round(coverage, 4),
        **dir_metrics,
        **horizon_mae,
        **calibration,
        **{f"trade_{k}": v for k, v in trade.items()},
    }


# ===================================================================
# 可視化
# ===================================================================
def plot_equity_curve(preds: torch.Tensor, actuals: torch.Tensor,
                      dates: pd.DatetimeIndex, config: dict):
    """エクイティカーブ + ドローダウン"""
    q_mid = preds.size(2) // 2
    pred_1d = preds[:, 0, q_mid].cpu().numpy()
    actual_1d = actuals[:, 0].cpu().numpy()

    cost = (config["SPREAD_PIPS"] + config["SLIPPAGE_PIPS"]) * config["PIP_SIZE"]
    cost_lr = cost / 150.0

    signals = np.zeros_like(pred_1d)
    signals[pred_1d > 0] = 1.0
    signals[pred_1d < 0] = -1.0

    pnl = signals * actual_1d - np.abs(signals) * cost_lr
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max

    # Buy & Hold (常にロング)
    bh_pnl = actual_1d - cost_lr
    bh_cum = np.cumsum(bh_pnl)

    plot_dates = dates[:len(cum_pnl)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.05})

    ax1.plot(plot_dates, cum_pnl, label="TFT モデル", color="#2196F3", linewidth=1.5)
    ax1.plot(plot_dates, bh_cum, label="Buy & Hold", color="#9E9E9E",
             linewidth=1, linestyle="--", alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("累積 PnL (log return)")
    ax1.set_title("エクイティカーブ")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.fill_between(plot_dates, drawdown, 0, color="#F44336", alpha=0.4)
    ax2.set_ylabel("ドローダウン")
    ax2.set_xlabel("日付")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_equity_curve.png")


def plot_monthly_returns(preds: torch.Tensor, actuals: torch.Tensor,
                         dates: pd.DatetimeIndex, config: dict):
    """月次リターンヒートマップ"""
    q_mid = preds.size(2) // 2
    pred_1d = preds[:, 0, q_mid].cpu().numpy()
    actual_1d = actuals[:, 0].cpu().numpy()

    cost = (config["SPREAD_PIPS"] + config["SLIPPAGE_PIPS"]) * config["PIP_SIZE"]
    cost_lr = cost / 150.0

    signals = np.zeros_like(pred_1d)
    signals[pred_1d > 0] = 1.0
    signals[pred_1d < 0] = -1.0
    pnl = signals * actual_1d - np.abs(signals) * cost_lr

    plot_dates = dates[:len(pnl)]
    df_pnl = pd.DataFrame({"date": plot_dates, "pnl": pnl})
    df_pnl["year"] = df_pnl["date"].dt.year
    df_pnl["month"] = df_pnl["date"].dt.month

    monthly = df_pnl.groupby(["year", "month"])["pnl"].sum().reset_index()
    pivot = monthly.pivot(index="year", columns="month", values="pnl")
    pivot.columns = [f"{m}月" for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "月次 PnL"})
    ax.set_title("月次リターン ヒートマップ")
    ax.set_ylabel("年")

    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_monthly_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_monthly_returns.png")


def plot_rolling_accuracy(preds: torch.Tensor, actuals: torch.Tensor,
                          dates: pd.DatetimeIndex, window: int = 60):
    """ローリング方向精度"""
    q_mid = preds.size(2) // 2
    pred_dir = (preds[:, 0, q_mid] > 0).cpu().numpy().astype(float)
    actual_dir = (actuals[:, 0] > 0).cpu().numpy().astype(float)
    correct = (pred_dir == actual_dir).astype(float)

    rolling_acc = pd.Series(correct).rolling(window, min_periods=20).mean().values
    plot_dates = dates[:len(rolling_acc)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_dates, rolling_acc, color="#2196F3", linewidth=1.2)
    ax.axhline(0.5, color="#F44336", linewidth=1, linestyle="--", alpha=0.7, label="ランダム (50%)")
    ax.axhline(np.mean(correct), color="#4CAF50", linewidth=1, linestyle=":",
               alpha=0.7, label=f"全期間平均 ({np.mean(correct):.1%})")
    ax.fill_between(plot_dates, 0.5, rolling_acc,
                     where=rolling_acc > 0.5, alpha=0.15, color="#4CAF50")
    ax.fill_between(plot_dates, 0.5, rolling_acc,
                     where=rolling_acc < 0.5, alpha=0.15, color="#F44336")
    ax.set_ylabel("方向精度")
    ax.set_title(f"ローリング方向精度 ({window}日窓)")
    ax.set_xlabel("日付")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_rolling_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_rolling_accuracy.png")


def plot_prediction_scatter(preds: torch.Tensor, actuals: torch.Tensor):
    """予測 vs 実績 散布図"""
    q_mid = preds.size(2) // 2
    pred_1d = preds[:, 0, q_mid].cpu().numpy()
    actual_1d = actuals[:, 0].cpu().numpy()

    # 方向一致で色分け
    correct = (pred_1d > 0) == (actual_1d > 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual_1d[correct], pred_1d[correct], alpha=0.4, s=15,
               c="#4CAF50", label=f"方向一致 ({correct.sum()})")
    ax.scatter(actual_1d[~correct], pred_1d[~correct], alpha=0.4, s=15,
               c="#F44336", label=f"方向不一致 ({(~correct).sum()})")

    lim = max(abs(actual_1d).max(), abs(pred_1d).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.5, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("実績 log_return")
    ax.set_ylabel("予測 log_return (q50)")
    ax.set_title("予測 vs 実績 散布図 (1日先)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_pred_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_pred_scatter.png")


def plot_quantile_fan(preds: torch.Tensor, actuals: torch.Tensor,
                      dates: pd.DatetimeIndex, last_n: int = 100):
    """分位点ファンチャート (直近 N サンプル)"""
    n = min(last_n, preds.size(0))
    q_mid = preds.size(2) // 2

    pred_q50 = preds[-n:, 0, q_mid].cpu().numpy()
    pred_q10 = preds[-n:, 0, 0].cpu().numpy()
    pred_q90 = preds[-n:, 0, -1].cpu().numpy()
    actual_1d = actuals[-n:, 0].cpu().numpy()

    plot_dates = dates[-n:]
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x, pred_q10, pred_q90, alpha=0.2, color="#2196F3",
                     label="80% 予測区間")
    ax.plot(x, pred_q50, color="#2196F3", linewidth=1.2, label="予測 (q50)")
    ax.plot(x, actual_1d, color="#333333", linewidth=1, alpha=0.8, label="実績")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("サンプル (直近)")
    ax.set_ylabel("log_return (1日先)")
    ax.set_title(f"分位点予測ファンチャート (直近 {n} サンプル)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # X軸に日付ラベル (間引き)
    tick_step = max(1, n // 10)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([d.strftime("%m/%d") for d in plot_dates[::tick_step]], rotation=45)

    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_quantile_fan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_quantile_fan.png")


def plot_regime_accuracy(preds: torch.Tensor, actuals: torch.Tensor,
                         df_test: pd.DataFrame):
    """市場レジーム別方向精度"""
    q_mid = preds.size(2) // 2
    pred_dir = (preds[:, 0, q_mid] > 0).cpu().numpy()
    actual_dir = (actuals[:, 0] > 0).cpu().numpy()
    correct = pred_dir == actual_dir

    if "market_regime" not in df_test.columns:
        print("  -> eval_regime_accuracy.png SKIPPED (no market_regime)")
        return

    regimes = df_test["market_regime"].values[:len(correct)]

    df_regime = pd.DataFrame({"regime": regimes, "correct": correct})
    stats = df_regime.groupby("regime")["correct"].agg(["mean", "count"]).reset_index()
    stats.columns = ["regime", "accuracy", "count"]
    stats = stats.sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"trend": "#4CAF50", "range": "#2196F3", "high_vol": "#F44336"}
    bars = ax.barh(stats["regime"], stats["accuracy"],
                    color=[colors.get(r, "#9E9E9E") for r in stats["regime"]])
    ax.axvline(0.5, color="red", linewidth=1, linestyle="--", alpha=0.5)

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{row["accuracy"]:.1%} (n={int(row["count"])})',
                va="center", fontsize=10)

    ax.set_xlabel("方向精度")
    ax.set_title("市場レジーム別 方向精度")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_regime_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_regime_accuracy.png")


def plot_horizon_accuracy(preds: torch.Tensor, actuals: torch.Tensor):
    """予測ホライズン別 方向精度 (累積リターンベース)"""
    q_mid = preds.size(2) // 2
    q50 = preds[:, :, q_mid]
    horizons = []
    accs = []

    for h in range(min(preds.size(1), actuals.size(1))):
        # 1〜(h+1)日の累積リターンで方向判定
        actual_cum = actuals[:, :h+1].sum(dim=1)
        pred_cum = q50[:, :h+1].sum(dim=1)
        acc = ((actual_cum > 0) == (pred_cum > 0)).float().mean().item()
        horizons.append(h + 1)
        accs.append(acc)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(horizons, accs, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.axhline(0.5, color="#F44336", linewidth=1, linestyle="--", alpha=0.7, label="ランダム")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.1%}", ha="center", fontsize=9)
    ax.set_xlabel("予測ホライズン (日)")
    ax.set_ylabel("方向精度")
    ax.set_title("予測ホライズン別 方向精度 (累積リターン)")
    ax.set_xticks(horizons)
    ax.set_ylim(0.3, max(accs) + 0.1)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_horizon_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_horizon_accuracy.png")


def plot_pnl_distribution(preds: torch.Tensor, actuals: torch.Tensor, config: dict):
    """日次PnL分布"""
    q_mid = preds.size(2) // 2
    pred_1d = preds[:, 0, q_mid].cpu().numpy()
    actual_1d = actuals[:, 0].cpu().numpy()

    cost = (config["SPREAD_PIPS"] + config["SLIPPAGE_PIPS"]) * config["PIP_SIZE"]
    cost_lr = cost / 150.0

    signals = np.zeros_like(pred_1d)
    signals[pred_1d > 0] = 1.0
    signals[pred_1d < 0] = -1.0
    pnl = signals * actual_1d - np.abs(signals) * cost_lr

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pnl, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(np.mean(pnl), color="#4CAF50", linewidth=1.5, linestyle="--",
               label=f"平均: {np.mean(pnl):.5f}")
    ax.axvline(np.median(pnl), color="#FF9800", linewidth=1.5, linestyle=":",
               label=f"中央値: {np.median(pnl):.5f}")

    # 勝率アノテーション
    win_rate = (pnl > 0).sum() / len(pnl)
    ax.text(0.98, 0.95, f"勝率: {win_rate:.1%}\n取引数: {len(pnl)}\nSharpe: {np.mean(pnl)/max(np.std(pnl),1e-10)*252**0.5:.2f}",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("日次 PnL (log return)")
    ax.set_ylabel("頻度")
    ax.set_title("日次PnL分布")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_pnl_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_pnl_distribution.png")


def plot_summary_dashboard(metrics: dict):
    """サマリーダッシュボード"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # 方向精度一覧
    ax = axes[0, 0]
    dir_keys = [k for k in metrics if k.startswith("direction_")]
    dir_labels = [k.replace("direction_", "") for k in dir_keys]
    dir_vals = [metrics[k] for k in dir_keys]
    colors = ["#4CAF50" if v > 0.5 else "#F44336" for v in dir_vals]
    bars = ax.bar(dir_labels, dir_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(0.5, color="red", linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, dir_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=9)
    ax.set_title("方向精度")
    ax.set_ylim(0, 1)

    # MAE 一覧
    ax = axes[0, 1]
    mae_keys = [k for k in metrics if k.startswith("mae")]
    mae_labels = [k.replace("mae_", "") if k != "mae" else "全体" for k in mae_keys]
    mae_vals = [metrics[k] for k in mae_keys]
    ax.bar(mae_labels, mae_vals, color="#2196F3", alpha=0.8, edgecolor="white")
    for i, (lbl, val) in enumerate(zip(mae_labels, mae_vals)):
        ax.text(i, val + 0.0001, f"{val:.4f}", ha="center", fontsize=9)
    ax.set_title("MAE (ホライズン別)")

    # トレーディング指標
    ax = axes[0, 2]
    trade_info = [
        ("PnL", metrics.get("trade_total_pnl", 0)),
        ("勝率", metrics.get("trade_win_rate", 0)),
        ("PF", metrics.get("trade_profit_factor", 0)),
        ("Sharpe", metrics.get("trade_sharpe_ratio", 0)),
        ("Max DD", metrics.get("trade_max_drawdown", 0)),
    ]
    ax.axis("off")
    table_data = [[name, f"{val:.4f}"] for name, val in trade_info]
    table = ax.table(cellText=table_data, colLabels=["指標", "値"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("トレーディング指標")

    # キャリブレーション
    ax = axes[1, 0]
    cal_keys = [k for k in metrics if k.startswith("calibration_")]
    if cal_keys:
        cal_labels = [k.replace("calibration_q", "Q") + "%" for k in cal_keys]
        cal_vals = [metrics[k] for k in cal_keys]
        ideal_vals = [int(k.replace("calibration_q", "")) / 100 for k in cal_keys]
        x = np.arange(len(cal_labels))
        ax.bar(x - 0.15, ideal_vals, 0.3, label="理想", color="#9E9E9E", alpha=0.5)
        ax.bar(x + 0.15, cal_vals, 0.3, label="実績", color="#2196F3", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cal_labels)
        ax.legend()
    ax.set_title("分位点キャリブレーション")

    # 取引数テキスト
    ax = axes[1, 1]
    ax.axis("off")
    n_trades = metrics.get("trade_n_trades", 0)
    coverage = metrics.get("quantile_coverage_80", 0)
    summary_text = (
        f"取引数: {n_trades}\n"
        f"80% カバレッジ: {coverage:.1%}\n"
        f"MAE: {metrics.get('mae', 0):.4f}\n"
        f"RMSE: {metrics.get('rmse', 0):.4f}"
    )
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=14, va="center", ha="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))
    ax.set_title("サマリー")

    # 空欄
    axes[1, 2].axis("off")

    fig.suptitle("TFT USD/JPY モデル評価ダッシュボード", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "eval_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> eval_dashboard.png")


# ===================================================================
# メイン
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="TFT Model Comprehensive Evaluation")
    parser.add_argument("--ckpt", type=str, nargs="+", default=None,
                        help="チェックポイントファイルパス (複数可)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="val_loss 上位 k 個のチェックポイントでアンサンブル")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    print(f"Device: {DEVICE}")
    print(f"Output: {EVAL_DIR}\n")

    # 1. データ準備
    print("=== 1. データ準備 ===")
    df = prepare_data(CONFIG)

    # 2. データセット作成
    print("\n=== 2. データセット作成 ===")
    training, validation, test = create_datasets(df, CONFIG)

    # テスト期間の DataFrame を保持 (レジーム分析用)
    n = len(df)
    val_end = int(n * (CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"]))
    test_df = df.iloc[val_end:]

    # 3. モデルロード
    print("\n=== 3. モデルロード ===")
    if args.ckpt:
        ckpt_paths = [Path(p) for p in args.ckpt]
    else:
        ckpt_dir = ARTIFACT_DIR / "checkpoints"
        ckpt_paths = find_best_checkpoints(ckpt_dir, top_k=args.top_k)

    models = []
    for p in ckpt_paths:
        print(f"  Loading: {p.name}")
        models.append(load_model(p))
    print(f"  Loaded {len(models)} model(s)")

    # 4. アンサンブル予測
    print("\n=== 4. 予測 ===")
    ens_preds, actuals, encoder_last, pred_time_idx = ensemble_predict(models, test, CONFIG)
    print(f"  予測形状: {ens_preds.shape} (samples, horizons, quantiles)")
    print(f"  実績形状: {actuals.shape}")

    # time_idx → 日付 マッピング (各サンプルの予測開始日)
    time_idx_to_date = dict(zip(df["time_idx"].values.astype(int), df.index))
    sample_dates = pd.DatetimeIndex([
        time_idx_to_date.get(int(ti), pd.NaT) for ti in pred_time_idx.cpu().numpy()
    ])
    print(f"  予測期間: {sample_dates[0].date()} ~ {sample_dates[-1].date()}")

    # 5. メトリクス計算
    print("\n=== 5. メトリクス計算 ===")
    metrics = compute_full_metrics(ens_preds, actuals, CONFIG)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # 6. レポート保存
    with open(EVAL_DIR / "eval_report.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  -> eval_report.json")

    # 7. 可視化
    print("\n=== 6. 可視化 ===")

    # レジーム分析用: 各サンプルに対応するレジームを取得
    sample_regimes = pd.Series([
        test_df.loc[d, "market_regime"] if d in test_df.index and "market_regime" in test_df.columns else "unknown"
        for d in sample_dates
    ])
    regime_df = pd.DataFrame({"market_regime": sample_regimes.values})

    plot_equity_curve(ens_preds, actuals, sample_dates, CONFIG)
    plot_monthly_returns(ens_preds, actuals, sample_dates, CONFIG)
    plot_rolling_accuracy(ens_preds, actuals, sample_dates)
    plot_prediction_scatter(ens_preds, actuals)
    plot_quantile_fan(ens_preds, actuals, sample_dates)
    plot_regime_accuracy(ens_preds, actuals, regime_df)
    plot_horizon_accuracy(ens_preds, actuals)
    plot_pnl_distribution(ens_preds, actuals, CONFIG)
    plot_summary_dashboard(metrics)

    print(f"\n{'='*50}")
    print(f"評価完了! 結果: {EVAL_DIR}")
    print(f"  方向精度 (1d): {metrics.get('direction_1d', 'N/A')}")
    print(f"  方向精度 (5d): {metrics.get('direction_5d', 'N/A')}")
    print(f"  Sharpe:        {metrics.get('trade_sharpe_ratio', 'N/A')}")
    print(f"  Profit Factor: {metrics.get('trade_profit_factor', 'N/A')}")
    print(f"  Max Drawdown:  {metrics.get('trade_max_drawdown', 'N/A')}")
    print(f"{'='*50}")

    return metrics


if __name__ == "__main__":
    main()
