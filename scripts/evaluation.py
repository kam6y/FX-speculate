# scripts/evaluation.py
"""共通バックテスト・評価モジュール

v6のバックテスト・評価ロジックを共通化。
コストモデル: スプレッド0.2pips(Bid/Ask時不要) + スリッページ0.1pips + API手数料0.002%
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CONFIG = {
    "PREDICT_HORIZON": 15,
    "PIP_SIZE": 0.01,
    "SPREAD_PIPS": 0.2,
    "SLIPPAGE_PIPS": 0.1,
    "API_FEE_RATE": 0.00002,  # 0.002% per side
    "POSITION_SIZE": 10000,
    "BAD_HOURS": [20, 21, 22, 23],
    "ATR_PERCENTILE": 30,
    "EVAL_USE_ATR_FILTER": True,
    "EVAL_USE_BAD_HOURS": True,
    "BAR_PER_YEAR": int(252 * 24 * 60),
    "MIN_TRADES_ABS": 300,
    "MIN_TRADES_RATE": 0.003,
    "MIN_SIDE_TRADES": 20,
}


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: list[float]
    eligible_bars: int
    use_bid_ask: bool
    total_cost_pips: float


def predict_with_thresholds(
    probs: np.ndarray, threshold_buy: float, threshold_sell: float
) -> np.ndarray:
    preds = np.zeros(len(probs), dtype=int)
    for i, (hold_p, buy_p, sell_p) in enumerate(probs):
        if buy_p > threshold_buy and buy_p > sell_p:
            preds[i] = 1
        elif sell_p > threshold_sell and sell_p > buy_p:
            preds[i] = 2
    return preds


def calculate_sharpe_ratio(
    returns: np.ndarray, bar_per_year: int, trade_rate: float
) -> float:
    if len(returns) == 0:
        return 0.0
    r = np.asarray(returns, dtype=float)
    if r.std() == 0:
        return float("inf") if r.mean() > 0 else 0.0
    periods_per_year = max(1.0, bar_per_year * trade_rate)
    return float(np.sqrt(periods_per_year) * (r.mean() / r.std()))


def calc_trade_pnl(
    preds: np.ndarray, future_returns_pips: np.ndarray, cost_pips: float
) -> np.ndarray:
    pnl = []
    for pred, fr in zip(preds, future_returns_pips):
        if pred == 1:
            pnl.append(fr - cost_pips)
        elif pred == 2:
            pnl.append(-fr - cost_pips)
    return np.array(pnl) if pnl else np.array([])


def trade_penalty(
    trade_count: int, total_count: int, buy_count: int, sell_count: int, config: dict,
) -> float:
    if total_count == 0:
        return 0.0
    min_trades = max(config["MIN_TRADES_ABS"], int(config["MIN_TRADES_RATE"] * total_count))
    rate_factor = min(1.0, trade_count / max(1, min_trades))
    side_factor = min(1.0, min(buy_count, sell_count) / max(1, config["MIN_SIDE_TRADES"]))
    balance_factor = min(buy_count, sell_count) / max(1, max(buy_count, sell_count))
    return rate_factor * side_factor * (0.5 + 0.5 * balance_factor)


def score_trading(
    preds: np.ndarray, future_returns_pips: np.ndarray, total_count: int, config: dict, cost_pips: float,
) -> float:
    trade_count = int((preds != 0).sum())
    buy_count = int((preds == 1).sum())
    sell_count = int((preds == 2).sum())
    trade_rate = trade_count / total_count if total_count > 0 else 0.0
    pnl_pips = calc_trade_pnl(preds, future_returns_pips, cost_pips)
    sharpe = calculate_sharpe_ratio(pnl_pips, config["BAR_PER_YEAR"], trade_rate)
    penalty = trade_penalty(trade_count, total_count, buy_count, sell_count, config)
    return sharpe * penalty - (1 - penalty)


def build_live_filter(
    index: pd.DatetimeIndex, df_features: pd.DataFrame, config: dict, atr_threshold: float,
) -> pd.Series:
    eligible = pd.Series(True, index=index)
    if config.get("EVAL_USE_BAD_HOURS", True):
        eligible &= ~index.hour.isin(config["BAD_HOURS"])
    if config.get("EVAL_USE_ATR_FILTER", True) and "volatility_atr" in df_features.columns:
        eligible &= df_features.loc[index, "volatility_atr"] >= atr_threshold
    return eligible


def run_backtest(
    predictions: np.ndarray, timestamps: pd.DatetimeIndex, price_df: pd.DataFrame,
    feature_df: pd.DataFrame, config: dict,
) -> BacktestResult:
    use_bid_ask = {"ask_close", "bid_close"}.issubset(price_df.columns)
    slippage_cost = config["SLIPPAGE_PIPS"] * config["PIP_SIZE"]
    spread_cost = 0.0 if use_bid_ask else config["SPREAD_PIPS"] * config["PIP_SIZE"]
    atr_threshold = config.get("atr_threshold", 0.0)
    eligible_mask = build_live_filter(timestamps, feature_df, config, atr_threshold)

    results = []
    equity_curve = [0.0]
    eligible_bars = 0

    for i, idx in enumerate(timestamps):
        if not eligible_mask.iloc[i]:
            continue
        exit_idx = idx + pd.Timedelta(minutes=config["PREDICT_HORIZON"])
        if exit_idx not in price_df.index:
            continue
        eligible_bars += 1
        pred = predictions[i]
        if pred == 0:
            continue

        if use_bid_ask:
            entry_buy = price_df.loc[idx, "ask_close"]
            exit_buy = price_df.loc[exit_idx, "bid_close"]
            entry_sell = price_df.loc[idx, "bid_close"]
            exit_sell = price_df.loc[exit_idx, "ask_close"]
        else:
            entry_buy = price_df.loc[idx, "close"]
            exit_buy = price_df.loc[exit_idx, "close"]
            entry_sell = entry_buy
            exit_sell = exit_buy

        api_fee_rate = config.get("API_FEE_RATE", 0.0)

        if pred == 1:
            entry_price, exit_price = entry_buy, exit_buy
            raw_pnl = (exit_price - entry_price) * config["POSITION_SIZE"]
            api_fee = (entry_price + exit_price) * config["POSITION_SIZE"] * api_fee_rate
            pnl = raw_pnl - slippage_cost * config["POSITION_SIZE"] - spread_cost * config["POSITION_SIZE"] - api_fee
            signal = "BUY"
        else:
            entry_price, exit_price = entry_sell, exit_sell
            raw_pnl = (entry_price - exit_price) * config["POSITION_SIZE"]
            api_fee = (entry_price + exit_price) * config["POSITION_SIZE"] * api_fee_rate
            pnl = raw_pnl - slippage_cost * config["POSITION_SIZE"] - spread_cost * config["POSITION_SIZE"] - api_fee
            signal = "SELL"

        results.append({
            "timestamp": idx, "signal": signal, "entry_price": entry_price,
            "exit_price": exit_price, "pnl": pnl, "win": pnl > 0,
        })
        equity_curve.append(equity_curve[-1] + pnl)

    if not results:
        raise ValueError("No trades generated")

    total_cost_pips = config["SLIPPAGE_PIPS"] + (0 if use_bid_ask else config["SPREAD_PIPS"])
    return BacktestResult(
        trades=pd.DataFrame(results), equity_curve=equity_curve,
        eligible_bars=eligible_bars, use_bid_ask=use_bid_ask, total_cost_pips=total_cost_pips,
    )


def compute_metrics(result: BacktestResult, config: dict = None) -> dict:
    if config is None:
        config = DEFAULT_CONFIG
    df = result.trades
    total_trades = len(df)
    wins = df["win"].sum()
    losses = total_trades - wins
    total_pnl = df["pnl"].sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_win = df[df["win"]]["pnl"].mean() if wins > 0 else 0
    avg_loss = df[~df["win"]]["pnl"].mean() if losses > 0 else 0
    gross_profit = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = df[df["pnl"] < 0]["pnl"].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")
    equity = pd.Series(result.equity_curve)
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = drawdown.min()
    trade_rate = total_trades / result.eligible_bars if result.eligible_bars > 0 else 0
    sharpe = calculate_sharpe_ratio(df["pnl"].values, config["BAR_PER_YEAR"], trade_rate)
    avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
    expected_trades_year = config["BAR_PER_YEAR"] * trade_rate
    annual_return = avg_trade_pnl * expected_trades_year
    calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else float("inf")
    return {
        "sharpe_ratio": sharpe, "total_pnl": total_pnl, "win_rate": win_rate,
        "profit_factor": profit_factor, "max_drawdown": max_drawdown,
        "calmar_ratio": calmar, "total_trades": total_trades,
        "avg_win": avg_win, "avg_loss": avg_loss, "trade_rate": trade_rate,
    }


def plot_equity_curve(result: BacktestResult, title: str = "Equity Curve") -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.equity_curve, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L (JPY)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
