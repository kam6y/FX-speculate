"""USD/JPY TFT 予測ダッシュボード。

Usage:
    uv run streamlit run dashboard/app.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ARTIFACT_DIR, PREDICTION_LENGTH, PREDICTIONS_DB
from data.events import get_all_event_dates

st.set_page_config(page_title="USD/JPY TFT 予測", layout="wide")

EVAL_DIR = ARTIFACT_DIR / "eval"

# ---------------------------------------------------------------------------
# データロード関数
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    """SQLite から予測結果を読み込む。"""
    if not PREDICTIONS_DB.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(PREDICTIONS_DB))
        df = pd.read_sql("SELECT * FROM predictions ORDER BY prediction_date DESC, horizon ASC", conn)
        conn.close()
        df["target_date"] = pd.to_datetime(df["target_date"])
        df["prediction_date"] = pd.to_datetime(df["prediction_date"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_eval_report() -> dict:
    """artifacts/eval/eval_report.json を読み込む。"""
    report_path = EVAL_DIR / "eval_report.json"
    if not report_path.exists():
        return {}
    try:
        with open(report_path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_upcoming_events(days_ahead: int = 30) -> pd.DataFrame:
    """今日から days_ahead 日以内の経済イベントを返す。"""
    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)

    years = sorted({today.year, cutoff.year})
    frames = [get_all_event_dates(y) for y in years]
    all_events = pd.concat(frames, ignore_index=True)
    mask = (all_events["date"] >= today) & (all_events["date"] <= cutoff)
    upcoming = all_events[mask].sort_values("date").reset_index(drop=True)
    return upcoming


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

EVENT_COLOR = {
    "FOMC": "#EF553B",
    "BOJ": "#636EFA",
    "NFP": "#00CC96",
    "CPI": "#AB63FA",
    "GDP": "#FFA15A",
    "ISM": "#19D3F3",
    "RETAIL": "#FF6692",
    "JACKSON_HOLE": "#B6E880",
}


def _get_latest_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """最新 prediction_date の予測行だけを返す。"""
    if df.empty:
        return df
    latest_date = df["prediction_date"].max()
    return df[df["prediction_date"] == latest_date].sort_values("horizon")


# ---------------------------------------------------------------------------
# パネル
# ---------------------------------------------------------------------------

def panel_prediction_chart(preds: pd.DataFrame) -> None:
    """予測チャート: ファンチャート付き。"""
    st.subheader("予測チャート")
    if preds.empty:
        st.info("データがありません。scripts/predict.py を実行してください。")
        return

    latest = _get_latest_predictions(preds)
    if latest.empty:
        st.info("データがありません。")
        return

    x_dates = latest["target_date"].tolist()
    median = latest["median"].tolist()
    q10 = latest["q10"].tolist()
    q90 = latest["q90"].tolist()

    fig = go.Figure()

    # 信頼区間 (ファン)
    fig.add_trace(go.Scatter(
        x=x_dates + x_dates[::-1],
        y=q90 + q10[::-1],
        fill="toself",
        fillcolor="rgba(99,110,250,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% 信頼区間",
        hoverinfo="skip",
    ))

    # 中央値
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=median,
        mode="lines+markers",
        line=dict(color="#636EFA", width=2),
        marker=dict(size=8),
        name="予測中央値 (log return)",
    ))

    # ゼロライン
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)

    fig.update_layout(
        title=f"USD/JPY 予測 (予測日: {latest['prediction_date'].iloc[0].date()})",
        xaxis_title="対象日",
        yaxis_title="予測 log return",
        height=400,
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def panel_direction_signals(preds: pd.DataFrame) -> None:
    """方向シグナル: ホライゾン別 UP/DOWN を st.metric で表示。"""
    st.subheader("方向シグナル")
    if preds.empty:
        st.info("データがありません。")
        return

    latest = _get_latest_predictions(preds)
    if latest.empty:
        st.info("データがありません。")
        return

    cols = st.columns(PREDICTION_LENGTH)
    for i, (_, row) in enumerate(latest.iterrows()):
        direction = row.get("direction", "N/A")
        median_val = row.get("median", 0.0)
        threshold = row.get("threshold", 0.0)
        delta = median_val - threshold

        arrow = "↑" if direction == "UP" else "↓"
        label = f"{int(row['horizon'])}日後 ({row['target_date'].strftime('%m/%d')})"
        cols[i].metric(
            label=label,
            value=f"{arrow} {direction}",
            delta=f"median={median_val:.5f}",
            delta_color="normal" if direction == "UP" else "inverse",
        )


def panel_direction_ratio(report: dict) -> None:
    """方向比率モニター: 実績 vs 予測 up 比率の棒グラフ。"""
    st.subheader("方向比率モニター (実績 vs 予測)")
    if not report or "horizons" not in report:
        st.info("データがありません。scripts/evaluate.py を実行してください。")
        return

    horizons = list(report["horizons"].keys())
    horizon_labels = [f"{i+1}d" for i in range(len(horizons))]
    actual_ratios = [report["horizons"][h]["actual_up_ratio"] for h in horizons]
    pred_ratios = [report["horizons"][h]["pred_up_ratio"] for h in horizons]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="実績",
        x=horizon_labels,
        y=actual_ratios,
        marker_color="#636EFA",
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="予測",
        x=horizon_labels,
        y=pred_ratios,
        marker_color="#EF553B",
        opacity=0.85,
    ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="上昇比率", range=[0, 1]),
        xaxis_title="予測ホライゾン",
        height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)


def panel_event_calendar(upcoming: pd.DataFrame) -> None:
    """イベントカレンダー: 直近の経済イベント一覧。"""
    st.subheader("イベントカレンダー (直近30日)")
    if upcoming.empty:
        st.info("直近30日以内に主要イベントはありません。")
        return

    display = upcoming.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d (%a)")
    display.columns = ["日付", "イベント"]

    # カラーバッジ付きで表示
    def color_event(row):
        color = EVENT_COLOR.get(row["イベント"], "#888888")
        return [f"color: {color}; font-weight: bold" if c == "イベント" else "" for c in display.columns]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(display) + 1) + 10, 400),
    )


def panel_feature_importance() -> None:
    """特徴量重要度: TFT VSN から読み込み。"""
    st.subheader("特徴量重要度 (Variable Selection Network)")

    importance_path = ARTIFACT_DIR / "feature_importance.json"
    if not importance_path.exists():
        st.info("データがありません。モデル学習後に artifacts/feature_importance.json が生成されます。")
        return

    try:
        with open(importance_path) as f:
            importance = json.load(f)
    except Exception as e:
        st.warning(f"読み込みエラー: {e}")
        return

    if not importance:
        st.info("データがありません。")
        return

    names = list(importance.keys())
    values = [float(v) for v in importance.values()]
    sorted_pairs = sorted(zip(values, names), reverse=True)
    values_sorted, names_sorted = zip(*sorted_pairs)

    fig = go.Figure(go.Bar(
        x=list(values_sorted),
        y=list(names_sorted),
        orientation="h",
        marker_color="#636EFA",
    ))
    fig.update_layout(
        xaxis_title="重要度スコア",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(names) * 22),
    )
    st.plotly_chart(fig, use_container_width=True)


def panel_attention_heatmap() -> None:
    """Attention ヒートマップ: 時系列 attention 重みを表示。"""
    st.subheader("Attention ヒートマップ")

    attn_path = ARTIFACT_DIR / "attention_weights.json"
    if not attn_path.exists():
        st.info("データがありません。モデル学習後に artifacts/attention_weights.json が生成されます。")
        return

    try:
        with open(attn_path) as f:
            attn_data = json.load(f)
    except Exception as e:
        st.warning(f"読み込みエラー: {e}")
        return

    if not attn_data:
        st.info("データがありません。")
        return

    # attn_data: {"weights": [[...], ...], "encoder_steps": [...], "decoder_steps": [...]}
    weights = np.array(attn_data.get("weights", []))
    encoder_steps = attn_data.get("encoder_steps", [f"t-{i}" for i in range(weights.shape[1] if weights.ndim > 1 else 0)])
    decoder_steps = attn_data.get("decoder_steps", [f"t+{i+1}" for i in range(weights.shape[0] if weights.ndim > 0 else 0)])

    if weights.ndim != 2 or weights.size == 0:
        st.info("データがありません。")
        return

    fig = go.Figure(go.Heatmap(
        z=weights,
        x=encoder_steps,
        y=decoder_steps,
        colorscale="Blues",
        colorbar=dict(title="Attention"),
    ))
    fig.update_layout(
        xaxis_title="エンコーダーステップ",
        yaxis_title="デコーダーステップ",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def panel_accuracy_history(report: dict) -> None:
    """過去予測の精度: eval レポートからメトリクスを表示。"""
    st.subheader("過去予測の精度 (テストセット)")
    if not report or "horizons" not in report:
        st.info("データがありません。scripts/evaluate.py を実行してください。")
        return

    horizons = report["horizons"]
    rows = []
    for h_key, metrics in horizons.items():
        h_num = int(h_key.split("_")[1])
        rows.append({
            "ホライゾン": f"{h_num}日後",
            "MAE": f"{metrics.get('mae', float('nan')):.6f}",
            "RMSE": f"{metrics.get('rmse', float('nan')):.6f}",
            "方向精度": f"{metrics.get('direction_accuracy', float('nan')):.1%}",
            "実績UP比率": f"{metrics.get('actual_up_ratio', float('nan')):.1%}",
            "予測UP比率": f"{metrics.get('pred_up_ratio', float('nan')):.1%}",
            "比率ギャップ": f"{metrics.get('direction_ratio_gap', float('nan')):.1%}",
        })

    df_report = pd.DataFrame(rows)
    st.dataframe(df_report, use_container_width=True, hide_index=True)

    # 方向精度の折れ線グラフ
    accuracies = [float(r["方向精度"].rstrip("%")) / 100 for r in rows]
    h_labels = [r["ホライゾン"] for r in rows]

    fig = go.Figure(go.Scatter(
        x=h_labels,
        y=accuracies,
        mode="lines+markers",
        line=dict(color="#00CC96", width=2),
        marker=dict(size=10),
        name="方向精度",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="ランダム基準 (50%)")
    fig.update_layout(
        yaxis=dict(title="方向精度", range=[0, 1], tickformat=".0%"),
        xaxis_title="予測ホライゾン",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# メインレイアウト
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("USD/JPY TFT 予測ダッシュボード")
    st.caption(f"最終更新: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # データロード
    preds_df = load_predictions()
    eval_report = load_eval_report()
    upcoming_events = load_upcoming_events(days_ahead=30)

    # --- 上段: 予測チャート + 方向シグナル ---
    panel_prediction_chart(preds_df)
    st.divider()
    panel_direction_signals(preds_df)
    st.divider()

    # --- 中段: 方向比率 + イベントカレンダー ---
    col_left, col_right = st.columns([3, 2])
    with col_left:
        panel_direction_ratio(eval_report)
    with col_right:
        panel_event_calendar(upcoming_events)

    st.divider()

    # --- 下段: 特徴量重要度 + Attention ヒートマップ ---
    col_feat, col_attn = st.columns(2)
    with col_feat:
        panel_feature_importance()
    with col_attn:
        panel_attention_heatmap()

    st.divider()

    # --- 最下段: 過去予測の精度 ---
    panel_accuracy_history(eval_report)

    # サイドバー: リフレッシュ
    with st.sidebar:
        st.header("設定")
        if st.button("データ更新"):
            st.cache_data.clear()
            st.rerun()

        if not preds_df.empty:
            all_dates = sorted(preds_df["prediction_date"].dt.date.unique(), reverse=True)
            st.subheader("予測履歴")
            st.write(f"記録件数: {len(preds_df)}")
            st.write(f"最新予測日: {all_dates[0]}")
            st.write(f"総予測日数: {len(all_dates)}")


if __name__ == "__main__":
    main()
