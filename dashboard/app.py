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
        with sqlite3.connect(str(PREDICTIONS_DB)) as conn:
            df = pd.read_sql("SELECT * FROM predictions ORDER BY prediction_date DESC, horizon ASC", conn)
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
def load_upcoming_events(today_str: str, days_ahead: int = 30) -> pd.DataFrame:
    """今日から days_ahead 日以内の経済イベントを返す。"""
    today = pd.Timestamp(today_str)
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
    st.caption(
        "TFT モデルによる USD/JPY の対数リターン予測。"
        "青線が中央値 (q50)、薄い帯が 90% 信頼区間 (q10–q90) を示す。"
        "帯が広いほど予測の不確実性が高い。"
    )
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
    """方向シグナル: ホライゾン別 UP/DOWN を信頼度付きで表示。"""
    st.subheader("方向シグナル")
    st.caption(
        "各ホライゾン (1〜5 営業日先) の方向予測。"
        "全分位点の加重平均 (direction signal) が閾値を上回れば UP、下回れば DOWN と判定する。"
        "信頼度 (HIGH/MEDIUM/LOW) はアンサンブル一致度・予測不確実性・シグナル強度の合成スコア。"
    )
    if preds.empty:
        st.info("データがありません。")
        return

    latest = _get_latest_predictions(preds)
    if latest.empty:
        st.info("データがありません。")
        return

    cols = st.columns(PREDICTION_LENGTH)
    for i, (_, row) in enumerate(latest.head(PREDICTION_LENGTH).iterrows()):
        direction = row.get("direction", "N/A")
        median_val = row.get("median", 0.0)
        dir_signal = row.get("direction_signal", median_val)
        conf_level = row.get("confidence_level", "")
        conf_score = row.get("confidence_score", None)
        should_trade = row.get("should_trade", None)

        if direction == "UP":
            arrow, delta_color = "↑", "normal"
        elif direction == "DOWN":
            arrow, delta_color = "↓", "inverse"
        else:
            arrow, delta_color = "－", "off"

        label = f"{int(row['horizon'])}日後 ({row['target_date'].strftime('%m/%d')})"
        cols[i].metric(
            label=label,
            value=f"{arrow} {direction}",
            delta=f"signal={dir_signal:.5f}",
            delta_color=delta_color,
        )
        if conf_level:
            trade_label = "Trade" if should_trade == 1 else "Skip"
            cols[i].caption(f"信頼度: {conf_level} ({trade_label})")


def panel_direction_ratio(report: dict) -> None:
    """方向比率モニター: 実績 vs 予測 up 比率の棒グラフ。"""
    st.subheader("方向比率モニター (実績 vs 予測)")
    st.caption(
        "テストセットにおける実績と予測の上昇比率を比較。"
        "両者が近いほどモデルのキャリブレーションが良好。"
        "大きな乖離はモデルが偏った予測をしている兆候。"
    )
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
    st.caption(
        "FOMC・BOJ・雇用統計 (NFP)・CPI など為替に影響する主要経済イベント。"
        "イベント前後はボラティリティが高まりやすく、予測の信頼区間が広がる傾向がある。"
    )
    if upcoming.empty:
        st.info("直近30日以内に主要イベントはありません。")
        return

    display = upcoming.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d (%a)")
    display = display.rename(columns={"date": "日付", "event_type": "イベント"})

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(display) + 1) + 10, 400),
    )


def panel_feature_importance() -> None:
    """特徴量重要度: TFT VSN から読み込み。"""
    st.subheader("特徴量重要度 (Variable Selection Network)")
    st.caption(
        "TFT 内部の Variable Selection Network が各入力特徴量に割り当てた重要度。"
        "スコアが高い特徴量ほどモデルの予測に強く寄与している。"
        "テクニカル・マクロ・イベント系のどの情報が効いているか把握できる。"
    )

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
    st.caption(
        "各予測ホライゾン (縦軸) がエンコーダーのどの時点 (横軸) に注目しているかを可視化。"
        "濃い色ほど注目度が高い。直近の時点に集中していれば短期情報重視、"
        "分散していれば長期パターンも活用していることを意味する。"
    )

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


def panel_prediction_tracking(preds: pd.DataFrame) -> None:
    """予測 vs 実績: 過去の予測の的中/外れを信頼度別に追跡。"""
    st.subheader("予測 vs 実績トラッキング")
    st.caption(
        "過去の予測に対して実際の値動きがどうだったかを追跡。"
        "信頼度が高い予測ほど的中率が高ければ、信頼度推定が有効に機能している。"
    )

    if preds.empty or "is_correct" not in preds.columns:
        st.info("データがありません。scripts/predict.py を実行してください。")
        return

    filled = preds[preds["is_correct"].notna()].copy()
    pending = preds[preds["is_correct"].isna()]

    if filled.empty:
        st.info(
            f"実績データがまだありません（{len(pending)} 件が待機中）。"
            "対象日を過ぎた後に predict.py を再実行すると実績が反映されます。"
        )
        return

    # --- 全体 vs 信頼度別の的中率サマリー ---
    has_confidence = "confidence_level" in filled.columns and filled["confidence_level"].notna().any()

    overall_acc = filled["is_correct"].mean()

    if has_confidence:
        high_data = filled[filled["confidence_level"] == "HIGH"]
        high_acc = high_data["is_correct"].mean() if not high_data.empty else None

        col_all, col_high = st.columns(2)
        col_all.metric("全体的中率", f"{overall_acc:.1%}", delta=f"{overall_acc - 0.5:+.1%} vs ランダム")
        if high_acc is not None:
            col_high.metric(
                "HIGH信頼の的中率",
                f"{high_acc:.1%}",
                delta=f"{high_acc - overall_acc:+.1%} vs 全体 (n={len(high_data)})",
            )
        else:
            col_high.metric("HIGH信頼の的中率", "N/A")
    else:
        st.metric("全体的中率", f"{overall_acc:.1%}", delta=f"{overall_acc - 0.5:+.1%} vs ランダム")

    # --- ホライゾン別 ---
    cols = st.columns(PREDICTION_LENGTH)
    for h in range(1, PREDICTION_LENGTH + 1):
        h_data = filled[filled["horizon"] == h]
        if h_data.empty:
            cols[h - 1].metric(f"{h}日後", "N/A")
        else:
            acc = h_data["is_correct"].mean()
            n = len(h_data)
            cols[h - 1].metric(f"{h}日後", f"{acc:.1%}", delta=f"n={n}")

    # --- 予測履歴テーブル ---
    st.markdown("#### 予測履歴")

    display = preds.sort_values(
        ["prediction_date", "horizon"], ascending=[False, True]
    ).copy()
    display["結果"] = display["is_correct"].map({1.0: "○", 0.0: "×"}).fillna("待機中")
    display["実績方向"] = display["actual_direction"].fillna("-")

    if has_confidence:
        display["信頼度"] = display["confidence_level"].fillna("-")
        table_cols = ["予測日", "対象日", "H", "予測方向", "実績方向", "信頼度", "結果"]
    else:
        table_cols = ["予測日", "対象日", "H", "予測方向", "実績方向", "結果"]

    table_df = display.rename(columns={
        "prediction_date": "予測日",
        "target_date": "対象日",
        "horizon": "H",
        "direction": "予測方向",
    })[table_cols]

    def highlight_result(row):
        if row["結果"] == "○":
            return ["background-color: rgba(0, 204, 150, 0.2)"] * len(row)
        elif row["結果"] == "×":
            return ["background-color: rgba(239, 85, 59, 0.2)"] * len(row)
        return [""] * len(row)

    styled = table_df.style.apply(highlight_result, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(table_df) + 1) + 10, 500),
    )

    # --- 的中率推移グラフ (全体 vs HIGH信頼) ---
    if len(filled["prediction_date"].unique()) >= 2:
        st.markdown("#### 的中率推移")
        by_date = (
            filled.groupby("prediction_date")["is_correct"]
            .mean()
            .reset_index()
            .sort_values("prediction_date")
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=by_date["prediction_date"],
            y=by_date["is_correct"],
            mode="lines+markers",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=8),
            name="全体",
        ))

        if has_confidence and not high_data.empty:
            high_by_date = (
                high_data.groupby("prediction_date")["is_correct"]
                .mean()
                .reset_index()
                .sort_values("prediction_date")
            )
            if len(high_by_date) >= 2:
                fig.add_trace(go.Scatter(
                    x=high_by_date["prediction_date"],
                    y=high_by_date["is_correct"],
                    mode="lines+markers",
                    line=dict(color="#00CC96", width=2),
                    marker=dict(size=8),
                    name="HIGH信頼のみ",
                ))

        fig.add_hline(
            y=0.5, line_dash="dash", line_color="gray",
            annotation_text="ランダム基準 (50%)",
        )
        fig.update_layout(
            yaxis=dict(title="的中率", range=[0, 1], tickformat=".0%"),
            xaxis_title="予測日",
            height=300,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)


def panel_confidence_detail(preds: pd.DataFrame) -> None:
    """信頼度詳細: 過去予測の信頼度別的中率。"""
    st.subheader("信頼度別パフォーマンス")
    st.caption(
        "過去の予測を信頼度レベル (HIGH/MEDIUM/LOW) 別に集計した的中率。"
        "HIGH の的中率が全体平均を上回っていれば、信頼度推定が有効に機能している。"
    )

    if preds.empty or "confidence_level" not in preds.columns:
        st.info("信頼度データがありません。最新の evaluate.py → predict.py を実行してください。")
        return

    filled = preds[preds["is_correct"].notna() & preds["confidence_level"].notna()].copy()
    if filled.empty:
        st.info("実績データがまだありません。")
        return

    rows = []
    for level in ["HIGH", "MEDIUM", "LOW"]:
        subset = filled[filled["confidence_level"] == level]
        if subset.empty:
            rows.append({"信頼度": level, "件数": 0, "的中率": "N/A"})
        else:
            acc = subset["is_correct"].mean()
            rows.append({
                "信頼度": level,
                "件数": len(subset),
                "的中率": f"{acc:.1%}",
            })

    tradeable = filled[filled["should_trade"] == 1]
    if not tradeable.empty:
        trade_acc = tradeable["is_correct"].mean()
        rows.append({
            "信頼度": "Trade対象のみ",
            "件数": len(tradeable),
            "的中率": f"{trade_acc:.1%}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def panel_accuracy_history(report: dict) -> None:
    """過去予測の精度: eval レポートからメトリクスを表示。"""
    st.subheader("過去予測の精度 (テストセット)")
    st.caption(
        "テストセットでの評価結果。MAE/RMSE は予測誤差の大きさ、方向精度は上昇/下降の的中率。"
        "方向精度が 50% (ランダム基準) を安定的に上回っているかが重要な判断材料。"
        "実績 UP 比率と予測 UP 比率の乖離 (比率ギャップ) が小さいほどキャリブレーションが良好。"
    )
    if not report or "horizons" not in report:
        st.info("データがありません。scripts/evaluate.py を実行してください。")
        return

    horizons = report["horizons"]
    rows = []
    accuracies_raw = []
    for h_key, metrics in horizons.items():
        h_num = int(h_key.split("_")[1])
        dir_acc = metrics.get('direction_accuracy', float('nan'))
        accuracies_raw.append(dir_acc)
        rows.append({
            "ホライゾン": f"{h_num}日後",
            "MAE": f"{metrics.get('mae', float('nan')):.6f}",
            "RMSE": f"{metrics.get('rmse', float('nan')):.6f}",
            "方向精度": f"{dir_acc:.1%}",
            "実績UP比率": f"{metrics.get('actual_up_ratio', float('nan')):.1%}",
            "予測UP比率": f"{metrics.get('pred_up_ratio', float('nan')):.1%}",
            "比率ギャップ": f"{metrics.get('direction_ratio_gap', float('nan')):.1%}",
        })

    df_report = pd.DataFrame(rows)
    st.dataframe(df_report, use_container_width=True, hide_index=True)

    # 方向精度の折れ線グラフ
    accuracies = accuracies_raw
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
    today_str = pd.Timestamp.today().normalize().isoformat()
    upcoming_events = load_upcoming_events(today_str, days_ahead=30)

    # --- 上段: 予測チャート + 方向シグナル ---
    panel_prediction_chart(preds_df)
    st.divider()
    panel_direction_signals(preds_df)
    st.divider()

    # --- 予測 vs 実績 ---
    panel_prediction_tracking(preds_df)
    st.divider()

    # --- 信頼度別パフォーマンス ---
    panel_confidence_detail(preds_df)
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
