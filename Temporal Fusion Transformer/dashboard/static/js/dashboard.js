/**
 * TFT USD/JPY Dashboard — Kintsukuroi Terminal
 */

// ─── Theme ───
const T = {
    ink: "#1A1A1A",
    ink60: "#6B6860",
    ink40: "#9B9890",
    sand: "#E8E6DD",
    cream: "#F4F3ED",
    gold: "#C4A265",
    up: "#2D6A4F",
    upBg: "rgba(45,106,79,0.06)",
    down: "#AE4A34",
    downBg: "rgba(174,74,52,0.06)",
    gridLine: "rgba(26,26,26,0.04)",
    font: "'IBM Plex Mono', monospace",
};

// ─── State ───
let backtestData = null;
let metricsData = null;
let predictionData = null;
let charts = {};

// ─── Chart.js Defaults ───
Chart.defaults.font.family = T.font;
Chart.defaults.font.size = 11;
Chart.defaults.color = T.ink40;

// ─── Chart.js Plugin: Horizontal Reference Lines ───
const hLinePlugin = {
    id: "horizontalLine",
    afterDraw(chart) {
        const lines = chart.options.plugins?.horizontalLine?.lines;
        if (!lines) return;
        const { ctx } = chart;
        const yAxis = chart.scales.y;
        lines.forEach(line => {
            const yPx = yAxis.getPixelForValue(line.value);
            if (yPx === undefined) return;
            ctx.save();
            ctx.beginPath();
            ctx.setLineDash(line.dash || [5, 3]);
            ctx.strokeStyle = line.color || T.ink40;
            ctx.lineWidth = line.width || 1;
            ctx.moveTo(chart.chartArea.left, yPx);
            ctx.lineTo(chart.chartArea.right, yPx);
            ctx.stroke();
            if (line.label) {
                ctx.fillStyle = line.color || T.ink40;
                ctx.font = `10px ${T.font}`;
                ctx.textAlign = "left";
                ctx.fillText(line.label, chart.chartArea.left + 6, yPx - 5);
            }
            ctx.restore();
        });
    }
};
Chart.register(hLinePlugin);

// ─── Init ───
document.addEventListener("DOMContentLoaded", () => {
    loadAlerts();
    loadPredictionHistory();
    loadSchedulerStatus();
    // 60秒ごとにスケジューラ状態を更新
    setInterval(loadSchedulerStatus, 60000);
});

// ─── Scheduler ───
async function loadSchedulerStatus() {
    try {
        const res = await fetch("/api/scheduler");
        const data = await res.json();
        const el = document.getElementById("scheduler-next");
        if (!el) return;
        if (data.status === "running" && data.jobs.length > 0) {
            el.textContent = "Next: " + data.jobs[0].next_run_display;
        } else {
            el.textContent = data.status;
        }
    } catch (e) { /* ignore */ }
}

// ─── Shared Chart Helpers ───
function xAxisConfig(allDates) {
    const monthStarts = new Set();
    let lastYM = "";
    allDates.forEach((d, i) => {
        const ym = d.slice(0, 7);
        if (ym !== lastYM) { lastYM = ym; monthStarts.add(i); }
    });
    return {
        ticks: {
            maxRotation: 0,
            autoSkip: false,
            callback: (val, idx) => (idx > 0 && monthStarts.has(idx)) ? allDates[idx].slice(0, 7) : null,
        },
        grid: { display: false },
        border: { display: false },
    };
}

const sharedTooltip = {
    backgroundColor: "rgba(26,26,26,0.92)",
    titleFont: { family: T.font, size: 11 },
    bodyFont: { family: T.font, size: 11 },
    cornerRadius: 6,
    padding: 10,
    displayColors: true,
    boxWidth: 8,
    boxHeight: 8,
    boxPadding: 4,
};

// ─── Metrics ───
function renderMetrics(m) {
    const grid = document.getElementById("metrics-grid");
    if (!grid) return;
    const placeholder = document.getElementById("metrics-placeholder");
    if (placeholder) placeholder.remove();
    const items = [
        { label: "1D Direction", value: fmtPct(m.ensemble_direction_1d || m.direction_accuracy), cls: accCls(m.ensemble_direction_1d || m.direction_accuracy), tip: "翌営業日の方向(上昇/下落)を正しく予測できた割合。50%超でランダム以上の予測力。" },
        { label: "Sharpe", value: fmt(m.trade_sharpe_ratio, 2), cls: m.trade_sharpe_ratio > 1 ? "metric-good" : "metric-warn", tip: "Sharpe = E[R] / σ(R)（平均リターン ÷ リターンの標準偏差）。リスク1単位あたりのリターンを示す。1.0以上で良好、2.0以上で優秀。" },
        { label: "Profit Factor", value: fmt(m.trade_profit_factor, 2), cls: m.trade_profit_factor > 1 ? "metric-good" : "metric-bad", tip: "総利益 ÷ 総損失。1.0超で利益優位。1.5以上が目安。値が大きいほど損小利大の傾向。" },
        { label: "Total PnL", value: fmt(m.trade_total_pnl, 2), cls: m.trade_total_pnl > 0 ? "metric-good" : "metric-bad", tip: "テスト期間の累計損益(対数リターン合計)。正の値で利益、負の値で損失。" },
        { label: "Max Drawdown", value: fmtPct(m.trade_max_drawdown), cls: m.trade_max_drawdown > -0.1 ? "metric-warn" : "metric-bad", tip: "ピークからの最大下落率。リスク許容の目安。-10%以内が望ましい。-15%超で要警戒。" },
        { label: "MAE 1D", value: fmt(m.mae_1d || m.mae, 4), cls: "", tip: "平均絶対誤差。予測リターンと実際のリターンの平均的なずれ。小さいほど予測精度が高い。" },
    ];
    grid.innerHTML = items.map((it, i) => `
        <div class="metric-card" style="animation: fadeUp 0.4s ease ${0.05 * i}s both">
            <div class="metric-label">${it.label}</div>
            <div class="metric-value ${it.cls}">${it.value}</div>
            ${it.sub ? `<div class="metric-sub">${it.sub}</div>` : ""}
            <div class="metric-tooltip">${it.tip}</div>
        </div>
    `).join("");
}

// ─── Prediction ───
async function runPrediction() {
    const btn = document.getElementById("btn-predict");
    const status = document.getElementById("predict-status");
    btn.disabled = true;
    status.innerHTML = '<span class="spinner"></span> Predicting...';
    try {
        const res = await fetch("/api/predict");
        if (!res.ok) { const err = await res.json(); throw new Error(err.error); }
        predictionData = await res.json();
        renderPrediction(predictionData);
        status.innerHTML = `<span class="status-dot"></span> ${new Date().toLocaleTimeString("ja-JP")}`;
        loadPredictionHistory();
    } catch (e) {
        status.textContent = e.message;
        console.error(e);
    } finally { btn.disabled = false; }
}

function renderPrediction(data) {
    if (!data?.horizons?.length) return;
    const h1 = data.horizons[0];
    const isUp = h1.direction === "UP";
    const cls = isUp ? "up" : "down";

    // Direction ring
    const ring = document.getElementById("direction-ring");
    ring.className = `direction-ring ${cls}`;

    const dirEl = document.getElementById("pred-direction");
    dirEl.className = `prediction-direction ${cls}`;
    dirEl.textContent = isUp ? "\u2191 UP" : "\u2193 DOWN";

    document.getElementById("pred-target-date").textContent = h1.target_date;
    document.getElementById("pred-log-return").textContent =
        `${h1.pred_log_return > 0 ? "+" : ""}${(h1.pred_log_return * 100).toFixed(3)}%`;

    // Meta
    document.getElementById("pred-price").textContent =
        data.current_price ? `\u00a5${data.current_price.toFixed(2)}` : "---";
    document.getElementById("pred-date").textContent = data.prediction_date;
    document.getElementById("pred-q-range").textContent =
        `${(h1.q10 * 100).toFixed(3)}% ~ ${(h1.q90 * 100).toFixed(3)}%`;

    // Confidence
    const conf = h1.confidence;
    const fill = document.getElementById("confidence-fill");
    fill.style.width = `${conf * 100}%`;
    fill.style.background = conf > 0.5 ? T.up : conf > 0.2 ? T.gold : T.down;
    document.getElementById("confidence-value").textContent = `${(conf * 100).toFixed(0)}%`;

    // Multi-horizon
    document.getElementById("horizon-tbody").innerHTML = data.horizons.map((h, i) => {
        const dc = h.direction === "UP" ? "up" : "down";
        return `<tr style="animation: slideIn 0.3s ease ${0.03 * i}s both">
            <td><strong>${h.horizon}D</strong></td>
            <td>${h.target_date}</td>
            <td><span class="dir-badge ${dc}">${h.direction}</span></td>
            <td>${(h.pred_log_return * 100).toFixed(3)}%</td>
            <td style="font-size:10px">${(h.q10 * 100).toFixed(2)}~${(h.q90 * 100).toFixed(2)}</td>
            <td>${(h.confidence * 100).toFixed(0)}%</td>
        </tr>`;
    }).join("");

    document.getElementById("prediction-hero").style.display = "grid";
    document.getElementById("prediction-empty").style.display = "none";
}

// ─── Backtest ───
async function loadBacktest() {
    const btn = document.getElementById("btn-backtest");
    const overlay = document.getElementById("backtest-loading");
    btn.disabled = true;
    overlay.style.display = "flex";
    try {
        // バックテスト(キャッシュ)・ライブエクイティ・メトリクスを並列取得
        const [btRes, liveRes, metricsRes] = await Promise.all([
            fetch("/api/backtest"),
            fetch("/api/live-equity"),
            fetch("/api/metrics"),
        ]);
        if (!btRes.ok) throw new Error("Backtest failed");
        backtestData = await btRes.json();

        let liveData = null;
        if (liveRes.ok) {
            liveData = await liveRes.json();
            if (!liveData.dates || liveData.dates.length === 0) liveData = null;
        }

        // Hide chart placeholders
        document.querySelectorAll('.chart-placeholder').forEach(el => el.classList.add('hidden'));

        // tft_metrics.json のアンサンブルメトリクスを表示
        if (metricsRes.ok) {
            metricsData = await metricsRes.json();
            renderMetrics(metricsData);
        }

        renderEquityCurve(backtestData, liveData);
        renderRollingAccuracy(backtestData, liveData);
        renderDrawdown(backtestData);
        renderMonthlyHeatmap(backtestData);
        renderConfidenceCalibration(backtestData);
        renderQuantileCalibration(backtestData);
        renderMultiHorizonAccuracy(backtestData);
    } catch (e) { console.error("Backtest:", e); }
    finally { btn.disabled = false; overlay.style.display = "none"; }
}

function renderEquityCurve(data, liveData) {
    const ctx = document.getElementById("chart-equity");
    if (!ctx) return;
    if (charts.equity) charts.equity.destroy();

    const datasets = [
        {
            label: "Backtest",
            data: data.equity_curve,
            borderColor: T.up,
            backgroundColor: T.upBg,
            fill: true,
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.25,
        },
        {
            label: "Buy & Hold",
            data: data.buy_hold,
            borderColor: T.ink40,
            borderWidth: 1,
            borderDash: [4, 3],
            pointRadius: 0,
            tension: 0.25,
            fill: false,
        },
    ];

    // ライブエクイティがあれば追加 (バックテスト末尾の後ろに配置)
    let allDates = [...data.dates];
    if (liveData && liveData.dates.length > 0) {
        // ライブデータ用のpadding (バックテスト期間は null)
        const liveEquity = new Array(data.dates.length).fill(null);
        const lastBt = data.equity_curve[data.equity_curve.length - 1] || 0;

        liveData.dates.forEach((d, i) => {
            if (!allDates.includes(d)) {
                allDates.push(d);
                // バックテストデータセットのpaddingも追加
                datasets[0].data.push(null);
                datasets[1].data.push(null);
            }
            liveEquity.push(lastBt + liveData.equity_curve[i]);
        });

        datasets.push({
            label: "Live",
            data: liveEquity,
            borderColor: T.gold,
            backgroundColor: "rgba(196,162,101,0.06)",
            fill: true,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
            borderDash: [],
        });
    }

    charts.equity = new Chart(ctx, {
        type: "line",
        data: {
            labels: allDates,
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { position: "top", align: "end", labels: { usePointStyle: true, boxWidth: 6, padding: 16 } },
                tooltip: { ...sharedTooltip,
                    callbacks: {
                        title: items => data.dates[items[0].dataIndex],
                        label: c => ` ${c.dataset.label}: ${(c.parsed.y * 100).toFixed(2)}%`
                    }
                },
                horizontalLine: {
                    lines: [{ value: 0, color: T.ink40, width: 0.8, dash: [3, 3] }]
                },
            },
            scales: {
                x: xAxisConfig(data.dates),
                y: {
                    ticks: { callback: v => (v * 100).toFixed(0) + "%" },
                    grid: { color: T.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

function renderDrawdown(data) {
    const ctx = document.getElementById("chart-drawdown");
    if (!ctx) return;
    if (charts.drawdown) charts.drawdown.destroy();

    charts.drawdown = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.dates,
            datasets: [{
                data: data.drawdown,
                borderColor: T.down,
                backgroundColor: T.downBg,
                fill: true,
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.25,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { ...sharedTooltip,
                    callbacks: {
                        title: items => data.dates[items[0].dataIndex],
                        label: c => ` DD: ${(c.parsed.y * 100).toFixed(2)}%`
                    }
                },
            },
            scales: {
                x: xAxisConfig(data.dates),
                y: {
                    ticks: { callback: v => (v * 100).toFixed(1) + "%" },
                    grid: { color: T.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

function renderRollingAccuracy(data, liveData) {
    const ctx = document.getElementById("chart-rolling-acc");
    if (!ctx) return;
    if (charts.rolling) charts.rolling.destroy();

    const vi = []; data.rolling_accuracy.forEach((v, i) => { if (v !== null) vi.push(i); });
    const acc = vi.map(i => data.rolling_accuracy[i]);
    const dates = vi.map(i => data.dates[i]);
    const avg = acc.reduce((a, b) => a + b, 0) / acc.length;
    const accMin = Math.min(...acc);
    const accMax = Math.max(...acc);
    const yMin = Math.floor((Math.min(accMin, 0.5) - 0.05) * 10) / 10;
    const yMax = Math.ceil((accMax + 0.05) * 10) / 10;

    charts.rolling = new Chart(ctx, {
        type: "line",
        data: {
            labels: dates,
            datasets: [{
                data: acc,
                borderColor: T.gold,
                backgroundColor: "rgba(196,162,101,0.06)",
                fill: true,
                borderWidth: 1.8,
                pointRadius: 0,
                tension: 0.3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { ...sharedTooltip,
                    callbacks: {
                        title: items => dates[items[0].dataIndex],
                        label: c => ` Accuracy: ${(c.parsed.y * 100).toFixed(1)}%`
                    }
                },
                horizontalLine: {
                    lines: [
                        { value: 0.5, color: T.down, width: 1.2, dash: [6, 4], label: "50% Random" },
                        { value: avg, color: T.up, width: 1, dash: [3, 3], label: `Avg ${(avg * 100).toFixed(1)}%` },
                    ]
                },
            },
            scales: {
                x: xAxisConfig(dates),
                y: {
                    min: yMin, max: yMax,
                    ticks: { stepSize: 0.1, callback: v => (v * 100).toFixed(0) + "%" },
                    grid: { color: T.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

function renderMonthlyHeatmap(data) {
    const el = document.getElementById("monthly-heatmap");
    if (!el || !data.monthly_pnl) return;

    const m = data.monthly_pnl;
    const years = [...new Set(m.map(d => d.year))].sort();
    const mos = Array.from({length:12}, (_,i) => i+1);
    const lookup = {}; m.forEach(d => { lookup[`${d.year}-${d.month}`] = d.pnl; });
    const maxAbs = Math.max(...m.map(d => Math.abs(d.pnl)), 0.01);

    let h = '<div class="heatmap-grid"><div class="heatmap-header"></div>';
    mos.forEach(mo => { h += `<div class="heatmap-header">${mo}月</div>`; });

    years.forEach(year => {
        h += `<div class="heatmap-year">${year}</div>`;
        mos.forEach(mo => {
            const k = `${year}-${mo}`, v = lookup[k];
            if (v !== undefined) {
                const t = Math.min(Math.abs(v) / maxAbs, 1);
                const bg = v > 0
                    ? `rgba(45,106,79,${0.1 + t * 0.55})`
                    : `rgba(174,74,52,${0.1 + t * 0.55})`;
                const tc = t > 0.45 ? "#fff" : T.ink;
                h += `<div class="heatmap-cell" style="background:${bg};color:${tc}">${(v*100).toFixed(1)}%</div>`;
            } else {
                h += `<div class="heatmap-cell" style="background:${T.cream};color:${T.ink40}">-</div>`;
            }
        });
    });
    h += "</div>";
    el.innerHTML = h;
}

// ─── Alerts ───
async function loadAlerts() {
    try {
        const res = await fetch("/api/alerts");
        renderAlerts(await res.json());
    } catch (e) { console.error("Alerts:", e); }
}

function renderAlerts(alerts) {
    const el = document.getElementById("alerts-container");
    if (!el) return;
    if (!alerts.length) {
        el.innerHTML = '<div class="alert-item info"><span class="alert-text">No active alerts</span></div>';
        return;
    }
    el.innerHTML = alerts.map(a => `
        <div class="alert-item ${a.severity}">
            <span class="alert-text">${a.message}</span>
            <span class="alert-time">${a.created_at}</span>
            <button class="btn btn-ghost" onclick="resolveAlert(${a.id})" style="padding:3px 8px;font-size:10px">resolve</button>
        </div>
    `).join("");
}

async function resolveAlert(id) {
    await fetch(`/api/alerts/${id}/resolve`, { method: "POST" });
    loadAlerts();
}

// ─── Prediction History ───
async function loadPredictionHistory() {
    try {
        const res = await fetch("/api/predictions/history");
        renderHistory(await res.json());
    } catch (e) { console.error("History:", e); }
}

function renderHistory(preds) {
    const tb = document.getElementById("history-tbody");
    if (!tb) return;
    if (!preds.length) {
        tb.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--ink-40);padding:24px;font-size:12px">No predictions yet</td></tr>';
        return;
    }
    tb.innerHTML = preds.map(p => {
        const dc = p.direction === "UP" ? "up" : "down";
        let badge = '<span class="correct-badge pending"></span>';
        if (p.is_correct === 1) badge = '<span class="correct-badge yes"></span>';
        else if (p.is_correct === 0) badge = '<span class="correct-badge no"></span>';
        return `<tr>
            <td>${p.target_date}</td>
            <td><span class="dir-badge ${dc}">${p.direction}</span></td>
            <td>${(p.pred_log_return * 100).toFixed(3)}%</td>
            <td>${p.actual_log_return !== null ? (p.actual_log_return * 100).toFixed(3) + "%" : "---"}</td>
            <td>${(p.confidence * 100).toFixed(0)}%</td>
            <td>${badge}</td>
        </tr>`;
    }).join("");
}

// ─── Analytics Charts ───

function renderConfidenceCalibration(data) {
    const ctx = document.getElementById("chart-conf-cal");
    if (!ctx || !data.confidence_calibration?.length) return;
    if (charts.confCal) charts.confCal.destroy();
    document.getElementById("placeholder-conf-cal")?.classList.add("hidden");

    const cal = data.confidence_calibration;
    const labels = cal.map((d, i) => `Q${i + 1}`);
    const accs = cal.map(d => d.accuracy);
    const counts = cal.map(d => d.count);

    const barColor = v => v >= 0.55 ? "rgba(45,106,79,0.55)" : v >= 0.50 ? "rgba(196,162,101,0.55)" : "rgba(174,74,52,0.55)";
    const barBorder = v => v >= 0.55 ? T.up : v >= 0.50 ? T.gold : T.down;
    const accMin = Math.min(...accs);
    const yMin = Math.max(Math.floor((Math.min(accMin, 0.5) - 0.1) * 10) / 10, 0);

    charts.confCal = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                data: accs,
                backgroundColor: accs.map(barColor),
                borderColor: accs.map(barBorder),
                borderWidth: 1,
                borderRadius: 3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...sharedTooltip,
                    callbacks: {
                        title: items => {
                            const i = items[0].dataIndex;
                            return `Confidence: ${cal[i].bin}`;
                        },
                        afterTitle: items => `n = ${counts[items[0].dataIndex]}`,
                        label: c => ` Accuracy: ${(c.parsed.y * 100).toFixed(1)}%`,
                    },
                },
                horizontalLine: {
                    lines: [{ value: 0.5, color: T.down, width: 1.2, dash: [6, 4], label: "50% Random" }],
                },
            },
            scales: {
                x: {
                    grid: { display: false }, border: { display: false },
                    ticks: { font: { size: 10 }, callback: (v, i) => `${labels[i]}\n(n=${counts[i]})` },
                },
                y: {
                    min: yMin, max: Math.min(Math.ceil((Math.max(...accs) + 0.1) * 10) / 10, 1),
                    ticks: { callback: v => (v * 100).toFixed(0) + "%", stepSize: 0.1 },
                    grid: { color: T.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

function renderQuantileCalibration(data) {
    const ctx = document.getElementById("chart-q-cal");
    if (!ctx || !data.quantile_calibration) return;

    const qc = data.quantile_calibration;
    if (qc.q10 == null && qc.q50 == null && qc.q90 == null) return;

    if (charts.qCal) charts.qCal.destroy();
    document.getElementById("placeholder-q-cal")?.classList.add("hidden");

    const labels = ["Q10", "Q50", "Q90", "80% CI"];
    const expected = [0.10, 0.50, 0.90, 0.80];
    const actual = [qc.q10 ?? 0, qc.q50 ?? 0, qc.q90 ?? 0, qc.coverage_80 ?? 0];

    const calColor = (a, e) => {
        const d = Math.abs(a - e);
        if (d < 0.05) return { bg: "rgba(45,106,79,0.50)", border: T.up };
        if (d < 0.10) return { bg: "rgba(196,162,101,0.50)", border: T.gold };
        return { bg: "rgba(174,74,52,0.50)", border: T.down };
    };
    const colors = actual.map((a, i) => calColor(a, expected[i]));

    charts.qCal = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "Target",
                    data: expected,
                    backgroundColor: "rgba(209,207,197,0.35)",
                    borderColor: T.stone,
                    borderWidth: 1.5,
                    borderRadius: 3,
                },
                {
                    label: "Actual",
                    data: actual,
                    backgroundColor: colors.map(c => c.bg),
                    borderColor: colors.map(c => c.border),
                    borderWidth: 1,
                    borderRadius: 3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "top", align: "end",
                    labels: { usePointStyle: true, boxWidth: 6, padding: 12, font: { size: 10 } },
                },
                tooltip: {
                    ...sharedTooltip,
                    callbacks: { label: c => ` ${c.dataset.label}: ${(c.parsed.y * 100).toFixed(1)}%` },
                },
            },
            scales: {
                x: { grid: { display: false }, border: { display: false } },
                y: {
                    min: 0, max: 1,
                    ticks: { callback: v => (v * 100).toFixed(0) + "%", stepSize: 0.2 },
                    grid: { color: T.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

function renderMultiHorizonAccuracy(data) {
    const ctx = document.getElementById("chart-horizon-acc");
    if (!ctx || !data.horizon_accuracy?.length) return;

    const labels = data.horizon_accuracy.map(h => h.horizon);
    const values = data.horizon_accuracy.map(h => h.accuracy);

    if (charts.horizonAcc) charts.horizonAcc.destroy();
    document.getElementById("placeholder-horizon-acc")?.classList.add("hidden");

    const barColor = v => v >= 0.55 ? "rgba(45,106,79,0.55)" : v >= 0.50 ? "rgba(196,162,101,0.55)" : "rgba(174,74,52,0.55)";
    const barBorder = v => v >= 0.55 ? T.up : v >= 0.50 ? T.gold : T.down;

    const vMin = Math.min(...values);
    const yMin = Math.max(Math.floor((Math.min(vMin, 0.5) - 0.1) * 10) / 10, 0);
    const yMax = Math.min(Math.ceil((Math.max(...values) + 0.1) * 10) / 10, 1);

    charts.horizonAcc = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: values.map(barColor),
                borderColor: values.map(barBorder),
                borderWidth: 1,
                borderRadius: 3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...sharedTooltip,
                    callbacks: { label: c => ` Accuracy: ${(c.parsed.y * 100).toFixed(1)}%` },
                },
                horizontalLine: {
                    lines: [{ value: 0.5, color: T.down, width: 1.2, dash: [6, 4], label: "50% Random" }],
                },
            },
            scales: {
                x: { grid: { display: false }, border: { display: false } },
                y: {
                    min: yMin, max: yMax,
                    ticks: { stepSize: 0.1, callback: v => (v * 100).toFixed(0) + "%" },
                    grid: { color: T.gridLine },
                    border: { display: false },
                },
            },
        },
    });
}

// ─── Helpers ───
function fmt(v, d = 2) { return v == null ? "---" : Number(v).toFixed(d); }
function fmtPct(v) { return v == null ? "---" : (v * 100).toFixed(1) + "%"; }
function accCls(v) { return v >= 0.65 ? "metric-good" : v >= 0.5 ? "metric-warn" : "metric-bad"; }
