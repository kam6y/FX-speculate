# Ralph Loop — モデルアーキテクチャ

## 全体パイプライン

```mermaid
flowchart TD
    A["USD/JPY 1分足 OHLCV\n881,522 samples"] --> B["特徴量生成\n99 features\nテクニカル指標 + OHLC"]

    B --> C["Train 60%\n528,913"]
    B --> D["Val 20%\n176,169"]
    B --> E["Test 20%\n176,170"]

    C --> F["ターゲット生成\nthreshold_pips = 3.0 / horizon = 15分\nBUY: +3pips以上 | HOLD: ±3pips以内 | SELL: -3pips以下"]

    F --> G["GPUWindowDataset\nwindow_size = 50\n全データ VRAM 常駐\n出力 : B, 50, 99"]

    G --> H["FXTransformer\n~16K params\n出力 : B, 3 確率"]

    H --> I["取引判定\nprob_threshold = 0.48\n+ ATRフィルタ 30th pctl\n+ 時間帯フィルタ 20-23時除外"]

    I --> J["最終結果  Seed=99\nSharpe 3.89 | Trades 116\nWin Rate 61.2% | PF 1.74\nP&L +21,208 JPY"]

    style A fill:#e1f5fe,stroke:#0288d1
    style H fill:#fff3e0,stroke:#f57c00
    style J fill:#e8f5e9,stroke:#2e7d32
```

## FXTransformer 内部構造

```mermaid
flowchart TD
    IN["入力\nB, 50, 99"] --> NORM

    subgraph NORM["Window Z-Score Normalization"]
        N1["各ウィンドウ内で独立に標準化\nx = (x - mean) / max(std, 1e-6)\n出力 : B, 50, 99"]
    end

    NORM --> PROJ

    subgraph PROJ["Input Projection"]
        P1["nn.Linear  99 → 32\n出力 : B, 50, 32"]
    end

    PROJ --> PE

    subgraph PE["Positional Encoding"]
        PE1["Sinusoidal PE\nsin / cos  max_len=200\nx = x + PE\n出力 : B, 50, 32"]
    end

    PE --> ENC

    subgraph ENC["TransformerEncoderLayer x 1"]
        direction TB
        MHA["Multi-Head Self-Attention\nnhead=2  head_dim=16\nAttention = softmax(QK^T / sqrt 16) V"]
        ADD1["Add & LayerNorm"]
        FFN["Feed-Forward Network\nLinear 32→128  ReLU  Dropout 0.25  Linear 128→32"]
        ADD2["Add & LayerNorm"]

        MHA --> ADD1 --> FFN --> ADD2
    end

    ENC --> LAST["Last Timestep\nx = x[:, -1, :]\n出力 : B, 32"]

    LAST --> DROP["Dropout  0.25"]

    DROP --> CLS["Classifier\nnn.Linear  32 → 3\n出力 : B, 3  logits"]

    CLS --> SOFT["Softmax\nP(HOLD)  P(BUY)  P(SELL)"]

    style IN fill:#e1f5fe,stroke:#0288d1
    style ENC fill:#fce4ec,stroke:#c62828
    style SOFT fill:#e8f5e9,stroke:#2e7d32
```

## 学習ループ

```mermaid
flowchart LR
    subgraph TRAIN["学習 max 200 epochs"]
        direction TB
        FWD["Forward Pass\nAMP float16"] --> LOSS["CrossEntropyLoss\nbalanced class weights"]
        LOSS --> BACK["Backward\nGradScaler"]
        BACK --> CLIP["Grad Clip\nmax_norm=1.0"]
        CLIP --> STEP["AdamW Step\nlr=3e-4"]
        STEP --> SCHED["CosineAnnealingLR"]
    end

    subgraph VAL["Validation"]
        direction TB
        VLOSS["Val Loss 計算"] --> CHECK{"val_loss < best?"}
        CHECK -->|Yes| SAVE["Save best state\nreset patience"]
        CHECK -->|No| INC["patience++"]
        INC --> STOP{"patience >= 20?"}
        STOP -->|Yes| DONE["Early Stop"]
        STOP -->|No| CONT["Continue"]
    end

    TRAIN --> VAL
    CONT --> TRAIN

    style TRAIN fill:#f3e5f5,stroke:#7b1fa2
    style VAL fill:#fffde7,stroke:#f9a825
```

## 取引判定フロー

```mermaid
flowchart TD
    PRED["モデル出力\nP(HOLD)  P(BUY)  P(SELL)"] --> CHK1{"ATR >= 30th\npercentile?"}

    CHK1 -->|No| HOLD1["HOLD  取引なし"]
    CHK1 -->|Yes| CHK2{"時間帯\n0:00-19:59?"}

    CHK2 -->|No  20-23時| HOLD2["HOLD  取引なし"]
    CHK2 -->|Yes| CHK3{"P(BUY) > 0.48\nかつ\nP(BUY) > P(SELL)?"}

    CHK3 -->|Yes| BUY["BUY  10,000通貨\n15分後に決済"]
    CHK3 -->|No| CHK4{"P(SELL) > 0.48\nかつ\nP(SELL) > P(BUY)?"}

    CHK4 -->|Yes| SELL["SELL  10,000通貨\n15分後に決済"]
    CHK4 -->|No| HOLD3["HOLD  取引なし"]

    style BUY fill:#e8f5e9,stroke:#2e7d32
    style SELL fill:#ffebee,stroke:#c62828
    style HOLD1 fill:#f5f5f5,stroke:#9e9e9e
    style HOLD2 fill:#f5f5f5,stroke:#9e9e9e
    style HOLD3 fill:#f5f5f5,stroke:#9e9e9e
```

## パラメータ数

| レイヤー | 計算式 | パラメータ数 |
|----------|--------|-------------|
| input_proj | 99x32 + 32 | 3,200 |
| PositionalEncoding | 固定バッファ | 0 |
| Self-Attention (Q,K,V,O) | 4x(32x32 + 32) | 4,224 |
| FFN | 32x128+128 + 128x32+32 | 8,352 |
| LayerNorm x 2 | 2x(32+32) | 128 |
| classifier | 32x3 + 3 | 99 |
| **合計** | | **16,003** |

## 学習設定

| 項目 | 値 |
|------|-----|
| Optimizer | AdamW (lr=3e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (balanced class weights) |
| Mixed Precision | AMP (float16 forward, float32 grad) |
| Gradient Clipping | max_norm=1.0 |
| Batch Size | 512 |
| Max Epochs | 200 |
| Early Stopping | patience=20 |
| Random Seed | 99 |

## コスト構造

| 項目 | 値 |
|------|-----|
| スプレッド | 0.2 pips (Bid/Ask時不要) |
| スリッページ | 0.1 pips |
| API手数料 | 0.002% per side |
| ポジションサイズ | 10,000通貨 |

## 最終成績

| 指標 | 値 |
|------|-----|
| Sharpe Ratio | **3.89** |
| Total P&L | **+21,208 JPY** |
| Total Trades | **116** |
| Win Rate | **61.2%** |
| Profit Factor | **1.74** |
| Max Drawdown | **-8,114 JPY** |
| Calmar Ratio | **8.26** |
| Avg Win | +703 JPY |
| Avg Loss | -639 JPY |
| Win/Loss Ratio | 1.10x |
