# TFT Horizon Accuracy 改善 実装計画

> **実行方法:** Ralph Loop で `Temporal Fusion Transformer/OPTIMIZATION_PROMPT.md` を使って反復実行する。
> 各イテレーションで1つの変更を行い、学習・検証を経て採用/リバートを判断する。

**Goal:** 1d方向精度(72.25%)を維持しつつ、全Horizon(1d-10d)の方向精度を底上げする

**Architecture:** 累積方向ペナルティLoss + マルチスケール特徴量の2軸。評価と学習の不整合を解消することが最大のレバレッジ。

**Tech Stack:** PyTorch Forecasting, PyTorch Lightning, ta (テクニカル指標)

**設計書:** `docs/superpowers/specs/2026-03-28-tft-horizon-accuracy-improvement-design.md`

---

## ファイル構成

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `Temporal Fusion Transformer/train_tft.py` | Modify | Loss関数改良、特徴量追加、CONFIG更新 |
| `Temporal Fusion Transformer/OPTIMIZATION_PROMPT.md` | Rewrite | Ralph Loop用プロンプト更新 |

## 実行順序

Ralph Loopの各イテレーションで1つずつ実装する。**必ずこの順序で進める。**

前のイテレーションの変更がコミット済みかどうかは git log で確認する。

---

### Task 1: マルチスケール特徴量の追加

**Files:**
- Modify: `Temporal Fusion Transformer/train_tft.py:263-297` (`add_technical_indicators`)
- Modify: `Temporal Fusion Transformer/train_tft.py:521-532` (`UNKNOWN_REALS_BASE`)

- [ ] **Step 1: `add_technical_indicators` に4つの特徴量を追加**

`train_tft.py` の `add_technical_indicators` 関数末尾（`return df` の直前）に以下を追加:

```python
    # マルチスケール特徴量 (Horizon精度改善用)
    df["return_5d"] = df["log_return"].rolling(5).sum()
    df["return_10d"] = df["log_return"].rolling(10).sum()
    df["rsi_5"] = ta.momentum.RSIIndicator(c, window=5).rsi()
    df["volatility_10d"] = df["log_return"].rolling(10).std()
```

- [ ] **Step 2: `UNKNOWN_REALS_BASE` に追加**

`UNKNOWN_REALS_BASE` リストの末尾に以下を追加:

```python
    "return_5d",
    "return_10d",
    "rsi_5",
    "volatility_10d",
```

- [ ] **Step 3: feature_schema.json を削除して学習実行**

```bash
rm -f "Temporal Fusion Transformer/artifacts/feature_schema.json" && cd "Temporal Fusion Transformer" && uv run python train_tft.py 2>&1 | tail -30
```

- [ ] **Step 4: メトリクス検証**

```bash
uv run python -c "import json; m=json.load(open('Temporal Fusion Transformer/artifacts/optimization_log.json'))[-1]['metrics']; print(f'ens_1d={m[\"ensemble_direction_1d\"]}, ens_5d={m[\"ensemble_direction_5d\"]}')"
```

判断基準:
- ens_1d >= 0.70 かつ ens_5d が前回比で悪化していなければ **採用 → コミット**
- ens_1d < 0.68 なら **リバート** (`git checkout -- "Temporal Fusion Transformer/train_tft.py"`)

- [ ] **Step 5: コミット**

```bash
git add "Temporal Fusion Transformer/train_tft.py"
git commit -m "feat: add multi-scale features (return_5d, return_10d, rsi_5, volatility_10d)"
```

---

### Task 2: 累積方向ペナルティ Loss の実装

**Files:**
- Modify: `Temporal Fusion Transformer/train_tft.py:54-75` (`DirectionAwareQuantileLoss`)
- Modify: `Temporal Fusion Transformer/train_tft.py:129` (CONFIG `DIRECTION_LOSS_WEIGHT`)

- [ ] **Step 1: CONFIG に `CUMULATIVE_DIRECTION_WEIGHT` を追加、`DIRECTION_LOSS_WEIGHT` を変更**

```python
    "DIRECTION_LOSS_WEIGHT": 0.3,              # 0.5→0.3 (累積ペナルティとの重複を相殺)
    "CUMULATIVE_DIRECTION_WEIGHT": 0.3,        # 新規: 累積方向ペナルティ重み
```

- [ ] **Step 2: `DirectionAwareQuantileLoss` を改良**

`__init__` に `cumulative_direction_weight` パラメータを追加し、`loss` メソッドに累積方向ペナルティを実装:

```python
class DirectionAwareQuantileLoss(QuantileLoss):
    """QuantileLoss に方向ペナルティを加えたカスタム損失。

    median 予測の符号が実績と一致しない場合に追加ペナルティを課す。
    これにより TFT が「0に近い安全な予測」ではなく方向性を学習する。
    累積方向ペナルティにより、全Horizonの方向精度を改善する。
    """

    def __init__(self, quantiles=None, direction_weight: float = 1.0,
                 cumulative_direction_weight: float = 0.0, **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)
        self.direction_weight = direction_weight
        self.cumulative_direction_weight = cumulative_direction_weight

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 標準 quantile loss
        ql = super().loss(y_pred, target)

        # R8-04: 符号ベース方向ペナルティ (個別ステップ)
        q_mid = y_pred.size(-1) // 2
        pred_median = y_pred[..., q_mid]
        sign_mismatch = (pred_median * target < 0).float()
        dir_penalty = sign_mismatch * (pred_median - target).abs()

        total = ql + self.direction_weight * dir_penalty.unsqueeze(-1)

        # 累積方向ペナルティ (全Horizon)
        if self.cumulative_direction_weight > 0 and pred_median.dim() >= 2:
            pred_cum = pred_median.cumsum(dim=-1)   # (batch, pred_len)
            target_cum = target.cumsum(dim=-1)      # (batch, pred_len)
            cum_mismatch = (pred_cum * target_cum < 0).float()
            cum_penalty = cum_mismatch * (pred_cum - target_cum).abs()
            # pred_len 全体の平均を各ステップに均等配分
            cum_mean = cum_penalty.mean(dim=-1, keepdim=True).expand_as(pred_median)
            total = total + self.cumulative_direction_weight * cum_mean.unsqueeze(-1)

        return total
```

- [ ] **Step 3: Loss インスタンス生成部分を更新**

`train_tft` 関数内の Loss 生成箇所（約688行目）を更新:

```python
        loss=DirectionAwareQuantileLoss(
            quantiles=config["QUANTILES"],
            direction_weight=config.get("DIRECTION_LOSS_WEIGHT", 0.0),
            cumulative_direction_weight=config.get("CUMULATIVE_DIRECTION_WEIGHT", 0.0),
        ),
```

- [ ] **Step 4: feature_schema.json を削除して学習実行**

```bash
rm -f "Temporal Fusion Transformer/artifacts/feature_schema.json" && cd "Temporal Fusion Transformer" && uv run python train_tft.py 2>&1 | tail -30
```

- [ ] **Step 5: メトリクス検証**

```bash
uv run python -c "import json; m=json.load(open('Temporal Fusion Transformer/artifacts/optimization_log.json'))[-1]['metrics']; print(f'ens_1d={m[\"ensemble_direction_1d\"]}, ens_5d={m[\"ensemble_direction_5d\"]}')"
```

判断基準:
- ens_1d >= 0.70 → **採用**
- ens_1d < 0.68 → **リバート** (Loss変更のみ。特徴量はTask 1で採用済みなら維持)

- [ ] **Step 6: コミット**

```bash
git add "Temporal Fusion Transformer/train_tft.py"
git commit -m "feat: add cumulative direction penalty loss for horizon accuracy improvement"
```

---

### Task 3: パラメータチューニング (Ralph Loop 反復)

Task 1, 2 が採用された後、Ralph Loop で以下のパラメータを微調整する。
**1イテレーション = 1パラメータ変更。**

- [ ] **チューニング対象:**

| パラメータ | 初期値 | 探索範囲 | 狙い |
|-----------|--------|---------|------|
| `DIRECTION_LOSS_WEIGHT` (α) | 0.3 | 0.1 - 0.5 | 1d精度と全体のバランス |
| `CUMULATIVE_DIRECTION_WEIGHT` (β) | 0.3 | 0.1 - 0.5 | Horizon精度の重み |

- [ ] **各イテレーションの手順:**

1. α または β を1つ変更
2. 学習実行: `rm -f "Temporal Fusion Transformer/artifacts/feature_schema.json" && cd "Temporal Fusion Transformer" && uv run python train_tft.py 2>&1 | tail -30`
3. メトリクス確認
4. 判断: 改善→採用+コミット、悪化→リバート

---

### Task 4: OPTIMIZATION_PROMPT.md の更新

Task 1-2 の実装後、Ralph Loop 用プロンプトを Round 9 として更新する。

- [ ] **Step 1: `OPTIMIZATION_PROMPT.md` を Round 9 用に書き換え**

以下の内容で上書き:

```markdown
# TFT 最適化ループ (Round 9) — Horizon精度チューニング

## 役割
`Temporal Fusion Transformer/train_tft.py` の全Horizon方向精度を改善する。
**train_tft.py のみ編集。1イテレーション = 1変更。**

## 累計 (R1-R8)
R8 で累積方向ペナルティLoss + マルチスケール特徴量を導入。
ベースライン: ens_1d=0.7225, ens_5d=0.3844

### 現在の CONFIG
(Task 1-2 実装後の値を記載)

### R1-R8 全「やるな」リスト
- CONFIG微調整は全て試行済み（R1-R7, 130 iter）
- 個別ステップ方向ペナルティのみでは Horizon精度は改善しない

## 目標
- ens_1d >= 0.70 を維持
- 全Horizon (direction_1d〜direction_10d) で改善
- direction_5d >= 0.45

## チューニング対象
1. **α (DIRECTION_LOSS_WEIGHT)**: 0.1 - 0.5 の範囲で微調整
2. **β (CUMULATIVE_DIRECTION_WEIGHT)**: 0.1 - 0.5 の範囲で微調整
3. **α, β の比率変更**: α=0.2, β=0.4 など片方を強調

## 手順
1. optimization_log.json の最新エントリを確認
2. 1つだけパラメータを変更（`# R9-XX:` コメント付与）
3. 実行: `rm -f "Temporal Fusion Transformer/artifacts/feature_schema.json" && cd "Temporal Fusion Transformer" && uv run python train_tft.py 2>&1 | tail -30`
4. 確認: メトリクス全体を確認（1d, 2d, 3d, 5d, 10d）
5. 判断: ens_1d >= 0.70 かつ direction_5d 改善 → 採用+コミット、ens_1d < 0.68 → revert

## 制約
- train_tft.py のみ、新規パッケージ禁止、GPU/CPU両対応、ログ維持
- エラー時は即revert
- **1イテレーション = 1パラメータ変更**

## 完了条件
`<promise>HORIZON OPTIMIZED</promise>`:
1. ens_1d >= 0.70 かつ direction_5d >= 0.45
2. 8 iter 連続改善なし
3. 30 iter 到達
```

- [ ] **Step 2: コミット**

```bash
git add "Temporal Fusion Transformer/OPTIMIZATION_PROMPT.md"
git commit -m "docs: update OPTIMIZATION_PROMPT.md for Round 9 (horizon accuracy tuning)"
```

---

## Ralph Loop 実行方法

Task 1-2 を手動またはインラインで実装した後、Task 3-4 のチューニングを Ralph Loop で回す:

```
/ralph-loop "Temporal Fusion Transformer/OPTIMIZATION_PROMPT.md の指示に従い、1イテレーションで1つの変更を実装・検証・判断せよ。" --max-iterations 30 --completion-promise "HORIZON OPTIMIZED"
```

## 成功基準

- ens_1d >= 0.70 (現行 0.7225 を維持)
- direction_5d >= 0.45 (現行 0.3844 から +6.5pt以上)
- 全 Horizon (direction_2d, 3d, 10d) で現行比改善
