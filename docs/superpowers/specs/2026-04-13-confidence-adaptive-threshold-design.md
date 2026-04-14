# 信頼度推定 + 適応的閾値システム設計

**日付**: 2026-04-13
**目的**: TFTモデルの方向精度向上と予測の実用性向上（信頼度による選択的予測）

> **Status (2026-04-14):** AdaptiveThreshold（ATRスケーリング + ABSTAIN）部分は実装検証の結果撤去済み（コミット `b481f13`）。ConfidenceEstimator のみが採用されている。以降の設計は歴史的資料。

## 背景

現在のTFT USD/JPY予測システムは方向精度58.6%（ランダム比+8.6pt）を達成しているが、ハイパーパラメータ・特徴量セットは最適化済みで、TFT本体の改善余地は限られている。

残る改善余地は「TFTの出力を活用する後段の仕組み」にある。

### 現状の課題

- H4/H5のキャリブレーションが悪い（ratio_gap 7-9%）
- 全予測を均等に扱っており、予測の信頼度に基づく判断ができない
- 閾値が固定で、市場状態の変化に対応していない

## アプローチ概要

TFTパイプラインを変更せず、後段に2つの仕組みを追加する：

- **アプローチA: 信頼度推定** — 予測ごとの信頼スコアを算出し、高信頼予測を選別
- **アプローチC: 適応的閾値** — 市場ボラティリティに応じて方向判定閾値を動的調整

## 設計詳細

### 1. 信頼度推定システム（アプローチA）

#### 1.1 信頼度シグナル

3つの独立したシグナルを算出し合成スコアにする。

**シグナル① アンサンブル一致度（ensemble_agreement）**

- top-5モデルの direction_signal をそれぞれ算出
- 5モデル中、同じ方向を示すモデル数を計測
- `agreement = max(n_up, n_down) / 5`
- 値域: 0.6（3/5）〜 1.0（5/5）

**シグナル② 分位点スプレッド（quantile_spread）**

- `spread = q90 - q10`（予測分布の幅）
- スプレッドが小さいほど信頼度が高い
- tune セットのスプレッド分布から percentile rank に変換: `spread_score = 1 - percentile_rank`
- 値域: 0.0〜1.0

**シグナル③ シグナル強度（signal_strength）**

- `raw_strength = abs(direction_signal - threshold) / threshold`
- `signal_strength = min(raw_strength, clip_max)` で上限をclip
- clip_max は固定値（例: 2.0）
- 正規化して0-1の範囲に変換

#### 1.2 合成信頼スコア

```
confidence = w1 * ensemble_agreement + w2 * spread_score + w3 * signal_strength
```

- w1 + w2 + w3 = 1.0（正規化制約）
- tune セットで 0.1 刻みのグリッドサーチ（探索空間: 約55パターン）
- 最適化指標: 信頼スコア上位70%の予測の方向精度

#### 1.3 信頼度閾値と棄権

- tune セットで「信頼スコア上位X%の方向精度」を X=50%〜100% で計算
- 精度-カバレッジ曲線を出力
- デフォルト棄権閾値: 「精度が全体平均+2pt以上となる最大カバレッジ」

#### 1.4 出力フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `confidence_score` | float | 0-1の合成スコア |
| `confidence_level` | str | HIGH / MEDIUM / LOW（3段階） |
| `should_trade` | bool | 信頼度閾値を超えているか |

confidence_level の境界:
- HIGH: 上位33%
- MEDIUM: 33%〜66%
- LOW: 下位33%

（境界は tune セットのスコア分布から決定）

### 2. 適応的閾値システム（アプローチC）

#### 2.1 ボラティリティスケーラー

固定閾値をベースラインとし、市場ボラティリティに応じてスケーリングする。

```
adaptive_threshold = base_threshold * volatility_scaler
```

**volatility_scaler の算出**:
- `current_atr` = 直近20営業日のATR（既存特徴量）
- `median_atr` = 過去1年（約250営業日）のATR中央値
- `raw_scaler = current_atr / median_atr`
- `volatility_scaler = clamp(raw_scaler, 0.7, 1.5)`

#### 2.2 ホライゾン別スケーリング

キャリブレーション状態に応じてスケーリング強度を調整する。

- H1-H3（ratio_gap < 4%）: `scaler = 1 + 0.5 * (raw_scaler - 1)`（半減）
- H4-H5（ratio_gap 7-9%）: `scaler = raw_scaler`（フルスケーリング）

#### 2.3 棄権ゾーン

direction_signal が閾値近傍にある場合を「判定不能」とする。

```
abstain_zone = adaptive_threshold * abstain_margin
if abs(direction_signal - adaptive_threshold) < abstain_zone:
    direction = "ABSTAIN"
```

- `abstain_margin`: tune セットで最適化（探索範囲: 0.01〜0.20、0.01刻み）
- 最適化指標: `accuracy - 1.0 * ratio_gap`（既存の閾値最適化と同じ指標）

### 3. A+C の統合

#### 3.1 最終判定フロー

```
TFT アンサンブル推論（既存）
        ↓
  5モデルの生出力（5分位点 × 5ホライゾン × 5モデル）
        ↓
  ┌─────────────┬──────────────────┐
  │ 信頼度推定(A) │  適応的閾値(C)     │
  │ ・ensemble   │  ・volatility     │
  │   agreement  │    scaler         │
  │ ・quantile   │  ・ホライゾン別    │
  │   spread     │    スケーリング    │
  │ ・signal     │  ・棄権ゾーン      │
  │   strength   │                   │
  └──────┬──────┴────────┬─────────┘
         ↓               ↓
    confidence_score   adaptive_threshold
         ↓               ↓
  ┌──────────────────────────────────┐
  │  最終判定ロジック                  │
  │  1. direction_signal vs           │
  │     adaptive_threshold で方向判定  │
  │  2. 棄権ゾーン内 → ABSTAIN        │
  │  3. confidence_score で            │
  │     HIGH/MEDIUM/LOW を付与         │
  │  4. LOW + ABSTAIN → 強い棄権      │
  └──────────────────────────────────┘
         ↓
  predictions.db / ダッシュボード
```

#### 3.2 判定の優先度

1. 適応的閾値で方向を判定（UP / DOWN / ABSTAIN）
2. 信頼度スコアを付与（独立に計算）
3. `should_trade` = direction が ABSTAIN でない AND confidence_level が LOW でない

### 4. ファイル変更の範囲

| ファイル | 変更内容 |
|---------|---------|
| `config.py` | 信頼度・適応閾値のパラメータ定数を追加 |
| `model/confidence.py` | **新規** — 信頼度推定ロジック（ConfidenceEstimator クラス） |
| `model/adaptive_threshold.py` | **新規** — 適応的閾値ロジック（AdaptiveThreshold クラス） |
| `scripts/evaluate.py` | 信頼度重み・適応閾値パラメータの最適化ステップを追加 |
| `scripts/predict.py` | 推論時に信頼度・適応閾値を適用、出力カラム追加 |
| `dashboard/app.py` | 信頼度表示パネル・適応閾値可視化を追加 |
| `tests/test_confidence.py` | **新規** — 信頼度推定のテスト |
| `tests/test_adaptive_threshold.py` | **新規** — 適応的閾値のテスト |

### 5. 評価方法

#### 5.1 テストセット指標

| 指標 | ベースライン | 改善目標 |
|------|------------|---------|
| 全体方向精度 | 58.6% | 適応閾値により微改善 |
| 非棄権予測の方向精度 | — | 全体+2pt以上 |
| H4/H5 ratio_gap | 7-9% | 3-5% |
| カバレッジ（非棄権率） | 100% | 70-85% |
| 高信頼予測の方向精度 | — | 63%+ |

#### 5.2 統計的有意性

- McNemar検定で方向精度の改善を検証
- ベースラインと改善後の正解/不正解ペアの入れ替わり比率を検定

#### 5.3 オーバーフィット対策

1. 最適化パラメータは合計4個のみ（信頼度重み3個 + abstain_margin 1個）
2. 粗いグリッド（重み0.1刻み、margin 0.01刻み）
3. 時系列分割を厳守（tune → test の一方向）
4. test セットは最後に1回だけ使用、再調整しない

### 6. 対象外（スコープ外）

- TFTモデル本体のアーキテクチャ変更
- 特徴量の追加・削除
- ハイパーパラメータの再最適化
- 残差補正モデル（アプローチB）— データ量の制約により見送り
