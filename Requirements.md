# Requirements.md — Conditional DDPM 時系列異常検知

> 実験の目的・背景・合格基準は [README.md](README.md) を参照。

## 1. データ仕様

### 2-1. 合成センサデータ
- **形式**: `ndarray / Tensor` の shape `[N, C, T]`
  - `N`: サンプル数
  - `C`: センサチャンネル数 (デフォルト 4)
  - `T`: 時系列長 (デフォルト 128)
- **正規化**: チャンネルごとに `z-score` 正規化を学習データの統計量で実施
- **データ分割**: 学習 70% / 検証 15% / テスト 15%

### 2-2. 正常信号の生成仕様
```
各チャンネル = Σ A_k * sin(ω_k * t + φ_k) + ε
  ω_k: チャンネルごとに異なる周波数 (製造ラインの回転数・振動周波数を模擬)
  A_k: ランダム振幅 [0.8, 1.2]
  φ_k: ランダム初期位相
  ε  ~ N(0, 0.05²)
```

### 2-3. 異常信号の生成仕様

| 異常タイプ | 物理的意味 | 発生箇所 |
|---|---|---|
| `spike` | センサ誤作動・衝撃 | ランダム区間 (幅3〜12) に振幅2.5〜5倍のスパイク |
| `drift` | 経年劣化・熱ドリフト | ランダム開始点から線形増加 (+1.5〜3.5) |
| `amplitude_shift` | 異常振動・設備負荷増大 | ランダム開始点から全振幅を2〜3.5倍にスケール |
| `frequency_shift` | 回転数異常 | ランダム開始点から高周波成分 (3.5 Hz相当) を加算 |

- 1サンプルにつき1異常タイプ、1チャンネルに異常を付加
- 異常は `T/4` ～ `3T/4` の間に発生 (端部は除外)

### 2-4. 条件ラベル
- 設備の稼働モードを模擬: `{0: 通常稼働, 1: 高負荷稼働}`
- モード 0: 上記仕様の正常信号
- モード 1: 全体的に振幅×1.3、ノイズ×1.5 の高負荷パターン

---

## 2. モデル仕様

### 3-1. Conditional 1D UNet (`model.py`)

```
入力:  x_t [B, C, T], t [B], condition [B]
出力:  ε_θ [B, C, T]  (予測ノイズ)
```

| コンポーネント | 仕様 |
|---|---|
| タイムステップ埋め込み | Sinusoidal (dim=64) → Linear(64, 256) → SiLU → Linear(256, 256) |
| 条件埋め込み | `nn.Embedding(num_conditions, 64)` |
| UNet 深度 | 4レベル (channel_mults = [1, 2, 4, 4]) |
| base_channels | 64 |
| 各レベルのブロック | ResBlock × 2 (+オプション SelfAttention) |
| ダウンサンプル | `Conv1d(stride=2)` |
| アップサンプル | `ConvTranspose1d(stride=2)` |
| ボトルネック | ResBlock → SelfAttention → ResBlock |
| 埋め込み注入 | scale + shift (Adaptive GroupNorm 方式) |
| 活性化関数 | SiLU |
| 正規化 | GroupNorm (groups=8、チャンネル数が8未満の場合は min(8, C)) |
| Dropout | 0.1 |

### 3-2. ResBlock 詳細
```
GroupNorm → SiLU → Conv1d(3, pad=1)
  → 加算: t_emb による scale+shift
  → 加算: cond_emb による scale+shift
  → GroupNorm → SiLU → Dropout → Conv1d(3, pad=1)
  → 加算: skip connection (チャンネル数変化時は Conv1d(1) で合わせる)
```

### 3-3. SelfAttention 詳細
```
GroupNorm → reshape to [B, T, C] → MultiheadAttention(num_heads=4) → reshape → 残差加算
適用レベル: ボトルネックと末尾2レベルのデコーダ
```

---

## 3. DDPM 仕様 (`ddpm.py`, `noise_schedule.py`)

### 4-1. ノイズスケジュール
- **方式**: コサインスケジュール (Nichol & Dhariwal, 2021)
- **総ステップ数** `T`: 1000
- `β_t` 上限: 0.9999

```python
alphas_cumprod(t) = cos(((t/T + s) / (1 + s)) * π/2)²  / alphas_cumprod(0)
s = 0.008  (オフセット、t=0 での過大ノイズを防ぐ)
```

### 4-2. 学習 (順拡散 + ノイズ予測)
```
t ~ Uniform(0, T)
ε ~ N(0, I)
x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
loss = Huber(ε, ε_θ(x_t, t, c))
```
- 損失関数: Huber loss (L2 より外れ値に頑健)

### 4-3. 推論 (部分ノイズ化 → 逆拡散)
```
1. x_{t*} = √ᾱ_{t*} · x₀ + √(1-ᾱ_{t*}) · ε,  ε ~ N(0, I)
2. for step in reversed(range(t*)):
     x₀_pred = (x_t - √(1-ᾱ_t) · ε_θ(x_t, t, c)) / √ᾱ_t
     x₀_pred = clamp(x₀_pred, -4, 4)
     posterior_mean を計算して x_{t-1} を得る
3. x̂₀ = x₀_pred at step=0
```
- デフォルト `t*` (`inference_T`): 200
- 複数回 (n_samples=5) 再構成して期待誤差を取ることでノイズ分散を抑える

---

## 4. 学習仕様 (`trainer.py`)

| ハイパーパラメータ | デフォルト値 | 説明 |
|---|---|---|
| `n_epochs` | 50 | エポック数 |
| `batch_size` | 64 | バッチサイズ |
| `lr` | 1e-4 | 初期学習率 |
| `optimizer` | Adam (β₁=0.9, β₂=0.999) | |
| `lr_scheduler` | CosineAnnealingLR (η_min = lr×0.1) | |
| `grad_clip` | 1.0 | 勾配クリッピング |
| `loss_type` | `huber` | `l1` / `l2` / `huber` から選択 |

- **学習データ**: 正常データのみ使用
- **検証**: 検証セット (正常) の loss を毎エポック記録。最良モデルを保存。

---

## 5. 評価仕様 (`evaluator.py`)

### 6-1. 異常スコア算出
```
score(x₀) = (1/n_samples) Σ_k ||x₀ - x̂₀^(k)||²  (チャンネル・時間軸の平均)
```

### 6-2. 評価指標
| 指標 | 算出方法 |
|---|---|
| AUROC | `sklearn.metrics.roc_auc_score` |
| 最適閾値 | Youden's J (TPR - FPR 最大化点) |
| Precision / Recall / F1 | 最適閾値での二値分類 |
| MSE (正常/異常 別) | 各グループの再構成 MSE 平均 |

### 6-3. t* 感度分析
- `inference_T` を `[50, 100, 150, 200, 250, 300, 400, 500]` で AUROC を計測
- 最良 `t*` を `results.json` に記録

### 6-4. 可視化出力 (outputs/)

| ファイル | 内容 |
|---|---|
| `training_curve.png` | エポックごとの学習・検証 loss |
| `roc_curve.png` | ROC 曲線 + AUROC 値 |
| `reconstruction.png` | 正常×3 / 異常×3 サンプルの元波形と再構成の重ね描き |
| `anomaly_score_dist.png` | 正常・異常の異常スコアヒストグラム (分布の分離度を視覚化) |
| `sensitivity_t_star.png` | t* vs AUROC の折れ線グラフ |

---

## 6. 設定管理 (`config.py`)

全ハイパーパラメータを `Config` dataclass に集約する。`main.py` で argparse から上書き可能にする。

```python
@dataclass
class Config:
    # データ
    n_train: int = 2000
    n_test: int = 500
    n_channels: int = 4
    seq_len: int = 128
    n_conditions: int = 2
    anomaly_ratio: float = 0.5   # テストデータ中の異常割合

    # モデル
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 4, 4)
    time_emb_dim: int = 64
    dropout: float = 0.1
    attn_levels: int = 2

    # DDPM
    T: int = 1000
    inference_T: int = 200
    n_recon_samples: int = 5

    # 学習
    n_epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-4
    loss_type: str = "huber"

    # システム
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs"
```

---

## 7. ディレクトリ構成

```
cddpm_anomaly/
├── CLAUDE.md
├── Requirements.md
├── requirements.txt
├── config.py
├── data_generator.py
├── noise_schedule.py
├── model.py
├── ddpm.py
├── trainer.py
├── evaluator.py
├── main.py
├── tests/
│   ├── test_data_generator.py
│   ├── test_noise_schedule.py
│   ├── test_model.py
│   ├── test_ddpm.py
│   ├── test_evaluator.py
│   └── test_integration.py
└── outputs/
    ├── model_best.pt
    ├── training_curve.png
    ├── roc_curve.png
    ├── reconstruction.png
    ├── anomaly_score_dist.png
    ├── sensitivity_t_star.png
    └── results.json
```

---

## 8. 依存ライブラリ (`requirements.txt`)

```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
black>=23.0.0
ruff>=0.1.0
pytest>=7.4.0
```

---

## 9. 将来の拡張ポイント (実験後に検討)

| 拡張 | 概要 |
|---|---|
| DDIM サンプリング | 推論ステップを 10〜50 に削減して高速化 |
| チャンネルごとの異常ローカライゼーション | `mean_error [B, C, T]` を使い異常チャンネル・時刻を特定 |
| 実データへの適用 | SMAP / MSL (NASA) や SMD などの公開センサデータセットで評価 |
| Classifier-Free Guidance | 条件なし/ありの予測を線形補間してより鮮明な再構成 |
| アンサンブル | 複数 `t*` の再構成誤差を合計してスコア安定化 |
