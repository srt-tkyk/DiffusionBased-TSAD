# CLAUDE.md — Conditional DDPM for Multivariate Time Series Anomaly Detection

> 実験の目的・背景は [README.md](README.md) を参照。
> 機能・技術仕様は [Requirements.md](Requirements.md) を参照。

---

## 実装の進め方 (Research → Plan → Implement サイクル)

### フェーズ 1: 実装前に必ず確認すること
- `Requirements.md` を読み、全セクションを把握する
- 各モジュールの依存関係を `requirements.txt` で確認してから実装を開始する
- 不明な設計判断は実装を止めてユーザーに確認する

### フェーズ 2: 実装順序
以下の順序で実装すること。前のモジュールが通過しない限り次に進まない。

```
1. data_generator.py     — 合成センサデータ生成
2. noise_schedule.py     — DDPMノイズスケジュール
3. model.py              — Conditional 1D UNet
4. ddpm.py               — DDPM学習・推論ロジック
5. trainer.py            — 学習ループ
6. evaluator.py          — 異常検知評価・可視化
7. main.py               — エントリーポイント
```

### フェーズ 3: 各モジュール完成後にテストを実行
```bash
python -m pytest tests/ -v
# 全テストが PASSED になってから次のモジュールへ
```

---

## コーディング規約

### 言語・バージョン
- Python 3.10+
- PyTorch 2.0+
- 型ヒントを全関数に付ける (`from __future__ import annotations` を先頭に)

### スタイル
- フォーマッター: `black` (line-length=100)
- linter: `ruff`
- docstring: Google スタイル (日本語可)
- クラス・関数には必ず docstring を書く

### 命名規則
```python
# テンソルの変数名は shape をコメントで明記する
x_0: torch.Tensor   # [B, C, T]  B=batch, C=channels, T=timesteps
x_t: torch.Tensor   # [B, C, T]
noise: torch.Tensor # [B, C, T]
t: torch.Tensor     # [B]        int64, diffusion timestep
condition: torch.Tensor  # [B]   int64, equipment mode label
```

### 禁止事項
- マジックナンバーを直接埋め込むこと → `config.py` の `Config` dataclass に集約する
- `print()` でのデバッグログ → `logging` モジュールを使う
- GPU/CPU の文字列ハードコード → `device` 引数として受け取る
- `model.eval()` / `torch.no_grad()` の付け忘れ (推論・評価時)

---

## アーキテクチャ上の重要な設計判断

### なぜ「部分ノイズ化」推論を使うか
- 完全なノイズ `x_T ~ N(0,I)` から逆拡散すると再構成は「平均的な正常信号」になり、
  入力信号との対応が失われる
- `t* << T` の部分ノイズ化により、入力の構造を保ちながら異常部分だけを正常化できる
- `t*` のチューニングが異常検知精度に大きく影響する (evaluator で感度分析を行う)

### 条件埋め込みの役割
- 同一設備でも稼働モード (高負荷/低負荷/起動中) でセンサパターンが異なる
- 条件なしの場合、モード変化を異常として誤検知するリスクがある
- `nn.Embedding(num_conditions, cond_dim)` をタイムステップ埋め込みと同様に
  各 ResBlock へ scale+shift として注入する

### 1D UNet でのスキップ接続
- ダウンサンプル前の特徴量をデコーダに渡す
- 長い時系列では解像度変化に注意: `F.interpolate` でサイズを合わせる

---

## テスト要件

### 必須テスト (tests/ に配置)
```
test_data_generator.py    — データ形状・値域チェック
test_noise_schedule.py    — α_bar の単調減少・端値チェック
test_model.py             — forward pass の出力 shape チェック
test_ddpm.py              — q_sample, p_loss, reconstruct の動作確認
test_evaluator.py         — AUROC が 0.5 以上になること (最低限)
```

### 統合テスト
- `test_integration.py`: データ生成 → 学習(5エポック) → 評価 が
  エラーなく完走し、AUROC が 0.5 以上であること

---

## 実行方法

```bash
# 依存インストール
pip install -r requirements.txt

# 合成データで学習・評価 (デフォルト設定)
python main.py

# 設定を変えて実行
python main.py --n_epochs 100 --n_channels 8 --inference_t 300

# テスト
python -m pytest tests/ -v
```

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `outputs/model_best.pt` | 最良モデルの重み |
| `outputs/training_curve.png` | 学習損失曲線 |
| `outputs/roc_curve.png` | ROC 曲線と AUROC |
| `outputs/reconstruction.png` | 正常/異常サンプルの再構成比較 |
| `outputs/anomaly_score_dist.png` | 異常スコア分布 |
| `outputs/sensitivity_t_star.png` | t* 感度分析 |
| `outputs/results.json` | AUROC・閾値・各種指標 |

---

## 既知の落とし穴

1. **GroupNorm の groups 引数**: `in_channels < groups` になるとエラー。
   `groups = min(8, in_channels)` で対応する。

2. **UNet のサイズ不一致**: ダウンサンプル後の T が奇数の場合、
   アップサンプル後にサイズがずれる。`F.interpolate(size=skip.shape[-1])` で吸収する。

3. **スキップ接続の順序**: `skips` リストへの push/pop 順序を
   エンコーダ/デコーダで対称に保つこと。

4. **推論時の `condition` 引数**: テストデータのラベルが学習時と同じ分布である
   前提。未知モードが来る場合は `num_conditions` を大きめに設定するか、
   デフォルト条件 (condition=0) にフォールバックする処理を入れる。

5. **`t_noise` の上限**: `t_noise <= T` であること。
   `T=1000` に対し `inference_T=200` 程度が推奨値。
