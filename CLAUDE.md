# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 機能・技術仕様の詳細は [Requirements.md](Requirements.md) を参照。

---

## プロジェクト概要

条件付き DDPM (Denoising Diffusion Probabilistic Models) を用いた多次元時系列異常検知システム。
工場センサデータを模した合成データで学習し、部分ノイズ化による再構成誤差で異常を検出する。

**合格基準**: AUROC ≥ 0.85 / MSE(正常) ≤ 0.05 / MSE比(異常/正常) ≥ 3.0

---

## コマンド

```bash
# 依存インストール
pip install -r requirements.txt

# 学習・評価 (デフォルト設定)
python main.py

# 設定を変えて実行
python main.py --n_epochs 100 --n_channels 8 --inference_t 300

# テスト (全件)
python -m pytest tests/ -v

# 単一テスト実行
python -m pytest tests/test_model.py -v

# 特定テスト関数
python -m pytest tests/test_model.py::test_forward_shape -v

# フォーマット・lint
black --line-length 100 .
ruff check .
```

---

## 実装順序 (厳守)

前のモジュールのテストが全 PASSED になってから次に進むこと。

```
1. config.py           — Config dataclass (全ハイパーパラメータ集約)
2. data_generator.py   — 合成センサデータ生成
3. noise_schedule.py   — コサインノイズスケジュール
4. model.py            — Conditional 1D UNet
5. ddpm.py             — DDPM 学習・推論ロジック
6. trainer.py          — 学習ループ
7. evaluator.py        — 異常検知評価・可視化
8. main.py             — エントリーポイント (argparse → Config 上書き)
```

---

## コーディング規約

- Python 3.10+ / PyTorch 2.0+
- 全関数に型ヒント (`from __future__ import annotations` を先頭に)
- フォーマッター: `black` (line-length=100)、linter: `ruff`
- docstring: Google スタイル (日本語可)、クラス・関数に必須
- テンソル変数名には shape をコメントで明記:
  ```python
  x_0: torch.Tensor   # [B, C, T]
  t: torch.Tensor     # [B] int64, diffusion timestep
  condition: torch.Tensor  # [B] int64, equipment mode label
  ```

### 禁止事項
- マジックナンバー直書き → `config.py` の `Config` dataclass に集約
- `print()` デバッグ → `logging` モジュールを使う
- GPU/CPU 文字列ハードコード → `device` 引数として受け取る
- 推論・評価時の `model.eval()` / `torch.no_grad()` 忘れ

---

## アーキテクチャ上の重要な設計判断

### 部分ノイズ化推論
- 完全ノイズ `x_T ~ N(0,I)` からの逆拡散では入力との対応が失われる
- `t* << T` (デフォルト 200/1000) の部分ノイズ化で入力構造を保ちつつ異常部分を正常化
- `t*` のチューニングが精度に大きく影響 → evaluator で感度分析を実施

### 条件埋め込み
- 設備の稼働モード差 (高負荷/低負荷) をモデルに伝えないと、モード変化を異常と誤検知する
- `nn.Embedding` → 各 ResBlock へ scale+shift として注入 (タイムステップ埋め込みと同方式)

### 1D UNet スキップ接続
- ダウンサンプル前の特徴量をデコーダに渡す
- 時系列長の不一致は `F.interpolate(size=skip.shape[-1])` で吸収

---

## 既知の落とし穴

1. **GroupNorm の groups 引数**: `in_channels < groups` でエラー → `groups = min(8, in_channels)`
2. **UNet サイズ不一致**: ダウンサンプル後の T が奇数 → アップサンプル後にずれる → `F.interpolate(size=skip.shape[-1])`
3. **スキップ接続の順序**: `skips` の push/pop 順序をエンコーダ/デコーダで対称に保つ
4. **推論時の condition**: 未知モードには `num_conditions` を大きめに設定するかデフォルト条件にフォールバック
5. **`inference_T` の上限**: `inference_T <= T` であること。`T=1000` に対し 200 が推奨

---

## テスト要件

| テストファイル | 検証内容 |
|---|---|
| `test_data_generator.py` | データ形状・値域チェック |
| `test_noise_schedule.py` | α_bar の単調減少・端値 |
| `test_model.py` | forward pass の出力 shape |
| `test_ddpm.py` | q_sample, p_loss, reconstruct の動作 |
| `test_evaluator.py` | AUROC ≥ 0.5 (最低限) |
| `test_integration.py` | データ生成→学習(5エポック)→評価がエラーなく完走、AUROC ≥ 0.5 |

---

## 出力ファイル (outputs/)

| ファイル | 内容 |
|---|---|
| `model_best.pt` | 最良モデルの重み |
| `training_curve.png` | 学習損失曲線 |
| `roc_curve.png` | ROC 曲線と AUROC |
| `reconstruction.png` | 正常/異常サンプルの再構成比較 |
| `anomaly_score_dist.png` | 異常スコア分布 |
| `sensitivity_t_star.png` | t* 感度分析 |
| `results.json` | AUROC・閾値・各種指標 |
