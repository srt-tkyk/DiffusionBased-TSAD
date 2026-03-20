"""プロジェクト全体のハイパーパラメータを管理する Config dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    """Conditional DDPM 時系列異常検知の全設定を集約する dataclass.

    Args:
        n_train: 学習用正常サンプル数.
        n_test: テスト用サンプル数 (正常+異常).
        n_channels: センサチャンネル数.
        seq_len: 時系列長.
        n_conditions: 設備稼働モード数.
        anomaly_ratio: テストデータ中の異常割合.
        base_channels: UNet の基本チャンネル数.
        channel_mults: UNet 各レベルのチャンネル倍率.
        time_emb_dim: タイムステップ埋め込み次元.
        dropout: Dropout 率.
        attn_levels: デコーダ末尾で SelfAttention を適用するレベル数.
        T: DDPM 総拡散ステップ数.
        inference_T: 推論時の部分ノイズ化ステップ数.
        n_recon_samples: 推論時の再構成サンプル数 (分散低減用).
        n_epochs: 学習エポック数.
        batch_size: バッチサイズ.
        lr: 初期学習率.
        loss_type: 損失関数の種類 ('huber' / 'l1' / 'l2').
        grad_clip: 勾配クリッピングの最大ノルム.
        seed: 乱数シード.
        device: 計算デバイス.
        output_dir: 出力ディレクトリ.
    """

    # データ
    n_train: int = 2000
    n_test: int = 500
    n_channels: int = 4
    seq_len: int = 128
    n_conditions: int = 2
    anomaly_ratio: float = 0.5

    # モデル
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4, 4)
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
    grad_clip: float = 1.0

    # システム
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    output_dir: str = "outputs"
