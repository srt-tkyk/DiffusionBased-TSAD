"""data_generator のテスト — データ形状・値域チェック."""

from __future__ import annotations

import torch

from config import Config
from data_generator import generate_data


def _small_config() -> Config:
    """テスト用の小さな Config を返す."""
    return Config(
        n_train=200,
        n_test=100,
        n_channels=4,
        seq_len=128,
        batch_size=32,
        seed=0,
    )


class TestGenerateData:
    """generate_data の出力を検証するテストクラス."""

    def setup_method(self) -> None:
        """各テスト前にデータを生成する."""
        self.cfg = _small_config()
        self.data = generate_data(self.cfg)

    def test_train_loader_shape(self) -> None:
        """学習バッチの shape が [B, C, T] であること."""
        batch, conds = next(iter(self.data.train_loader))
        assert batch.ndim == 3
        assert batch.shape[1] == self.cfg.n_channels
        assert batch.shape[2] == self.cfg.seq_len
        assert conds.ndim == 1

    def test_val_loader_exists(self) -> None:
        """検証 DataLoader が空でないこと."""
        batch, _ = next(iter(self.data.val_loader))
        assert batch.shape[1] == self.cfg.n_channels

    def test_test_data_shape(self) -> None:
        """テストデータの shape が [N_test, C, T] であること."""
        assert self.data.test_data.shape == (
            self.cfg.n_test,
            self.cfg.n_channels,
            self.cfg.seq_len,
        )

    def test_test_labels_shape(self) -> None:
        """テストラベルの shape が [N_test] であること."""
        assert self.data.test_labels.shape == (self.cfg.n_test,)

    def test_test_labels_values(self) -> None:
        """テストラベルが 0 と 1 のみであること."""
        unique = torch.unique(self.data.test_labels)
        assert set(unique.tolist()).issubset({0, 1})

    def test_anomaly_ratio(self) -> None:
        """異常データの割合が設定通りであること."""
        n_anomaly = (self.data.test_labels == 1).sum().item()
        expected = int(self.cfg.n_test * self.cfg.anomaly_ratio)
        assert n_anomaly == expected

    def test_normalization(self) -> None:
        """mean / std テンソルの shape が [C] であること."""
        assert self.data.mean.shape == (self.cfg.n_channels,)
        assert self.data.std.shape == (self.cfg.n_channels,)

    def test_no_nan(self) -> None:
        """データに NaN が含まれないこと."""
        assert not torch.isnan(self.data.test_data).any()
        for batch, _ in self.data.train_loader:
            assert not torch.isnan(batch).any()
            break

    def test_conditions_range(self) -> None:
        """条件ラベルが [0, n_conditions) の範囲であること."""
        assert self.data.test_conditions.min() >= 0
        assert self.data.test_conditions.max() < self.cfg.n_conditions
