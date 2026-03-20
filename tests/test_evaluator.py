"""evaluator のテスト — AUROC が 0.5 以上になること (最低限)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from config import Config
from ddpm import DDPM
from evaluator import compute_anomaly_scores
from model import ConditionalUNet1d


def _small_config() -> Config:
    """テスト用の小さな Config を返す."""
    return Config(
        n_channels=2,
        seq_len=32,
        base_channels=16,
        channel_mults=(1, 2),
        time_emb_dim=16,
        dropout=0.0,
        attn_levels=1,
        n_conditions=2,
        T=100,
        inference_T=10,
        n_recon_samples=1,
        device="cpu",
    )


class TestComputeAnomalyScores:
    """compute_anomaly_scores のテスト."""

    def setup_method(self) -> None:
        """テスト用 DDPM を準備."""
        self.cfg = _small_config()
        model = ConditionalUNet1d(self.cfg)
        self.ddpm = DDPM(model, self.cfg)
        self.ddpm.eval()

    def test_output_shape(self) -> None:
        """出力のサイズが入力サンプル数と一致すること."""
        N = 10
        data = torch.randn(N, self.cfg.n_channels, self.cfg.seq_len)
        conditions = torch.randint(0, self.cfg.n_conditions, (N,))
        scores = compute_anomaly_scores(self.ddpm, data, conditions, self.cfg)
        assert scores.shape == (N,)

    def test_scores_non_negative(self) -> None:
        """異常スコアが非負であること (MSE なので)."""
        N = 10
        data = torch.randn(N, self.cfg.n_channels, self.cfg.seq_len)
        conditions = torch.randint(0, self.cfg.n_conditions, (N,))
        scores = compute_anomaly_scores(self.ddpm, data, conditions, self.cfg)
        assert (scores >= 0).all()

    def test_auroc_above_chance(self) -> None:
        """明確に異なる分布を持つデータで AUROC ≥ 0.5 となること.

        未学習モデルでもこのテストは構造上の健全性を確認する。
        正常データとスケーリングされた異常データを用いる。
        """
        N = 50
        normal = torch.randn(N, self.cfg.n_channels, self.cfg.seq_len) * 0.1
        anomaly = torch.randn(N, self.cfg.n_channels, self.cfg.seq_len) * 5.0

        data = torch.cat([normal, anomaly], dim=0)
        conditions = torch.zeros(2 * N, dtype=torch.long)
        labels = np.array([0] * N + [1] * N)

        scores = compute_anomaly_scores(self.ddpm, data, conditions, self.cfg)
        auroc = roc_auc_score(labels, scores)
        assert auroc >= 0.5, f"AUROC = {auroc:.4f}, expected >= 0.5"
