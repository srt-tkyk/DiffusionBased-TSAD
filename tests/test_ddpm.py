"""ddpm のテスト — q_sample, p_loss, reconstruct の動作確認."""

from __future__ import annotations

import torch

from config import Config
from ddpm import DDPM
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
        n_recon_samples=2,
    )


def _make_ddpm(config: Config) -> DDPM:
    """Config から DDPM を生成."""
    model = ConditionalUNet1d(config)
    return DDPM(model, config)


class TestQSample:
    """q_sample のテスト."""

    def setup_method(self) -> None:
        """共通の DDPM を準備."""
        self.cfg = _small_config()
        self.ddpm = _make_ddpm(self.cfg)

    def test_output_shape(self) -> None:
        """出力 shape が入力と同じであること."""
        x_0 = torch.randn(4, self.cfg.n_channels, self.cfg.seq_len)
        t = torch.randint(0, self.cfg.T, (4,))
        x_t, noise = self.ddpm.q_sample(x_0, t)
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape

    def test_t_zero_close_to_original(self) -> None:
        """t=0 のとき x_t ≈ x_0 であること (ノイズがほぼゼロ)."""
        x_0 = torch.randn(4, self.cfg.n_channels, self.cfg.seq_len)
        t = torch.zeros(4, dtype=torch.long)
        x_t, _ = self.ddpm.q_sample(x_0, t)
        # t=0 では alphas_cumprod[0] ≈ 1 なので x_t ≈ x_0
        assert torch.allclose(x_t, x_0, atol=0.2)

    def test_custom_noise(self) -> None:
        """指定したノイズが使用されること."""
        x_0 = torch.randn(2, self.cfg.n_channels, self.cfg.seq_len)
        t = torch.tensor([50, 50])
        noise = torch.zeros_like(x_0)
        x_t, returned_noise = self.ddpm.q_sample(x_0, t, noise=noise)
        assert torch.equal(returned_noise, noise)


class TestPLoss:
    """p_loss のテスト."""

    def setup_method(self) -> None:
        """共通の DDPM を準備."""
        self.cfg = _small_config()
        self.ddpm = _make_ddpm(self.cfg)

    def test_loss_is_scalar(self) -> None:
        """損失がスカラーであること."""
        x_0 = torch.randn(4, self.cfg.n_channels, self.cfg.seq_len)
        cond = torch.randint(0, self.cfg.n_conditions, (4,))
        loss = self.ddpm.p_loss(x_0, cond, loss_type="huber")
        assert loss.ndim == 0

    def test_loss_positive(self) -> None:
        """損失が正であること."""
        x_0 = torch.randn(4, self.cfg.n_channels, self.cfg.seq_len)
        cond = torch.randint(0, self.cfg.n_conditions, (4,))
        for loss_type in ["huber", "l1", "l2"]:
            loss = self.ddpm.p_loss(x_0, cond, loss_type=loss_type)
            assert loss.item() > 0, f"loss_type={loss_type} should be positive"

    def test_loss_backward(self) -> None:
        """損失の backward が通ること."""
        x_0 = torch.randn(4, self.cfg.n_channels, self.cfg.seq_len)
        cond = torch.randint(0, self.cfg.n_conditions, (4,))
        loss = self.ddpm.p_loss(x_0, cond, loss_type="huber")
        loss.backward()
        # 勾配が計算されていることを確認
        for p in self.ddpm.model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break


class TestReconstruct:
    """reconstruct のテスト."""

    def test_output_shape(self) -> None:
        """再構成結果の shape が入力と同じであること."""
        cfg = _small_config()
        ddpm = _make_ddpm(cfg)
        ddpm.eval()

        x_0 = torch.randn(2, cfg.n_channels, cfg.seq_len)
        cond = torch.randint(0, cfg.n_conditions, (2,))
        recon = ddpm.reconstruct(x_0, cond)
        assert recon.shape == x_0.shape

    def test_no_nan(self) -> None:
        """再構成結果に NaN がないこと."""
        cfg = _small_config()
        ddpm = _make_ddpm(cfg)
        ddpm.eval()

        x_0 = torch.randn(2, cfg.n_channels, cfg.seq_len)
        cond = torch.randint(0, cfg.n_conditions, (2,))
        recon = ddpm.reconstruct(x_0, cond)
        assert not torch.isnan(recon).any()
