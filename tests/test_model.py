"""model のテスト — forward pass の出力 shape チェック."""

from __future__ import annotations

import torch

from config import Config
from model import ConditionalUNet1d, SinusoidalTimestepEmbedding, ResBlock, SelfAttention1d


def _small_config() -> Config:
    """テスト用の小さな Config を返す."""
    return Config(
        n_channels=4,
        seq_len=128,
        base_channels=32,
        channel_mults=(1, 2, 4, 4),
        time_emb_dim=32,
        dropout=0.0,
        attn_levels=2,
        n_conditions=2,
    )


class TestSinusoidalEmbedding:
    """SinusoidalTimestepEmbedding のテスト."""

    def test_output_shape(self) -> None:
        """出力 shape が [B, dim] であること."""
        emb = SinusoidalTimestepEmbedding(64)
        t = torch.randint(0, 1000, (8,))
        out = emb(t)
        assert out.shape == (8, 64)


class TestResBlock:
    """ResBlock のテスト."""

    def test_same_channels(self) -> None:
        """入出力チャンネルが同じ場合の shape."""
        block = ResBlock(32, 32, 128, dropout=0.0)
        x = torch.randn(2, 32, 64)
        t_emb = torch.randn(2, 128)
        c_emb = torch.randn(2, 128)
        out = block(x, t_emb, c_emb)
        assert out.shape == (2, 32, 64)

    def test_different_channels(self) -> None:
        """入出力チャンネルが異なる場合の shape."""
        block = ResBlock(32, 64, 128, dropout=0.0)
        x = torch.randn(2, 32, 64)
        t_emb = torch.randn(2, 128)
        c_emb = torch.randn(2, 128)
        out = block(x, t_emb, c_emb)
        assert out.shape == (2, 64, 64)


class TestSelfAttention:
    """SelfAttention1d のテスト."""

    def test_output_shape(self) -> None:
        """出力 shape が入力と同じであること."""
        attn = SelfAttention1d(32, num_heads=4)
        x = torch.randn(2, 32, 64)
        out = attn(x)
        assert out.shape == x.shape


class TestConditionalUNet1d:
    """ConditionalUNet1d のテスト."""

    def setup_method(self) -> None:
        """モデルとテストデータを準備."""
        self.cfg = _small_config()
        self.model = ConditionalUNet1d(self.cfg)
        self.model.eval()

    def test_forward_shape(self) -> None:
        """forward の出力 shape が入力と同じ [B, C, T] であること."""
        B = 4
        x = torch.randn(B, self.cfg.n_channels, self.cfg.seq_len)  # [B, C, T]
        t = torch.randint(0, self.cfg.T, (B,))  # [B]
        cond = torch.randint(0, self.cfg.n_conditions, (B,))  # [B]

        with torch.no_grad():
            out = self.model(x, t, cond)

        assert out.shape == (B, self.cfg.n_channels, self.cfg.seq_len)

    def test_forward_different_seq_len(self) -> None:
        """seq_len=64 でも動作すること."""
        cfg = _small_config()
        cfg.seq_len = 64
        model = ConditionalUNet1d(cfg)
        model.eval()

        B = 2
        x = torch.randn(B, cfg.n_channels, cfg.seq_len)
        t = torch.randint(0, cfg.T, (B,))
        cond = torch.randint(0, cfg.n_conditions, (B,))

        with torch.no_grad():
            out = model(x, t, cond)

        assert out.shape == (B, cfg.n_channels, cfg.seq_len)

    def test_batch_size_1(self) -> None:
        """バッチサイズ 1 でも動作すること."""
        x = torch.randn(1, self.cfg.n_channels, self.cfg.seq_len)
        t = torch.randint(0, self.cfg.T, (1,))
        cond = torch.randint(0, self.cfg.n_conditions, (1,))

        with torch.no_grad():
            out = self.model(x, t, cond)

        assert out.shape == (1, self.cfg.n_channels, self.cfg.seq_len)

    def test_output_not_zero(self) -> None:
        """出力がゼロでないこと (初期化直後でも非ゼロの出力が期待される)."""
        x = torch.randn(2, self.cfg.n_channels, self.cfg.seq_len)
        t = torch.randint(0, self.cfg.T, (2,))
        cond = torch.randint(0, self.cfg.n_conditions, (2,))

        with torch.no_grad():
            out = self.model(x, t, cond)

        assert out.abs().sum() > 0
