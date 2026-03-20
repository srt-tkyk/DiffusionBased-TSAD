"""Conditional 1D UNet for DDPM ノイズ予測.

Requirements.md セクション 2 の仕様に基づく実装。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class SinusoidalTimestepEmbedding(nn.Module):
    """正弦波ベースのタイムステップ埋め込み.

    Args:
        dim: 埋め込み次元数.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """タイムステップを埋め込みベクトルに変換する.

        Args:
            t: タイムステップ [B].

        Returns:
            埋め込みベクトル [B, dim].
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]
        return emb


class TimestepMLPEmbedding(nn.Module):
    """Sinusoidal → MLP のタイムステップ埋め込み.

    Args:
        time_emb_dim: sinusoidal 埋め込み次元.
        out_dim: 出力次元.
    """

    def __init__(self, time_emb_dim: int, out_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """タイムステップを MLP 埋め込みに変換する.

        Args:
            t: タイムステップ [B].

        Returns:
            埋め込みベクトル [B, out_dim].
        """
        return self.mlp(self.sinusoidal(t))


class ResBlock(nn.Module):
    """Adaptive GroupNorm 方式の ResBlock.

    GroupNorm → SiLU → Conv1d → (scale+shift by t_emb and cond_emb) →
    GroupNorm → SiLU → Dropout → Conv1d → skip connection.

    Args:
        in_ch: 入力チャンネル数.
        out_ch: 出力チャンネル数.
        emb_dim: 埋め込み次元 (time + condition).
        dropout: Dropout 率.
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        # time embedding → scale + shift (2 * out_ch)
        self.time_proj = nn.Linear(emb_dim, out_ch * 2)
        # condition embedding → scale + shift (2 * out_ch)
        self.cond_proj = nn.Linear(emb_dim, out_ch * 2)

        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        # Skip connection (チャンネル数変化時は 1x1 conv)
        self.skip_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor
    ) -> torch.Tensor:
        """ResBlock forward pass.

        Args:
            x: 入力 [B, in_ch, T].
            t_emb: タイムステップ埋め込み [B, emb_dim].
            cond_emb: 条件埋め込み [B, emb_dim].

        Returns:
            出力 [B, out_ch, T].
        """
        h = self.conv1(F.silu(self.norm1(x)))  # [B, out_ch, T]

        # Time embedding scale+shift
        t_params = self.time_proj(t_emb)[:, :, None]  # [B, 2*out_ch, 1]
        t_scale, t_shift = t_params.chunk(2, dim=1)
        h = h * (1 + t_scale) + t_shift

        # Condition embedding scale+shift
        c_params = self.cond_proj(cond_emb)[:, :, None]  # [B, 2*out_ch, 1]
        c_scale, c_shift = c_params.chunk(2, dim=1)
        h = h * (1 + c_scale) + c_shift

        h = self.conv2(self.dropout(F.silu(self.norm2(h))))  # [B, out_ch, T]
        return h + self.skip_conv(x)


class SelfAttention1d(nn.Module):
    """1D Self-Attention.

    GroupNorm → reshape [B, T, C] → MultiheadAttention → reshape → 残差加算.

    Args:
        channels: チャンネル数.
        num_heads: アテンションヘッド数.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-Attention forward pass.

        Args:
            x: 入力 [B, C, T].

        Returns:
            出力 [B, C, T].
        """
        b, c, t = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 1)  # [B, T, C]
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1)  # [B, C, T]
        return x + h


class Downsample(nn.Module):
    """Conv1d stride=2 によるダウンサンプリング.

    Args:
        channels: チャンネル数.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ダウンサンプル.

        Args:
            x: 入力 [B, C, T].

        Returns:
            出力 [B, C, T//2].
        """
        return self.conv(x)


class Upsample(nn.Module):
    """ConvTranspose1d stride=2 によるアップサンプリング.

    Args:
        channels: チャンネル数.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """アップサンプル.

        Args:
            x: 入力 [B, C, T].

        Returns:
            出力 [B, C, T*2].
        """
        return self.conv(x)


class ConditionalUNet1d(nn.Module):
    """条件付き 1D UNet (DDPM ノイズ予測).

    Args:
        config: プロジェクト設定.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        base_ch = config.base_channels
        ch_mults = config.channel_mults
        emb_dim = base_ch * 4  # 256
        n_levels = len(ch_mults)

        # Embeddings
        self.time_emb = TimestepMLPEmbedding(config.time_emb_dim, emb_dim)
        self.cond_emb = nn.Sequential(
            nn.Embedding(config.n_conditions, config.time_emb_dim),
            nn.Linear(config.time_emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv1d(config.n_channels, base_ch, kernel_size=3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_ch
        encoder_channels = [ch]  # skip connection 用のチャンネル数記録

        for level in range(n_levels):
            out_ch = base_ch * ch_mults[level]
            for _ in range(2):  # ResBlock × 2 per level
                self.encoder_blocks.append(ResBlock(ch, out_ch, emb_dim, config.dropout))
                ch = out_ch
                encoder_channels.append(ch)

            if level < n_levels - 1:
                self.downsamples.append(Downsample(ch))
                encoder_channels.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        # Bottleneck
        self.bottleneck = nn.ModuleList(
            [
                ResBlock(ch, ch, emb_dim, config.dropout),
                SelfAttention1d(ch),
                ResBlock(ch, ch, emb_dim, config.dropout),
            ]
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()

        for level in reversed(range(n_levels)):
            out_ch = base_ch * ch_mults[level]

            if level < n_levels - 1:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

            for i in range(2 + (1 if level < n_levels - 1 else 0)):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, emb_dim, config.dropout)
                )
                ch = out_ch

                # SelfAttention: ボトルネック側から attn_levels レベル分のデコーダに適用
                use_attn = level >= n_levels - config.attn_levels
                if use_attn:
                    self.decoder_attns.append(SelfAttention1d(ch))
                else:
                    self.decoder_attns.append(nn.Identity())

        # Output
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv1d(ch, config.n_channels, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """ノイズ予測の forward pass.

        Args:
            x: ノイズ付き入力 [B, C, T].
            t: 拡散タイムステップ [B].
            condition: 条件ラベル [B].

        Returns:
            予測ノイズ [B, C, T].
        """
        t_emb = self.time_emb(t)  # [B, emb_dim]
        c_emb = self.cond_emb(condition)  # [B, emb_dim]

        h = self.init_conv(x)  # [B, base_ch, T]
        skips = [h]

        # Encoder
        block_idx = 0
        for level in range(len(self.config.channel_mults)):
            for _ in range(2):
                h = self.encoder_blocks[block_idx](h, t_emb, c_emb)
                skips.append(h)
                block_idx += 1

            h = self.downsamples[level](h)
            if not isinstance(self.downsamples[level], nn.Identity):
                skips.append(h)

        # Bottleneck
        h = self.bottleneck[0](h, t_emb, c_emb)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, t_emb, c_emb)

        # Decoder
        block_idx = 0
        attn_idx = 0
        for i, level in enumerate(reversed(range(len(self.config.channel_mults)))):
            h = self.upsamples[i](h)

            n_blocks = 2 + (1 if level < len(self.config.channel_mults) - 1 else 0)
            for _ in range(n_blocks):
                skip = skips.pop()
                # サイズ不一致の場合は interpolate で吸収
                if h.shape[-1] != skip.shape[-1]:
                    h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
                h = torch.cat([h, skip], dim=1)  # [B, ch + skip_ch, T']
                h = self.decoder_blocks[block_idx](h, t_emb, c_emb)
                h = self.decoder_attns[attn_idx](h)
                block_idx += 1
                attn_idx += 1

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h
