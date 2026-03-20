"""DDPM 学習・推論ロジック.

順拡散 (q_sample)、損失計算 (p_loss)、
部分ノイズ化 → 逆拡散による再構成 (reconstruct) を提供する。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model import ConditionalUNet1d
from noise_schedule import cosine_schedule

logger = logging.getLogger(__name__)


class DDPM(nn.Module):
    """Conditional DDPM の学習・推論を管理するモジュール.

    Args:
        model: ノイズ予測 UNet.
        config: プロジェクト設定.
    """

    def __init__(self, model: ConditionalUNet1d, config: Config) -> None:
        super().__init__()
        self.model = model
        self.config = config

        sched = cosine_schedule(config.T)
        # バッファとして登録 (device 移動時に自動的に移る)
        for key, val in sched.items():
            self.register_buffer(key, val)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """順拡散: x_0 にステップ t 分のノイズを付加する.

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

        Args:
            x_0: 元信号 [B, C, T].
            t: 拡散ステップ [B].
            noise: 使用するノイズ (None の場合はランダム生成) [B, C, T].

        Returns:
            (x_t, noise): ノイズ付き信号とノイズのタプル.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]  # [B, 1, 1]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]  # [B, 1, 1]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise  # [B, C, T]
        return x_t, noise

    def p_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        loss_type: str = "huber",
    ) -> torch.Tensor:
        """学習損失を計算する.

        Args:
            x_0: 正常信号 [B, C, T].
            condition: 条件ラベル [B].
            loss_type: 損失関数の種類 ('huber' / 'l1' / 'l2').

        Returns:
            スカラー損失値.
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.config.T, (B,), device=x_0.device)

        x_t, noise = self.q_sample(x_0, t)
        noise_pred = self.model(x_t, t, condition)  # [B, C, T]

        if loss_type == "huber":
            loss = F.smooth_l1_loss(noise_pred, noise)
        elif loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        return loss

    @torch.no_grad()
    def _p_sample_step(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """逆拡散の 1 ステップ.

        Args:
            x_t: 現在のノイズ付き信号 [B, C, T].
            t_idx: 現在のステップインデックス (0-indexed).
            condition: 条件ラベル [B].

        Returns:
            x_{t-1} [B, C, T].
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)

        # ε_θ の予測
        noise_pred = self.model(x_t, t_tensor, condition)

        # x_0 の予測: x_0_pred = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t_idx]
        x_0_pred = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        x_0_pred = x_0_pred.clamp(-4, 4)

        # Posterior mean
        mean = (
            self.posterior_mean_coef1[t_idx] * x_0_pred
            + self.posterior_mean_coef2[t_idx] * x_t
        )

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t_idx]
            return mean + variance.sqrt() * noise
        else:
            return mean

    @torch.no_grad()
    def reconstruct(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        inference_T: int | None = None,
        n_samples: int | None = None,
    ) -> torch.Tensor:
        """部分ノイズ化 → 逆拡散で再構成する.

        Args:
            x_0: 入力信号 [B, C, T].
            condition: 条件ラベル [B].
            inference_T: 部分ノイズ化ステップ数 (None で config.inference_T).
            n_samples: 再構成サンプル数 (None で config.n_recon_samples).

        Returns:
            再構成信号 [B, C, T] (n_samples 回の平均).
        """
        self.model.eval()

        if inference_T is None:
            inference_T = self.config.inference_T
        if n_samples is None:
            n_samples = self.config.n_recon_samples

        assert inference_T <= self.config.T, (
            f"inference_T ({inference_T}) must be <= T ({self.config.T})"
        )

        reconstructions = []
        for _ in range(n_samples):
            # 部分ノイズ化: x_{t*} = √ᾱ_{t*} · x_0 + √(1-ᾱ_{t*}) · ε
            t_star = torch.full((x_0.shape[0],), inference_T - 1, device=x_0.device, dtype=torch.long)
            x_t, _ = self.q_sample(x_0, t_star)

            # 逆拡散
            for step in reversed(range(inference_T)):
                x_t = self._p_sample_step(x_t, step, condition)

            reconstructions.append(x_t)

        # 複数回再構成の平均
        return torch.stack(reconstructions).mean(dim=0)  # [B, C, T]
