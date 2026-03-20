"""DDPM コサインノイズスケジュール (Nichol & Dhariwal, 2021)."""

from __future__ import annotations

import math

import torch


def cosine_schedule(T: int, s: float = 0.008, beta_max: float = 0.9999) -> dict[str, torch.Tensor]:
    """コサインスケジュールに基づく DDPM パラメータを計算する.

    Args:
        T: 総拡散ステップ数.
        s: オフセット (t=0 での過大ノイズを防ぐ).
        beta_max: β_t の上限値.

    Returns:
        以下のキーを持つ dict:
            - betas: β_t [T]
            - alphas: α_t = 1 - β_t [T]
            - alphas_cumprod: ᾱ_t = Π_{i=0}^{t} α_i [T]
            - sqrt_alphas_cumprod: √ᾱ_t [T]
            - sqrt_one_minus_alphas_cumprod: √(1-ᾱ_t) [T]
            - posterior_variance: β̃_t [T]
            - posterior_mean_coef1: 係数1 for posterior mean [T]
            - posterior_mean_coef2: 係数2 for posterior mean [T]
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]

    # β_t = 1 - ᾱ_t / ᾱ_{t-1}, clamp to beta_max
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = betas.clamp(max=beta_max)

    alphas = 1.0 - betas
    alphas_cumprod = alphas_cumprod[1:]  # [T], t=0..T-1 に対応

    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # Posterior mean coefficients
    posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)

    return {
        "betas": betas.float(),
        "alphas": alphas.float(),
        "alphas_cumprod": alphas_cumprod.float(),
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod.float(),
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod.float(),
        "posterior_variance": posterior_variance.float(),
        "posterior_mean_coef1": posterior_mean_coef1.float(),
        "posterior_mean_coef2": posterior_mean_coef2.float(),
    }
