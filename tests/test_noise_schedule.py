"""noise_schedule のテスト — α_bar の単調減少・端値チェック."""

from __future__ import annotations

import torch

from noise_schedule import cosine_schedule


class TestCosineSchedule:
    """cosine_schedule の出力を検証するテストクラス."""

    def setup_method(self) -> None:
        """共通パラメータでスケジュールを生成."""
        self.T = 1000
        self.sched = cosine_schedule(self.T)

    def test_output_keys(self) -> None:
        """必要なキーがすべて含まれていること."""
        expected_keys = {
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        }
        assert set(self.sched.keys()) == expected_keys

    def test_shapes(self) -> None:
        """すべてのテンソルが [T] の shape であること."""
        for key, val in self.sched.items():
            assert val.shape == (self.T,), f"{key} shape mismatch: {val.shape}"

    def test_alphas_cumprod_monotonically_decreasing(self) -> None:
        """ᾱ_t が単調減少であること."""
        ac = self.sched["alphas_cumprod"]
        diff = ac[1:] - ac[:-1]
        assert (diff <= 0).all(), "alphas_cumprod is not monotonically decreasing"

    def test_alphas_cumprod_range(self) -> None:
        """ᾱ_t が (0, 1] の範囲であること."""
        ac = self.sched["alphas_cumprod"]
        assert ac[0] > 0.9, f"alphas_cumprod[0] too small: {ac[0]}"
        assert ac[-1] > 0, f"alphas_cumprod[-1] should be > 0: {ac[-1]}"
        assert ac[-1] < 0.1, f"alphas_cumprod[-1] too large: {ac[-1]}"

    def test_betas_range(self) -> None:
        """β_t が (0, beta_max] の範囲であること."""
        betas = self.sched["betas"]
        assert (betas > 0).all(), "betas should be positive"
        assert (betas <= 0.9999).all(), "betas should be <= beta_max"

    def test_alphas_plus_betas(self) -> None:
        """α_t + β_t = 1 であること."""
        total = self.sched["alphas"] + self.sched["betas"]
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)

    def test_sqrt_consistency(self) -> None:
        """sqrt 値が alphas_cumprod と整合すること."""
        ac = self.sched["alphas_cumprod"]
        assert torch.allclose(self.sched["sqrt_alphas_cumprod"], ac.sqrt(), atol=1e-5)
        assert torch.allclose(
            self.sched["sqrt_one_minus_alphas_cumprod"], (1.0 - ac).sqrt(), atol=1e-5
        )

    def test_small_T(self) -> None:
        """T=10 でもエラーなく動作すること."""
        sched = cosine_schedule(10)
        assert sched["betas"].shape == (10,)
