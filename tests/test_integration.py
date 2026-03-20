"""統合テスト — データ生成→学習(5エポック)→評価がエラーなく完走し、AUROC ≥ 0.5."""

from __future__ import annotations

import tempfile

from config import Config
from main import main


class TestIntegration:
    """統合テスト."""

    def test_full_pipeline(self) -> None:
        """データ生成→学習→評価が完走し AUROC ≥ 0.5 であること."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                n_train=128,
                n_test=64,
                n_channels=2,
                seq_len=32,
                base_channels=16,
                channel_mults=(1, 2),
                time_emb_dim=16,
                dropout=0.0,
                attn_levels=1,
                n_conditions=2,
                T=50,
                inference_T=10,
                n_recon_samples=1,
                n_epochs=5,
                batch_size=32,
                lr=1e-3,
                loss_type="huber",
                seed=42,
                device="cpu",
                output_dir=tmpdir,
            )

            results = main(config)

            assert "auroc" in results
            assert results["auroc"] >= 0.5, f"AUROC = {results['auroc']:.4f}, expected >= 0.5"
            assert "sensitivity_t_star" in results
            assert len(results["sensitivity_t_star"]) > 0
