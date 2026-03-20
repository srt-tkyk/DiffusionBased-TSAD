"""エントリーポイント.

合成データ生成 → DDPM 学習 → 評価 のパイプラインを実行する。
argparse で Config のフィールドを上書き可能。
"""

from __future__ import annotations

import argparse
import logging

import torch

from config import Config
from data_generator import generate_data
from ddpm import DDPM
from evaluator import evaluate
from model import ConditionalUNet1d
from trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> Config:
    """コマンドライン引数を解析して Config を生成する.

    Returns:
        Config オブジェクト.
    """
    parser = argparse.ArgumentParser(description="Conditional DDPM for Time Series Anomaly Detection")

    # データ
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--n_channels", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_conditions", type=int, default=2)
    parser.add_argument("--anomaly_ratio", type=float, default=0.5)

    # モデル
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_emb_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_levels", type=int, default=2)

    # DDPM
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--inference_t", type=int, default=200, dest="inference_T")
    parser.add_argument("--n_recon_samples", type=int, default=5)

    # 学習
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default="huber", choices=["huber", "l1", "l2"])
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # システム
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    config = Config(
        n_train=args.n_train,
        n_test=args.n_test,
        n_channels=args.n_channels,
        seq_len=args.seq_len,
        n_conditions=args.n_conditions,
        anomaly_ratio=args.anomaly_ratio,
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim,
        dropout=args.dropout,
        attn_levels=args.attn_levels,
        T=args.T,
        inference_T=args.inference_T,
        n_recon_samples=args.n_recon_samples,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss_type,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir=args.output_dir,
    )

    return config


def main(config: Config | None = None) -> dict:
    """メインパイプライン.

    Args:
        config: Config オブジェクト (None の場合は argparse から生成).

    Returns:
        評価結果の dict.
    """
    if config is None:
        config = parse_args()

    # 再現性
    torch.manual_seed(config.seed)

    logger.info("Config: %s", config)
    logger.info("Device: %s", config.device)

    # 1. データ生成
    logger.info("=== Phase 1: Data Generation ===")
    data_split = generate_data(config)

    # 2. モデル構築
    logger.info("=== Phase 2: Model Construction ===")
    unet = ConditionalUNet1d(config)
    ddpm = DDPM(unet, config)

    param_count = sum(p.numel() for p in ddpm.parameters())
    logger.info("Model parameters: %d (%.2f M)", param_count, param_count / 1e6)

    # 3. 学習
    logger.info("=== Phase 3: Training ===")
    train(ddpm, data_split.train_loader, data_split.val_loader, config)

    # 4. 評価
    logger.info("=== Phase 4: Evaluation ===")
    results = evaluate(
        ddpm,
        data_split.test_data,
        data_split.test_labels,
        data_split.test_conditions,
        config,
    )

    logger.info("=== Done ===")
    logger.info("AUROC: %.4f", results["auroc"])

    return results


if __name__ == "__main__":
    main()
