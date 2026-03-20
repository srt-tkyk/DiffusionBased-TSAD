"""学習ループ.

DDPM モデルの学習・検証・最良モデル保存を行う。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from ddpm import DDPM

logger = logging.getLogger(__name__)


def train(
    ddpm: DDPM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
) -> dict[str, list[float]]:
    """DDPM を学習する.

    Args:
        ddpm: DDPM モジュール.
        train_loader: 学習用 DataLoader.
        val_loader: 検証用 DataLoader.
        config: プロジェクト設定.

    Returns:
        学習履歴 dict: {"train_loss": [...], "val_loss": [...]}.
    """
    device = torch.device(config.device)
    ddpm = ddpm.to(device)

    optimizer = Adam(ddpm.parameters(), lr=config.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.1)

    os.makedirs(config.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(config.n_epochs):
        # --- 学習 ---
        ddpm.train()
        train_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.n_epochs}", leave=False)
        for batch_data, batch_cond in pbar:
            batch_data = batch_data.to(device)  # [B, C, T]
            batch_cond = batch_cond.to(device)  # [B]

            optimizer.zero_grad()
            loss = ddpm.p_loss(batch_data, batch_cond, loss_type=config.loss_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), config.grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # --- 検証 ---
        ddpm.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_data, batch_cond in val_loader:
                batch_data = batch_data.to(device)
                batch_cond = batch_cond.to(device)
                loss = ddpm.p_loss(batch_data, batch_cond, loss_type=config.loss_type)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        logger.info(
            "Epoch %d/%d - train_loss: %.4f, val_loss: %.4f, lr: %.2e",
            epoch + 1,
            config.n_epochs,
            avg_train_loss,
            avg_val_loss,
            scheduler.get_last_lr()[0],
        )

        # 最良モデル保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = Path(config.output_dir) / "model_best.pt"
            torch.save(ddpm.state_dict(), save_path)
            logger.info("Best model saved (val_loss=%.4f)", best_val_loss)

    # 学習曲線プロット
    _plot_training_curve(history, config.output_dir)

    return history


def _plot_training_curve(history: dict[str, list[float]], output_dir: str) -> None:
    """学習曲線を PNG に保存する.

    Args:
        history: 学習履歴.
        output_dir: 出力先ディレクトリ.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Training curve saved to %s/training_curve.png", output_dir)
