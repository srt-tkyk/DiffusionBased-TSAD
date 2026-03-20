"""異常検知評価・可視化モジュール.

再構成誤差に基づく異常スコア算出、AUROC / F1 等の評価指標、
t* 感度分析、各種可視化プロットを提供する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from config import Config
from ddpm import DDPM

logger = logging.getLogger(__name__)


def compute_anomaly_scores(
    ddpm: DDPM,
    data: torch.Tensor,
    conditions: torch.Tensor,
    config: Config,
    inference_T: int | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """再構成誤差ベースの異常スコアを算出する.

    score(x_0) = mean(||x_0 - x̂_0||²) over channels and timesteps.

    Args:
        ddpm: 学習済み DDPM.
        data: テスト入力 [N, C, T].
        conditions: 条件ラベル [N].
        config: プロジェクト設定.
        inference_T: 部分ノイズ化ステップ数 (None で config.inference_T).
        batch_size: 推論バッチサイズ.

    Returns:
        異常スコア [N].
    """
    device = torch.device(config.device)
    ddpm = ddpm.to(device)
    ddpm.eval()

    scores_list: list[np.ndarray] = []

    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = data[start:end].to(device)  # [B, C, T]
        cond = conditions[start:end].to(device)  # [B]

        recon = ddpm.reconstruct(batch, cond, inference_T=inference_T)  # [B, C, T]
        mse = ((batch - recon) ** 2).mean(dim=(1, 2))  # [B]
        scores_list.append(mse.cpu().numpy())

    return np.concatenate(scores_list)


def evaluate(
    ddpm: DDPM,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    test_conditions: torch.Tensor,
    config: Config,
) -> dict:
    """テストデータで評価を行い、結果を保存する.

    Args:
        ddpm: 学習済み DDPM.
        test_data: テスト入力 [N, C, T].
        test_labels: ラベル (0=正常, 1=異常) [N].
        test_conditions: 条件ラベル [N].
        config: プロジェクト設定.

    Returns:
        評価結果の dict.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_np = test_labels.numpy()

    # --- 異常スコア算出 ---
    logger.info("Computing anomaly scores (inference_T=%d)...", config.inference_T)
    scores = compute_anomaly_scores(ddpm, test_data, test_conditions, config)

    # --- AUROC ---
    auroc = roc_auc_score(labels_np, scores)
    logger.info("AUROC: %.4f", auroc)

    # --- 最適閾値 (Youden's J) ---
    fpr, tpr, thresholds = roc_curve(labels_np, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])

    # --- Precision / Recall / F1 ---
    preds = (scores >= best_threshold).astype(int)
    precision = float(precision_score(labels_np, preds, zero_division=0))
    recall = float(recall_score(labels_np, preds, zero_division=0))
    f1 = float(f1_score(labels_np, preds, zero_division=0))

    # --- MSE (正常 / 異常) ---
    normal_mask = labels_np == 0
    anomaly_mask = labels_np == 1
    mse_normal = float(scores[normal_mask].mean()) if normal_mask.any() else 0.0
    mse_anomaly = float(scores[anomaly_mask].mean()) if anomaly_mask.any() else 0.0
    mse_ratio = mse_anomaly / max(mse_normal, 1e-8)

    results = {
        "auroc": auroc,
        "best_threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse_normal": mse_normal,
        "mse_anomaly": mse_anomaly,
        "mse_ratio": mse_ratio,
    }

    logger.info(
        "Results: AUROC=%.4f, F1=%.4f, MSE_normal=%.4f, MSE_anomaly=%.4f, MSE_ratio=%.2f",
        auroc,
        f1,
        mse_normal,
        mse_anomaly,
        mse_ratio,
    )

    # --- 可視化 ---
    _plot_roc_curve(fpr, tpr, auroc, output_dir)
    _plot_anomaly_score_dist(scores, labels_np, best_threshold, output_dir)
    _plot_reconstruction(ddpm, test_data, test_labels, test_conditions, config, output_dir)

    # --- t* 感度分析 ---
    sensitivity_results = _sensitivity_analysis(
        ddpm, test_data, test_labels, test_conditions, config
    )
    results["sensitivity_t_star"] = sensitivity_results

    # 最良 t* を記録
    best_t_star_entry = max(sensitivity_results, key=lambda x: x["auroc"])
    results["best_inference_T"] = best_t_star_entry["inference_T"]
    results["best_auroc"] = best_t_star_entry["auroc"]

    # --- results.json 保存 ---
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s/results.json", output_dir)

    return results


def _plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auroc: float, output_dir: Path) -> None:
    """ROC 曲線をプロットする.

    Args:
        fpr: 偽陽性率.
        tpr: 真陽性率.
        auroc: AUROC 値.
        output_dir: 出力先.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_anomaly_score_dist(
    scores: np.ndarray, labels: np.ndarray, threshold: float, output_dir: Path
) -> None:
    """異常スコア分布をプロットする.

    Args:
        scores: 異常スコア.
        labels: ラベル.
        threshold: 最適閾値.
        output_dir: 出力先.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 0], bins=50, alpha=0.6, label="Normal", density=True)
    ax.hist(scores[labels == 1], bins=50, alpha=0.6, label="Anomaly", density=True)
    ax.axvline(threshold, color="r", linestyle="--", label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_score_dist.png", dpi=150)
    plt.close(fig)


def _plot_reconstruction(
    ddpm: DDPM,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    test_conditions: torch.Tensor,
    config: Config,
    output_dir: Path,
    n_samples: int = 3,
) -> None:
    """正常 / 異常サンプルの元波形と再構成を比較プロットする.

    Args:
        ddpm: 学習済み DDPM.
        test_data: テスト入力.
        test_labels: ラベル.
        test_conditions: 条件ラベル.
        config: 設定.
        output_dir: 出力先.
        n_samples: 各カテゴリのプロット数.
    """
    device = torch.device(config.device)

    normal_idx = torch.where(test_labels == 0)[0][:n_samples]
    anomaly_idx = torch.where(test_labels == 1)[0][:n_samples]
    all_idx = torch.cat([normal_idx, anomaly_idx])

    x = test_data[all_idx].to(device)
    cond = test_conditions[all_idx].to(device)

    ddpm.eval()
    recon = ddpm.reconstruct(x, cond).cpu()
    x = x.cpu()

    total = len(all_idx)
    n_ch = config.n_channels
    fig, axes = plt.subplots(total, n_ch, figsize=(4 * n_ch, 3 * total))
    if total == 1:
        axes = axes[None, :]
    if n_ch == 1:
        axes = axes[:, None]

    for i in range(total):
        label_str = "Normal" if i < len(normal_idx) else "Anomaly"
        for c in range(n_ch):
            ax = axes[i, c]
            ax.plot(x[i, c].numpy(), alpha=0.7, label="Original")
            ax.plot(recon[i, c].numpy(), alpha=0.7, label="Reconstructed")
            ax.set_title(f"{label_str} - Ch{c}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction.png", dpi=150)
    plt.close(fig)


def _sensitivity_analysis(
    ddpm: DDPM,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    test_conditions: torch.Tensor,
    config: Config,
) -> list[dict]:
    """t* 感度分析を行う.

    Args:
        ddpm: 学習済み DDPM.
        test_data: テスト入力.
        test_labels: ラベル.
        test_conditions: 条件ラベル.
        config: 設定.

    Returns:
        各 inference_T に対する AUROC のリスト.
    """
    t_star_values = [50, 100, 150, 200, 250, 300, 400, 500]
    # T を超えるものは除外
    t_star_values = [t for t in t_star_values if t <= config.T]
    labels_np = test_labels.numpy()

    results: list[dict] = []
    for t_star in tqdm(t_star_values, desc="Sensitivity analysis"):
        scores = compute_anomaly_scores(
            ddpm, test_data, test_conditions, config, inference_T=t_star
        )
        try:
            auroc = float(roc_auc_score(labels_np, scores))
        except ValueError:
            auroc = 0.0
        results.append({"inference_T": t_star, "auroc": auroc})
        logger.info("t*=%d, AUROC=%.4f", t_star, auroc)

    # プロット
    _plot_sensitivity(results, Path(config.output_dir))

    return results


def _plot_sensitivity(results: list[dict], output_dir: Path) -> None:
    """t* vs AUROC の折れ線グラフを描画する.

    Args:
        results: 感度分析結果.
        output_dir: 出力先.
    """
    t_stars = [r["inference_T"] for r in results]
    aurocs = [r["auroc"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_stars, aurocs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("inference_T (t*)")
    ax.set_ylabel("AUROC")
    ax.set_title("Sensitivity Analysis: t* vs AUROC")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_t_star.png", dpi=150)
    plt.close(fig)
