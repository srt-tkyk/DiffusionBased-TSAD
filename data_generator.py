"""合成センサデータの生成モジュール.

正常信号 (正弦波合成 + ガウスノイズ) と 4 種類の異常パターンを生成し、
学習 / 検証 / テスト に分割して返す。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config

logger = logging.getLogger(__name__)

AnomalyType = Literal["spike", "drift", "amplitude_shift", "frequency_shift"]
ANOMALY_TYPES: list[AnomalyType] = ["spike", "drift", "amplitude_shift", "frequency_shift"]


@dataclass
class DataSplit:
    """学習 / 検証 / テスト 分割されたデータを保持する.

    Attributes:
        train_loader: 学習用 DataLoader (正常データのみ).
        val_loader: 検証用 DataLoader (正常データのみ).
        test_data: テスト用入力テンソル [N_test, C, T].
        test_labels: テスト用ラベル (0=正常, 1=異常) [N_test].
        test_conditions: テスト用条件ラベル [N_test].
        mean: チャンネルごとの平均 [C].
        std: チャンネルごとの標準偏差 [C].
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_data: torch.Tensor
    test_labels: torch.Tensor
    test_conditions: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


def _generate_normal_signal(
    n_samples: int,
    n_channels: int,
    seq_len: int,
    condition: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """正常信号を生成する.

    Args:
        n_samples: サンプル数.
        n_channels: チャンネル数.
        seq_len: 時系列長.
        condition: 稼働モード (0=通常, 1=高負荷).
        rng: NumPy 乱数生成器.

    Returns:
        正常信号の ndarray [n_samples, n_channels, seq_len].
    """
    t = np.linspace(0, 2 * np.pi, seq_len, endpoint=False)  # [T]
    signals = np.zeros((n_samples, n_channels, seq_len), dtype=np.float32)

    for i in range(n_samples):
        for c in range(n_channels):
            # チャンネルごとに異なる周波数を 2〜3 本合成
            n_harmonics = rng.integers(2, 4)
            for _ in range(n_harmonics):
                freq = rng.uniform(0.5, 4.0)
                amp = rng.uniform(0.8, 1.2)
                phase = rng.uniform(0, 2 * np.pi)
                signals[i, c] += amp * np.sin(freq * t + phase)

            # ガウスノイズ
            noise_std = 0.05
            if condition == 1:
                noise_std *= 1.5
            signals[i, c] += rng.normal(0, noise_std, seq_len)

        # 高負荷モードは全体振幅を 1.3 倍
        if condition == 1:
            signals[i] *= 1.3

    return signals


def _inject_anomaly(
    signal: np.ndarray,
    anomaly_type: AnomalyType,
    seq_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """1 サンプルに異常を注入する.

    Args:
        signal: 入力信号 [C, T] (in-place で変更).
        anomaly_type: 異常タイプ.
        seq_len: 時系列長.
        rng: NumPy 乱数生成器.

    Returns:
        異常が注入された信号 [C, T].
    """
    n_channels = signal.shape[0]
    target_ch = rng.integers(0, n_channels)

    # 異常は T/4 〜 3T/4 の間に発生
    quarter = seq_len // 4
    start = rng.integers(quarter, 3 * quarter)

    if anomaly_type == "spike":
        width = rng.integers(3, 13)
        end = min(start + width, seq_len)
        amp_factor = rng.uniform(2.5, 5.0)
        signal[target_ch, start:end] += amp_factor * rng.choice([-1, 1])

    elif anomaly_type == "drift":
        drift_mag = rng.uniform(1.5, 3.5)
        drift = np.linspace(0, drift_mag, seq_len - start, dtype=np.float32)
        signal[target_ch, start:] += drift

    elif anomaly_type == "amplitude_shift":
        scale = rng.uniform(2.0, 3.5)
        signal[target_ch, start:] *= scale

    elif anomaly_type == "frequency_shift":
        t_arr = np.arange(seq_len - start, dtype=np.float32) / seq_len
        high_freq = np.sin(2 * np.pi * 3.5 * t_arr * seq_len / (2 * np.pi))
        signal[target_ch, start:] += high_freq.astype(np.float32)

    return signal


def generate_data(config: Config) -> DataSplit:
    """Config に基づいて合成データを生成し、分割して返す.

    Args:
        config: プロジェクト設定.

    Returns:
        DataSplit オブジェクト.
    """
    rng = np.random.default_rng(config.seed)

    # --- 学習 + 検証データ (正常のみ) ---
    n_total_normal = config.n_train
    conditions_normal = rng.integers(0, config.n_conditions, size=n_total_normal)

    all_normal: list[np.ndarray] = []
    for cond in range(config.n_conditions):
        mask = conditions_normal == cond
        n_cond = int(mask.sum())
        if n_cond > 0:
            all_normal.append(
                _generate_normal_signal(n_cond, config.n_channels, config.seq_len, cond, rng)
            )

    normal_data = np.concatenate(all_normal, axis=0)  # [N_train, C, T]

    # Z-score 正規化 (チャンネルごと)
    mean = normal_data.mean(axis=(0, 2), keepdims=True)  # [1, C, 1]
    std = normal_data.std(axis=(0, 2), keepdims=True) + 1e-8  # [1, C, 1]
    normal_data = (normal_data - mean) / std

    # シャッフルして学習 / 検証に分割 (70% / 15% → 学習82% / 検証18% of normal)
    n_val = max(1, int(n_total_normal * 0.15 / 0.85))
    n_train = n_total_normal - n_val

    indices = rng.permutation(n_total_normal)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_data = torch.tensor(normal_data[train_idx], dtype=torch.float32)
    train_conds = torch.tensor(conditions_normal[train_idx], dtype=torch.long)
    val_data = torch.tensor(normal_data[val_idx], dtype=torch.float32)
    val_conds = torch.tensor(conditions_normal[val_idx], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_data, train_conds),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_conds),
        batch_size=config.batch_size,
        shuffle=False,
    )

    # --- テストデータ (正常 + 異常) ---
    n_anomaly = int(config.n_test * config.anomaly_ratio)
    n_normal_test = config.n_test - n_anomaly

    test_conditions = rng.integers(0, config.n_conditions, size=config.n_test)

    # 正常テストデータ
    test_normal_list: list[np.ndarray] = []
    for cond in range(config.n_conditions):
        mask = test_conditions[:n_normal_test] == cond
        n_cond = int(mask.sum())
        if n_cond > 0:
            test_normal_list.append(
                _generate_normal_signal(n_cond, config.n_channels, config.seq_len, cond, rng)
            )
    test_normal = np.concatenate(test_normal_list, axis=0) if test_normal_list else np.empty(
        (0, config.n_channels, config.seq_len), dtype=np.float32
    )

    # 異常テストデータ (正常信号に異常を注入)
    test_anomaly_list: list[np.ndarray] = []
    for cond in range(config.n_conditions):
        mask = test_conditions[n_normal_test:] == cond
        n_cond = int(mask.sum())
        if n_cond > 0:
            anomaly_signals = _generate_normal_signal(
                n_cond, config.n_channels, config.seq_len, cond, rng
            )
            for j in range(n_cond):
                atype = ANOMALY_TYPES[rng.integers(0, len(ANOMALY_TYPES))]
                _inject_anomaly(anomaly_signals[j], atype, config.seq_len, rng)
            test_anomaly_list.append(anomaly_signals)
    test_anomaly = np.concatenate(test_anomaly_list, axis=0) if test_anomaly_list else np.empty(
        (0, config.n_channels, config.seq_len), dtype=np.float32
    )

    # 結合・正規化
    test_all = np.concatenate([test_normal, test_anomaly], axis=0)  # [N_test, C, T]
    test_all = (test_all - mean) / std
    labels = np.concatenate(
        [np.zeros(n_normal_test, dtype=np.int64), np.ones(n_anomaly, dtype=np.int64)]
    )

    # mean / std を保存用に squeeze
    mean_t = torch.tensor(mean.squeeze(), dtype=torch.float32)  # [C]
    std_t = torch.tensor(std.squeeze(), dtype=torch.float32)  # [C]

    logger.info(
        "Data generated: train=%d, val=%d, test=%d (normal=%d, anomaly=%d)",
        len(train_data),
        len(val_data),
        len(test_all),
        n_normal_test,
        n_anomaly,
    )

    return DataSplit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_data=torch.tensor(test_all, dtype=torch.float32),
        test_labels=torch.tensor(labels, dtype=torch.long),
        test_conditions=torch.tensor(test_conditions, dtype=torch.long),
        mean=mean_t,
        std=std_t,
    )
