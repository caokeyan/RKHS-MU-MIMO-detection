"""优化目标：-log(f_{s1} / sum_b f_b) 及 RKHS 二次项。"""
from __future__ import annotations

import numpy as np


def log_margin_from_scores(
    scores: np.ndarray, labels: np.ndarray, eps: float = 1e-300
) -> np.ndarray:
    """逐样本 log(f_label / sum_b f_b)，与 J_data 的单项一致。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    s = scores[np.arange(len(labels)), labels]
    denom = np.sum(scores, axis=1)
    return np.log(np.maximum(s, eps) / np.maximum(denom, eps))


def softmax_ce_from_scores(
    scores: np.ndarray, labels: np.ndarray, eps: float = 1e-300
) -> float:
    """
    scores: (n, 16) 正数 f_a(y_j)。
    labels: (n,) 整数 s_1。
    返回平均 -log(f_{label} / sum_b f_b)。
    """
    s = scores[np.arange(len(labels)), labels]
    denom = np.sum(scores, axis=1)
    return float(-np.mean(np.log(np.maximum(s, eps) / np.maximum(denom, eps))))


def rkhs_penalty(alpha: np.ndarray, K: np.ndarray) -> float:
    """alpha: (16, N), K: (N,N) -> sum_a alpha_a^T K alpha_a。"""
    val = 0.0
    for a in range(alpha.shape[0]):
        val += alpha[a] @ K @ alpha[a]
    return float(val)


def total_objective(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: np.ndarray | None = None,
    K: np.ndarray | None = None,
    lam: float = 0.0,
) -> float:
    loss = softmax_ce_from_scores(scores, labels)
    if lam > 0 and alpha is not None and K is not None:
        loss += lam * rkhs_penalty(alpha, K)
    return loss
