"""
方案 A：导频直接估 H_eff，线性 MMSE 后硬判 X_1。

导频模型 Y_p = H_eff X_p + N_p，LS 得 Ĥ_eff；数据 Y = H_eff X + N。
"""
from __future__ import annotations

import numpy as np

from system import K, M, hard_slice_symbol

# K=40 时需 T>=K；取 2K 以保证 LS 残差自由度
DEFAULT_PILOT_LENGTH = max(80, 2 * K)


def ls_estimate_heff(Y_p: np.ndarray, X_p: np.ndarray) -> np.ndarray:
    """Y_p: (M, T), X_p: (K, T) -> H_hat (M, K)。"""
    return Y_p @ np.linalg.pinv(X_p)


def ls_estimate_channel(Y_p: np.ndarray, X_p: np.ndarray) -> np.ndarray:
    """别名：返回 Ĥ_eff。"""
    return ls_estimate_heff(Y_p, X_p)


def estimate_n0_from_residual(Y_p: np.ndarray, H_hat: np.ndarray, X_p: np.ndarray) -> float:
    """LS 残差估 N₀；需 T > K。"""
    resid = Y_p - H_hat @ X_p
    M_rx, T = Y_p.shape
    dof = M_rx * (T - K)
    if dof <= 0:
        return 0.0
    return float(np.sum(np.abs(resid) ** 2) / dof)


def mmse_equalize(
    y: np.ndarray,
    H_hat: np.ndarray,
    n0_hat: float,
) -> np.ndarray:
    """线性 MMSE 软估计 x_hat: (n, K)，含对角偏置校正。"""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[None, :]
    G = H_hat.conj().T @ H_hat + n0_hat * np.eye(K, dtype=np.complex128)
    W = np.linalg.solve(G, H_hat.conj().T)
    x_hat = (W @ y.T).T
    x_hat = x_hat / np.diag(W @ H_hat)
    return x_hat


def mmse_detect_x1(
    y: np.ndarray,
    H_hat: np.ndarray,
    n0_hat: float,
) -> int | np.ndarray:
    """MMSE 均衡 + 硬判，只输出 X_1 索引。"""
    y_arr = np.asarray(y)
    single = y_arr.ndim == 1
    x_hat = mmse_equalize(y_arr, H_hat, n0_hat)
    if single:
        return hard_slice_symbol(x_hat[0, 0])
    return np.array([hard_slice_symbol(x_hat[i, 0]) for i in range(x_hat.shape[0])])


def mmse_detect_s1(y: np.ndarray, H_hat: np.ndarray, n0_hat: float) -> int | np.ndarray:
    """别名：检测第一路符号 X_1。"""
    return mmse_detect_x1(y, H_hat, n0_hat)


def generate_pilots(n_pilot: int | None = None) -> np.ndarray:
    """导频 X_p: (K, T)，默认 T=max(80,2K)，DFT 行正交。"""
    T = n_pilot or DEFAULT_PILOT_LENGTH
    if T < K:
        raise ValueError(f"导频长度 T={T} 需 >= K={K}")
    t = np.arange(T, dtype=np.float64)
    k = np.arange(K, dtype=np.float64)
    X_p = np.exp(-2j * np.pi * np.outer(k, t) / T) / np.sqrt(T)
    return X_p.astype(np.complex128)


def batch_pilot_estimates(
    H_eff: np.ndarray,
    X_p: np.ndarray,
    n0: float,
    rng: np.random.Generator,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    每个样本独立导频 LS 估 (Ĥ_eff, N̂₀)，形状 (n,M,K) 与 (n,)。
    符合块衰落：各相干块单独发导频再传数据。
    """
    T = X_p.shape[1]
    std = np.sqrt(n0 / 2)
    noise_p = std * (
        rng.standard_normal((n, M, T)) + 1j * rng.standard_normal((n, M, T))
    )
    Y_p = (H_eff @ X_p)[None, :, :] + noise_p
    X_pinv = np.linalg.pinv(X_p)
    H_hat = Y_p @ X_pinv
    resid = Y_p - np.einsum("nmk,kt->nmt", H_hat, X_p, optimize=True)
    dof = M * (T - K)
    n0_hat = np.sum(np.abs(resid) ** 2, axis=(1, 2)) / dof
    return H_hat, n0_hat


def mmse_detect_x1_batch(
    y: np.ndarray,
    H_hat: np.ndarray,
    n0_hat: np.ndarray,
) -> np.ndarray:
    """y: (n,M)，H_hat: (n,M,K)，n0_hat: (n,) -> X_1 索引 (n,)。"""
    n = y.shape[0]
    est = np.empty(n, dtype=np.int64)
    for i in range(n):
        est[i] = mmse_detect_x1(y[i], H_hat[i], float(n0_hat[i]))
    return est


def estimate_heff_from_pilots(
    Y_p: np.ndarray,
    X_p: np.ndarray,
) -> tuple[np.ndarray, float]:
    """方案 A：由导频 LS 得 (Ĥ_eff, N̂₀)。"""
    H_hat = ls_estimate_heff(Y_p, X_p)
    n0_hat = estimate_n0_from_residual(Y_p, H_hat, X_p)
    return H_hat, n0_hat
