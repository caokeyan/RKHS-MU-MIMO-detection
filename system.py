"""
128×40 MU-MIMO（方案 A）：Y = H_eff X + N，只检测 X_1。

- H_eff = H @ W，形状 (M, K) = (128, 40)，仿真中不分离 H、W。
- MLD 使用真 H_eff（大 K 下为 MC 边际近似）；MMSE+LS 用导频直接估 H_eff。
"""
from __future__ import annotations

import numpy as np

M = 128  # BS 接收天线数
K = 40   # 并发流数 / X 维数
MOD_ORDER = 16
BITS_PER_SYMBOL = 4
Es = 1.0

CONSTELLATION = np.array(
    [
        -3 - 3j,
        -3 - 1j,
        -3 + 3j,
        -3 + 1j,
        -1 - 3j,
        -1 - 1j,
        -1 + 3j,
        -1 + 1j,
        3 - 3j,
        3 - 1j,
        3 + 3j,
        3 + 1j,
        1 - 3j,
        1 - 1j,
        1 + 3j,
        1 + 1j,
    ],
    dtype=np.complex128,
) / np.sqrt(10)


def n0_from_snr_db(snr_db: float) -> float:
    return Es / (10 ** (snr_db / 10.0))


def generate_heff(rng: np.random.Generator) -> np.ndarray:
    """随机 Rayleigh 等效信道 H_eff: (M, K)。"""
    return (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) / np.sqrt(2)


def generate_channel(rng: np.random.Generator) -> np.ndarray:
    """别名：返回 H_eff (M, K)。"""
    return generate_heff(rng)


def generate_samples(
    n: int,
    H_eff: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Y = H_eff X + N（按样本行向量 y = x @ H_eff.T）。

    返回 y (n,M), x_idx (n,K), x1_idx (n,)。
    """
    x_idx = rng.integers(0, MOD_ORDER, size=(n, K))
    return generate_samples_from_indices(x_idx, H_eff, snr_db, rng)


def generate_samples_from_indices(
    x_idx: np.ndarray,
    H_eff: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """固定符号索引，仅按 SNR 改变噪声（扫 SNR 时 H、X 不变）。"""
    n0 = n0_from_snr_db(snr_db)
    std = np.sqrt(n0 / 2)
    x = CONSTELLATION[x_idx]
    noise = std * (rng.standard_normal((x_idx.shape[0], M)) + 1j * rng.standard_normal((x_idx.shape[0], M)))
    y = (x @ H_eff.T) + noise
    return y, x_idx, x_idx[:, 0]


def generate_mixed_snr(
    n: int,
    H_eff: np.ndarray,
    snr_min: float,
    snr_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """每样本独立 SNR，返回 snr_db (n,)。"""
    x_idx = rng.integers(0, MOD_ORDER, size=(n, K))
    x = CONSTELLATION[x_idx]
    y_clean = x @ H_eff.T
    snr_db = rng.uniform(snr_min, snr_max, size=n)
    n0 = Es / (10 ** (snr_db / 10.0))
    std = np.sqrt(n0 / 2)[:, None]
    noise = std * (
        rng.standard_normal((n, M)) + 1j * rng.standard_normal((n, M))
    )
    y = y_clean + noise
    return y, x_idx, x_idx[:, 0], snr_db


def y_to_features(y: np.ndarray) -> np.ndarray:
    """复观测 -> 实特征 (n, 2M) = [Re y, Im y]。"""
    y = np.asarray(y)
    return np.concatenate([y.real, y.imag], axis=-1).astype(np.float64)


def idx_to_symbol(idx: int | np.ndarray) -> np.ndarray:
    return CONSTELLATION[np.asarray(idx)]


def symbol_to_idx(s: complex, tol: float = 1e-6) -> int:
    d = np.abs(CONSTELLATION - s)
    return int(np.argmin(d))


def hard_slice_symbol(x: complex) -> int:
    return symbol_to_idx(x)


_PAM_TO_LEVEL = {-3: 0, -1: 1, 1: 2, 3: 3}
_GRAY4_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int32)
_BINARY4_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)


def idx_to_bits(
    idx: int | np.ndarray,
    mapping: str = "gray",
) -> np.ndarray:
    """符号索引 -> 4 bit，形状 (n, 4)。"""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    s = CONSTELLATION[idx]
    scale = np.sqrt(10.0)
    re = np.round(np.real(s) * scale).astype(int)
    im = np.round(np.imag(s) * scale).astype(int)
    re_lv = np.array([_PAM_TO_LEVEL[int(v)] for v in re])
    im_lv = np.array([_PAM_TO_LEVEL[int(v)] for v in im])
    g = _GRAY4_BITS if mapping == "gray" else _BINARY4_BITS
    bits = np.empty((len(idx), BITS_PER_SYMBOL), dtype=np.int32)
    bits[:, :2] = g[re_lv]
    bits[:, 2:] = g[im_lv]
    return bits if idx.ndim else bits[0]


def bit_ber(
    true_idx: np.ndarray,
    est_idx: np.ndarray,
    mapping: str = "gray",
) -> float:
    """bit 误码率。"""
    b_true = idx_to_bits(true_idx, mapping)
    b_est = idx_to_bits(est_idx, mapping)
    if b_true.ndim == 1:
        return float(np.mean(b_true != b_est))
    return float(np.mean(b_true != b_est))
