"""
128×40 MU-MIMO（方案 A）：Y = H_eff X + N，只检测 X_1。

支持 square QAM：默认 16-QAM，可 set_modulation(64) 切换到 64-QAM。
"""
from __future__ import annotations

import numpy as np

M = 128  # BS 接收天线数
K = 40   # 并发流数 / X 维数
Es = 1.0

# 下面由 set_modulation 填充
MOD_ORDER = 16
BITS_PER_SYMBOL = 4
CONSTELLATION = np.zeros(16, dtype=np.complex128)
_I_LEVEL = np.zeros(16, dtype=np.int32)
_Q_LEVEL = np.zeros(16, dtype=np.int32)
_PAM_BITS = 2


def make_square_qam(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单位能量方形 M-QAM 星座 + I/Q 电平索引。"""
    L = int(round(np.sqrt(order)))
    if L * L != order:
        raise ValueError("仅支持完全平方阶数的方形 QAM，如 16/64")
    vals = (np.arange(L, dtype=np.float64) * 2.0) - (L - 1)
    syms: list[complex] = []
    i_lv: list[int] = []
    q_lv: list[int] = []
    for qi, qv in enumerate(vals):
        for ii, iv in enumerate(vals):
            syms.append(complex(iv, qv))
            i_lv.append(ii)
            q_lv.append(qi)
    pts = np.asarray(syms, dtype=np.complex128)
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts, np.asarray(i_lv, dtype=np.int32), np.asarray(q_lv, dtype=np.int32)


def set_modulation(order: int = 16) -> None:
    """切换全局调制（并同步已 import 的依赖模块中的同名常量）。"""
    global MOD_ORDER, BITS_PER_SYMBOL, CONSTELLATION, _I_LEVEL, _Q_LEVEL, _PAM_BITS
    order = int(order)
    pts, i_lv, q_lv = make_square_qam(order)
    MOD_ORDER = order
    BITS_PER_SYMBOL = int(np.log2(order))
    CONSTELLATION = pts
    _I_LEVEL, _Q_LEVEL = i_lv, q_lv
    _PAM_BITS = BITS_PER_SYMBOL // 2

    # 同步其它模块里 `from system import MOD_ORDER` 得到的绑定
    import sys

    for name in (
        "mld",
        "mmse",
        "kernel_rkhs",
        "rkhs_mld_approx",
        "rkhs_cond_detector",
        "rkhs_nn_detector",
        "cnn_detector",
        "test_oracle_rkhs",
        "run_experiment",
        "objective",
        "run_extended_exp",
        "midterm_final",
    ):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if hasattr(mod, "MOD_ORDER"):
            setattr(mod, "MOD_ORDER", MOD_ORDER)
        if hasattr(mod, "CONSTELLATION"):
            setattr(mod, "CONSTELLATION", CONSTELLATION)
        if hasattr(mod, "BITS_PER_SYMBOL"):
            setattr(mod, "BITS_PER_SYMBOL", BITS_PER_SYMBOL)


# 默认 16-QAM
set_modulation(16)


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
    *,
    nonlin_mode: str = "none",
    nonlin_beta: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_idx = rng.integers(0, MOD_ORDER, size=(n, K))
    return generate_samples_from_indices(
        x_idx,
        H_eff,
        snr_db,
        rng,
        nonlin_mode=nonlin_mode,
        nonlin_beta=nonlin_beta,
    )


def apply_rx_nonlinearity(
    y: np.ndarray,
    *,
    mode: str = "none",
    beta: float = 0.35,
) -> np.ndarray:
    """
    接收端无记忆非线性（兼容性场景）。

    - soft_clip / kerr / mzm：光/射频启发
    - hard_clip：硬幅度限幅
    - phase_noise：乘性相位抖动
    - iq_imbalance：I/Q 增益与正交误差
    """
    y = np.asarray(y, dtype=np.complex128)
    if mode in ("none", "", "linear"):
        return y
    if mode == "soft_clip":
        b = float(beta) if float(beta) > 0 else 0.35
        mag2 = np.abs(y) ** 2
        return (y / np.sqrt(1.0 + b * mag2)).astype(np.complex128)
    if mode == "kerr":
        b = float(beta) if float(beta) > 0 else 0.05
        return (y + b * (np.abs(y) ** 2) * y).astype(np.complex128)
    if mode == "mzm":
        b = float(beta) if float(beta) > 0 else 0.9
        a = np.abs(y)
        ph = np.exp(1j * np.angle(y))
        rms = float(np.sqrt(np.mean(a**2))) + 1e-12
        u = np.clip(b * a / rms, 0.0, 1.0)
        return (rms * np.abs(np.sin(0.5 * np.pi * u)) * ph).astype(np.complex128)
    if mode == "hard_clip":
        # β 为相对 rms 的限幅门限
        b = float(beta) if float(beta) > 0 else 1.5
        a = np.abs(y)
        rms = float(np.sqrt(np.mean(a**2))) + 1e-12
        thr = b * rms
        scale = np.minimum(1.0, thr / (a + 1e-12))
        return (y * scale).astype(np.complex128)
    if mode == "phase_noise":
        # β 为相位标准差（弧度）
        b = float(beta) if float(beta) > 0 else 0.15
        rng = np.random.default_rng(0)
        th = b * rng.standard_normal(y.shape)
        return (y * np.exp(1j * th)).astype(np.complex128)
    if mode == "iq_imbalance":
        # β∈(0,1) 控制幅度失衡；固定小正交误差
        b = float(beta) if float(beta) > 0 else 0.1
        g = 1.0 + b
        eps = 0.05
        i = y.real
        q = y.imag
        return ((g * i) + 1j * (q + eps * i)).astype(np.complex128)
    raise ValueError(f"未知非线性模式: {mode}")


def generate_samples_from_indices(
    x_idx: np.ndarray,
    H_eff: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    nonlin_mode: str = "none",
    nonlin_beta: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n0 = n0_from_snr_db(snr_db)
    std = np.sqrt(n0 / 2)
    x = CONSTELLATION[x_idx]
    noise = std * (
        rng.standard_normal((x_idx.shape[0], M))
        + 1j * rng.standard_normal((x_idx.shape[0], M))
    )
    y = (x @ H_eff.T) + noise
    y = apply_rx_nonlinearity(y, mode=nonlin_mode, beta=nonlin_beta)
    return y, x_idx, x_idx[:, 0]


def generate_mixed_snr(
    n: int,
    H_eff: np.ndarray,
    snr_min: float,
    snr_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    y = np.asarray(y)
    return np.concatenate([y.real, y.imag], axis=-1).astype(np.float64)


def idx_to_symbol(idx: int | np.ndarray) -> np.ndarray:
    return CONSTELLATION[np.asarray(idx)]


def symbol_to_idx(s: complex, tol: float = 1e-6) -> int:
    d = np.abs(CONSTELLATION - s)
    return int(np.argmin(d))


def hard_slice_symbol(x: complex) -> int:
    return symbol_to_idx(x)


def _level_to_bits(level: np.ndarray, nbits: int, *, gray: bool) -> np.ndarray:
    lv = np.asarray(level, dtype=np.int64)
    if gray:
        g = lv ^ (lv >> 1)
    else:
        g = lv
    out = np.empty((len(lv), nbits), dtype=np.int32)
    for b in range(nbits):
        out[:, b] = (g >> (nbits - 1 - b)) & 1
    return out


def idx_to_bits(
    idx: int | np.ndarray,
    mapping: str = "gray",
) -> np.ndarray:
    """符号索引 -> bit，形状 (n, BITS_PER_SYMBOL)。"""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    gray = mapping != "binary"
    bi = _level_to_bits(_I_LEVEL[idx], _PAM_BITS, gray=gray)
    bq = _level_to_bits(_Q_LEVEL[idx], _PAM_BITS, gray=gray)
    bits = np.concatenate([bi, bq], axis=1)
    return bits if np.ndim(idx) else bits[0]


def bit_ber(
    true_idx: np.ndarray,
    est_idx: np.ndarray,
    mapping: str = "gray",
) -> float:
    b_true = idx_to_bits(true_idx, mapping)
    b_est = idx_to_bits(est_idx, mapping)
    if b_true.ndim == 1:
        return float(np.mean(b_true != b_est))
    return float(np.mean(b_true != b_est))
