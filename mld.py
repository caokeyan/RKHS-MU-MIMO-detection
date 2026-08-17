"""边际 MLD：f_a^*(y) 与 argmax 检测（方案 A：真 H_eff，只判 X_1）。

- 小 K：对 x_{2:K} 精确枚举（16^{K-1} ≤ 上限时）
- 大 K：高斯干扰近似（默认）——将 x_{2:K} 视为独立单位能量，
  干扰+噪声协方差 R = N_0 I + H_I H_I^H，软分为
  log f_a ∝ -1/2 (y - h_1 s_a)^H R^{-1} (y - h_1 s_a)
  （N_0 I 时退化为原精确 MLD 的 -||·||^2/(2 N_0)）
- 可选 mode=\"mc\"：均匀蒙特卡洛边际（大 K 下通常远差于高斯近似）
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from system import CONSTELLATION, Es, K, MOD_ORDER

# 精确枚举上限
_MAX_EXACT_OTHER = 65536
_OTHER_CHUNK = 8192
_DEFAULT_MC_OTHER = 8192


def _n_other_exact() -> int | None:
    if K - 1 > 8:
        return None
    n = MOD_ORDER ** (K - 1)
    return int(n) if n <= _MAX_EXACT_OTHER else None


_N_OTHER_EXACT = _n_other_exact()
_USE_EXACT = _N_OTHER_EXACT is not None

_OTHER_INDICES: np.ndarray | None
if _USE_EXACT:
    _OTHER_INDICES = np.array(
        list(itertools.product(range(MOD_ORDER), repeat=K - 1)),
        dtype=np.int32,
    )
else:
    _OTHER_INDICES = None


@dataclass
class GaussianMldCache:
    """大 K 高斯干扰代理：缓存 h_1 与干扰 Gram H_I H_I^H。"""

    h1: np.ndarray          # (M,)
    interf_gram: np.ndarray  # (M, M) = H_I @ H_I^H * Es
    mode: str = "gaussian"


def _sample_other_indices(n_mc: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, MOD_ORDER, size=(n_mc, K - 1), dtype=np.int32)


def _hy_from_other(H_eff: np.ndarray, other_idx: np.ndarray) -> np.ndarray:
    n_other = other_idx.shape[0]
    M_rx = H_eff.shape[0]
    hy = np.empty((MOD_ORDER, n_other, M_rx), dtype=np.complex128)
    for a in range(MOD_ORDER):
        s_mat = np.empty((n_other, K), dtype=np.complex128)
        s_mat[:, 0] = CONSTELLATION[a]
        for j in range(K - 1):
            s_mat[:, 1 + j] = CONSTELLATION[other_idx[:, j]]
        hy[a] = s_mat @ H_eff.T
    return hy


def precompute_mld_hy(
    H_eff: np.ndarray,
    *,
    mode: str | None = None,
    n_mc: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray | GaussianMldCache:
    """
    预计算 MLD 缓存（同一 H_eff 扫多 SNR 可复用）。

    - exact：返回 hy (16, 16^{K-1}, M)
    - gaussian（大 K 默认）：返回 GaussianMldCache
    - mc：返回 hy (16, n_mc, M)
    """
    if mode is None:
        mode = "exact" if _USE_EXACT else "gaussian"

    if mode == "exact":
        if not _USE_EXACT or _OTHER_INDICES is None:
            raise ValueError(f"K={K} 无法精确枚举，请用 mode='gaussian'")
        return _hy_from_other(H_eff, _OTHER_INDICES)

    if mode == "gaussian":
        H_eff = np.asarray(H_eff)
        h1 = H_eff[:, 0].copy()
        H_I = H_eff[:, 1:]
        # E[|x|^2]=Es（星座已归一）；独立流 → Cov = Es * H_I H_I^H
        interf_gram = (Es * (H_I @ H_I.conj().T)).astype(np.complex128)
        # 数值稳定：轻微对角加载在评分时按 n0 加入
        return GaussianMldCache(h1=h1, interf_gram=interf_gram, mode="gaussian")

    if mode == "mc":
        n_hyp = int(n_mc) if n_mc is not None else _DEFAULT_MC_OTHER
        if rng is None:
            rng = np.random.default_rng(0)
        return _hy_from_other(H_eff, _sample_other_indices(n_hyp, rng))

    raise ValueError(f"未知 MLD mode={mode!r}")


def _gaussian_log_scores(
    y: np.ndarray,
    cache: GaussianMldCache,
    n0: float,
) -> np.ndarray:
    """
    log f_a(y) = -1/2 (y - h1 s_a)^H R^{-1} (y - h1 s_a)，
    R = n0 I + interf_gram。R 与 a 无关，softmax 时 logdet 可略。
    """
    y = np.asarray(y, dtype=np.complex128)
    n_batch, M_rx = y.shape
    R = cache.interf_gram + float(n0) * np.eye(M_rx, dtype=np.complex128)
    # Cholesky 白化：R = L L^H → metric = ||L^{-1}(y-μ)||^2
    L = np.linalg.cholesky(R)
    # white_y: (n, M)
    white_y = np.linalg.solve(L, y.T).T
    white_h = np.linalg.solve(L, cache.h1)

    log_scores = np.empty((n_batch, MOD_ORDER), dtype=np.float64)
    for a in range(MOD_ORDER):
        resid = white_y - white_h * CONSTELLATION[a]
        d2 = np.sum(np.abs(resid) ** 2, axis=1)
        log_scores[:, a] = -0.5 * d2
    return log_scores


def _marginal_log_scores_vectorized(
    y: np.ndarray,
    hy_cache: np.ndarray,
    inv2n0: float,
    *,
    y_batch: int = 128,
) -> np.ndarray:
    n_batch, _ = y.shape
    acc = np.full((n_batch, MOD_ORDER), -np.inf, dtype=np.float64)
    y_norm = np.sum(np.abs(y) ** 2, axis=1)
    n_other = hy_cache.shape[1]

    for i0 in range(0, n_batch, y_batch):
        i1 = min(i0 + y_batch, n_batch)
        yb = y[i0:i1]
        yn = y_norm[i0:i1][:, None, None]

        for start in range(0, n_other, _OTHER_CHUNK):
            hc = hy_cache[:, start : start + _OTHER_CHUNK, :]
            hn = np.sum(np.abs(hc) ** 2, axis=2)[None, :, :]
            cross = np.einsum("nm,acm->nac", yb, np.conj(hc), optimize=True)
            d2 = yn + hn - 2.0 * np.real(cross)
            logw = -d2 * inv2n0
            m = np.max(logw, axis=2)
            ls = m + np.log(np.sum(np.exp(logw - m[:, :, None]), axis=2))
            acc[i0:i1] = np.logaddexp(acc[i0:i1], ls)

    return acc


def _logsumexp_marginal_for_a_legacy(
    y: np.ndarray,
    H_eff: np.ndarray,
    a: int,
    inv2n0: float,
    *,
    other_idx: np.ndarray | None = None,
) -> np.ndarray:
    n_batch = y.shape[0]
    s1 = CONSTELLATION[a]
    acc = np.full(n_batch, -np.inf)

    if other_idx is None:
        if not _USE_EXACT or _OTHER_INDICES is None:
            raise ValueError("大 K 下请先调用 precompute_mld_hy(mode='gaussian')")
        other_idx = _OTHER_INDICES

    for start in range(0, len(other_idx), _OTHER_CHUNK):
        sub = other_idx[start : start + _OTHER_CHUNK]
        s_mat = np.empty((len(sub), K), dtype=np.complex128)
        s_mat[:, 0] = s1
        for j in range(K - 1):
            s_mat[:, 1 + j] = CONSTELLATION[sub[:, j]]
        Hy = s_mat @ H_eff.T
        diff = y[:, None, :] - Hy[None, :, :]
        d2 = np.sum(np.abs(diff) ** 2, axis=-1)
        logw = -d2 * inv2n0
        m = np.max(logw, axis=1)
        ls = m + np.log(np.sum(np.exp(logw - m[:, None]), axis=1))
        acc = np.logaddexp(acc, ls)

    return acc


def marginal_scores(
    y: np.ndarray,
    H_eff: np.ndarray,
    n0: float,
    *,
    log_domain: bool = True,
    hy_cache: np.ndarray | GaussianMldCache | None = None,
) -> np.ndarray:
    """
    f_a^*(y) 的对数/正数边际得分（只对 X_1=a）。
    hy_cache: precompute_mld_hy(H_eff) 的返回值。
    """
    y = np.asarray(y)
    single = y.ndim == 1
    if single:
        y = y[None, :]

    if hy_cache is None:
        hy_cache = precompute_mld_hy(H_eff)

    if isinstance(hy_cache, GaussianMldCache):
        log_scores = _gaussian_log_scores(y, hy_cache, n0)
    else:
        inv2n0 = 0.5 / n0
        log_scores = _marginal_log_scores_vectorized(y, hy_cache, inv2n0)

    if log_domain:
        scores = log_scores
    else:
        scores = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))

    return scores[0] if single else scores


def detect_from_scores(scores: np.ndarray) -> np.ndarray:
    return np.argmax(scores, axis=-1)


def ber(true: np.ndarray, est: np.ndarray) -> float:
    """X_1 符号索引误码率。"""
    return float(np.mean(true != est))


def marginal_mld_detect(
    y: np.ndarray,
    H_eff: np.ndarray,
    n0: float,
    *,
    hy_cache: np.ndarray | GaussianMldCache | None = None,
) -> np.ndarray:
    scores = marginal_scores(y, H_eff, n0, log_domain=True, hy_cache=hy_cache)
    return detect_from_scores(scores)


def marginal_mld_ber(
    y: np.ndarray,
    x1_true: np.ndarray,
    H_eff: np.ndarray,
    n0: float,
) -> float:
    return ber(x1_true, marginal_mld_detect(y, H_eff, n0))


def marginal_mld_ber_snr(
    y: np.ndarray,
    x1_true: np.ndarray,
    H_eff: np.ndarray,
    snr_db: float,
) -> float:
    from system import n0_from_snr_db

    return marginal_mld_ber(y, x1_true, H_eff, n0_from_snr_db(snr_db))


def joint_mld_detect(y: np.ndarray, H_eff: np.ndarray, n0: float) -> np.ndarray:
    """联合 MLD 的 X_1（仅小 K 参考）。"""
    if not _USE_EXACT or MOD_ORDER ** K > _MAX_EXACT_OTHER * MOD_ORDER:
        raise ValueError(f"joint MLD 不可用于 K={K}（搜索空间 16^{K}）")
    y = np.asarray(y)
    single = y.ndim == 1
    if single:
        y = y[None, :]
    best_idx = np.zeros(y.shape[0], dtype=np.int64)
    best_metric = np.full(y.shape[0], np.inf)

    for s_idx_tuple in itertools.product(range(MOD_ORDER), repeat=K):
        s = CONSTELLATION[list(s_idx_tuple)]
        resid = y - (H_eff @ s)
        metric = np.sum(np.abs(resid) ** 2, axis=1)
        better = metric < best_metric
        best_metric = np.where(better, metric, best_metric)
        best_idx = np.where(better, s_idx_tuple[0], best_idx)

    return best_idx[0] if single else best_idx
