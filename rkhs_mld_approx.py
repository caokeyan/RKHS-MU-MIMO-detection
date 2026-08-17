"""
用 RKHS / Adaptive-MKL 最小化贝叶斯后验损失，逼近后验 f^*(y)。

决策形式（无 PIC/MMSE 残差底座）：
    logits(y) = K_η(φ(y), C) α^T
    f̂ = softmax(logits),  â = argmax logits

贝叶斯后验损失（empirical）：
    J_hard(f) = -log( f_{X_1}(y) / ∑_b f_b(y) )
可选把 Ĥ 写进损失（可实现，不用真 H / 真 f*）：
    p̂_a(y) = softmax( z_rob,a(y; Ĥ) )     # plug-in 后验
    J_plugin = w · CE(p̂, f) + (1-w) · J_hard(f) + λ ‖f‖_H²

特征 φ：
  - blind： [Re y, Im y]
  - struct：白化充分统计 z_a(y;H,N₀)
  - struct_hat：z_rob(y;Ĥ)（稳健协方差）

训练目标 target：
  - fstar：真 f* 监督（可逼近性，特权信息）
  - hard：仅 J_hard
  - plugin：损失里用 Ĥ 的 plug-in 后验（可实现主线）
"""
from __future__ import annotations

import numpy as np

from kernel_rkhs import (
    ADAPTIVE_MKL_RATIOS,
    RICH_ADAPTIVE_MKL_RATIOS,
    _fit_adam,
    _fit_lbfgs_limited_local,
    _label_diagonal_init,
    build_base_kernels,
    build_kernel_matrix,
    combine_kernels,
    fit_adaptive_mkl_alpha,
    gamma_theory_rkhs,
    lam_theory_rkhs,
    solve_alpha_from_logits,
)
from mld import GaussianMldCache, _gaussian_log_scores, marginal_scores, precompute_mld_hy
from mmse import (
    DEFAULT_PILOT_LENGTH,
    estimate_n0_from_residual,
    generate_pilots,
    ls_estimate_heff,
)
from objective import softmax_ce_from_scores
from system import Es, K, M, MOD_ORDER, n0_from_snr_db, y_to_features


def estimate_heff_block(
    H_eff: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    pilot_mult: float = 1.0,
) -> tuple[np.ndarray, float, int]:
    """
    块衰落导频 LS → (Ĥ, N̂₀, T_pilot)。
    pilot_mult>1 加长导频以降低 σ_e²。
    """
    n0 = n0_from_snr_db(float(snr_db))
    T = max(DEFAULT_PILOT_LENGTH, int(np.ceil(pilot_mult * DEFAULT_PILOT_LENGTH)))
    T = max(T, 2 * K)
    X_p = generate_pilots(T)
    std = np.sqrt(n0 / 2)
    noise = std * (
        rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))
    )
    Y_p = H_eff @ X_p + noise
    H_hat = ls_estimate_heff(Y_p, X_p)
    n0_hat = estimate_n0_from_residual(Y_p, H_hat, X_p)
    if n0_hat <= 0:
        n0_hat = n0
    return H_hat, float(n0_hat), int(T)


def sigma_e2_pilot(n0_hat: float, T_pilot: int) -> float:
    """正交归一导频下，每列信道估计误差方差近似 n0·K/T。"""
    return float(n0_hat) * float(K) / max(float(T_pilot), 1.0)


def struct_z_features(
    y: np.ndarray,
    H_eff: np.ndarray,
    n0: float,
    hy_cache: np.ndarray | GaussianMldCache | None = None,
) -> np.ndarray:
    """
    MLD（高斯干扰）充分统计：z_a(y) = log f_a^{GA}(y; H, N₀)，形状 (n, 16)。
    """
    if hy_cache is None:
        hy_cache = precompute_mld_hy(H_eff)
    z = marginal_scores(y, H_eff, float(n0), log_domain=True, hy_cache=hy_cache)
    return (z - z.max(axis=1, keepdims=True)).astype(np.float64)


def robust_struct_z_features(
    y: np.ndarray,
    H_hat: np.ndarray,
    n0_hat: float,
    sigma_e2: float,
) -> np.ndarray:
    """
    CSI 不确定度对角加载：
      R = N₀ I + Es Ĥ_I Ĥ_I^H + (K-1) σ_e² I
    再算 GA 对数软分 z_a（16 维）。这是对 f^*(y;H) 在估信道下的稳健充分统计。
    """
    H_hat = np.asarray(H_hat)
    h1 = H_hat[:, 0].copy()
    H_I = H_hat[:, 1:]
    M_rx = H_hat.shape[0]
    interf = (Es * (H_I @ H_I.conj().T)).astype(np.complex128)
    load = float(sigma_e2) * float(K - 1)
    cache = GaussianMldCache(
        h1=h1,
        interf_gram=interf + load * np.eye(M_rx, dtype=np.complex128),
        mode="gaussian",
    )
    z = _gaussian_log_scores(np.asarray(y), cache, float(n0_hat))
    return (z - z.max(axis=1, keepdims=True)).astype(np.float64)


def compact_csi_features(
    H_hat: np.ndarray,
    n0_hat: float | np.ndarray,
    sigma_e2: float | np.ndarray,
) -> np.ndarray:
    """
    紧凑 CSI 描述（只用 Ĥ，不用真 H）：能量、干扰、流功率、N₀、σ_e²。
    H_hat: (M,K) 或 (n,M,K) → (d,) 或 (n,d)
    """
    H = np.asarray(H_hat)
    single = H.ndim == 2
    if single:
        H = H[None, ...]
    n = H.shape[0]
    h1 = H[:, :, 0]
    H_I = H[:, :, 1:]
    e1 = np.sum(np.abs(h1) ** 2, axis=1)
    eI = np.sum(np.abs(H_I) ** 2, axis=(1, 2))
    eH = e1 + eI
    col = np.sum(np.abs(H) ** 2, axis=1)
    col_n = col / (eH[:, None] + 1e-12)
    n0v = (
        np.full(n, float(n0_hat))
        if np.ndim(n0_hat) == 0
        else np.asarray(n0_hat, dtype=np.float64)
    )
    sev = (
        np.full(n, float(sigma_e2))
        if np.ndim(sigma_e2) == 0
        else np.asarray(sigma_e2, dtype=np.float64)
    )
    feats = np.concatenate(
        [
            np.log(e1 + 1e-12)[:, None],
            np.log(eI + 1e-12)[:, None],
            np.log(eH + 1e-12)[:, None],
            (e1 / (eH + 1e-12))[:, None],
            np.log(n0v + 1e-12)[:, None],
            np.log(sev + 1e-12)[:, None],
            col_n,
        ],
        axis=1,
    ).astype(np.float64)
    return feats[0] if single else feats


def robust_struct_z_features_batch(
    y: np.ndarray,
    H_hat: np.ndarray,
    n0_hat: np.ndarray,
    sigma_e2: np.ndarray | float,
) -> np.ndarray:
    """逐样本 Ĥ 的稳健 z，(n,16)。"""
    y = np.asarray(y)
    H_hat = np.asarray(H_hat)
    n = y.shape[0]
    n0_hat = np.asarray(n0_hat, dtype=np.float64)
    if np.ndim(sigma_e2) == 0:
        se2 = np.full(n, float(sigma_e2))
    else:
        se2 = np.asarray(sigma_e2, dtype=np.float64)
    out = np.empty((n, MOD_ORDER), dtype=np.float64)
    for i in range(n):
        out[i] = robust_struct_z_features(
            y[i : i + 1], H_hat[i], float(n0_hat[i]), float(se2[i])
        )[0]
    return out


def estimate_heff_per_sample(
    H_eff: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    n: int,
    *,
    pilot_mult: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    逐样本独立导频 LS（特征阶段不使用真 H）。
    返回 H_hat (n,M,K), n0_hat (n,), T, sigma_e2 (n,)
    注：仿真生成导频噪声时仍用真 H_eff（物理信道），与 MMSE+LS 设定一致。
    """
    from mmse import batch_pilot_estimates

    n0 = n0_from_snr_db(float(snr_db))
    T = max(DEFAULT_PILOT_LENGTH, int(np.ceil(pilot_mult * DEFAULT_PILOT_LENGTH)))
    T = max(T, 2 * K)
    X_p = generate_pilots(T)
    H_hat, n0_hat = batch_pilot_estimates(H_eff, X_p, n0, rng, n)
    n0_hat = np.maximum(np.asarray(n0_hat, dtype=np.float64), 1e-12)
    se2 = n0_hat * float(K) / float(T)
    s = float(snr_db)
    if s >= 10.0:
        se2 = se2 * 0.15
    elif s >= 8.0:
        se2 = se2 * 0.4
    return H_hat, n0_hat, int(T), se2


def blind_y_features(y: np.ndarray) -> np.ndarray:
    return y_to_features(np.atleast_2d(y)).astype(np.float64)


def _fit_soft_label_adaptive_mkl(
    base_kernels: list[np.ndarray],
    soft_labels: np.ndarray,
    hard_labels: np.ndarray,
    lam: float,
    *,
    soft_weight: float = 0.7,
    adam_epochs: int = 1200,
    lbfgs_maxiter: int = 800,
    val_frac: float = 0.15,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    贝叶斯型软+硬 Adaptive-MKL：
      L = w · CE_soft(p̂, softmax(Kα)) + (1-w) · J_hard + λ reg
    p̂ 可以是真 f*（特权）或 plug-in softmax(z_rob(·;Ĥ))（可实现）。
    部署只需 α,η。
    """
    import torch

    soft = np.asarray(soft_labels, dtype=np.float64)
    soft = soft / (soft.sum(axis=1, keepdims=True) + 1e-300)
    hard = np.asarray(hard_labels, dtype=np.int64)
    n = base_kernels[0].shape[0]
    n_k = len(base_kernels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(32, int(n * val_frac))
    va, tr = perm[:n_val], perm[n_val:]

    Ks_np: list[np.ndarray] = []
    scales = np.empty(n_k, dtype=np.float64)
    for m, Km in enumerate(base_kernels):
        trc = float(np.trace(Km).real)
        scales[m] = n / max(trc, 1e-12)
        Ks_np.append(Km * scales[m])
    Ks_t = [torch.from_numpy(Km.astype(np.float32)) for Km in Ks_np]
    soft_t = torch.from_numpy(soft.astype(np.float32))
    hard_t = torch.from_numpy(hard)

    # 用 f* 的 log 作对角暖启动量级
    from kernel_rkhs import _label_diagonal_init, solve_alpha_from_logits

    a0 = _label_diagonal_init(hard).astype(np.float32) * 0.05
    alpha_m = torch.tensor(
        np.stack([a0 for _ in range(n_k)], axis=0) / max(n_k, 1),
        dtype=torch.float32,
        requires_grad=True,
    )
    theta = torch.zeros(n_k, dtype=torch.float32)
    for m in range(n_k):
        theta[m] = 0.3 * (n_k - 1 - m) / max(n_k - 1, 1)
    theta = theta.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam(
        [{"params": [alpha_m], "lr": 0.05}, {"params": [theta], "lr": 0.03}]
    )
    best_state = None
    best_val = float("inf")
    stale = 0
    eps = 1e-6
    w = float(np.clip(soft_weight, 0.0, 1.0))

    def _logits_reg():
        eta = torch.softmax(theta, dim=0)
        logits = torch.zeros(n, MOD_ORDER, dtype=torch.float32)
        reg = torch.zeros((), dtype=torch.float32)
        for m in range(n_k):
            am = alpha_m[m]
            logits = logits + (Ks_t[m] @ am.T)
            reg = reg + (1.0 / (eta[m] + eps)) * torch.sum(am * (am @ Ks_t[m]))
        return logits, eta, reg

    for _ in range(int(adam_epochs)):
        opt.zero_grad()
        logits, eta, reg = _logits_reg()
        log_p = torch.log_softmax(logits, dim=1)
        # soft CE = -∑ f* log p
        ce_soft = -(soft_t[tr] * log_p[tr]).sum(dim=1).mean()
        ce_hard = torch.nn.functional.cross_entropy(logits[tr], hard_t[tr])
        ent = -torch.sum(eta * torch.log(eta + eps))
        loss = w * ce_soft + (1.0 - w) * ce_hard + float(lam) * reg - 0.02 * ent
        loss.backward()
        opt.step()

        with torch.no_grad():
            logits_v, _, _ = _logits_reg()
            # 验证用硬 CE（与部署 BER 更对齐）
            val = float(torch.nn.functional.cross_entropy(logits_v[va], hard_t[va]))
        if val < best_val - 1e-5:
            best_val = val
            best_state = (
                alpha_m.detach().cpu().numpy().copy(),
                theta.detach().cpu().numpy().copy(),
            )
            stale = 0
        else:
            stale += 1
            if stale >= 80:
                break

    if best_state is None:
        am_np = alpha_m.detach().cpu().numpy()
        th_np = theta.detach().cpu().numpy()
    else:
        am_np, th_np = best_state
    eta_np = np.exp(th_np - th_np.max())
    eta_np /= eta_np.sum()
    K_eta = combine_kernels(Ks_np, eta_np)
    resid_sum = np.zeros((n, MOD_ORDER), dtype=np.float64)
    for m in range(n_k):
        resid_sum += Ks_np[m] @ am_np[m].T
    try:
        alpha_eff = solve_alpha_from_logits(resid_sum, K_eta, float(lam))
    except np.linalg.LinAlgError:
        alpha_eff = am_np[int(np.argmax(eta_np))]

    # 固定 η，用 soft+hard 再精修单组 α
    if lbfgs_maxiter > 0:
        alpha_eff = _refine_soft_hard_alpha(
            alpha_eff, K_eta, soft, hard, float(lam),
            soft_weight=w, adam_epochs=min(400, adam_epochs),
            lbfgs_maxiter=int(lbfgs_maxiter),
        )

    stats = {
        "val_j_data": float(best_val),
        "kernel_scales": scales.copy(),
        "eta_entropy": float(-np.sum(eta_np * np.log(eta_np + 1e-12))),
        "mode": "soft_distill_mkl",
        "soft_weight": w,
    }
    if verbose:
        print(f"  soft-distill η_H={stats['eta_entropy']:.3f} valHardCE={best_val:.4f} w={w:.2f}")
    return alpha_eff, eta_np, K_eta, stats


def _refine_soft_hard_alpha(
    alpha: np.ndarray,
    K: np.ndarray,
    soft: np.ndarray,
    hard: np.ndarray,
    lam: float,
    *,
    soft_weight: float,
    adam_epochs: int,
    lbfgs_maxiter: int,
) -> np.ndarray:
    import torch
    from scipy.optimize import minimize

    n = K.shape[0]
    K_t = torch.from_numpy(K.astype(np.float32))
    soft_t = torch.from_numpy(soft.astype(np.float32))
    hard_t = torch.from_numpy(hard.astype(np.int64))
    a = torch.tensor(alpha.astype(np.float32), requires_grad=True)
    opt = torch.optim.Adam([a], lr=0.05)
    best, best_v = a.detach().cpu().numpy().copy(), float("inf")
    stale = 0
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(32, int(0.15 * n))
    va, tr = perm[:n_val], perm[n_val:]
    w = float(soft_weight)

    for _ in range(int(adam_epochs)):
        opt.zero_grad()
        logits = K_t @ a.T
        log_p = torch.log_softmax(logits, dim=1)
        ce_s = -(soft_t[tr] * log_p[tr]).sum(1).mean()
        ce_h = torch.nn.functional.cross_entropy(logits[tr], hard_t[tr])
        reg = float(lam) * torch.sum(a * (a @ K_t))
        (w * ce_s + (1 - w) * ce_h + reg).backward()
        opt.step()
        with torch.no_grad():
            v = float(torch.nn.functional.cross_entropy(K_t[va] @ a.T, hard_t[va]))
        if v < best_v - 1e-5:
            best_v = v
            best = a.detach().cpu().numpy().copy()
            stale = 0
        else:
            stale += 1
            if stale >= 50:
                break

    def fun(x):
        aa = x.reshape(MOD_ORDER, n)
        logits = K @ aa.T
        g = np.clip(logits, -40, 40)
        z = g.max(1, keepdims=True)
        f = np.exp(g - z)
        f /= f.sum(1, keepdims=True) + 1e-300
        ce_s = -np.mean(np.sum(soft * np.log(f + 1e-300), axis=1))
        ce_h = -np.mean(np.log(f[np.arange(n), hard] + 1e-300))
        reg = float(lam) * sum(aa[c] @ K @ aa[c] for c in range(MOD_ORDER))
        return float(w * ce_s + (1 - w) * ce_h + reg)

    def jac(x):
        aa = x.reshape(MOD_ORDER, n)
        logits = K @ aa.T
        g = np.clip(logits, -40, 40)
        z = g.max(1, keepdims=True)
        e = np.exp(g - z)
        f = e / (e.sum(1, keepdims=True) + 1e-300)
        # soft: ∂CE/∂logit = f - soft; hard: f - onehot
        dg_s = f - soft
        dg_h = f.copy()
        dg_h[np.arange(n), hard] -= 1.0
        dg = (w * dg_s + (1 - w) * dg_h) / n
        grad = dg.T @ K
        for c in range(MOD_ORDER):
            grad[c] += 2.0 * float(lam) * (K @ aa[c])
        return grad.ravel()

    res = minimize(
        fun, best.ravel(), method="L-BFGS-B", jac=jac,
        options={"maxiter": int(lbfgs_maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    return res.x.reshape(MOD_ORDER, n)


def _normalize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def _normalize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _tune_hard_expr(
    X: np.ndarray,
    ylab: np.ndarray,
    n0: float,
    lam_c: float,
    ratios: tuple[float, ...],
    *,
    adam_epochs: int,
    lbfgs_maxiter: int,
    gamma_scale0: float = 1.0,
    verbose: bool = False,
) -> tuple[float, float, float, float]:
    """
    验证集最小化 J_hard，选 (γ 乘子, λ·c 乘子)。
    返回 (gamma, lam, g_scale, c_scale)。
    """
    n = X.shape[0]
    g0 = gamma_theory_rkhs(n0, X)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(64, int(0.15 * n))
    va, tr = perm[:n_val], perm[n_val:]

    g_grid = (0.7, 1.0, 1.4)
    c_grid = (0.5, 1.0)
    # 快扫：缩短优化；最优再由外层全量重训
    ae = max(250, int(adam_epochs) // 3)
    lb = max(150, int(lbfgs_maxiter) // 3)

    best = (float("inf"), float(gamma_scale0), 1.0)
    for gs in g_grid:
        for cs in c_grid:
            gamma = float(g0) * float(gs)
            lam = lam_theory_rkhs(n0, n, c=float(lam_c) * float(cs))
            base_k = build_base_kernels(X, gamma, ms_ratios=ratios)
            alpha, eta, K, _ = fit_adaptive_mkl_alpha(
                base_k,
                ylab,
                float(lam),
                adam_epochs=ae,
                lbfgs_maxiter=lb,
                verbose=False,
            )
            logits = K @ alpha.T
            f = _softmax_rows(logits)
            # 只用 holdout 上的贝叶斯后验损失
            val_j = softmax_ce_from_scores(f[va], ylab[va])
            if val_j < best[0] - 1e-5:
                best = (float(val_j), float(gs), float(cs))
            if verbose:
                print(
                    f"    tune g×{gs:.2f} c×{cs:.2f}: valJ={val_j:.4f}",
                    flush=True,
                )
    _, gs_b, cs_b = best
    gamma = float(g0) * gs_b
    lam = lam_theory_rkhs(n0, n, c=float(lam_c) * cs_b)
    return gamma, lam, gs_b, cs_b


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    g = np.clip(logits, -40.0, 40.0)
    z = g.max(axis=1, keepdims=True)
    e = np.exp(g - z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-300)


def _kernel_context_for_nn(
    K: np.ndarray, labels: np.ndarray, *, gamma: float, n0: float, bw: float
) -> np.ndarray:
    """固定维上下文，供超网络预测全局 α。"""
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    eig = np.linalg.eigvalsh(K)
    eig_top = np.sort(eig)[-min(32, n) :][::-1]
    if len(eig_top) < 32:
        eig_top = np.pad(
            eig_top,
            (0, 32 - len(eig_top)),
            constant_values=eig_top[-1] if len(eig_top) else 0.0,
        )
    hist = np.bincount(labels.astype(np.int64), minlength=MOD_ORDER).astype(np.float64)
    hist = hist / max(hist.sum(), 1.0)
    row_sum = K.sum(axis=1)
    return np.concatenate(
        [
            [float(gamma), float(n0), float(bw), float(n), float(row_sum.mean()), float(row_sum.std())],
            eig_top,
            hist,
        ]
    ).astype(np.float32)


def _nn_warmstart_alpha(
    K: np.ndarray,
    labels: np.ndarray,
    *,
    gamma: float,
    n0: float,
    lam: float,
    epochs: int = 250,
    patience: int = 40,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """
    超网络 α = NN(ctx)，损失 = J_hard(softmax(Kα^T)) + λ ‖α‖_K^2。
    返回 (α_nn, val_J)。
    """
    import torch
    import torch.nn as nn

    labels = np.asarray(labels, dtype=np.int64)
    n = K.shape[0]
    bw = float(np.sqrt(np.maximum(np.median(np.diag(K)), 1e-12)))
    ctx_np = _kernel_context_for_nn(K, labels, gamma=gamma, n0=n0, bw=bw)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(48, int(0.15 * n))
    va, tr = perm[:n_val], perm[n_val:]

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            out = MOD_ORDER * n
            self.net = nn.Sequential(
                nn.Linear(ctx_np.shape[0], 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, out),
            )

        def forward(self, ctx: torch.Tensor) -> torch.Tensor:
            return self.net(ctx).view(MOD_ORDER, n)

    net = _Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    K_t = torch.from_numpy(K.astype(np.float32))
    lab_t = torch.from_numpy(labels)
    ctx = torch.from_numpy(ctx_np)
    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(int(epochs)):
        opt.zero_grad()
        alpha = net(ctx)
        logits = K_t @ alpha.T
        ce = nn.functional.cross_entropy(logits[tr], lab_t[tr])
        reg = float(lam) * torch.sum(alpha * (alpha @ K_t))
        (ce + reg).backward()
        opt.step()
        with torch.no_grad():
            a_v = net(ctx)
            v = float(nn.functional.cross_entropy(K_t[va] @ a_v.T, lab_t[va]))
        if v < best_val - 1e-5:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        a_nn = net(ctx).cpu().numpy()
    return a_nn.astype(np.float64), float(best_val)


def _val_j_alpha(alpha: np.ndarray, K: np.ndarray, labels: np.ndarray, va: np.ndarray) -> float:
    logits = K @ alpha.T
    f = _softmax_rows(logits[va])
    return float(softmax_ce_from_scores(f, labels[va]))


def _softmax_weights_from_j(js: list[float], *, temp: float = 0.08) -> np.ndarray:
    """val-J 越低权重越大：w ∝ exp(-(J-Jmin)/τ)。"""
    arr = np.asarray(js, dtype=np.float64)
    z = -(arr - float(arr.min())) / max(float(temp), 1e-6)
    z -= z.max()
    w = np.exp(z)
    s = float(w.sum())
    return w / s if s > 0 else np.full(len(js), 1.0 / max(len(js), 1))


def _aggregate_via_logit_proj(
    cand_logits: list[tuple[str, np.ndarray, float]],
    K: np.ndarray,
    labels: np.ndarray,
    lam: float,
    *,
    va: np.ndarray,
    adam_epochs: int = 200,
    lbfgs_maxiter: int = 150,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
    RKHS 聚合：先对候选 logits 按 val-J 加权平均，再投影回 H_K：
        L̄ = Σ w_i L_i,   α = argmin ‖Kα^T - L̄‖ + λ‖α‖_K
    凸组合 + 核投影，仍是单一 RKHS 决策函数。
    """
    names = [c[0] for c in cand_logits]
    js = [float(c[2]) for c in cand_logits]
    w = _softmax_weights_from_j(js, temp=0.06)
    L = np.zeros_like(cand_logits[0][1], dtype=np.float64)
    for wi, (_, Li, _) in zip(w, cand_logits):
        L += float(wi) * np.asarray(Li, dtype=np.float64)
    try:
        a_agg = solve_alpha_from_logits(L, K, float(lam))
    except np.linalg.LinAlgError:
        best_i = int(np.argmin(js))
        a_agg = solve_alpha_from_logits(
            np.asarray(cand_logits[best_i][1], dtype=np.float64), K, float(lam)
        )
    a_agg = _fit_adam(
        a_agg, K, labels, float(lam), "softmax",
        epochs=int(adam_epochs), patience=40, verbose=False,
    )
    if lbfgs_maxiter > 0:
        a_agg = _fit_lbfgs_limited_local(
            a_agg, K, labels, float(lam), maxiter=int(lbfgs_maxiter)
        )
    j_agg = _val_j_alpha(a_agg, K, labels, va)
    tag = "proj[" + "+".join(f"{n}:{wi:.2f}" for n, wi in zip(names, w)) + "]"
    return a_agg, w, tag, float(j_agg)


class RKHSApproxMLDDetector:
    """
    RKHS 逼近 MLD：logits = K_η α^T。
    接口对齐：fit / scores / detect / last_fit_stats / gamma / lam。
    """

    def __init__(
        self,
        *,
        feature_mode: str = "struct",
        target: str = "fstar",
        lam_c: float = 0.1,
        kernel_mode: str = "adaptive",
        ms_ratios: tuple[float, ...] = ADAPTIVE_MKL_RATIOS,
        pilot_mult: float = 1.0,
        robust_csi: bool = True,
        plugin_soft_weight: float | None = None,
        gamma_scale: float = 1.0,
        expr_tune: bool = False,
        lock_ms_ratios: bool = False,
        use_nn: bool = False,
        aggregate: bool = True,
        n_mkl_bags: int = 2,
        stack_rkhs: bool = False,
    ) -> None:
        if feature_mode not in ("blind", "struct", "struct_hat"):
            raise ValueError("feature_mode 须为 blind/struct/struct_hat")
        if target not in ("fstar", "hard", "plugin"):
            raise ValueError("target 须为 fstar/hard/plugin")
        self.feature_mode = feature_mode
        self.target = target
        self.lam_c = float(lam_c)
        self.kernel_mode = kernel_mode
        self.ms_ratios = tuple(ms_ratios)
        self.pilot_mult = float(pilot_mult)
        self.robust_csi = bool(robust_csi)
        self.plugin_soft_weight = plugin_soft_weight
        self.gamma_scale = float(gamma_scale)
        self.expr_tune = bool(expr_tune)
        self.lock_ms_ratios = bool(lock_ms_ratios)
        self.use_nn = bool(use_nn)
        self.aggregate = bool(aggregate)
        self.n_mkl_bags = int(max(1, n_mkl_bags))
        self.stack_rkhs = bool(stack_rkhs)
        self.gamma: float | None = None
        self.lam: float | None = None
        self.eta: np.ndarray | None = None
        self.kernel_scales: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.alpha_m: np.ndarray | None = None  # Adaptive-MKL 分核系数 (n_k,C,n)
        # 二阶堆叠 RKHS：φ₂=[z_rob, softmax(L₁)]
        self.stack_active: bool = False
        self.X2_centers: np.ndarray | None = None
        self.feat2_mean: np.ndarray | None = None
        self.feat2_std: np.ndarray | None = None
        self.gamma2: float | None = None
        self.lam2: float | None = None
        self.eta2: np.ndarray | None = None
        self.kernel_scales2: np.ndarray | None = None
        self.alpha2: np.ndarray | None = None
        self.ms_ratios2: tuple[float, ...] = ADAPTIVE_MKL_RATIOS
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.X_centers: np.ndarray | None = None
        self.H_eff: np.ndarray | None = None
        self.H_hat: np.ndarray | None = None
        self.n0_hat: float | None = None
        self.T_pilot: int | None = None
        self.sigma_e2: float | None = None
        self.hy_cache = None
        self.hy_hat_cache = None
        self.snr_db: float | None = None
        self.last_fit_stats: dict[str, float] = {}

    def _features(self, y: np.ndarray) -> np.ndarray:
        assert self.H_eff is not None and self.snr_db is not None
        n0 = n0_from_snr_db(self.snr_db)
        if self.feature_mode == "blind":
            return blind_y_features(y)
        if self.feature_mode == "struct":
            return struct_z_features(y, self.H_eff, n0, self.hy_cache)
        # struct_hat
        assert self.H_hat is not None and self.n0_hat is not None
        if self.robust_csi and self.sigma_e2 is not None:
            return robust_struct_z_features(
                y, self.H_hat, self.n0_hat, float(self.sigma_e2)
            )
        return struct_z_features(y, self.H_hat, self.n0_hat, self.hy_hat_cache)

    def fit(
        self,
        y_train: np.ndarray,
        s1_train: np.ndarray,
        *,
        H_eff: np.ndarray,
        snr_db: float,
        f_star_train: np.ndarray | None = None,
        adam_epochs: int = 1200,
        lbfgs_maxiter: int = 1000,
        verbose: bool = False,
    ) -> float:
        self.H_eff = np.asarray(H_eff)
        self.snr_db = float(snr_db)
        n = len(y_train)
        n0 = n0_from_snr_db(self.snr_db)
        ylab = np.asarray(s1_train, dtype=np.int64)

        if self.feature_mode == "struct":
            self.hy_cache = precompute_mld_hy(self.H_eff)
        elif self.feature_mode == "struct_hat" or self.target == "plugin":
            # struct_hat 特征或 plugin 损失都需要 Ĥ（不要求加长导频）
            self.H_hat, self.n0_hat, self.T_pilot = estimate_heff_block(
                self.H_eff,
                self.snr_db,
                np.random.default_rng(0),
                pilot_mult=self.pilot_mult,
            )
            se2 = sigma_e2_pilot(self.n0_hat, self.T_pilot)
            # 高 SNR 信道误差已小，过强对角加载会伤充分统计 → 衰减 σ_e²
            s = float(snr_db)
            if s >= 10.0:
                se2 *= 0.15
            elif s >= 8.0:
                se2 *= 0.4
            self.sigma_e2 = float(se2)
            # 极高 SNR：若衰减后仍过大，关闭稳健（退回标准 GA(Ĥ)）
            if s >= 12.0:
                self.robust_csi = False
            self.hy_hat_cache = precompute_mld_hy(self.H_hat)

        X_raw = self._features(y_train)
        X, mean, std = _normalize_fit(X_raw)
        self.feat_mean, self.feat_std = mean, std
        self.X_centers = X
        g0 = gamma_theory_rkhs(n0, X)
        self.gamma = float(g0) * float(self.gamma_scale)
        self.lam = lam_theory_rkhs(n0, n, c=self.lam_c)

        # struct_hat：Adaptive-MKL 主表达；高 SNR 用标准 6 核防崩，低中 SNR 用加密库
        ratios = self.ms_ratios
        if self.feature_mode == "struct_hat":
            if float(snr_db) >= 10.0:
                ratios = ADAPTIVE_MKL_RATIOS
            elif (
                len(ratios) < 8
                or ratios == ADAPTIVE_MKL_RATIOS
                or ratios == (0.25, 0.5, 1.0, 2.0)
                or ratios == (0.15, 0.5, 1.0, 2.0)
            ):
                ratios = RICH_ADAPTIVE_MKL_RATIOS
        elif not self.lock_ms_ratios:
            if float(snr_db) >= 10.0:
                ratios = (0.25, 0.5, 1.0, 2.0)
            elif float(snr_db) >= 8.0:
                ratios = (0.15, 0.5, 1.0, 2.0)
        self.ms_ratios = ratios
        self.lock_ms_ratios = True if self.feature_mode == "struct_hat" else self.lock_ms_ratios
        self.alpha_m = None

        if self.target == "fstar":
            if f_star_train is None:
                raise ValueError("target='fstar' 需要 f_star_train")
            f_star = np.asarray(f_star_train, dtype=np.float64)

            if self.feature_mode == "struct_hat":
                # 可实现蒸馏：软标签 CE + 硬 CE（非闭式 log-f* 岭回归）
                base_k = build_base_kernels(X, self.gamma, ms_ratios=ratios)
                soft_w = 0.55 if float(snr_db) < 8.0 else (0.35 if float(snr_db) < 10.0 else 0.2)
                alpha, eta, K, stats = _fit_soft_label_adaptive_mkl(
                    base_k,
                    f_star,
                    ylab,
                    float(self.lam),
                    soft_weight=soft_w,
                    adam_epochs=int(adam_epochs),
                    lbfgs_maxiter=int(lbfgs_maxiter),
                    verbose=verbose,
                )
                self.alpha = alpha
                self.eta = eta
                self.kernel_scales = np.asarray(stats["kernel_scales"], dtype=np.float64)
                logits = K @ self.alpha.T
                pick = f"approx_distill_{self.feature_mode}"
                if self.robust_csi:
                    pick += "_robust"
            else:
                # 真 H 结构特征：闭式逼近 log f*
                log_p = np.log(np.maximum(f_star, 1e-300))
                log_p = log_p - log_p.max(axis=1, keepdims=True)
                if self.kernel_mode == "adaptive":
                    base_k = build_base_kernels(X, self.gamma, ms_ratios=ratios)
                    eta = np.full(len(ratios), 1.0 / len(ratios))
                    scales = np.empty(len(ratios))
                    Ks = []
                    for m, Km in enumerate(base_k):
                        trc = float(np.trace(Km).real)
                        scales[m] = n / max(trc, 1e-12)
                        Ks.append(Km * scales[m])
                    K = combine_kernels(Ks, eta)
                    self.kernel_scales = scales
                    self.eta = eta
                else:
                    K = build_kernel_matrix(
                        X, self.gamma, kernel_mode=self.kernel_mode, ms_ratios=ratios
                    )
                    self.eta = None
                    self.kernel_scales = None
                self.alpha = solve_alpha_from_logits(log_p, K, float(self.lam))
                logits = K @ self.alpha.T
                pick = f"approx_fstar_{self.feature_mode}"
        elif self.target == "plugin":
            # 损失里用 Ĥ：p̂ = softmax(z_rob(y;Ĥ))，最小化 w·CE(p̂,f)+(1-w)·J_hard
            assert self.H_hat is not None and self.n0_hat is not None
            if self.robust_csi and self.sigma_e2 is not None:
                z_hat = robust_struct_z_features(
                    y_train, self.H_hat, self.n0_hat, float(self.sigma_e2)
                )
            else:
                z_hat = struct_z_features(
                    y_train, self.H_hat, self.n0_hat, self.hy_hat_cache
                )
            p_hat = _softmax_rows(z_hat)
            if self.plugin_soft_weight is None:
                # Ĥ plug-in 后验有偏：默认只作轻正则，避免抬高 J_hard / BER
                soft_w = 0.15
            else:
                soft_w = float(self.plugin_soft_weight)
            base_k = build_base_kernels(X, self.gamma, ms_ratios=ratios)
            alpha, eta, K, stats = _fit_soft_label_adaptive_mkl(
                base_k,
                p_hat,
                ylab,
                float(self.lam),
                soft_weight=soft_w,
                adam_epochs=int(adam_epochs),
                lbfgs_maxiter=int(lbfgs_maxiter),
                verbose=verbose,
            )
            self.alpha = alpha
            self.eta = eta
            self.kernel_scales = np.asarray(stats["kernel_scales"], dtype=np.float64)
            logits = K @ self.alpha.T
            pick = f"approx_plugin_{self.feature_mode}_w{soft_w:.2f}"
            if self.robust_csi:
                pick += "_robust"
        else:
            # hard：最小化经验贝叶斯后验损失 J_hard + RKHS 正则
            do_tune = bool(self.expr_tune)
            if do_tune:
                gamma, lam, gs, cs = _tune_hard_expr(
                    X,
                    ylab,
                    n0,
                    self.lam_c,
                    ratios,
                    adam_epochs=int(adam_epochs),
                    lbfgs_maxiter=int(lbfgs_maxiter),
                    gamma_scale0=float(self.gamma_scale),
                    verbose=verbose,
                )
                self.gamma = float(gamma)
                self.lam = float(lam)
                self.gamma_scale = float(gs)
            else:
                gs, cs = float(self.gamma_scale), 1.0
                self.gamma = float(g0) * gs
                self.lam = lam_theory_rkhs(n0, n, c=self.lam_c * cs)

            base_k = build_base_kernels(X, self.gamma, ms_ratios=ratios)
            # 高 SNR：多核 L-BFGS 易崩，缩短精修；仍保留分核 α_m 推理
            mkl_lbfgs = int(lbfgs_maxiter)
            if float(snr_db) >= 10.0:
                mkl_lbfgs = min(mkl_lbfgs, 200)
            alpha, eta, K, stats = fit_adaptive_mkl_alpha(
                base_k,
                ylab,
                float(self.lam),
                adam_epochs=int(adam_epochs),
                lbfgs_maxiter=mkl_lbfgs,
                verbose=verbose,
                keep_multi_alpha=True,
                ent_reg=0.03 if float(snr_db) >= 8.0 else 0.02,
            )
            self.alpha = alpha
            self.eta = eta
            self.alpha_m = stats.get("alpha_m")
            self.kernel_scales = np.asarray(stats.get("kernel_scales", []), dtype=np.float64)
            if self.kernel_scales.size == 0:
                self.kernel_scales = None
            if self.alpha_m is not None:
                logits = np.zeros((n, MOD_ORDER), dtype=np.float64)
                for m, Km in enumerate(base_k):
                    Km_s = Km * float(self.kernel_scales[m]) if self.kernel_scales is not None else Km
                    logits += Km_s @ self.alpha_m[m].T
            else:
                logits = K @ self.alpha.T
            # 崩溃回退：仅高 SNR 且 train SER 异常高时改用合并 α_eff
            tr_ser = float(np.mean(np.argmax(logits, 1) != ylab))
            if float(snr_db) >= 8.0 and tr_ser > 0.25 and self.alpha is not None:
                logits = K @ self.alpha.T
                self.alpha_m = None
                pick = f"approx_hard_mkl_fallback_{self.feature_mode}"
            else:
                pick = f"approx_hard_mkl{len(ratios)}_{self.feature_mode}_g{gs:.2f}_c{cs:.2f}"
            if self.feature_mode == "struct_hat" and self.robust_csi:
                pick += "_robust"
            self._last_c_scale = float(cs)

            # —— RKHS 内聚合：MKL / NN / 对角 / 多种子 bag 的 α 凸组合（同一 K_η）——
            if (self.use_nn or self.aggregate) and K is not None and self.alpha is not None:
                rng_v = np.random.default_rng(1)
                perm = rng_v.permutation(n)
                n_val = max(48, int(0.15 * n))
                va = perm[:n_val]
                # 把多核 logits 压成 K_η 上的 α_eff，便于与其它 α 聚合
                if self.alpha_m is not None:
                    try:
                        a_mkl = solve_alpha_from_logits(logits, K, float(self.lam))
                    except np.linalg.LinAlgError:
                        a_mkl = self.alpha
                else:
                    a_mkl = self.alpha
                j_mkl = _val_j_alpha(a_mkl, K, ylab, va)

                pool_L: list[tuple[str, np.ndarray, float]] = []
                # 主 MKL logits
                if self.alpha_m is not None:
                    L_mkl = logits
                else:
                    L_mkl = K @ a_mkl.T
                pool_L.append(("mkl", L_mkl, j_mkl))
                pool_a: list[tuple[str, np.ndarray, float]] = [("mkl", a_mkl, j_mkl)]

                # Adaptive-MKL 多种子 bag → 多个 α_eff / logits
                n_bags = self.n_mkl_bags if self.aggregate else 1
                for b in range(1, n_bags):
                    try:
                        a_b, eta_b, K_b, st_b = fit_adaptive_mkl_alpha(
                            base_k,
                            ylab,
                            float(self.lam),
                            adam_epochs=max(250, int(adam_epochs) // 3),
                            lbfgs_maxiter=min(200, int(lbfgs_maxiter)),
                            verbose=False,
                            keep_multi_alpha=False,
                            ent_reg=0.03 if float(snr_db) >= 8.0 else 0.02,
                            seed=int(b * 17 + 3),
                        )
                        L_b = K @ a_b.T
                        j_b = _val_j_alpha(a_b, K, ylab, va)
                        pool_L.append((f"bag{b}", L_b, j_b))
                        pool_a.append((f"bag{b}", a_b, j_b))
                    except Exception:
                        pass

                if self.use_nn:
                    try:
                        a_nn, _ = _nn_warmstart_alpha(
                            K,
                            ylab,
                            gamma=float(self.gamma),
                            n0=float(n0),
                            lam=float(self.lam),
                            epochs=min(300, max(120, int(adam_epochs) // 4)),
                            patience=35,
                        )
                        a_ref = _fit_adam(
                            a_nn, K, ylab, float(self.lam), "softmax",
                            epochs=min(500, int(adam_epochs) // 2),
                            patience=60, verbose=False,
                        )
                        if int(lbfgs_maxiter) > 0:
                            a_ref = _fit_lbfgs_limited_local(
                                a_ref, K, ylab, float(self.lam),
                                maxiter=min(400, int(lbfgs_maxiter)),
                            )
                        j_nn = _val_j_alpha(a_ref, K, ylab, va)
                        pool_L.append(("nn", K @ a_ref.T, j_nn))
                        pool_a.append(("nn", a_ref, j_nn))
                    except Exception:
                        pass

                    a_diag = _label_diagonal_init(ylab)
                    a_d = _fit_adam(
                        a_diag, K, ylab, float(self.lam), "softmax",
                        epochs=min(400, int(adam_epochs) // 2), patience=50, verbose=False,
                    )
                    if int(lbfgs_maxiter) > 0:
                        a_d = _fit_lbfgs_limited_local(
                            a_d, K, ylab, float(self.lam),
                            maxiter=min(300, int(lbfgs_maxiter)),
                        )
                    j_d = _val_j_alpha(a_d, K, ylab, va)
                    pool_L.append(("diag", K @ a_d.T, j_d))
                    pool_a.append(("diag", a_d, j_d))

                j_best_single = min(c[2] for c in pool_a)
                best_single = min(pool_a, key=lambda c: c[2])

                if self.aggregate and len(pool_L) >= 2:
                    a_agg, w_agg, tag_agg, j_agg = _aggregate_via_logit_proj(
                        pool_L, K, ylab, float(self.lam), va=va,
                        adam_epochs=min(250, int(adam_epochs) // 3),
                        lbfgs_maxiter=min(200, int(lbfgs_maxiter)),
                    )
                    # 严格：聚合须明显不差于最优单模型
                    if j_agg <= j_best_single + 5e-4:
                        self.alpha = a_agg
                        self.alpha_m = None
                        logits = K @ self.alpha.T
                        pick = f"approx_hard_{tag_agg}_{self.feature_mode}"
                        if self.feature_mode == "struct_hat" and self.robust_csi:
                            pick += "_robust"
                        if verbose:
                            print(
                                f"  RKHS-proj {tag_agg} valJ={j_agg:.4f} "
                                f"(best_single={best_single[0]}:{j_best_single:.4f})",
                                flush=True,
                            )
                    else:
                        if best_single[0] != "mkl":
                            self.alpha = best_single[1]
                            self.alpha_m = None
                            logits = K @ self.alpha.T
                        pick = f"approx_hard_mkl+{best_single[0]}_{self.feature_mode}"
                        if self.feature_mode == "struct_hat" and self.robust_csi:
                            pick += "_robust"
                else:
                    if best_single[0] != "mkl" and best_single[2] < j_mkl - 1e-4:
                        self.alpha = best_single[1]
                        self.alpha_m = None
                        logits = K @ self.alpha.T
                        pick = f"approx_hard_mkl+{best_single[0]}_{self.feature_mode}"
                        if self.feature_mode == "struct_hat" and self.robust_csi:
                            pick += "_robust"

        # —— 二阶堆叠 RKHS：φ₂ = [z_rob, softmax(L₁)]，再最小化 J ——
        self.stack_active = False
        if (
            self.stack_rkhs
            and self.target == "hard"
            and self.feature_mode == "struct_hat"
            and float(snr_db) >= 6.0  # 低 SNR 堆叠易过拟合
        ):
            soft1 = _softmax_rows(logits)
            X2_raw = np.concatenate([X, soft1], axis=1)
            X2, m2, s2 = _normalize_fit(X2_raw)
            self.feat2_mean, self.feat2_std = m2, s2
            self.X2_centers = X2
            self.ms_ratios2 = ADAPTIVE_MKL_RATIOS
            g2 = gamma_theory_rkhs(n0, X2)
            lam2 = lam_theory_rkhs(n0, n, c=max(self.lam_c, 0.15))  # 略强正则
            self.gamma2, self.lam2 = float(g2), float(lam2)
            base2 = build_base_kernels(X2, self.gamma2, ms_ratios=self.ms_ratios2)
            a2, eta2, K2, st2 = fit_adaptive_mkl_alpha(
                base2,
                ylab,
                float(self.lam2),
                adam_epochs=max(400, int(adam_epochs) // 2),
                lbfgs_maxiter=min(400, int(lbfgs_maxiter)),
                verbose=verbose,
                keep_multi_alpha=False,
                ent_reg=0.02,
                seed=11,
            )
            self.alpha2 = a2
            self.eta2 = eta2
            self.kernel_scales2 = np.asarray(st2.get("kernel_scales", []), dtype=np.float64)
            if self.kernel_scales2.size == 0:
                self.kernel_scales2 = None
            logits2 = K2 @ self.alpha2.T
            # 用验证 BER 选（与部署目标一致），J 作次要
            rng_s = np.random.default_rng(5)
            perm_s = rng_s.permutation(n)
            n_val_s = max(64, int(0.2 * n))
            va_s = perm_s[:n_val_s]
            ber1 = float(np.mean(np.argmax(logits[va_s], 1) != ylab[va_s]))
            ber2 = float(np.mean(np.argmax(logits2[va_s], 1) != ylab[va_s]))
            j1 = float(softmax_ce_from_scores(_softmax_rows(logits[va_s]), ylab[va_s]))
            j2 = float(softmax_ce_from_scores(_softmax_rows(logits2[va_s]), ylab[va_s]))
            if ber2 < ber1 - 1e-4 or (ber2 <= ber1 + 1e-4 and j2 < j1 - 1e-3):
                self.stack_active = True
                logits = logits2
                pick = f"{pick}+stack2"
                if verbose:
                    print(
                        f"  stack2 ON valBER {ber1:.4f}→{ber2:.4f} J {j1:.4f}→{j2:.4f}",
                        flush=True,
                    )
            elif verbose:
                print(
                    f"  stack2 OFF (valBER1={ber1:.4f} BER2={ber2:.4f})",
                    flush=True,
                )

        f = _softmax_rows(logits)
        train_ce = softmax_ce_from_scores(f, ylab)
        train_ser = float(np.mean(np.argmax(logits, 1) != ylab))
        self.last_fit_stats = {
            "train_j_data": float(train_ce),
            "train_ser": train_ser,
            "gamma": float(self.gamma),
            "lam": float(self.lam),
            "n_centers": float(n),
            "alpha_init_pick": pick,
            "mode": f"rkhs_approx_mld_{self.feature_mode}_{self.target}",
            "feature_dim": float(X.shape[1]),
            "gamma_scale": float(self.gamma_scale),
            "stack_active": float(self.stack_active),
        }
        if hasattr(self, "_last_c_scale"):
            self.last_fit_stats["c_scale"] = float(self._last_c_scale)
        if self.sigma_e2 is not None:
            self.last_fit_stats["sigma_e2"] = float(self.sigma_e2)
        if self.eta is not None:
            self.last_fit_stats["eta_entropy"] = float(
                -np.sum(self.eta * np.log(self.eta + 1e-12))
            )
        if verbose:
            print(
                f"  RKHS≈MLD [{self.feature_mode}/{self.target}"
                f"{'/robust' if self.robust_csi and self.feature_mode=='struct_hat' else ''}]: "
                f"SER={train_ser:.4f} J={train_ce:.4f} d={X.shape[1]}",
                flush=True,
            )
        return float(train_ce)

    def _stage1_logits_from_X(self, X: np.ndarray) -> np.ndarray:
        """已归一化的一阶段特征 → logits。"""
        if self.alpha_m is not None and self.kernel_scales is not None:
            base_k = build_base_kernels(
                X, float(self.gamma), self.X_centers, ms_ratios=self.ms_ratios
            )
            logits = np.zeros((X.shape[0], MOD_ORDER), dtype=np.float64)
            for m, Km in enumerate(base_k):
                Km_s = Km * float(self.kernel_scales[m])
                logits += Km_s @ self.alpha_m[m].T
            return logits
        if self.alpha is None:
            raise RuntimeError("先 fit()")
        if self.kernel_mode == "adaptive" and self.eta is not None:
            base_k = build_base_kernels(
                X, float(self.gamma), self.X_centers, ms_ratios=self.ms_ratios
            )
            if self.kernel_scales is not None and len(self.kernel_scales) == len(base_k):
                base_k = [Km * float(s) for Km, s in zip(base_k, self.kernel_scales)]
            K = combine_kernels(base_k, self.eta)
        else:
            K = build_kernel_matrix(
                X,
                float(self.gamma),
                self.X_centers,
                kernel_mode=self.kernel_mode,
                ms_ratios=self.ms_ratios,
            )
        return K @ self.alpha.T

    def _logits(self, y: np.ndarray) -> np.ndarray:
        if self.X_centers is None:
            raise RuntimeError("先 fit()")
        X = _normalize_apply(self._features(y), self.feat_mean, self.feat_std)
        logits1 = self._stage1_logits_from_X(X)
        if not self.stack_active or self.alpha2 is None or self.X2_centers is None:
            return logits1
        soft1 = _softmax_rows(logits1)
        X2_raw = np.concatenate([X, soft1], axis=1)
        X2 = _normalize_apply(X2_raw, self.feat2_mean, self.feat2_std)
        base2 = build_base_kernels(
            X2, float(self.gamma2), self.X2_centers, ms_ratios=self.ms_ratios2
        )
        if self.kernel_scales2 is not None and len(self.kernel_scales2) == len(base2):
            base2 = [Km * float(s) for Km, s in zip(base2, self.kernel_scales2)]
        if self.eta2 is not None:
            K2 = combine_kernels(base2, self.eta2)
        else:
            K2 = combine_kernels(base2, np.full(len(base2), 1.0 / len(base2)))
        return K2 @ self.alpha2.T

    def scores(self, y: np.ndarray) -> np.ndarray:
        return _softmax_rows(self._logits(y))

    def detect(self, y: np.ndarray) -> np.ndarray:
        return np.argmax(self._logits(y), axis=-1)
