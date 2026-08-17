"""
核展开检测：logits_a = Σ_i α_{a,i} k(x_i, x)，输出经 softplus 或 softmax。
推荐 output_mode='softmax'（与 J_data 的归一化比值一致，更易学尖分布）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from objective import rkhs_penalty, softmax_ce_from_scores
from system import MOD_ORDER, y_to_features

# 多尺度 RBF：γ_s = γ_base × ratio；ratio<1 更宽，>1 更窄
DEFAULT_MULTISCALE_RATIOS: tuple[float, ...] = (0.25, 1.0, 4.0)
# SimpleMKL / 自适应多核基核库（相对 γ_theory）
ADAPTIVE_MKL_RATIOS: tuple[float, ...] = (0.05, 0.15, 0.5, 1.0, 2.0, 8.0)
# 加密尺度库：让 η 在更密的带宽上选（struct_hat 主线）
RICH_ADAPTIVE_MKL_RATIOS: tuple[float, ...] = (
    0.03, 0.06, 0.12, 0.25, 0.5, 0.8, 1.0, 1.5, 2.5, 5.0, 10.0,
)


def estimate_n0_from_y(y: np.ndarray) -> float:
    """仅由接收样本 y 粗估 N₀（训练期可用，不用 CSI）。"""
    X = y_to_features(np.atleast_2d(y))
    per_dim_var = np.var(X, axis=0)
    return float(max(np.median(per_dim_var), 1e-12))


def gamma_from_n0(n0: float) -> float:
    """y 空间高斯核带宽 1/N₀（用于 f_a^* / MLD，不直接套到归一化 φ 上的 RKHS）。"""
    return 1.0 / max(float(n0), 1e-12)


def gamma_rkhs_from_n0(
    n0: float,
    X_norm: np.ndarray,
    *,
    n0_ref: float = 1.0,
    scale_clip: tuple[float, float] = (0.25, 4.0),
) -> float:
    """
    RKHS 核（中心在训练 y_k、特征 φ(y) 已 z-score）：
    f_a^* 用 exp(-‖y-h‖²/N₀)；φ 空间典型间距 ~ 2·median_bw² →
    γ = 1/(2·bw²)·√(N₀_ref/N₀)。勿用裸 1/N₀，否则高 SNR 时 K(y_test,y_train)≈0。
    """
    bw = median_bandwidth(X_norm)
    g0 = 1.0 / (2.0 * bw * bw + 1e-12)
    n0 = max(float(n0), 1e-12)
    lo, hi = scale_clip
    scale = float(np.clip(np.sqrt(n0_ref / n0), lo, hi))
    return g0 * scale


def gamma_from_features(
    n0: float,
    X_norm: np.ndarray,
    *,
    n0_ref: float = 1.0,
    scale_clip: tuple[float, float] = (0.25, 4.0),
) -> float:
    """γ_base：median 带宽 × √(N₀_ref/N₀)（仅用于 median/调参候选）。"""
    bw = median_bandwidth(X_norm)
    g0 = 1.0 / (2.0 * bw * bw)
    n0 = max(float(n0), 1e-12)
    lo, hi = scale_clip
    scale = float(np.clip(np.sqrt(n0_ref / n0), lo, hi))
    return g0 * scale


def gamma_from_noise(
    n0: float,
    X_norm: np.ndarray,
    *,
    n0_ref: float = 1.0,
    scale_clip: tuple[float, float] = (0.25, 4.0),
) -> float:
    """盲/RKHS：γ 与 N₀、特征 median_bw 联合标定。"""
    return gamma_rkhs_from_n0(
        n0, X_norm, n0_ref=n0_ref, scale_clip=scale_clip
    )


def gamma_theory_rkhs(
    n0: float,
    X_norm: np.ndarray,
    *,
    n0_ref: float = 1.0,
    scale_clip: tuple[float, float] = (0.25, 4.0),
) -> float:
    """
    理论 RKHS 带宽（推荐盲法，不用 holdout 网格）：
    φ 空间 RBF 基准 1/(2·bw²)，再乘 √(N₀/N₀_ref)。
    高 SNR（N₀↓）时 scale↓ → γ↓ → 核加宽，与 f_a^* 变尖时仍保持 K_te 连通一致。
    """
    bw = median_bandwidth(X_norm)
    g0 = 1.0 / (2.0 * bw * bw + 1e-12)
    n0 = max(float(n0), 1e-12)
    lo, hi = scale_clip
    scale = float(np.clip(np.sqrt(n0 / n0_ref), lo, hi))
    return g0 * scale


def lam_theory_rkhs(
    n0: float,
    n: int,
    *,
    c: float = 0.1,
    n0_lam_floor: float = 0.1,
) -> float:
    """
    Ridge 理论标定：λ = c·N₀_eff/n，N₀_eff = max(N₀, n0_lam_floor)。
    高 SNR 时 N₀ 很小，若直接用 c·N₀/n 在 n≈2000 上过小，Adam/L-BFGS 易失败（train SER≫1）。
    默认 n0_lam_floor=0.1 与 SNR≈10 dB 同级，仅抬高正则不下调。
    """
    n = max(int(n), 1)
    n0_eff = max(float(n0), float(n0_lam_floor))
    return float(c * n0_eff / n)


def median_bandwidth(X: np.ndarray) -> float:
    n = min(X.shape[0], 400)
    idx = np.random.default_rng(0).choice(X.shape[0], n, replace=False)
    sub = X[idx]
    d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=2)
    med = np.median(d2[d2 > 0])
    return float(np.sqrt(0.5 * med + 1e-12))


def rbf_kernel(X: np.ndarray, gamma: float, Y: np.ndarray | None = None) -> np.ndarray:
    if Y is None:
        Y = X
    d2 = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * d2)


def build_kernel_matrix(
    X: np.ndarray,
    gamma_base: float,
    Y: np.ndarray | None = None,
    *,
    kernel_mode: str = "single",
    ms_ratios: tuple[float, ...] = DEFAULT_MULTISCALE_RATIOS,
    eta: np.ndarray | None = None,
) -> np.ndarray:
    """
    核矩阵。
    - single：单 RBF
    - multiscale：等权多尺度
    - adaptive：加权多核 K=∑ η_m K_m（η 在单纯形上；缺省等权）
    """
    if kernel_mode == "single":
        return rbf_kernel(X, gamma_base, Y)
    if kernel_mode in ("multiscale", "adaptive"):
        ratios = tuple(ms_ratios)
        n_k = len(ratios)
        if eta is None:
            w = np.full(n_k, 1.0 / n_k, dtype=np.float64)
        else:
            w = np.asarray(eta, dtype=np.float64).ravel()
            if w.shape[0] != n_k:
                raise ValueError(f"eta 长度 {w.shape[0]} 与基核数 {n_k} 不一致")
            w = np.maximum(w, 0.0)
            s = float(w.sum())
            w = w / s if s > 0 else np.full(n_k, 1.0 / n_k)
        n_rows = X.shape[0]
        n_cols = X.shape[0] if Y is None else Y.shape[0]
        K = np.zeros((n_rows, n_cols), dtype=np.float64)
        for wi, ratio in zip(w, ratios):
            if wi <= 0:
                continue
            K += wi * rbf_kernel(X, gamma_base * ratio, Y)
        return K
    raise ValueError(f"未知 kernel_mode={kernel_mode!r}")


def build_base_kernels(
    X: np.ndarray,
    gamma_base: float,
    Y: np.ndarray | None = None,
    *,
    ms_ratios: tuple[float, ...] = ADAPTIVE_MKL_RATIOS,
) -> list[np.ndarray]:
    """预计算基核列表 {K_m}，供自适应多核组合。"""
    return [rbf_kernel(X, gamma_base * float(r), Y) for r in ms_ratios]


def combine_kernels(base_kernels: list[np.ndarray], eta: np.ndarray) -> np.ndarray:
    """K = ∑ η_m K_m，η≥0 且归一化到 ∑η=1。"""
    eta = np.asarray(eta, dtype=np.float64).ravel()
    eta = np.maximum(eta, 0.0)
    s = float(eta.sum())
    if s <= 0:
        eta = np.full(len(base_kernels), 1.0 / len(base_kernels))
    else:
        eta = eta / s
    K = np.zeros_like(base_kernels[0], dtype=np.float64)
    for w, Km in zip(eta, base_kernels):
        if w > 0:
            K += w * Km
    return K


def fit_adaptive_mkl_alpha(
    base_kernels: list[np.ndarray],
    labels: np.ndarray,
    lam: float,
    *,
    adam_epochs: int = 1500,
    lbfgs_maxiter: int = 1200,
    patience: int = 80,
    lr_alpha: float = 0.05,
    lr_eta: float = 0.05,
    ent_reg: float = 0.02,
    val_frac: float = 0.15,
    seed: int = 0,
    verbose: bool = False,
    keep_multi_alpha: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    SimpleMKL 风格自适应多核（分核系数 + 凸组合权重）：

        f = ∑_m K_m α_m^T
        min  CE(y, softmax(f))
             + λ ∑_m (1/η_m) ∑_a α_{m,a}^T K_m α_{m,a}
             - μ H(η)
        s.t. η = softmax(θ) ∈ Δ

    基核先做 trace 归一化。默认 keep_multi_alpha=True：推理保留 ∑_m K_m α_m，
    不把多核压成单组 α_eff（更贴合 Adaptive-MKL）。
    """
    labels = np.asarray(labels, dtype=np.int64)
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
    labels_t = torch.from_numpy(labels)

    a0 = _label_diagonal_init(labels).astype(np.float32)
    alpha_m = torch.tensor(
        np.stack([a0 for _ in range(n_k)], axis=0) / max(n_k, 1),
        dtype=torch.float32,
        requires_grad=True,
    )
    theta = torch.zeros(n_k, dtype=torch.float32)
    for m in range(n_k):
        theta[m] = 0.35 * (n_k - 1 - m) / max(n_k - 1, 1)
    theta = theta.clone().detach().requires_grad_(True)

    # 阶段1：联合学 (α_m, η)；阶段2：固定 η 精修 α_m（更贴 SimpleMKL 交替）
    phase1 = max(int(adam_epochs * 0.65), 200)
    phase2 = max(int(adam_epochs) - phase1, 100)

    opt = torch.optim.Adam(
        [
            {"params": [alpha_m], "lr": float(lr_alpha)},
            {"params": [theta], "lr": float(lr_eta)},
        ]
    )

    best_state = None
    best_val = float("inf")
    stale = 0
    eps = 1e-6

    def _forward(am, th):
        eta = torch.softmax(th, dim=0)
        logits = torch.zeros(n, MOD_ORDER, dtype=torch.float32)
        reg = torch.zeros((), dtype=torch.float32)
        for m in range(n_k):
            logits = logits + (Ks_t[m] @ am[m].T)
            reg = reg + (1.0 / (eta[m] + eps)) * torch.sum(am[m] * (am[m] @ Ks_t[m]))
        return logits, eta, reg

    for ep in range(phase1):
        opt.zero_grad()
        logits, eta, reg = _forward(alpha_m, theta)
        ce = torch.nn.functional.cross_entropy(logits[tr], labels_t[tr])
        ent = -torch.sum(eta * torch.log(eta + eps))
        loss = ce + float(lam) * reg - float(ent_reg) * ent
        loss.backward()
        opt.step()

        with torch.no_grad():
            logits_v, _, _ = _forward(alpha_m, theta)
            val_ce = float(
                torch.nn.functional.cross_entropy(logits_v[va], labels_t[va])
            )
        if val_ce < best_val - 1e-5:
            best_val = val_ce
            best_state = (
                alpha_m.detach().cpu().numpy().copy(),
                theta.detach().cpu().numpy().copy(),
            )
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        with torch.no_grad():
            alpha_m.copy_(torch.from_numpy(best_state[0]))
            theta.copy_(torch.from_numpy(best_state[1]))

    # 阶段2：固定 η，只精修分核 α_m
    theta.requires_grad_(False)
    opt2 = torch.optim.Adam([alpha_m], lr=float(lr_alpha) * 0.7)
    stale = 0
    for _ep in range(phase2):
        opt2.zero_grad()
        logits, eta, reg = _forward(alpha_m, theta)
        ce = torch.nn.functional.cross_entropy(logits[tr], labels_t[tr])
        loss = ce + float(lam) * reg
        loss.backward()
        opt2.step()
        with torch.no_grad():
            logits_v, _, _ = _forward(alpha_m, theta)
            val_ce = float(
                torch.nn.functional.cross_entropy(logits_v[va], labels_t[va])
            )
        if val_ce < best_val - 1e-5:
            best_val = val_ce
            best_state = (
                alpha_m.detach().cpu().numpy().copy(),
                theta.detach().cpu().numpy().copy(),
            )
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        am_np, th_np = best_state
    else:
        am_np = alpha_m.detach().cpu().numpy()
        th_np = theta.detach().cpu().numpy()

    eta_np = np.exp(th_np - th_np.max())
    eta_np = eta_np / eta_np.sum()

    # 可选：固定 η 后对拼接 α_m 做有限 L-BFGS（多核目标）
    if lbfgs_maxiter > 0 and keep_multi_alpha:
        am_np = _refine_multi_alpha_lbfgs(
            am_np, Ks_np, eta_np, labels, float(lam), maxiter=min(int(lbfgs_maxiter), 400)
        )

    logits_sum = np.zeros((n, MOD_ORDER), dtype=np.float64)
    for m in range(n_k):
        logits_sum += Ks_np[m] @ am_np[m].T

    K_eta = combine_kernels(Ks_np, eta_np)
    if keep_multi_alpha:
        # 兼容旧接口：α_eff 仅作备份；主推理用 alpha_m
        try:
            alpha_eff = solve_alpha_from_logits(logits_sum, K_eta, float(lam))
        except np.linalg.LinAlgError:
            alpha_eff = am_np[int(np.argmax(eta_np))]
    else:
        try:
            alpha_eff = solve_alpha_from_logits(logits_sum, K_eta, float(lam))
        except np.linalg.LinAlgError:
            alpha_eff = am_np[int(np.argmax(eta_np))]
        if lbfgs_maxiter > 0:
            alpha_eff = _fit_adam(
                alpha_eff, K_eta, labels, float(lam), "softmax",
                epochs=min(600, adam_epochs), patience=80, verbose=False,
            )
            alpha_eff = _fit_lbfgs_limited_local(
                alpha_eff, K_eta, labels, float(lam), maxiter=int(lbfgs_maxiter)
            )
        logits_sum = K_eta @ alpha_eff.T

    f_tr = np.exp(logits_sum - logits_sum.max(axis=1, keepdims=True))
    f_tr /= f_tr.sum(axis=1, keepdims=True) + 1e-300
    stats = {
        "val_j_data": float(best_val),
        "train_ser": float(np.mean(np.argmax(f_tr, 1) != labels)),
        "train_j_data": float(softmax_ce_from_scores(f_tr, labels)),
        "eta": eta_np.copy(),
        "kernel_scales": scales.copy(),
        "n_kernels": float(n_k),
        "mode": "adaptive_mkl_multi" if keep_multi_alpha else "adaptive_mkl",
        "eta_entropy": float(-np.sum(eta_np * np.log(eta_np + 1e-12))),
        "alpha_m": am_np if keep_multi_alpha else None,
        "keep_multi_alpha": bool(keep_multi_alpha),
    }
    if verbose:
        top = ", ".join(f"{e:.3f}" for e in eta_np)
        print(
            f"  Adaptive-MKL η=[{top}] H={stats['eta_entropy']:.3f} "
            f"valCE={best_val:.4f} trSER={stats['train_ser']:.3f} "
            f"multi={keep_multi_alpha}"
        )
    return alpha_eff, eta_np, K_eta, stats


def _refine_multi_alpha_lbfgs(
    alpha_m: np.ndarray,
    Ks: list[np.ndarray],
    eta: np.ndarray,
    labels: np.ndarray,
    lam: float,
    *,
    maxiter: int = 300,
) -> np.ndarray:
    """固定 η，对分核 α_m 做有限步 L-BFGS，目标=J_hard+多核正则。"""
    n_k, C, n = alpha_m.shape
    labels = np.asarray(labels, dtype=np.int64)
    eta = np.asarray(eta, dtype=np.float64)
    eta = np.maximum(eta, 1e-8)
    eta = eta / eta.sum()

    def pack(am: np.ndarray) -> np.ndarray:
        return am.ravel()

    def unpack(x: np.ndarray) -> np.ndarray:
        return x.reshape(n_k, C, n)

    def fun(x: np.ndarray) -> float:
        am = unpack(x)
        logits = np.zeros((n, C), dtype=np.float64)
        reg = 0.0
        for m in range(n_k):
            logits += Ks[m] @ am[m].T
            for c in range(C):
                reg += (1.0 / eta[m]) * float(am[m, c] @ Ks[m] @ am[m, c])
        g = np.clip(logits, -40, 40)
        z = g.max(1, keepdims=True)
        f = np.exp(g - z)
        f /= f.sum(1, keepdims=True) + 1e-300
        ce = -np.mean(np.log(f[np.arange(n), labels] + 1e-300))
        return float(ce + float(lam) * reg)

    def jac(x: np.ndarray) -> np.ndarray:
        am = unpack(x)
        logits = np.zeros((n, C), dtype=np.float64)
        for m in range(n_k):
            logits += Ks[m] @ am[m].T
        g = np.clip(logits, -40, 40)
        z = g.max(1, keepdims=True)
        e = np.exp(g - z)
        f = e / (e.sum(1, keepdims=True) + 1e-300)
        dg = f.copy()
        dg[np.arange(n), labels] -= 1.0
        dg /= n
        grad = np.zeros_like(am)
        for m in range(n_k):
            # ∂CE/∂α_m = dg^T @ K_m ; reg: 2/η_m K_m α
            grad[m] = dg.T @ Ks[m]
            for c in range(C):
                grad[m, c] += 2.0 * float(lam) / eta[m] * (Ks[m] @ am[m, c])
        return grad.ravel()

    res = minimize(
        fun,
        pack(alpha_m),
        method="L-BFGS-B",
        jac=jac,
        options={
            "maxiter": int(maxiter),
            "maxfun": int(max(800, maxiter * 20)),
            "ftol": 1e-11,
            "gtol": 1e-7,
        },
    )
    return unpack(res.x)


def _fit_lbfgs_limited_local(
    alpha_init: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    lam: float,
    *,
    maxiter: int = 1200,
) -> np.ndarray:
    def fun(x: np.ndarray) -> float:
        v, _ = _loss_grad_numpy(x, K, labels, lam, "softmax", margin_mu=0.0, log_tau=None)
        return float(v)

    def jac(x: np.ndarray) -> np.ndarray:
        _, g = _loss_grad_numpy(x, K, labels, lam, "softmax", margin_mu=0.0, log_tau=None)
        return g

    res = minimize(
        fun,
        alpha_init.ravel(),
        method="L-BFGS-B",
        jac=jac,
        options={
            "maxiter": int(maxiter),
            "maxfun": int(max(2000, maxiter * 30)),
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )
    return res.x.reshape(MOD_ORDER, -1)

def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return np.log1p(np.exp(x))


def _softplus_grad(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logits(K: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """g: (n, 16)。"""
    return K @ alpha.T


def log_margin_rows(g: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """逐样本 log(f_label / sum_b f_b)，softmax logits。"""
    g = np.clip(g, -40.0, 40.0)
    z = g.max(axis=1, keepdims=True)
    e = np.exp(g - z)
    p = e / (e.sum(axis=1, keepdims=True) + 1e-300)
    return np.log(p[np.arange(len(labels)), labels] + 1e-300)


def project_logits_margin(g: np.ndarray, label: int, tau: float) -> np.ndarray:
    """投影 logit：使 p_label >= tau（最小改动：只抬 label 维）。"""
    g = np.clip(g.copy(), -40.0, 40.0)
    z = float(g.max())
    e = np.exp(g - z)
    p = e / (e.sum() + 1e-300)
    if p[label] >= tau - 1e-12:
        return g
    g[label] += float(np.log(tau / max(p[label], 1e-300)))
    return np.clip(g, -40.0, 40.0)


def solve_alpha_from_logits(
    G: np.ndarray,
    K: np.ndarray,
    lam: float,
) -> np.ndarray:
    """给定目标 logits G (n,16)，Ridge 解 α 使 K@αᵀ≈G。"""
    n = K.shape[0]
    reg = K + (2.0 * lam * n) * np.eye(n, dtype=np.float64)
    alpha = np.empty((MOD_ORDER, n), dtype=np.float64)
    for a in range(MOD_ORDER):
        alpha[a] = np.linalg.solve(reg, G[:, a])
    return alpha


def apsm_margin_project_global(
    alpha: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    lam: float,
    tau: float,
    *,
    n_sweeps: int = 1,
    log_tau: float | None = None,
) -> np.ndarray:
    """全局 APSM：逐样本投影 logit 后 Ridge 解整表 α。"""
    log_tau = float(np.log(tau)) if log_tau is None else log_tau
    labels = labels.astype(np.int64)
    alpha = np.asarray(alpha, dtype=np.float64).copy()
    for _ in range(max(1, n_sweeps)):
        G = _logits(K, alpha)
        for n, a in enumerate(labels):
            if log_margin_rows(G[n : n + 1], labels[n : n + 1])[0] < log_tau:
                G[n] = project_logits_margin(G[n], int(a), tau)
        alpha = solve_alpha_from_logits(G, K, lam)
    return alpha


def apsm_margin_project_local(
    alpha: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    tau: float,
    *,
    n_sweeps: int = 2,
    column_lr: float = 0.12,
    log_tau: float | None = None,
) -> np.ndarray:
    """
    局部 APSM：仅更新违反 margin 的锚点列 α[:,n]，避免全局解 α 冲掉其它样本约束。
    """
    log_tau = float(np.log(tau)) if log_tau is None else log_tau
    labels = labels.astype(np.int64)
    alpha = np.asarray(alpha, dtype=np.float64).copy()
    n = K.shape[0]
    for _ in range(max(1, n_sweeps)):
        G = _logits(K, alpha)
        viol = np.where(log_margin_rows(G, labels) < log_tau)[0]
        if viol.size == 0:
            break
        for idx in viol:
            gn = G[idx]
            gp = project_logits_margin(gn, int(labels[idx]), tau)
            diff = gp - gn
            knn = max(float(K[idx, idx]), 1e-6)
            alpha[:, idx] += column_lr * diff / knn
        G = _logits(K, alpha)
    return alpha


def apsm_margin_project(
    alpha: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    lam: float,
    tau: float,
    *,
    mode: str = "local",
    n_sweeps: int = 1,
    column_lr: float = 0.12,
) -> np.ndarray:
    if mode == "global":
        return apsm_margin_project_global(
            alpha, K, labels, lam, tau, n_sweeps=n_sweeps
        )
    if mode == "local":
        return apsm_margin_project_local(
            alpha, K, labels, tau, n_sweeps=n_sweeps, column_lr=column_lr
        )
    raise ValueError(f"未知 apsm_mode={mode!r}")


def anneal_margin_tau(
    tau_start: float, tau_end: float, step: int, total_steps: int
) -> float:
    """线性退火 τ：前期易满足交集，后期逼近目标 τ。"""
    if total_steps <= 1:
        return float(tau_end)
    t = float(np.clip(step / max(total_steps - 1, 1), 0.0, 1.0))
    return float(tau_start + (tau_end - tau_start) * t)


def _scores_from_logits(g: np.ndarray, output_mode: str) -> np.ndarray:
    if output_mode == "softmax":
        g = np.clip(g, -40.0, 40.0)
        z = g.max(axis=1, keepdims=True)
        e = np.exp(g - z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-300)
    if output_mode == "softplus":
        return _softplus(g)
    raise ValueError(f"未知 output_mode={output_mode!r}")


def _softmax_margin_shortfall(
    g: np.ndarray, labels: np.ndarray, log_tau: float
) -> tuple[float, np.ndarray]:
    """mean relu(log τ - log margin)；及对 g 的梯度 (n,16)。"""
    g = np.clip(g, -40.0, 40.0)
    z = g.max(axis=1, keepdims=True)
    e = np.exp(g - z)
    f = e / (e.sum(axis=1, keepdims=True) + 1e-300)
    lm = np.log(f[np.arange(len(labels)), labels] + 1e-300)
    gap = np.maximum(log_tau - lm, 0.0)
    loss = float(gap.mean())
    if loss == 0.0:
        return 0.0, np.zeros_like(g)
    mask = (gap > 0).astype(np.float64)
    n = len(labels)
    dg = f.copy()
    n_idx = np.arange(n)
    a_star = labels
    dg[n_idx, a_star] -= 1.0
    dg = (dg * mask[:, None]) / n
    return loss, dg


def _loss_grad_numpy(
    alpha_flat: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    lam: float,
    output_mode: str,
    *,
    margin_mu: float = 0.0,
    log_tau: float | None = None,
) -> tuple[float, np.ndarray]:
    n = K.shape[0]
    alpha = alpha_flat.reshape(MOD_ORDER, n)
    g = _logits(K, alpha)
    n_idx = np.arange(n)
    a_star = labels

    if output_mode == "softmax":
        g_clip = np.clip(g, -40.0, 40.0)
        z = g_clip.max(axis=1, keepdims=True)
        e = np.exp(g_clip - z)
        f = e / (e.sum(axis=1, keepdims=True) + 1e-300)
        log_z = np.log(e.sum(axis=1)) + z.ravel()
        ce = float(np.mean(log_z - g_clip[n_idx, a_star]))
        dg = f.copy()
        dg[n_idx, a_star] -= 1.0
        dg /= n
        grad_alpha = dg.T @ K + 2.0 * lam * (K @ alpha.T).T
        if margin_mu > 0.0 and log_tau is not None:
            m_loss, dm = _softmax_margin_shortfall(g_clip, labels, log_tau)
            ce += margin_mu * m_loss
            grad_alpha += margin_mu * (dm.T @ K)
    elif output_mode == "softplus":
        f = _softplus(g)
        S = f.sum(axis=1) + 1e-300
        log_sum = np.log(S)
        log_fs1 = np.log(f[n_idx, a_star] + 1e-300)
        ce = float(np.mean(log_sum - log_fs1))
        spg = _softplus_grad(g)
        dg = spg / S[:, None]
        dg[n_idx, a_star] -= spg[n_idx, a_star] / (f[n_idx, a_star] + 1e-300)
        dg /= n
        grad_alpha = dg.T @ K + 2.0 * lam * (K @ alpha.T).T
    else:
        raise ValueError(f"未知 output_mode={output_mode!r}")

    reg = lam * sum(alpha[a] @ K @ alpha[a] for a in range(MOD_ORDER))
    return ce + reg, grad_alpha.ravel()


def _loss_torch(
    alpha: torch.Tensor,
    K: torch.Tensor,
    labels: torch.Tensor,
    lam: float,
    output_mode: str,
    *,
    margin_mu: float = 0.0,
    margin_tau: float | None = None,
) -> torch.Tensor:
    g = K @ alpha.T
    if output_mode == "softmax":
        log_z = torch.logsumexp(g, dim=1)
        ce = (log_z - g[torch.arange(len(labels)), labels]).mean()
        if margin_mu > 0.0 and margin_tau is not None:
            log_margin = g[torch.arange(len(labels)), labels] - log_z
            ce = ce + margin_mu * torch.relu(
                torch.log(torch.tensor(margin_tau, dtype=g.dtype, device=g.device))
                - log_margin
            ).mean()
    elif output_mode == "softplus":
        f = F.softplus(g)
        log_sum = torch.log(f.sum(1) + 1e-300)
        log_fs1 = torch.log(f[torch.arange(len(labels)), labels] + 1e-300)
        ce = (log_sum - log_fs1).mean()
    else:
        raise ValueError(f"未知 output_mode={output_mode!r}")
    reg = lam * sum(alpha[a] @ K @ alpha[a] for a in range(MOD_ORDER))
    return ce + reg


def _fit_adam(
    alpha_init: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    lam: float,
    output_mode: str,
    *,
    epochs: int = 2500,
    lr: float = 0.08,
    patience: int = 120,
    verbose: bool = False,
    log_every: int = 200,
    use_apspm: bool = False,
    margin_tau: float = 0.7,
    margin_tau_start: float = 0.2,
    margin_tau_end: float = 0.7,
    margin_mu: float = 0.0,
    use_margin_loss: bool = False,
    apsm_mode: str = "local",
    apsm_every: int = 150,
    apsm_sweeps: int = 2,
    apsm_column_lr: float = 0.12,
) -> np.ndarray:
    n = K.shape[0]
    K_t = torch.from_numpy(K)
    lab_t = torch.from_numpy(labels.astype(np.int64))
    alpha = torch.tensor(alpha_init, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([alpha], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=40, min_lr=1e-5
    )

    best_loss = float("inf")
    best = alpha_init.copy()
    stale = 0

    mu = margin_mu if use_margin_loss else 0.0
    for ep in range(epochs):
        tau_ep = anneal_margin_tau(
            margin_tau_start, margin_tau_end, ep, epochs
        )
        opt.zero_grad()
        loss = _loss_torch(
            alpha,
            K_t,
            lab_t,
            lam,
            output_mode,
            margin_mu=mu,
            margin_tau=tau_ep if mu > 0 else None,
        )
        loss.backward()
        opt.step()
        sched.step(loss.detach())
        lv = float(loss.detach())
        if lv < best_loss - 1e-8:
            best_loss = lv
            best = alpha.detach().numpy().copy()
            stale = 0
        else:
            stale += 1
        if (
            use_apspm
            and output_mode == "softmax"
            and apsm_every > 0
            and (ep + 1) % apsm_every == 0
        ):
            with torch.no_grad():
                a_np = alpha.detach().numpy()
                a_np = apsm_margin_project(
                    a_np,
                    K,
                    labels,
                    lam,
                    tau_ep,
                    mode=apsm_mode,
                    n_sweeps=apsm_sweeps,
                    column_lr=apsm_column_lr,
                )
                alpha.data.copy_(torch.from_numpy(a_np))
        if verbose and (ep == 0 or (ep + 1) % log_every == 0):
            print(f"      Adam ep {ep + 1}/{epochs}, loss={lv:.6f}", flush=True)
        if stale >= patience:
            if verbose:
                print(f"    Adam 早停 @ ep={ep}, loss={best_loss:.6f}")
            break
    # 勿用最后一轮权重覆盖早停期间记录的 best（否则高 SNR 易掉进坏局部解）
    if use_apspm and output_mode == "softmax":
        best = apsm_margin_project(
            best,
            K,
            labels,
            lam,
            margin_tau_end,
            mode=apsm_mode,
            n_sweeps=max(2, apsm_sweeps),
            column_lr=apsm_column_lr,
        )
    return best


def _fit_lbfgs(
    alpha_init: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    lam: float,
    output_mode: str,
    *,
    verbose: bool = False,
) -> np.ndarray:
    def fun(x):
        v, _ = _loss_grad_numpy(
            x, K, labels, lam, output_mode, margin_mu=0.0, log_tau=None
        )
        return v

    def jac(x):
        _, g = _loss_grad_numpy(
            x, K, labels, lam, output_mode, margin_mu=0.0, log_tau=None
        )
        return g

    res = minimize(
        fun,
        alpha_init.ravel(),
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": 3000, "maxfun": 80000, "ftol": 1e-15, "gtol": 1e-10},
    )
    if verbose:
        print(f"    L-BFGS-B: success={res.success}, nit={res.nit}, fun={res.fun:.6f}")
    return res.x.reshape(MOD_ORDER, -1)


def _label_diagonal_init(labels: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """α_{label(j), j} 偏大，给优化一个合理起点。"""
    n = len(labels)
    alpha = np.zeros((MOD_ORDER, n), dtype=np.float64)
    for j, a in enumerate(labels):
        alpha[int(a), j] = strength
    return alpha


def alpha_diagnostics(alpha: np.ndarray | None) -> dict[str, float]:
    """检查 RKHS 是否学到非平凡解（避免 α≡0 或近零）。"""
    if alpha is None:
        return {
            "alpha_fro": 0.0,
            "alpha_max": 0.0,
            "alpha_active_frac": 0.0,
        }
    aa = np.asarray(alpha, dtype=np.float64)
    return {
        "alpha_fro": float(np.linalg.norm(aa, ord="fro")),
        "alpha_max": float(np.max(np.abs(aa))),
        "alpha_active_frac": float(np.mean(np.abs(aa) > 1e-4)),
    }


class RKHSDetector:
    def __init__(
        self,
        gamma: float | None = None,
        lam: float | None = None,
        *,
        lam_c: float = 0.1,
        kernel_mode: str = "single",
        ms_ratios: tuple[float, ...] = DEFAULT_MULTISCALE_RATIOS,
        output_mode: str = "softmax",
        use_apspm: bool = False,
        margin_tau: float = 0.7,
        margin_tau_start: float = 0.3,
        margin_tau_end: float | None = None,
        margin_mu: float = 0.4,
        use_margin_loss: bool | None = None,
        apsm_mode: str = "local",
        apsm_every: int = 150,
        apsm_sweeps: int = 2,
        apsm_column_lr: float = 0.12,
        gamma_mode: str = "noise",
        tune_hyperparams: bool = False,
        fast_tune: bool = False,
        n_restarts: int = 3,
        adam_epochs: int = 2500,
        tune_adam_epochs: int = 1500,
    ):
        self.gamma = gamma
        self.lam = lam
        self.lam_c = lam_c
        if output_mode not in ("softplus", "softmax"):
            raise ValueError("output_mode 须为 'softplus' 或 'softmax'")
        if kernel_mode not in ("single", "multiscale"):
            raise ValueError("kernel_mode 须为 'single' 或 'multiscale'")
        if gamma_mode not in ("noise", "median", "tune", "noise_tune", "theory"):
            raise ValueError("gamma_mode 须为 noise / median / tune / noise_tune / theory")
        self.output_mode = output_mode
        self.kernel_mode = kernel_mode
        self.ms_ratios = tuple(ms_ratios)
        if apsm_mode not in ("local", "global"):
            raise ValueError("apsm_mode 须为 'local' 或 'global'")
        self.use_apspm = use_apspm
        self.margin_tau = float(margin_tau)
        self.margin_tau_start = float(margin_tau_start)
        self.margin_tau_end = float(
            margin_tau if margin_tau_end is None else margin_tau_end
        )
        self.margin_mu = float(margin_mu)
        self.use_margin_loss = (
            use_apspm if use_margin_loss is None else bool(use_margin_loss)
        )
        self.apsm_mode = apsm_mode
        self.apsm_every = int(apsm_every)
        self.apsm_sweeps = int(apsm_sweeps)
        self.apsm_column_lr = float(apsm_column_lr)
        self.gamma_mode = gamma_mode
        self.tune_hyperparams = tune_hyperparams
        self.fast_tune = fast_tune
        self.n_restarts = max(1, n_restarts)
        self.adam_epochs = adam_epochs
        self.tune_adam_epochs = tune_adam_epochs

        self.alpha: np.ndarray | None = None
        self.Y_train_feat: np.ndarray | None = None
        self.K_train: np.ndarray | None = None
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self._lam_fitted: float = lam if lam is not None else lam_c
        self.last_fit_stats: dict[str, float] = {}

    def _resolve_lam(self, n: int, n0: float | None = None) -> float:
        """λ = c/n 或理论 c·N₀/n；若构造时传入 lam 则固定用 lam。"""
        if self.lam is not None:
            return self.lam
        if self.gamma_mode == "theory" and n0 is not None:
            return lam_theory_rkhs(n0, n, c=self.lam_c)
        return self.lam_c / n

    def _build_K(
        self, X: np.ndarray, Y: np.ndarray | None = None, *, gamma: float | None = None
    ) -> np.ndarray:
        g = gamma if gamma is not None else self.gamma
        assert g is not None
        return build_kernel_matrix(
            X, g, Y, kernel_mode=self.kernel_mode, ms_ratios=self.ms_ratios
        )

    def _normalize_fit(self, X: np.ndarray) -> np.ndarray:
        self.feat_mean = X.mean(axis=0)
        self.feat_std = X.std(axis=0)
        self.feat_std[self.feat_std < 1e-8] = 1.0
        return (X - self.feat_mean) / self.feat_std

    def _normalize_apply(self, X: np.ndarray) -> np.ndarray:
        assert self.feat_mean is not None and self.feat_std is not None
        return (X - self.feat_mean) / self.feat_std

    def _default_gamma(self, X: np.ndarray) -> float:
        bw = median_bandwidth(X)
        return 1.0 / (2.0 * bw * bw)

    def _gamma_candidates(
        self, X: np.ndarray, snr_db: float | None = None
    ) -> list[float]:
        if self.gamma is not None:
            return [self.gamma]
        if self.gamma_mode in ("noise", "noise_tune") and snr_db is not None:
            from system import n0_from_snr_db

            g0 = gamma_rkhs_from_n0(n0_from_snr_db(float(snr_db)), X)
        else:
            g0 = self._default_gamma(X)
        if self.gamma_mode == "noise_tune" and snr_db is not None and float(snr_db) >= 8.0:
            scales = (0.35, 0.5, 1.0, 1.5, 2.0) if self.fast_tune else (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
        elif self.fast_tune:
            scales = (0.5, 1.0, 2.0)
        else:
            scales = (0.25, 0.5, 1.0, 2.0, 4.0)
        return [float(g0 * s) for s in scales]

    def _tune_gamma_on_val(
        self,
        X_tr: np.ndarray,
        s1_tr: np.ndarray,
        X_val: np.ndarray,
        s1_val: np.ndarray,
        lam: float,
        verbose: bool,
        snr_db: float | None = None,
    ) -> float:
        """固定 λ=c/n，仅在验证集上选 γ。"""
        best_g, best_val = None, float("inf")
        for gamma in self._gamma_candidates(X_tr, snr_db=snr_db):
            K_tr = self._build_K(X_tr, gamma=gamma)
            K_vt = self._build_K(X_val, X_tr, gamma=gamma)
            alpha, _ = self._fit_once(
                K_tr,
                s1_tr,
                lam,
                verbose=False,
                adam_epochs=self.tune_adam_epochs,
                do_lbfgs=not self.fast_tune,
                n_restarts=1 if self.fast_tune else self.n_restarts,
                K_val=K_vt,
                s1_val=s1_val,
            )
            f_val = _scores_from_logits(_logits(K_vt, alpha), self.output_mode)
            val_ce = softmax_ce_from_scores(f_val, s1_val)
            if val_ce < best_val:
                best_val = val_ce
                best_g = gamma
        if verbose and best_g is not None:
            print(f"  验证集选 γ={best_g:.4e}（λ={lam:.2e}=c/n 固定）, val CE={best_val:.4f}")
        return best_g  # type: ignore[return-value]

    def _featurize(self, y: np.ndarray) -> np.ndarray:
        return y_to_features(np.atleast_2d(y))

    def _resolve_gamma(
        self,
        X: np.ndarray,
        y_train: np.ndarray,
        snr_db: float | None,
        verbose: bool,
    ) -> float:
        if self.gamma is not None:
            return self.gamma
        from system import n0_from_snr_db

        if snr_db is not None:
            n0 = n0_from_snr_db(float(snr_db))
            n0_src = f"SNR={snr_db:.1f} dB"
        else:
            n0 = estimate_n0_from_y(y_train)
            n0_src = "y 样本方差"

        if self.gamma_mode in ("noise", "theory"):
            if self.gamma_mode == "theory":
                gamma = gamma_theory_rkhs(n0, X)
                gamma_note = "1/(2·bw²)·√(N₀/N₀_ref)"
            else:
                gamma = gamma_rkhs_from_n0(n0, X)
                gamma_note = "1/(2·bw²)·√(N₀_ref/N₀)"
            if verbose:
                bw = median_bandwidth(X)
                if self.kernel_mode == "multiscale":
                    g_list = [gamma * r for r in self.ms_ratios]
                    print(
                        f"  γ_base={gamma:.4e} ← {gamma_note}, "
                        f"bw={bw:.3f}, N₀={n0:.2e} ({n0_src}) | "
                        f"多尺度 γ∈{[f'{g:.2e}' for g in g_list]}"
                    )
                else:
                    print(
                        f"  γ={gamma:.4e} ← {gamma_note}, "
                        f"bw={bw:.3f}, N₀={n0:.2e} ({n0_src})"
                    )
            return gamma
        if self.gamma_mode == "median":
            gamma = self._default_gamma(X)
            if verbose:
                print(f"  γ={gamma:.4e} ← median bandwidth")
            return gamma
        if self.gamma_mode == "noise_tune":
            raise ValueError("noise_tune 须配合 tune_hyperparams=True")
        raise ValueError(f"未知 gamma_mode={self.gamma_mode!r}")

    def _fit_once(
        self,
        K: np.ndarray,
        labels: np.ndarray,
        lam: float,
        *,
        verbose: bool,
        adam_epochs: int | None = None,
        do_lbfgs: bool = True,
        K_val: np.ndarray | None = None,
        s1_val: np.ndarray | None = None,
        grad_tol: float = 0.02,
        n_restarts: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """多起点 Adam→L-BFGS；有验证核 K_val 时按验证 CE 选最优（缓解坏局部极小）。"""
        adam_epochs = adam_epochs if adam_epochs is not None else self.adam_epochs
        n_restart_use = self.n_restarts if n_restarts is None else max(1, n_restarts)
        n = K.shape[0]
        rng_init = np.random.default_rng(1)
        inits = [
            ("diag", _label_diagonal_init(labels)),
            ("rand", 0.08 * rng_init.standard_normal((MOD_ORDER, n))),
            ("rand+", 0.15 * rng_init.standard_normal((MOD_ORDER, n))),
        ][: n_restart_use]

        best_alpha, best_score = None, float("inf")
        fallback_alpha, fallback_score = None, float("inf")
        for i, (name, a0) in enumerate(inits):
            if verbose:
                print(f"    起点 {i + 1}/{len(inits)} ({name}): Adam…", flush=True)
            a_adam = _fit_adam(
                a0,
                K,
                labels,
                lam,
                self.output_mode,
                epochs=adam_epochs,
                verbose=verbose,
                log_every=200,
                use_apspm=self.use_apspm,
                margin_tau_start=self.margin_tau_start,
                margin_tau_end=self.margin_tau_end,
                margin_mu=self.margin_mu,
                use_margin_loss=self.use_margin_loss,
                apsm_mode=self.apsm_mode,
                apsm_every=self.apsm_every,
                apsm_sweeps=self.apsm_sweeps,
                apsm_column_lr=self.apsm_column_lr,
            )
            a_final = a_adam
            if self.use_apspm and self.output_mode == "softmax":
                a_final = apsm_margin_project(
                    a_final,
                    K,
                    labels,
                    lam,
                    self.margin_tau_end,
                    mode=self.apsm_mode,
                    n_sweeps=self.apsm_sweeps,
                    column_lr=self.apsm_column_lr,
                )
            if do_lbfgs:
                if verbose:
                    print(f"    起点 {i + 1}/{len(inits)} ({name}): L-BFGS-B…", flush=True)
                a_final = _fit_lbfgs(
                    a_final, K, labels, lam, self.output_mode, verbose=verbose
                )
                if self.use_apspm and self.output_mode == "softmax":
                    a_final = apsm_margin_project(
                        a_final,
                        K,
                        labels,
                        lam,
                        self.margin_tau_end,
                        mode=self.apsm_mode,
                        n_sweeps=self.apsm_sweeps,
                        column_lr=self.apsm_column_lr,
                    )
            loss, grad = _loss_grad_numpy(
                a_final.ravel(), K, labels, lam, self.output_mode
            )
            gnorm = float(np.linalg.norm(grad))
            if K_val is not None and s1_val is not None:
                f_val = _scores_from_logits(_logits(K_val, a_final), self.output_mode)
                pick_score = softmax_ce_from_scores(f_val, s1_val)
                pick_name = "val CE"
            else:
                pick_score = loss
                pick_name = "J_full"
            if pick_score < fallback_score:
                fallback_score = pick_score
                fallback_alpha = a_final
            diag = alpha_diagnostics(a_final)
            if diag["alpha_fro"] < 1e-3:
                if verbose:
                    print(
                        f"    起点 {i + 1} ({name}): α 近零 (||α||={diag['alpha_fro']:.2e}), 丢弃"
                    )
                continue
            if gnorm > grad_tol:
                if verbose:
                    print(f"    起点 {i + 1} ({name}): 未收敛 ||g||={gnorm:.2e}, 丢弃")
                continue
            if pick_score < best_score:
                best_score = pick_score
                best_alpha = a_final
            if verbose:
                print(
                    f"    起点 {i + 1} ({name}): J_full={loss:.4f}, ||g||={gnorm:.2e}, "
                    f"{pick_name}={pick_score:.4f}"
                )

        if best_alpha is None:
            if verbose:
                print(
                    f"    警告: 无起点满足梯度阈值，退回验证 CE 最优解 "
                    f"(score={fallback_score:.4f})"
                )
            best_alpha = fallback_alpha
            best_score = fallback_score
        return best_alpha, best_score

    def fit(
        self,
        y_train: np.ndarray,
        s1_train: np.ndarray,
        *,
        verbose: bool = True,
        val_frac: float = 0.15,
        snr_db: float | None = None,
        adam_epochs: int | None = None,
        do_lbfgs: bool = True,
        n_restarts: int | None = None,
    ) -> float:
        X_raw = self._featurize(y_train)
        X = self._normalize_fit(X_raw)
        labels = s1_train.astype(np.int64)
        n = X.shape[0]
        from system import n0_from_snr_db

        n0_fit = (
            n0_from_snr_db(float(snr_db))
            if snr_db is not None
            else estimate_n0_from_y(y_train)
        )
        lam = self._resolve_lam(n, n0=n0_fit)

        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        n_val = max(20, int(n * val_frac))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr, s1_tr = X[tr_idx], labels[tr_idx]
        X_val, s1_val = X[val_idx], labels[val_idx]

        if (
            self.gamma_mode in ("tune", "noise_tune")
            and self.tune_hyperparams
            and self.gamma is None
        ):
            gamma = self._tune_gamma_on_val(
                X_tr, s1_tr, X_val, s1_val, lam, verbose, snr_db=snr_db
            )
        else:
            gamma = self._resolve_gamma(X, y_train, snr_db, verbose)

        self.gamma = gamma
        self._lam_fitted = lam
        if verbose:
            km = self.kernel_mode
            print(f"  构建核矩阵 K ({n}×{n}, {km})…", flush=True)
        K = self._build_K(X, gamma=gamma)
        self.K_train = K
        self.Y_train_feat = X

        K_val = self._build_K(X_val, X, gamma=gamma)
        if verbose:
            if self.use_apspm:
                apsm_s = (
                    f", APSM {self.apsm_mode} τ={self.margin_tau_start}"
                    f"→{self.margin_tau_end}, μ={self.margin_mu}"
                )
            elif self.use_margin_loss:
                apsm_s = f", margin μ={self.margin_mu}, τ→{self.margin_tau_end}"
            else:
                apsm_s = ""
            print(
                f"  最终训练: n={n}, kernel={self.kernel_mode}, γ_base={gamma:.4e}, "
                f"λ={lam:.2e}, output={self.output_mode}{apspm_s}"
            )

        alpha, train_loss = self._fit_once(
            K,
            labels,
            lam,
            verbose=verbose,
            K_val=K_val,
            s1_val=s1_val,
            adam_epochs=adam_epochs,
            do_lbfgs=do_lbfgs,
            n_restarts=n_restarts,
        )
        self.alpha = alpha

        f_final = self._scores_matrix(y_train)
        train_ce = softmax_ce_from_scores(f_final, s1_train)
        est_tr = np.argmax(f_final, axis=1)
        train_ser = float(np.mean(est_tr != s1_train))
        diag = alpha_diagnostics(self.alpha)
        g_tr = _logits(self.K_train, self.alpha)
        lm_tr = log_margin_rows(g_tr, labels)
        frac_ok = float(np.mean(lm_tr >= np.log(self.margin_tau_end)))
        self.last_fit_stats = {
            "gamma": float(gamma),
            "lam": float(lam),
            "output_mode": self.output_mode,
            "kernel_mode": self.kernel_mode,
            "use_apspm": self.use_apspm,
            "margin_tau": self.margin_tau,
            "train_log_margin_mean": float(lm_tr.mean()),
            "train_margin_frac_ge_tau": frac_ok,
            "train_j_data": float(train_ce),
            "train_j_full": float(train_loss),
            **diag,
            "train_ser": train_ser,
        }
        if verbose:
            print(
                f"  训练 J_full={train_loss:.6f}, J_data={train_ce:.6f}, "
                f"SER={train_ser:.4f}, ||α||_F={diag['alpha_fro']:.3f}"
            )
        return train_ce

    def _scores_matrix(self, y: np.ndarray) -> np.ndarray:
        assert self.alpha is not None and self.gamma is not None
        X = self._normalize_apply(self._featurize(y))
        K = self._build_K(X, self.Y_train_feat)
        g = _logits(K, self.alpha)
        return _scores_from_logits(g, self.output_mode)

    def scores(self, y: np.ndarray) -> np.ndarray:
        return self._scores_matrix(y)

    def detect(self, y: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(y), axis=-1)
