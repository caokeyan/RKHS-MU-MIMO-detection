"""
残差自适应多核检测：前端软分（MMSE / PIC / GA）+ Adaptive-MKL 残差。

final_logits = base_logits + K_η(φ) α^T

- 前端 mmse：导频 Ĥ 上线性 MMSE 软分
- 前端 pic：软并行干扰消除后再对 X₁ 软分（默认；通常强于纯 MMSE）
- 前端 ga：真 H 上高斯干扰软 MLD（上界对照）
- 前端 auto：在 mmse/pic/ga_hat 中按验证集 SER 选 base

特征 φ：MMSE/PIC 均衡量 + 多前端 logits + 置信度，避免 2M 高维。
"""
from __future__ import annotations

import numpy as np

from kernel_rkhs import (
    ADAPTIVE_MKL_RATIOS,
    build_base_kernels,
    combine_kernels,
    gamma_theory_rkhs,
    lam_theory_rkhs,
)
from mld import GaussianMldCache, marginal_scores, precompute_mld_hy
from mmse import (
    estimate_n0_from_residual,
    generate_pilots,
    ls_estimate_heff,
    mmse_equalize,
)
from objective import softmax_ce_from_scores
from system import CONSTELLATION, MOD_ORDER, M, n0_from_snr_db


def estimate_heff_block(
    H_eff: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """块衰落：整块共用一次导频 LS → (Ĥ, N̂₀)。"""
    n0 = n0_from_snr_db(float(snr_db))
    X_p = generate_pilots()
    T = X_p.shape[1]
    std = np.sqrt(n0 / 2)
    noise = std * (
        rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))
    )
    Y_p = H_eff @ X_p + noise
    H_hat = ls_estimate_heff(Y_p, X_p)
    n0_hat = estimate_n0_from_residual(Y_p, H_hat, X_p)
    if n0_hat <= 0:
        n0_hat = n0
    return H_hat, float(n0_hat)


def mmse_x1_soft_logits(
    y: np.ndarray,
    H_hat: np.ndarray,
    n0_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MMSE 均衡后对 X_1 的星座软分：
    log f_a ∝ -|x̂₁ - s_a|² / (2 N̂₀)
    返回 logits (n,16) 与 x̂₁ (n,)。
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[None, :]
    x_hat = mmse_equalize(y, H_hat, float(n0_hat))
    x1 = x_hat[:, 0]
    d2 = np.abs(x1[:, None] - CONSTELLATION[None, :]) ** 2
    logits = -0.5 * d2 / max(float(n0_hat), 1e-12)
    return logits.astype(np.float64), x1


def ga_mld_soft_logits(
    y: np.ndarray,
    H_eff: np.ndarray,
    n0: float,
    hy_cache: np.ndarray | GaussianMldCache | None = None,
) -> np.ndarray:
    """高斯干扰 / 精确边际 MLD 对数软分 (n,16)。"""
    if hy_cache is None:
        hy_cache = precompute_mld_hy(H_eff)
    return marginal_scores(y, H_eff, n0, log_domain=True, hy_cache=hy_cache)


def _soft_symbol_mean(x_hat: np.ndarray, n0: float) -> np.ndarray:
    """对均衡输出做星座软均值 E[x|x̂]。"""
    d2 = np.abs(np.asarray(x_hat)[..., None] - CONSTELLATION) ** 2
    logits = -0.5 * d2 / max(float(n0), 1e-12)
    z = logits.max(axis=-1, keepdims=True)
    p = np.exp(logits - z)
    p /= p.sum(axis=-1, keepdims=True) + 1e-300
    return (p * CONSTELLATION).sum(axis=-1)


def pic_x1_soft_logits(
    y: np.ndarray,
    H_hat: np.ndarray,
    n0_hat: float,
    n_iter: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    软 PIC：反复估计干扰软均值并消除，再对 X₁ 匹配滤波软分。
    返回 logits (n,16) 与 x̂₁ (n,)。
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[None, :]
    n0 = float(n0_hat)
    H = np.asarray(H_hat)
    h1 = H[:, 0]
    nh = max(float(np.vdot(h1, h1).real), 1e-12)
    x = mmse_equalize(y, H, n0)
    x1 = x[:, 0]
    for _ in range(int(n_iter)):
        ex = _soft_symbol_mean(x, n0)
        y1 = y - ex[:, 1:] @ H[:, 1:].T
        x1 = (y1 @ np.conj(h1)) / nh
        y_o = y - np.outer(x1, h1)
        xo = mmse_equalize(y_o, H, n0)
        x = xo.copy()
        x[:, 0] = x1
    d2 = np.abs(x1[:, None] - CONSTELLATION[None, :]) ** 2
    logits = -0.5 * d2 / max(n0, 1e-12)
    return logits.astype(np.float64), x1


def _margin(logits: np.ndarray) -> np.ndarray:
    """top1 - top2 置信度。"""
    part = np.partition(logits, -2, axis=1)
    return (logits.max(axis=1) - part[:, -2])[:, None]


def _frontend_features_rich(
    x_mmse: np.ndarray,
    x_pic: np.ndarray,
    logits_mmse: np.ndarray,
    logits_pic: np.ndarray,
    logits_ga_hat: np.ndarray,
) -> np.ndarray:
    """多前端紧凑特征。"""
    return np.concatenate(
        [
            x_mmse.real[:, None],
            x_mmse.imag[:, None],
            x_pic.real[:, None],
            x_pic.imag[:, None],
            logits_mmse,
            logits_pic,
            logits_ga_hat,
            _margin(logits_mmse),
            _margin(logits_pic),
            _margin(logits_ga_hat),
        ],
        axis=1,
    ).astype(np.float64)


def _normalize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def _normalize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    g = np.clip(logits, -40.0, 40.0)
    z = g.max(axis=1, keepdims=True)
    e = np.exp(g - z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-300)


class ResidualAdaptiveMKLDetector:
    """
    final = base_frontend + Adaptive-MKL 残差。
    接口对齐 RKHSNNDetector：fit / scores / detect / last_fit_stats。
    """

    def __init__(
        self,
        *,
        frontend: str = "auto",
        lam_c: float = 0.1,
        ms_ratios: tuple[float, ...] = ADAPTIVE_MKL_RATIOS,
        resid_scale: float = 1.0,
        pic_iters: int = 4,
    ) -> None:
        if frontend not in ("mmse", "pic", "ga", "auto"):
            raise ValueError("frontend 须为 'mmse' / 'pic' / 'ga' / 'auto'")
        self.frontend = frontend
        self.lam_c = float(lam_c)
        self.ms_ratios = tuple(ms_ratios)
        self.resid_scale = float(resid_scale)
        self._active_resid_scale = float(resid_scale)
        self.pic_iters = int(pic_iters)
        self.base_frontend: str = "pic" if frontend == "auto" else frontend
        self.gamma: float | None = None
        self.lam: float | None = None
        self.eta: np.ndarray | None = None
        self.kernel_scales: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.X_centers: np.ndarray | None = None
        self.H_eff: np.ndarray | None = None
        self.H_hat: np.ndarray | None = None
        self.n0_hat: float | None = None
        self.hy_cache = None
        self.hy_hat_cache = None
        self.snr_db: float | None = None
        self.last_fit_stats: dict[str, float] = {}

    def _compute_frontends(
        self,
        y: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        reuse_hat: bool = True,
    ) -> dict[str, np.ndarray]:
        assert self.H_eff is not None and self.snr_db is not None
        n0 = n0_from_snr_db(self.snr_db)
        if (not reuse_hat) or self.H_hat is None or self.n0_hat is None:
            rng = rng or np.random.default_rng(0)
            self.H_hat, self.n0_hat = estimate_heff_block(self.H_eff, self.snr_db, rng)
            self.hy_hat_cache = precompute_mld_hy(self.H_hat)

        logits_m, x_m = mmse_x1_soft_logits(y, self.H_hat, self.n0_hat)
        logits_p, x_p = pic_x1_soft_logits(
            y, self.H_hat, self.n0_hat, n_iter=self.pic_iters
        )
        logits_gh = ga_mld_soft_logits(y, self.H_hat, self.n0_hat, self.hy_hat_cache)

        out: dict[str, np.ndarray] = {
            "mmse": logits_m,
            "pic": logits_p,
            "ga_hat": logits_gh,
            "x_mmse": x_m,
            "x_pic": x_p,
        }
        if self.frontend == "ga" or self.base_frontend == "ga":
            if self.hy_cache is None:
                self.hy_cache = precompute_mld_hy(self.H_eff)
            out["ga"] = ga_mld_soft_logits(y, self.H_eff, n0, self.hy_cache)
        return out

    def _base_and_feat(
        self,
        y: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        reuse_hat: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        packs = self._compute_frontends(y, rng=rng, reuse_hat=reuse_hat)
        name = self.base_frontend
        if name == "ga":
            base = packs["ga"]
        elif name == "mmse":
            base = packs["mmse"]
        elif name == "ga_hat":
            base = packs["ga_hat"]
        else:
            base = packs["pic"]
        feat = _frontend_features_rich(
            packs["x_mmse"],
            packs["x_pic"],
            packs["mmse"],
            packs["pic"],
            packs["ga_hat"],
        )
        return base, feat

    def fit(
        self,
        y_train: np.ndarray,
        s1_train: np.ndarray,
        *,
        H_eff: np.ndarray,
        snr_db: float,
        adam_epochs: int = 1200,
        lbfgs_maxiter: int = 1000,
        verbose: bool = False,
    ) -> float:
        self.H_eff = np.asarray(H_eff)
        self.snr_db = float(snr_db)
        rng = np.random.default_rng(0)
        n = len(y_train)
        n0 = n0_from_snr_db(self.snr_db)
        ylab = s1_train.astype(np.int64)

        packs = self._compute_frontends(y_train, rng=rng, reuse_hat=False)
        perm = rng.permutation(n)
        n_val = max(64, int(0.2 * n))
        va_idx = perm[:n_val]

        # auto：验证集选最优 base
        if self.frontend == "auto":
            cands = ("pic", "mmse", "ga_hat")
            best_name, best_ser = "pic", 1.0
            for name in cands:
                ser = float(np.mean(np.argmax(packs[name][va_idx], 1) != ylab[va_idx]))
                if ser < best_ser - 1e-12:
                    best_name, best_ser = name, ser
            self.base_frontend = best_name
        elif self.frontend == "ga":
            self.base_frontend = "ga"
            if "ga" not in packs:
                packs = self._compute_frontends(y_train, rng=rng, reuse_hat=True)
        else:
            self.base_frontend = self.frontend

        if self.base_frontend == "ga":
            base_t = packs["ga"]
        else:
            base_t = packs[self.base_frontend]

        feat_raw = _frontend_features_rich(
            packs["x_mmse"],
            packs["x_pic"],
            packs["mmse"],
            packs["pic"],
            packs["ga_hat"],
        )
        X, mean, std = _normalize_fit(feat_raw)
        self.feat_mean, self.feat_std = mean, std
        self.X_centers = X
        self.gamma = gamma_theory_rkhs(n0, X)
        lam_c_eff = self.lam_c * (
            2.5 if float(snr_db) >= 10.0 else (1.2 if float(snr_db) >= 8.0 else 0.8)
        )
        self.lam = lam_theory_rkhs(n0, n, c=lam_c_eff)

        ratios = self.ms_ratios
        if float(snr_db) >= 10.0:
            ratios = (0.25, 0.5, 1.0, 2.0)
        elif float(snr_db) >= 8.0:
            ratios = (0.15, 0.5, 1.0, 2.0)
        else:
            ratios = (0.25, 0.5, 1.0, 2.0, 4.0)
        self.ms_ratios = ratios

        self._active_resid_scale = float(self.resid_scale)
        front_ser_va = float(np.mean(np.argmax(base_t[va_idx], 1) != ylab[va_idx]))

        base_kernels_full = build_base_kernels(X, self.gamma, ms_ratios=ratios)
        alpha, eta, K, stats = _fit_residual_adaptive_mkl(
            base_kernels_full,
            ylab,
            base_t,
            float(self.lam),
            adam_epochs=int(adam_epochs),
            lbfgs_maxiter=int(lbfgs_maxiter),
            resid_scale=self.resid_scale,
            val_frac=0.2,
            verbose=verbose,
        )

        logits_full = base_t + self._active_resid_scale * (K @ alpha.T)
        resid_ser_va = float(np.mean(np.argmax(logits_full[va_idx], 1) != ylab[va_idx]))
        used_residual = True
        if resid_ser_va > front_ser_va * 1.02 + 1e-9:
            alpha = np.zeros_like(alpha)
            used_residual = False
            self._active_resid_scale = 0.0
            logits_full = base_t.copy()

        self.alpha = alpha
        self.eta = eta
        self.kernel_scales = np.asarray(stats["kernel_scales"], dtype=np.float64)

        f = _softmax_rows(logits_full)
        train_ce = softmax_ce_from_scores(f, s1_train)
        train_ser = float(np.mean(np.argmax(f, 1) != s1_train))
        tag = self.base_frontend
        self.last_fit_stats = {
            "train_j_data": float(train_ce),
            "train_ser": train_ser,
            "val_j_data": float(stats.get("val_j_data", train_ce)),
            "gamma": float(self.gamma),
            "lam": float(self.lam),
            "n_centers": float(n),
            "alpha_init_pick": (
                f"residual_{tag}" if used_residual else f"base_only_{tag}"
            ),
            "mode": f"residual_adaptive_mkl_{tag}",
            "eta_entropy": float(stats.get("eta_entropy", np.nan)),
            "frontend_ber": float(np.mean(np.argmax(base_t, 1) != s1_train)),
            "val_front_ser": front_ser_va,
            "val_resid_ser": resid_ser_va,
            "used_residual": float(used_residual),
        }
        for i, e in enumerate(eta):
            self.last_fit_stats[f"eta_{i}"] = float(e)
        if verbose:
            print(
                f"  Residual-{tag}: frontBER={self.last_fit_stats['frontend_ber']:.4f} "
                f"→ SER={train_ser:.4f} J={train_ce:.4f} "
                f"val {front_ser_va:.3f}→{resid_ser_va:.3f} "
                f"{'OK' if used_residual else 'FALLBACK'}",
                flush=True,
            )
        return float(train_ce)

    def _residual_logits(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.alpha is None or self.X_centers is None:
            raise RuntimeError("先 fit()")
        base, feat_raw = self._base_and_feat(y, reuse_hat=True)
        X = _normalize_apply(feat_raw, self.feat_mean, self.feat_std)
        base_k = build_base_kernels(
            X, float(self.gamma), self.X_centers, ms_ratios=self.ms_ratios
        )
        if self.kernel_scales is not None and len(self.kernel_scales) == len(base_k):
            base_k = [Km * float(s) for Km, s in zip(base_k, self.kernel_scales)]
        K = combine_kernels(base_k, self.eta)
        resid = self._active_resid_scale * (K @ self.alpha.T)
        return base, resid

    def scores(self, y: np.ndarray) -> np.ndarray:
        base, resid = self._residual_logits(y)
        return _softmax_rows(base + resid)

    def detect(self, y: np.ndarray) -> np.ndarray:
        base, resid = self._residual_logits(y)
        return np.argmax(base + resid, axis=-1)


def _fit_residual_adaptive_mkl(
    base_kernels: list[np.ndarray],
    labels: np.ndarray,
    base_logits: np.ndarray,
    lam: float,
    *,
    adam_epochs: int,
    lbfgs_maxiter: int,
    resid_scale: float,
    ent_reg: float = 0.02,
    val_frac: float = 0.15,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    在固定 base_logits 上学习残差 Adaptive-MKL：
    logits = base + scale * ∑_m K_m α_mᵀ
    """
    import torch
    from kernel_rkhs import _label_diagonal_init, solve_alpha_from_logits
    from kernel_rkhs import _fit_adam, _fit_lbfgs_limited_local

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
    base_t = torch.from_numpy(base_logits.astype(np.float32))

    a0 = _label_diagonal_init(labels).astype(np.float32) * 0.05
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
        [
            {"params": [alpha_m], "lr": 0.05},
            {"params": [theta], "lr": 0.03},
        ]
    )
    best_state = None
    best_val = float("inf")
    stale = 0
    eps = 1e-6
    scale = float(resid_scale)

    for _ in range(int(adam_epochs)):
        opt.zero_grad()
        eta = torch.softmax(theta, dim=0)
        resid = torch.zeros(n, MOD_ORDER, dtype=torch.float32)
        reg = torch.zeros((), dtype=torch.float32)
        for m in range(n_k):
            am = alpha_m[m]
            resid = resid + (Ks_t[m] @ am.T)
            reg = reg + (1.0 / (eta[m] + eps)) * torch.sum(am * (am @ Ks_t[m]))
        logits = base_t + scale * resid
        ce = torch.nn.functional.cross_entropy(logits[tr], labels_t[tr])
        ent = -torch.sum(eta * torch.log(eta + eps))
        loss = ce + float(lam) * reg - float(ent_reg) * ent
        loss.backward()
        opt.step()

        with torch.no_grad():
            eta_v = torch.softmax(theta, dim=0)
            resid_v = torch.zeros(n, MOD_ORDER, dtype=torch.float32)
            for m in range(n_k):
                resid_v = resid_v + (Ks_t[m] @ alpha_m[m].T)
            val_ce = float(
                torch.nn.functional.cross_entropy(
                    base_t[va] + scale * resid_v[va], labels_t[va]
                )
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

    # 合并为单组 α_eff：K_η α_effᵀ ≈ ∑ K_m α_mᵀ
    resid_sum = np.zeros((n, MOD_ORDER), dtype=np.float64)
    for m in range(n_k):
        resid_sum += Ks_np[m] @ am_np[m].T
    try:
        alpha_eff = solve_alpha_from_logits(resid_sum, K_eta, float(lam))
    except np.linalg.LinAlgError:
        alpha_eff = am_np[int(np.argmax(eta_np))]

    # 固定 η、base，精修 α（残差 CE）
    if lbfgs_maxiter > 0:
        alpha_eff = _refine_residual_alpha(
            alpha_eff, K_eta, labels, base_logits, float(lam),
            scale=scale, adam_epochs=min(400, adam_epochs),
            lbfgs_maxiter=int(lbfgs_maxiter),
        )

    stats = {
        "val_j_data": float(best_val),
        "eta": eta_np.copy(),
        "kernel_scales": scales.copy(),
        "eta_entropy": float(-np.sum(eta_np * np.log(eta_np + 1e-12))),
        "mode": "residual_adaptive_mkl",
    }
    if verbose:
        print(f"  residual-MKL η_H={stats['eta_entropy']:.3f} valCE={best_val:.4f}")
    return alpha_eff, eta_np, K_eta, stats


def _refine_residual_alpha(
    alpha: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    base_logits: np.ndarray,
    lam: float,
    *,
    scale: float,
    adam_epochs: int,
    lbfgs_maxiter: int,
) -> np.ndarray:
    """对 logits=base+scale*Kαᵀ 做 Adam→LBFGS。"""
    import torch

    n = K.shape[0]
    K_t = torch.from_numpy(K.astype(np.float32))
    base_t = torch.from_numpy(base_logits.astype(np.float32))
    labels_t = torch.from_numpy(labels.astype(np.int64))
    a = torch.tensor(alpha.astype(np.float32), requires_grad=True)
    opt = torch.optim.Adam([a], lr=0.05)
    best, best_v = a.detach().cpu().numpy().copy(), float("inf")
    stale = 0
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(32, int(0.15 * n))
    va, tr = perm[:n_val], perm[n_val:]

    for _ in range(int(adam_epochs)):
        opt.zero_grad()
        logits = base_t + float(scale) * (K_t @ a.T)
        ce = torch.nn.functional.cross_entropy(logits[tr], labels_t[tr])
        reg = float(lam) * torch.sum(a * (a @ K_t))
        (ce + reg).backward()
        opt.step()
        with torch.no_grad():
            v = float(
                torch.nn.functional.cross_entropy(
                    base_t[va] + float(scale) * (K_t[va] @ a.T), labels_t[va]
                )
            )
        if v < best_v - 1e-5:
            best_v = v
            best = a.detach().cpu().numpy().copy()
            stale = 0
        else:
            stale += 1
            if stale >= 60:
                break

    # L-BFGS on full data
    from scipy.optimize import minimize
    from kernel_rkhs import _loss_grad_numpy

    def fun(x):
        aa = x.reshape(MOD_ORDER, n)
        logits = base_logits + float(scale) * (K @ aa.T)
        # reuse CE via softmax path of _loss_grad but with shifted logits:
        # approximate by fitting aa on residual target
        g = logits
        g = np.clip(g, -40, 40)
        z = g.max(1, keepdims=True)
        f = np.exp(g - z)
        f /= f.sum(1, keepdims=True) + 1e-300
        ce = -np.mean(np.log(f[np.arange(n), labels] + 1e-300))
        reg = float(lam) * sum(aa[c] @ K @ aa[c] for c in range(MOD_ORDER))
        return float(ce + reg)

    def jac(x):
        aa = x.reshape(MOD_ORDER, n)
        logits = base_logits + float(scale) * (K @ aa.T)
        g = np.clip(logits, -40, 40)
        z = g.max(1, keepdims=True)
        e = np.exp(g - z)
        f = e / (e.sum(1, keepdims=True) + 1e-300)
        dg = f.copy()
        dg[np.arange(n), labels] -= 1.0
        dg /= n
        # d(ce)/d(aa) = scale * dg^T path through K: ∂L/∂α = scale * (dg^T @ K) per class row
        # L = ce(base + scale K α^T) → ∂ce/∂α = scale * (K @ dg)_cols → α_grad[c] = scale * (K @ dg[:,c])
        grad = float(scale) * (dg.T @ K)  # (16, n)
        for c in range(MOD_ORDER):
            grad[c] += 2.0 * float(lam) * (K @ aa[c])
        return grad.ravel()

    res = minimize(
        fun, best.ravel(), method="L-BFGS-B", jac=jac,
        options={"maxiter": int(lbfgs_maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    return res.x.reshape(MOD_ORDER, n)
