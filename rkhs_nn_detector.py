"""
RKHS + NN：函数类仍是核展开 f_a(y) = sum_k alpha_{a,k} K(phi(y), phi(y_k))，
系数矩阵 alpha 由神经网络 alpha = NN(context) 给出，再用同一 J_data 训练 NN。

与 RKHSDetector 的区别：先用超网络 warm-start α，再对 α 做与盲 RKHS 相同的全数据 Adam→L-BFGS。
"""
from __future__ import annotations

import numpy as np

from scipy.optimize import minimize

from kernel_rkhs import (
    ADAPTIVE_MKL_RATIOS,
    DEFAULT_MULTISCALE_RATIOS,
    _fit_adam,
    _fit_lbfgs,
    _label_diagonal_init,
    _loss_grad_numpy,
    build_base_kernels,
    build_kernel_matrix,
    fit_adaptive_mkl_alpha,
    gamma_theory_rkhs,
    lam_theory_rkhs,
    median_bandwidth,
)


def _fit_lbfgs_limited(
    alpha_init: np.ndarray,
    K: np.ndarray,
    labels: np.ndarray,
    lam: float,
    *,
    maxiter: int = 3000,
    verbose: bool = False,
) -> np.ndarray:
    """L-BFGS-B，可限制 maxiter（fast 模式加速）。"""

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
    if verbose:
        print(f"    L-BFGS-B: success={res.success}, nit={res.nit}, fun={res.fun:.6f}")
    return res.x.reshape(MOD_ORDER, -1)
from objective import softmax_ce_from_scores

def _val_ce_alpha(
    alpha: np.ndarray, K: np.ndarray, val_idx: np.ndarray, labels: np.ndarray
) -> float:
    logits = K[val_idx] @ alpha.T
    g = np.clip(logits, -40.0, 40.0)
    z = g.max(axis=1, keepdims=True)
    f = np.exp(g - z)
    f /= f.sum(axis=1, keepdims=True) + 1e-300
    return float(softmax_ce_from_scores(f, labels[val_idx]))
from system import MOD_ORDER, n0_from_snr_db, y_to_features
from mmse import batch_pilot_estimates, generate_pilots

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError("rkhs_nn_detector 需要 PyTorch") from e


def _normalize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def _normalize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _heff_to_feat(H: np.ndarray, n: int) -> np.ndarray:
    """
    CSI 特征：只用目标流信道 h_1（检测 X_1 足够，且避免 2MK 维爆炸）。
    H (M,K) 或 (n,M,K) → (n, 2M)。
    """
    H = np.asarray(H)
    if H.ndim == 2:
        h1 = H[:, 0]
        v = np.concatenate([h1.real, h1.imag]).astype(np.float64)
        return np.tile(v[None, :], (n, 1))
    h1 = H[:, :, 0]  # (n, M)
    return np.concatenate([h1.real, h1.imag], axis=1).astype(np.float64)


def _build_features(
    y: np.ndarray,
    *,
    H_hat: np.ndarray | None = None,
) -> np.ndarray:
    """盲：仅 φ(y)；CSI：[φ(y), vec(Ĥ)]。"""
    yf = y_to_features(y)
    if H_hat is None:
        return yf
    return np.concatenate([yf, _heff_to_feat(H_hat, yf.shape[0])], axis=1)

def _kernel_context(K: np.ndarray, labels: np.ndarray, *, gamma: float, n0: float, bw: float) -> np.ndarray:
    """固定维上下文，供超网络预测 alpha（与样本 y 无关的全局 alpha）。"""
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    eig = np.linalg.eigvalsh(K)
    eig_top = np.sort(eig)[-min(32, n) :][::-1]
    if len(eig_top) < 32:
        eig_top = np.pad(eig_top, (0, 32 - len(eig_top)), constant_values=eig_top[-1] if len(eig_top) else 0.0)
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


class _AlphaHyperNet(nn.Module):
    def __init__(self, ctx_dim: int, n_centers: int, n_class: int = MOD_ORDER) -> None:
        super().__init__()
        out_dim = n_class * n_centers
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
        )
        self.n_centers = n_centers
        self.n_class = n_class

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx).view(self.n_class, self.n_centers)


class RKHSNNDetector:
    """
    核 RKHS 检测器，alpha 由 NN 生成；训练损失为 J_data + lambda * ||alpha||_K^2。
    """

    def __init__(
        self,
        *,
        lam_c: float = 0.1,
        max_centers: int = 0,
        lr: float = 1e-3,
        kernel_mode: str = "single",
        ms_ratios: tuple[float, ...] | None = None,
        alpha_refine_epochs: int = 500,
        use_csi: bool = False,
    ) -> None:
        self.lam_c = float(lam_c)
        self.max_centers = int(max_centers)
        self.alpha_refine_epochs = int(alpha_refine_epochs)
        self.lr = float(lr)
        self.use_csi = bool(use_csi)
        if kernel_mode not in ("single", "multiscale", "adaptive"):
            raise ValueError("kernel_mode 须为 'single' / 'multiscale' / 'adaptive'")
        self.kernel_mode = kernel_mode
        if ms_ratios is None:
            ms_ratios = (
                ADAPTIVE_MKL_RATIOS if kernel_mode == "adaptive" else DEFAULT_MULTISCALE_RATIOS
            )
        self.ms_ratios = tuple(ms_ratios)
        self.gamma: float | None = None
        self.lam: float | None = None
        self.eta: np.ndarray | None = None  # 自适应多核权重
        self.kernel_scales: np.ndarray | None = None  # 基核 trace 归一化系数
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.Y_train_feat: np.ndarray | None = None
        self.K_train: np.ndarray | None = None
        self.center_idx: np.ndarray | None = None
        self.net: _AlphaHyperNet | None = None
        self.alpha: np.ndarray | None = None
        self.ctx_np: np.ndarray | None = None
        self._H_eff_ref: np.ndarray | None = None  # 推理时再估 Ĥ 用
        self.device = torch.device("cpu")
        self.last_fit_stats: dict[str, float] = {}
        self._snr_db_fit: float | None = None

    def _subsample_centers(
        self, y_train: np.ndarray, s1_train: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(y_train)
        cap = self.max_centers
        if cap <= 0 or n <= cap:
            idx = np.arange(n, dtype=np.intp)
        else:
            idx = rng.choice(n, size=cap, replace=False)
        return y_train[idx], s1_train[idx], idx

    def fit(
        self,
        y_train: np.ndarray,
        s1_train: np.ndarray,
        *,
        snr_db: float | None = None,
        val_frac: float = 0.15,
        epochs: int = 200,
        patience: int = 30,
        alpha_adam_epochs: int | None = None,
        do_lbfgs: bool = True,
        lbfgs_maxiter: int = 3000,
        use_nn_warmstart: bool = True,
        alpha_inits: tuple[str, ...] = ("diag", "nn"),
        skip_second_init_if_diag_good: bool = True,
        diag_good_val_ce: float = 0.35,
        verbose: bool = False,
        gamma_override: float | None = None,
        lam_override: float | None = None,
        H_eff: np.ndarray | None = None,
    ) -> float:
        rng = np.random.default_rng(0)
        n0 = n0_from_snr_db(float(snr_db)) if snr_db is not None else 1.0
        self._snr_db_fit = float(snr_db) if snr_db is not None else None

        H_hat_all = None
        if self.use_csi:
            if H_eff is None:
                raise ValueError("use_csi=True 时需要传入 H_eff 以生成导频 Ĥ")
            self._H_eff_ref = np.asarray(H_eff)
            X_p = generate_pilots()
            H_hat_all, _ = batch_pilot_estimates(
                self._H_eff_ref, X_p, n0, rng, len(y_train)
            )

        X_all_raw = _build_features(y_train, H_hat=H_hat_all)
        X_all, mean, std = _normalize_fit(X_all_raw)
        self.feat_mean, self.feat_std = mean, std

        y_c, s1_c, idx = self._subsample_centers(y_train, s1_train, rng)
        self.center_idx = idx
        n = len(y_c)
        self.lam = (
            float(lam_override)
            if lam_override is not None and lam_override > 0
            else lam_theory_rkhs(n0, n, c=self.lam_c)
        )
        X = X_all[idx]
        self.Y_train_feat = X
        self.gamma = (
            float(gamma_override)
            if gamma_override is not None and gamma_override > 0
            else gamma_theory_rkhs(n0, X)
        )
        bw = median_bandwidth(X)

        # —— 自适应多核（SimpleMKL 凸组合）——
        if self.kernel_mode == "adaptive":
            base = build_base_kernels(X, self.gamma, ms_ratios=self.ms_ratios)
            adam_ep = (
                int(alpha_adam_epochs)
                if alpha_adam_epochs is not None
                else max(self.alpha_refine_epochs, 800)
            )
            alpha, eta, K, stats = fit_adaptive_mkl_alpha(
                base,
                s1_c.astype(np.int64),
                float(self.lam),
                adam_epochs=adam_ep,
                lbfgs_maxiter=int(lbfgs_maxiter) if do_lbfgs else 0,
                patience=max(40, patience),
                ent_reg=0.02,
                verbose=verbose,
            )
            self.alpha = alpha
            self.eta = eta
            self.kernel_scales = np.asarray(stats.get("kernel_scales"), dtype=np.float64)
            self.K_train = K
            self.net = None
            self.last_fit_stats = {
                **{k: v for k, v in stats.items() if k not in ("eta", "kernel_scales")},
                "gamma": float(self.gamma),
                "lam": float(self.lam),
                "n_centers": float(n),
                "alpha_init_pick": "adaptive_mkl",
                "alpha_adam_epochs": float(adam_ep),
                "alpha_lbfgs": float(do_lbfgs),
                "lbfgs_maxiter": float(lbfgs_maxiter),
                "nn_warmstart": 0.0,
                "mode": "rkhs_nn_adaptive_mkl",
            }
            for i, e in enumerate(eta):
                self.last_fit_stats[f"eta_{i}"] = float(e)
            if verbose:
                print(
                    f"  RKHS–NN Adaptive-MKL: J={stats['train_j_data']:.4f} "
                    f"SER={stats['train_ser']:.4f} n={n}",
                    flush=True,
                )
            return float(stats["train_j_data"])

        K = build_kernel_matrix(
            X, self.gamma, kernel_mode=self.kernel_mode, ms_ratios=self.ms_ratios
        )
        self.K_train = K
        self.eta = None
        ctx_np = _kernel_context(K, s1_c, gamma=self.gamma, n0=n0, bw=bw)
        self.ctx_np = ctx_np
        ctx_dim = ctx_np.shape[0]

        perm = rng.permutation(n)
        n_val = max(32, int(n * val_frac))
        va, tr = perm[:n_val], perm[n_val:]

        K_t = torch.from_numpy(K.astype(np.float32)).to(self.device)
        labels = torch.from_numpy(s1_c.astype(np.int64)).to(self.device)
        ctx = torch.from_numpy(ctx_np).to(self.device)

        labels_np = s1_c.astype(np.int64)
        a_diag = _label_diagonal_init(labels_np)
        a_nn: np.ndarray | None = None

        if use_nn_warmstart and epochs > 0:
            net = _AlphaHyperNet(ctx_dim, n).to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            best_state = None
            nn_best_val = float("inf")
            stale = 0
            for _ in range(epochs):
                opt.zero_grad()
                alpha = net(ctx)
                logits = K_t @ alpha.T
                ce = nn.functional.cross_entropy(logits[tr], labels[tr])
                reg = self.lam * torch.sum(alpha * torch.mm(alpha, K_t))
                loss = ce + reg
                loss.backward()
                opt.step()
                with torch.no_grad():
                    alpha_v = net(ctx)
                    logits_v = K_t @ alpha_v.T
                    val_ce = float(nn.functional.cross_entropy(logits_v[va], labels[va]))
                if val_ce < nn_best_val - 1e-5:
                    nn_best_val = val_ce
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            self.net = net
            with torch.no_grad():
                a_nn = net(ctx).cpu().numpy()
        else:
            self.net = None

        init_map: dict[str, np.ndarray] = {"diag": a_diag}
        if a_nn is not None and "nn" in alpha_inits:
            init_map["nn"] = a_nn

        adam_ep = (
            int(alpha_adam_epochs)
            if alpha_adam_epochs is not None
            else max(self.alpha_refine_epochs, 800)
        )
        lam_f = float(self.lam)
        best_alpha = None
        best_val = float("inf")
        best_tag = "diag"
        for tag in alpha_inits:
            if tag not in init_map:
                continue
            if (
                tag == "nn"
                and skip_second_init_if_diag_good
                and best_alpha is not None
                and best_val <= diag_good_val_ce
            ):
                break
            a0 = init_map[tag]
            a_fit = _fit_adam(
                a0,
                K,
                labels_np,
                lam_f,
                "softmax",
                epochs=adam_ep,
                patience=120,
                verbose=verbose and tag == "diag",
                log_every=400,
            )
            if do_lbfgs:
                if lbfgs_maxiter >= 3000:
                    a_fit = _fit_lbfgs(
                        a_fit, K, labels_np, lam_f, "softmax", verbose=verbose and tag == "diag"
                    )
                else:
                    a_fit = _fit_lbfgs_limited(
                        a_fit,
                        K,
                        labels_np,
                        lam_f,
                        maxiter=lbfgs_maxiter,
                        verbose=verbose and tag == "diag",
                    )
            vce = _val_ce_alpha(a_fit, K, va, labels_np)
            if vce < best_val:
                best_val = vce
                best_alpha = a_fit
                best_tag = tag
        assert best_alpha is not None
        self.alpha = best_alpha

        logits_np = K @ self.alpha.T
        f_tr = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        f_tr /= f_tr.sum(axis=1, keepdims=True) + 1e-300
        train_ce = softmax_ce_from_scores(f_tr, s1_c)
        self.last_fit_stats = {
            "train_j_data": float(train_ce),
            "train_ser": float(np.mean(np.argmax(f_tr, 1) != s1_c)),
            "val_j_data": float(best_val),
            "alpha_init_pick": best_tag,
            "gamma": float(self.gamma),
            "lam": float(self.lam),
            "n_centers": float(n),
            "alpha_adam_epochs": float(adam_ep),
            "alpha_lbfgs": float(do_lbfgs),
            "lbfgs_maxiter": float(lbfgs_maxiter),
            "nn_warmstart": float(use_nn_warmstart and epochs > 0),
            "mode": "rkhs_nn",
        }
        if verbose:
            print(
                f"  RKHS–NN: J_data={train_ce:.4f}, SER={self.last_fit_stats['train_ser']:.4f}, "
                f"n_center={n}, val={best_val:.4f}",
                flush=True,
            )
        return train_ce

    def _logits(self, y: np.ndarray) -> np.ndarray:
        if self.Y_train_feat is None or self.gamma is None:
            raise RuntimeError("先调用 fit()")
        H_hat = None
        if self.use_csi:
            if self._H_eff_ref is None or self._snr_db_fit is None:
                raise RuntimeError("CSI 模式缺少 H_eff / snr")
            n0 = n0_from_snr_db(self._snr_db_fit)
            X_p = generate_pilots()
            H_hat, _ = batch_pilot_estimates(
                self._H_eff_ref, X_p, n0, np.random.default_rng(1), len(y)
            )
        X = _normalize_apply(
            _build_features(y, H_hat=H_hat), self.feat_mean, self.feat_std
        )
        if self.kernel_mode == "adaptive" and self.eta is not None:
            base = build_base_kernels(
                X, self.gamma, self.Y_train_feat, ms_ratios=self.ms_ratios
            )
            if self.kernel_scales is not None and len(self.kernel_scales) == len(base):
                base = [Km * float(s) for Km, s in zip(base, self.kernel_scales)]
            from kernel_rkhs import combine_kernels

            K = combine_kernels(base, self.eta)
        else:
            K = build_kernel_matrix(
                X,
                self.gamma,
                self.Y_train_feat,
                kernel_mode=self.kernel_mode,
                ms_ratios=self.ms_ratios,
                eta=self.eta,
            )
        if self.alpha is not None:
            return K @ self.alpha.T
        if self.net is None or self.ctx_np is None:
            raise RuntimeError("fit() 未得到 alpha 且无可用的 NN")
        ctx = torch.from_numpy(self.ctx_np).to(self.device)
        self.net.eval()
        with torch.no_grad():
            alpha = self.net(ctx).cpu().numpy()
        return K @ alpha.T

    def scores(self, y: np.ndarray) -> np.ndarray:
        g = self._logits(y)
        g = np.clip(g, -40.0, 40.0)
        z = g.max(axis=1, keepdims=True)
        e = np.exp(g - z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-300)

    def detect(self, y: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(y), axis=1)
