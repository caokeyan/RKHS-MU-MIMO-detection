"""
可实现条件核（无真 H / 无真 f*，块衰落）：

  Ĥ：整块一次导频 LS（与 MMSE+LS 同设定）
  φ(y) = [ z_rob(y;Ĥ) ,  rel(y;Ĥ) ]
  rel = [margin, entropy, 残差能量, MF 相关]
  K = K_z ⊙ K_rel   或拼接后 Adaptive-MKL
  监督：仅硬标签 CE

逐样本独立导频在块衰落下会额外注入 CE 噪声，故不用。
"""
from __future__ import annotations

import numpy as np

from kernel_rkhs import (
    ADAPTIVE_MKL_RATIOS,
    build_base_kernels,
    combine_kernels,
    fit_adaptive_mkl_alpha,
    gamma_theory_rkhs,
    lam_theory_rkhs,
    rbf_kernel,
)
from mmse import mmse_equalize
from objective import softmax_ce_from_scores
from rkhs_mld_approx import (
    estimate_heff_block,
    robust_struct_z_features,
    sigma_e2_pilot,
    struct_z_features,
    _normalize_fit,
    _normalize_apply,
    _softmax_rows,
)
from system import CONSTELLATION, MOD_ORDER, n0_from_snr_db


def reliability_features(
    y: np.ndarray,
    H_hat: np.ndarray,
    n0_hat: float,
    z: np.ndarray,
) -> np.ndarray:
    """随 y 变化的可靠度 / 几何特征（Ĥ 固定）。"""
    y = np.asarray(y)
    z = np.asarray(z)
    # margin & entropy from z
    part = np.partition(z, -2, axis=1)
    margin = (z.max(axis=1) - part[:, -2])[:, None]
    g = np.clip(z, -40, 40)
    e = np.exp(g - g.max(1, keepdims=True))
    p = e / (e.sum(1, keepdims=True) + 1e-300)
    ent = (-np.sum(p * np.log(p + 1e-300), axis=1))[:, None]
    # MMSE 残差能量
    x_hat = mmse_equalize(y, H_hat, float(n0_hat))
    resid = y - x_hat @ H_hat.T
    r2 = np.sum(np.abs(resid) ** 2, axis=1)[:, None]
    # MF for user 1
    h1 = H_hat[:, 0]
    nh = max(float(np.vdot(h1, h1).real), 1e-12)
    mf = (y @ np.conj(h1)) / nh
    mf_feat = np.stack([mf.real, mf.imag, np.abs(mf) ** 2], axis=1)
    # nearest constellation distance of x1_hat
    d2 = np.min(np.abs(x_hat[:, 0:1] - CONSTELLATION[None, :]) ** 2, axis=1)[:, None]
    return np.concatenate(
        [
            margin,
            ent,
            np.log(r2 + 1e-12),
            np.log(d2 + 1e-12),
            mf_feat,
        ],
        axis=1,
    ).astype(np.float64)


class CondHatRKHSDetector:
    """块 Ĥ + z_rob × rel 条件核；硬标签。"""

    def __init__(
        self,
        *,
        lam_c: float = 0.1,
        ms_ratios: tuple[float, ...] = ADAPTIVE_MKL_RATIOS,
        pilot_mult: float = 1.0,
        robust_csi: bool = True,
        product_kernel: bool = True,
    ) -> None:
        self.lam_c = float(lam_c)
        self.ms_ratios = tuple(ms_ratios)
        self.pilot_mult = float(pilot_mult)
        self.robust_csi = bool(robust_csi)
        self.product_kernel = bool(product_kernel)
        self.gamma = self.gamma_rel = self.lam = None
        self.eta = self.kernel_scales = self.alpha = None
        self.z_mean = self.z_std = self.r_mean = self.r_std = None
        self.Z_centers = self.R_centers = self.X_centers = None
        self.H_eff = self.H_hat = None
        self.n0_hat = self.sigma_e2 = self.snr_db = None
        self.last_fit_stats: dict[str, float] = {}

    def _z_and_rel(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.H_hat is not None and self.n0_hat is not None
        if self.robust_csi and self.sigma_e2 is not None:
            z = robust_struct_z_features(y, self.H_hat, self.n0_hat, float(self.sigma_e2))
        else:
            z = struct_z_features(y, self.H_hat, self.n0_hat)
        rel = reliability_features(y, self.H_hat, float(self.n0_hat), z)
        return z, rel

    def _product_kernels(self, Z, R, Zc=None, Rc=None):
        Zc = Z if Zc is None else Zc
        Rc = R if Rc is None else Rc
        out = []
        for ratio in self.ms_ratios:
            Kz = rbf_kernel(Z, float(self.gamma) * float(ratio), Zc)
            Kr = rbf_kernel(R, float(self.gamma_rel) * float(ratio), Rc)
            out.append(Kz * Kr)
        return out

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
        f_star_train: np.ndarray | None = None,
    ) -> float:
        del f_star_train
        self.H_eff = np.asarray(H_eff)
        self.snr_db = float(snr_db)
        n = len(y_train)
        n0 = n0_from_snr_db(self.snr_db)
        ylab = np.asarray(s1_train, dtype=np.int64)

        self.H_hat, self.n0_hat, T = estimate_heff_block(
            self.H_eff, self.snr_db, np.random.default_rng(0), pilot_mult=self.pilot_mult
        )
        se2 = sigma_e2_pilot(self.n0_hat, T)
        s = float(snr_db)
        if s >= 10.0:
            se2 *= 0.15
        elif s >= 8.0:
            se2 *= 0.4
        self.sigma_e2 = float(se2)

        Z_raw, R_raw = self._z_and_rel(y_train)
        Z, self.z_mean, self.z_std = _normalize_fit(Z_raw)
        R, self.r_mean, self.r_std = _normalize_fit(R_raw)
        self.Z_centers, self.R_centers = Z, R
        self.X_centers = np.concatenate([Z, R], axis=1)
        self.gamma = gamma_theory_rkhs(n0, Z)
        self.gamma_rel = gamma_theory_rkhs(n0, R)
        self.lam = lam_theory_rkhs(n0, n, c=self.lam_c)

        ratios = self.ms_ratios
        if s >= 10.0:
            ratios = (0.25, 0.5, 1.0, 2.0)
        elif s >= 8.0:
            ratios = (0.15, 0.5, 1.0, 2.0)
        self.ms_ratios = ratios

        base_k = (
            self._product_kernels(Z, R)
            if self.product_kernel
            else build_base_kernels(self.X_centers, self.gamma, ms_ratios=ratios)
        )
        alpha, eta, K, stats = fit_adaptive_mkl_alpha(
            base_k, ylab, float(self.lam),
            adam_epochs=int(adam_epochs), lbfgs_maxiter=int(lbfgs_maxiter), verbose=verbose,
        )
        self.alpha, self.eta = alpha, eta
        self.kernel_scales = np.asarray(stats["kernel_scales"], dtype=np.float64)
        logits = K @ self.alpha.T
        f = _softmax_rows(logits)
        train_ser = float(np.mean(np.argmax(logits, 1) != ylab))
        train_ce = softmax_ce_from_scores(f, ylab)
        pick = "approx_hard_cond_rel_prod" if self.product_kernel else "approx_hard_cond_rel_cat"
        self.last_fit_stats = {
            "train_j_data": float(train_ce),
            "train_ser": train_ser,
            "gamma": float(self.gamma),
            "lam": float(self.lam),
            "n_centers": float(n),
            "alpha_init_pick": pick,
            "mode": "rkhs_cond_rel_hard",
            "feature_dim": float(self.X_centers.shape[1]),
            "sigma_e2": float(self.sigma_e2),
        }
        if verbose:
            print(f"  CondRel-RKHS: SER={train_ser:.4f} J={train_ce:.4f}", flush=True)
        return float(train_ce)

    def _logits(self, y: np.ndarray) -> np.ndarray:
        if self.alpha is None:
            raise RuntimeError("先 fit()")
        Z_raw, R_raw = self._z_and_rel(y)
        Z = _normalize_apply(Z_raw, self.z_mean, self.z_std)
        R = _normalize_apply(R_raw, self.r_mean, self.r_std)
        if self.product_kernel:
            base_k = self._product_kernels(Z, R, self.Z_centers, self.R_centers)
        else:
            X = np.concatenate([Z, R], axis=1)
            base_k = build_base_kernels(
                X, float(self.gamma), self.X_centers, ms_ratios=self.ms_ratios
            )
        if self.kernel_scales is not None and len(self.kernel_scales) == len(base_k):
            base_k = [Km * float(s) for Km, s in zip(base_k, self.kernel_scales)]
        K = combine_kernels(base_k, self.eta)
        return K @ self.alpha.T

    def scores(self, y: np.ndarray) -> np.ndarray:
        return _softmax_rows(self._logits(y))

    def detect(self, y: np.ndarray) -> np.ndarray:
        return np.argmax(self._logits(y), axis=-1)
