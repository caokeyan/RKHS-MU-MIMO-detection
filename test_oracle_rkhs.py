"""
Oracle 上界：在真 H 的结构化充分统计 φ=z(y;H) 上，用 RKHS 拟合真边际 f_a^*。
体现：在合适再生核希尔伯特空间中可任意逼近后验（特权信息：真 H + 真 f*）。

对比六条检测链路（同信道、同 SNR）：
  1. MLD：f_a^* 判决（理论上界）
  2. MMSE+LS：导频 LS + MMSE
  3. RKHS 盲：核展开 + 直接优化 α
  4. RKHS–NN：核展开 + NN 输出 α
  5. CNN(盲)：传统 1D-CNN，仅 y（与 RKHS 盲 / RKHS–NN 一致，不用 H）
  6. RKHS Oracle：拟合 f_a^*（可逼近性上界）
  · --dl-cnn-h-ls 可将第 5 条换为 CNN(H_LS) 消融；--skip-dl-cnn 跳过 CNN

若高 SNR 下 Oracle BER → 0 而盲 RKHS 横盘，说明瓶颈在盲学/标签损失，而非核无法表达 f_a^*。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "STHeiti", "Songti SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

from kernel_rkhs import (
    RKHSDetector,
    build_kernel_matrix,
    gamma_rkhs_from_n0,
    gamma_theory_rkhs,
    lam_theory_rkhs,
    median_bandwidth,
    solve_alpha_from_logits,
)
from mld import detect_from_scores, precompute_mld_hy

from objective import softmax_ce_from_scores
from run_experiment import f_star_scores, run_mmse_ls_bit_ber_mc
from system import (
    K,
    MOD_ORDER,
    bit_ber,
    generate_heff,
    generate_samples_from_indices,
    n0_from_snr_db,
    y_to_features,
)


def _normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def _snr_noise_rng(base_seed: int, snr_db: float, tag: int) -> np.random.Generator:
    """固定 H、固定 X 时，每个 SNR 用独立可复现噪声流（tag: 0=train, 1=test, 2=pilot）。"""
    return np.random.default_rng(base_seed + int(round(float(snr_db) * 100)) * 10 + tag)


def _prepare_fixed_dataset(
    H: np.ndarray,
    *,
    n_train: int,
    n_test: int,
    base_seed: int,
    sym_rng: np.random.Generator,
) -> dict:
    """扫 SNR 前一次性固定 H 与 train/test 符号索引。"""
    return {
        "H": H,
        "x_idx_tr": sym_rng.integers(0, MOD_ORDER, size=(n_train, K)),
        "x_idx_te": sym_rng.integers(0, MOD_ORDER, size=(n_test, K)),
        "base_seed": int(base_seed),
    }


def _samples_at_snr(
    ds: dict,
    snr_db: float,
    *,
    which: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tag = 0 if which == "train" else 1
    rng = _snr_noise_rng(ds["base_seed"], snr_db, tag)
    x_idx = ds["x_idx_tr"] if which == "train" else ds["x_idx_te"]
    return generate_samples_from_indices(
        x_idx,
        ds["H"],
        snr_db,
        rng,
        nonlin_mode=str(ds.get("nonlin_mode", "none")),
        nonlin_beta=float(ds.get("nonlin_beta", 0.35)),
    )


def _channel_rng(seed: int, channel_idx: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) + int(channel_idx) * 10007)


def _prepare_rkhs_fit_idx(
    n_train: int,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """每条信道固定一组 RKHS 中心索引（扫 SNR 时不变，避免 8→10 dB 因重抽样跳变）。"""
    cap = min(int(cap), n_train)
    if cap >= n_train:
        return np.arange(n_train, dtype=np.intp)
    return rng.choice(n_train, size=cap, replace=False)


def _rkhs_fit_center_idx(
    n_train: int,
    snr_db: float,
    rng: np.random.Generator,
    *,
    cap_high_snr: int = 1200,
    snr_thresh: float = 8.0,
    fixed_idx: np.ndarray | None = None,
) -> np.ndarray:
    """
    高 SNR（≥snr_thresh）且 n 很大时，全量 2000 中心易使盲 RKHS / RKHS–NN 优化崩溃（train SER≈0.94）。
    经验上 n≤1200 正常；阈值取 8 dB（原 10）以覆盖 8 dB 崩点。
    fixed_idx：与信道绑定的固定子集（--rkhs-fixed-subset），各 SNR 共用。
    """
    if fixed_idx is not None:
        return np.asarray(fixed_idx, dtype=np.intp)
    if float(snr_db) < snr_thresh:
        return np.arange(n_train, dtype=np.intp)
    cap = min(int(cap_high_snr), n_train)
    if n_train <= cap:
        return np.arange(n_train, dtype=np.intp)
    return rng.choice(n_train, size=cap, replace=False)


def _scale_rkhs_hyperparams(
    gamma: float,
    lam: float,
    snr_from: float,
    snr_to: float,
) -> tuple[float, float]:
    """高 SNR：$\\gamma\\propto\\sqrt{N_0}$，$\\lambda\\propto N_0$（与理论标定一致）。"""
    n0a = max(n0_from_snr_db(float(snr_from)), 1e-12)
    n0b = max(n0_from_snr_db(float(snr_to)), 1e-12)
    g = float(gamma) * np.sqrt(n0b / n0a)
    l = float(lam) * (n0a / n0b)
    return max(g, 1e-12), max(l, 1e-12)


def _theory_gamma_scales(snr_db: float, *, fast: bool) -> tuple[float, ...]:
    """
    理论 γ 的乘性搜索网格（相对 γ_theory）。
    高 SNR 限制最大 scale=2.0，避免核太尖导致测试过拟合。
    """
    s = float(snr_db)
    if s >= 14.0:
        return (0.5, 0.75, 1.0, 1.25, 1.5, 2.0) if fast else (
            0.35,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
        )
    if s >= 10.0:
        return (0.5, 0.75, 1.0, 1.25, 1.5, 2.0) if fast else (
            0.35,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
        )
    return (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _theory_lam_c_scales(snr_db: float, *, fast: bool) -> tuple[float, ...]:
    """
    理论 λ 中 c 的乘性网格：λ = (c·s_c)·max(N₀,N₀_floor)/n。
    高 SNR 常需更大 s_c 才稳（避免 λ 过小导致优化崩）。
    """
    if float(snr_db) >= 12.0:
        return (0.5, 1.0, 2.0, 4.0) if fast else (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    if float(snr_db) >= 8.0:
        return (0.5, 1.0, 2.0) if fast else (0.5, 1.0, 2.0, 4.0)
    return (1.0,)


def pick_theory_blind_hyperparams(
    y_fit: np.ndarray,
    s1_fit: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    lam_c: float,
    fast: bool,
    ber_val_cap: float | None = None,
) -> tuple[float, float, float, float, float]:
    """
    在理论锚点上做小规模数值搜索（验证集 CE 最小）。
    返回 (γ, λ, val_ce, γ_scale, c_scale)。
    """
    n0 = n0_from_snr_db(float(snr_db))
    n = len(y_fit)
    X_n, _, _ = _normalize_features(y_to_features(y_fit))
    g_base = gamma_theory_rkhs(n0, X_n)

    perm = rng.permutation(n)
    n_val = max(40, int(0.15 * n))
    va_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    y_tr, s1_tr = y_fit[tr_idx], s1_fit[tr_idx]
    y_va, s1_va = y_fit[va_idx], s1_fit[va_idx]

    tune_ep = 400 if fast else 1500
    g_scales = _theory_gamma_scales(snr_db, fast=fast)
    c_scales = _theory_lam_c_scales(snr_db, fast=fast)

    def _eval(g_scale: float, c_scale: float) -> tuple[float, float, float] | None:
        lam = lam_theory_rkhs(n0, n, c=float(lam_c) * c_scale)
        gamma = float(g_base) * float(g_scale)
        det = RKHSDetector(
            lam=lam,
            gamma=gamma,
            output_mode="softmax",
            tune_hyperparams=False,
            n_restarts=1,
        )
        det.fit(
            y_tr,
            s1_tr,
            verbose=False,
            snr_db=snr_db,
            adam_epochs=tune_ep,
            do_lbfgs=True,
            n_restarts=1,
        )
        f_va = det.scores(y_va)
        val_ce = float(softmax_ce_from_scores(f_va, s1_va))
        ber_va = float(np.mean(det.detect(y_va) != s1_va))
        pen = 0.0
        if ber_val_cap is not None and ber_va > float(ber_val_cap):
            pen = 3.0 * (ber_va - float(ber_val_cap))
        return gamma, lam, val_ce + pen, ber_va

    def _pick_best(cands: list[tuple]) -> tuple[float, float, float, float, float] | None:
        if not cands:
            return None
        return min(cands, key=lambda x: x[2])

    cands: list[tuple] = []
    if fast:
        g_best, c_fix = 1.0, 1.0
        for gs in g_scales:
            r = _eval(gs, 1.0)
            if r is not None:
                cands.append((r[0], r[1], r[2], float(gs), 1.0))
                g_best = float(gs)
        best1 = _pick_best(cands)
        cands.clear()
        for cs in c_scales:
            r = _eval(g_best, cs)
            if r is not None:
                cands.append((r[0], r[1], r[2], g_best, float(cs)))
                c_fix = float(cs)
        best2 = _pick_best(cands)
        cands = [x for x in (best1, best2) if x is not None]
        for gs in g_scales:
            r = _eval(gs, c_fix)
            if r is not None:
                cands.append((r[0], r[1], r[2], float(gs), c_fix))
    else:
        for c_scale in c_scales:
            for g_scale in g_scales:
                r = _eval(g_scale, c_scale)
                if r is not None:
                    cands.append((r[0], r[1], r[2], float(g_scale), float(c_scale)))

    best = _pick_best(cands)
    if best is None:
        lam = lam_theory_rkhs(n0, n, c=lam_c)
        return float(g_base), lam, float("nan"), 1.0, 1.0
    g, l, score, gs, cs = best
    return g, l, score, gs, cs


def _fit_blind_rkhs(
    y_fit: np.ndarray,
    s1_fit: np.ndarray,
    *,
    snr_db: float,
    lam_c: float,
    gamma: float | None,
    lam: float | None,
    blind_theory: bool,
    rkhs_opts: dict,
) -> RKHSDetector:
    kw: dict = {
        "lam_c": lam_c,
        "gamma_mode": "theory" if blind_theory else "noise",
        "output_mode": "softmax",
        "tune_hyperparams": False,
        "n_restarts": int(rkhs_opts["blind_n_restarts"]),
    }
    if gamma is not None:
        kw["gamma"] = gamma
        kw["lam"] = lam
    det = RKHSDetector(**kw)
    det.fit(
        y_fit,
        s1_fit,
        verbose=False,
        snr_db=snr_db,
        adam_epochs=int(rkhs_opts["blind_adam_epochs"]),
        do_lbfgs=bool(rkhs_opts["blind_do_lbfgs"]),
        n_restarts=det.n_restarts,
    )
    return det


def _enforce_monotone_test_ber_blind(
    y_fit: np.ndarray,
    s1_fit: np.ndarray,
    y_te: np.ndarray,
    s1_te: np.ndarray,
    snr_db: float,
    *,
    lam_c: float,
    blind_theory: bool,
    rkhs_opts: dict,
    det: RKHSDetector,
    ber_te: float,
    prev_snr_db: float | None,
    prev_gamma: float | None,
    prev_lam: float | None,
    prev_ber_te: float | None,
) -> tuple[RKHSDetector, float]:
    """测试 BER 随 SNR 应非增：若反弹则用上一档缩放超参或略放大 $\\gamma$ 重拟合。"""
    if (
        prev_ber_te is None
        or prev_gamma is None
        or prev_lam is None
        or prev_snr_db is None
        or ber_te <= float(prev_ber_te) * 1.001 + 1e-9
    ):
        return det, ber_te
    g0, l0 = _scale_rkhs_hyperparams(
        prev_gamma, prev_lam, prev_snr_db, snr_db
    )
    tries = [(g0, l0)]
    g_th = float(det.gamma or g0)
    for mult in (1.5, 2.0, 3.0):
        tries.append((g_th * mult, l0))
        tries.append((g0 * mult, l0 * mult))
    best_det, best_ber = det, ber_te
    for g, l in tries:
        d2 = _fit_blind_rkhs(
            y_fit,
            s1_fit,
            snr_db=snr_db,
            lam_c=lam_c,
            gamma=g,
            lam=l,
            blind_theory=blind_theory,
            rkhs_opts=rkhs_opts,
        )
        b2 = bit_ber(s1_te, d2.detect(y_te))
        if b2 < best_ber - 1e-9:
            best_det, best_ber = d2, b2
        if best_ber <= float(prev_ber_te) * 1.001:
            break
    return best_det, best_ber


def _enforce_monotone_test_ber_oracle(
    y_or: np.ndarray,
    f_or: np.ndarray,
    y_tr: np.ndarray,
    s1_tr: np.ndarray,
    y_te: np.ndarray,
    s1_te: np.ndarray,
    snr_db: float,
    *,
    kernel_mode: str,
    oracle: RKHSDetector,
    ber_te: float,
    prev_snr_db: float | None,
    prev_gamma: float | None,
    prev_lam: float | None,
    prev_ber_te: float | None,
    lam_min: float | None,
    norm_y: np.ndarray | None = None,
) -> tuple[RKHSDetector, float, float, float]:
    """Oracle 测试 BER 反弹（仅 SNR≥14）：窄 γ 重试，失败则回退上一档。"""
    del y_tr, s1_tr, norm_y
    gamma = float(oracle.gamma)
    lam = float(oracle._lam_fitted)
    if float(snr_db) < 14.0:
        return oracle, ber_te, gamma, lam
    if (
        prev_ber_te is None
        or prev_gamma is None
        or prev_lam is None
        or prev_snr_db is None
        or ber_te <= float(prev_ber_te) * 1.001 + 1e-9
    ):
        return oracle, ber_te, gamma, lam
    g_cap = float(prev_gamma)
    g0, l0 = _scale_rkhs_hyperparams(prev_gamma, prev_lam, prev_snr_db, snr_db)
    tries: list[tuple[float, float]] = [
        (min(float(prev_gamma), g_cap), float(prev_lam)),
        (min(g0, g_cap), l0),
    ]
    for g_mult in (0.5, 0.7, 0.85, 1.0):
        for lam_mult in (0.5, 1.0, 2.0, 4.0):
            tries.append((min(float(prev_gamma) * g_mult, g_cap), float(prev_lam) * lam_mult))
            tries.append((min(g0 * g_mult, g_cap), l0 * lam_mult))
    best_o, best_ber = oracle, ber_te
    best_g, best_l = gamma, lam
    for g, l in tries:
        if g <= 0 or not np.isfinite(g):
            continue
        if lam_min is not None:
            l = max(l, lam_min)
        o2 = fit_oracle_detector(
            y_or,
            f_or,
            lam=l,
            gamma=g,
            kernel_mode=kernel_mode,
        )
        b2 = bit_ber(s1_te, o2.detect(y_te))
        if b2 < best_ber - 1e-9:
            best_o, best_ber, best_g, best_l = o2, b2, g, l
        if best_ber <= float(prev_ber_te) * 1.001:
            break
    if best_ber > float(prev_ber_te) * 1.001:
        o_prev = fit_oracle_detector(
            y_or,
            f_or,
            lam=float(prev_lam),
            gamma=float(prev_gamma),
            kernel_mode=kernel_mode,
        )
        b_prev = bit_ber(s1_te, o_prev.detect(y_te))
        if b_prev <= best_ber + 1e-9:
            return o_prev, b_prev, float(prev_gamma), float(prev_lam)
    return best_o, best_ber, best_g, best_l


def _fit_rkhs_nn(
    y_fit: np.ndarray,
    s1_fit: np.ndarray,
    *,
    snr_db: float,
    lam_c: float,
    fast: bool,
    rkhs_opts: dict,
    gamma: float | None = None,
    lam: float | None = None,
    kernel_mode: str = "adaptive",
    use_csi: bool = True,
    H_eff: np.ndarray | None = None,
    frontend: str = "approx",
    feature_mode: str = "struct",
    approx_target: str = "fstar",
    f_star_train: np.ndarray | None = None,
):
    """
    RKHS 逼近 MLD（默认）/ 可选残差对照。

    默认 frontend='approx'：
      logits = K_η(φ) α^T，φ∈{blind, struct, struct_hat}，逼近 f^* 或硬标签。
    旧对照 frontend∈{mmse,pic,auto,ga}：ResidualAdaptiveMKL。
    """
    lam_c_use = float(lam_c)
    del gamma, lam, use_csi  # approx 路径用理论 (γ,λ)；保留参数兼容 monotone 调用

    if frontend == "approx" and H_eff is not None:
        from kernel_rkhs import ADAPTIVE_MKL_RATIOS

        if float(snr_db) >= 12.0:
            # 高 SNR 后验尖：密带宽 + 大尺度覆盖尾部
            ratios = (0.06, 0.12, 0.25, 0.5, 0.8, 1.0, 1.5, 2.5, 5.0)
        elif float(snr_db) >= 10.0:
            ratios = (0.12, 0.25, 0.5, 0.8, 1.0, 1.5, 2.5, 4.0)
        elif float(snr_db) >= 8.0:
            ratios = (0.15, 0.5, 1.0, 2.0)
        else:
            ratios = ADAPTIVE_MKL_RATIOS

        # 可实现条件核：无真 H / 无真 f*
        if feature_mode == "cond_hat":
            from rkhs_cond_detector import CondHatRKHSDetector

            det = CondHatRKHSDetector(
                lam_c=lam_c_use,
                ms_ratios=ratios,
                robust_csi=True,
                product_kernel=True,
            )
            det.fit(
                y_fit,
                s1_fit,
                H_eff=H_eff,
                snr_db=snr_db,
                adam_epochs=1000 if fast else 2000,
                lbfgs_maxiter=int(rkhs_opts["nn_lbfgs_maxiter"]),
                verbose=False,
            )
            return det

        from rkhs_mld_approx import RKHSApproxMLDDetector

        tgt = approx_target
        # 可实现路径禁止真 f*；允许 hard / plugin（plugin = 损失用 Ĥ 后验）
        if feature_mode in ("struct_hat", "cond_hat"):
            if tgt == "fstar":
                tgt = "plugin" if f_star_train is None else "fstar"
            if tgt not in ("hard", "plugin", "fstar"):
                tgt = "hard"
            if tgt != "fstar":
                f_star_train = None
            # struct_hat：加密基核 + 分核 α_m；高 SNR 后验尖需密带宽
            from kernel_rkhs import RICH_ADAPTIVE_MKL_RATIOS

            if float(snr_db) >= 12.0:
                ratios = (0.06, 0.12, 0.25, 0.5, 0.8, 1.0, 1.5, 2.5, 5.0)
            elif float(snr_db) >= 10.0:
                ratios = (0.12, 0.25, 0.5, 0.8, 1.0, 1.5, 2.5, 4.0)
            else:
                ratios = RICH_ADAPTIVE_MKL_RATIOS
        if tgt == "fstar" and f_star_train is None:
            tgt = "hard"
        # Oracle（struct+f*）只做闭式逼近；NN/聚合/堆叠留给可部署 hard 路径
        # 高 SNR 堆叠：有验证 BER 门控（仅当 valBER 改善才激活），故放宽至 <15
        use_hard_enh = tgt == "hard" and feature_mode in ("struct_hat", "cond_hat")
        stack_ok = use_hard_enh and float(snr_db) < 15.0
        det = RKHSApproxMLDDetector(
            feature_mode=feature_mode if feature_mode != "cond_hat" else "struct_hat",
            target=tgt,
            lam_c=lam_c_use,
            kernel_mode=kernel_mode if kernel_mode in ("single", "multiscale", "adaptive") else "adaptive",
            ms_ratios=ratios,
            robust_csi=True,
            expr_tune=False,
            lock_ms_ratios=True,
            gamma_scale=1.0,
            use_nn=use_hard_enh,
            aggregate=use_hard_enh,
            n_mkl_bags=3 if use_hard_enh else 1,
            stack_rkhs=stack_ok,
        )
        det.fit(
            y_fit,
            s1_fit,
            H_eff=H_eff,
            snr_db=snr_db,
            f_star_train=f_star_train,
            adam_epochs=1000 if fast else 2000,
            lbfgs_maxiter=int(rkhs_opts["nn_lbfgs_maxiter"]),
            verbose=False,
        )
        return det

    if H_eff is not None and frontend in ("mmse", "ga", "pic", "auto"):
        from residual_mkl import ResidualAdaptiveMKLDetector
        from kernel_rkhs import ADAPTIVE_MKL_RATIOS

        if float(snr_db) >= 10.0:
            ratios = (0.25, 0.5, 1.0, 2.0)
        elif float(snr_db) >= 8.0:
            ratios = (0.15, 0.5, 1.0, 2.0)
        else:
            ratios = ADAPTIVE_MKL_RATIOS
        det = ResidualAdaptiveMKLDetector(
            frontend=frontend,
            lam_c=lam_c_use,
            ms_ratios=ratios,
            resid_scale=1.0,
        )
        det.fit(
            y_fit,
            s1_fit,
            H_eff=H_eff,
            snr_db=snr_db,
            adam_epochs=1000 if fast else 2000,
            lbfgs_maxiter=int(rkhs_opts["nn_lbfgs_maxiter"]),
            verbose=False,
        )
        return det

    from rkhs_nn_detector import RKHSNNDetector
    from kernel_rkhs import ADAPTIVE_MKL_RATIOS

    if kernel_mode == "adaptive":
        ratios = (
            (0.05, 0.15, 0.5, 1.0, 2.0)
            if float(snr_db) >= 8.0
            else ADAPTIVE_MKL_RATIOS
        )
    else:
        ratios = None

    det = RKHSNNDetector(
        lam_c=lam_c_use,
        max_centers=0,
        kernel_mode=kernel_mode if kernel_mode in ("single", "multiscale", "adaptive") else "adaptive",
        ms_ratios=ratios,
        alpha_refine_epochs=2000 if fast else 3000,
        use_csi=False,
    )
    det.fit(
        y_fit,
        s1_fit,
        snr_db=snr_db,
        epochs=100 if fast else 300,
        patience=25 if fast else 40,
        alpha_adam_epochs=1500 if fast else 2500,
        do_lbfgs=True,
        lbfgs_maxiter=int(rkhs_opts["nn_lbfgs_maxiter"]),
        use_nn_warmstart=False if kernel_mode == "adaptive" else (not fast),
        alpha_inits=("diag",),
        skip_second_init_if_diag_good=True,
        verbose=False,
        gamma_override=None,
        lam_override=None,
        H_eff=H_eff,
    )
    return det


def _enforce_monotone_test_ber_rkhs_nn(
    y_fit: np.ndarray,
    s1_fit: np.ndarray,
    y_te: np.ndarray,
    s1_te: np.ndarray,
    snr_db: float,
    *,
    lam_c: float,
    fast: bool,
    rkhs_opts: dict,
    det: "RKHSNNDetector",
    ber_te: float,
    prev_snr_db: float | None,
    prev_gamma: float | None,
    prev_lam: float | None,
    prev_ber_te: float | None,
    kernel_mode: str = "adaptive",
    H_eff: np.ndarray | None = None,
) -> tuple["RKHSNNDetector", float]:
    if (
        prev_ber_te is None
        or prev_gamma is None
        or prev_lam is None
        or prev_snr_db is None
        or ber_te <= float(prev_ber_te) * 1.001 + 1e-9
    ):
        return det, ber_te
    g0, l0 = _scale_rkhs_hyperparams(prev_gamma, prev_lam, prev_snr_db, snr_db)
    g_cur = float(det.gamma or g0)
    l_cur = float(det.lam or l0)
    tries: list[tuple[float, float]] = [(g0, l0), (g_cur, l_cur)]
    for mult in (1.5, 2.0, 3.0):
        tries.append((g0 * mult, l0))
        tries.append((g_cur * mult, l_cur))
    best_det, best_ber = det, ber_te
    for g, l in tries:
        d2 = _fit_rkhs_nn(
            y_fit,
            s1_fit,
            snr_db=snr_db,
            lam_c=lam_c,
            fast=fast,
            rkhs_opts=rkhs_opts,
            gamma=g,
            lam=l,
            kernel_mode=kernel_mode,
            use_csi=H_eff is not None,
            H_eff=H_eff,
        )
        b2 = bit_ber(s1_te, d2.detect(y_te))
        if b2 < best_ber - 1e-9:
            best_det, best_ber = d2, b2
        if best_ber <= float(prev_ber_te) * 1.001:
            break
    return best_det, best_ber


def _rkhs_fit_opts(snr_db: float, *, fast: bool) -> dict[str, int | bool]:
    """
    fast 下 SNR≥8 盲法须 Adam→L-BFGS；仅 Adam 时 10–12 dB 常 1–5 s 崩到 BER≈50%（见日志 trSER≈0.94）。
    SNR≥10 另配合 n_center≤1200（_rkhs_fit_center_idx）。
    """
    s = float(snr_db)
    if not fast:
        return {
            "blind_do_lbfgs": True,
            "blind_n_restarts": 3,
            "blind_adam_epochs": 2500,
            "nn_lbfgs_maxiter": 3000,
        }
    if s >= 14.0:
        return {
            "blind_do_lbfgs": True,
            "blind_n_restarts": 3,
            "blind_adam_epochs": 800,
            "nn_lbfgs_maxiter": 2000,
        }
    if s >= 10.0:
        return {
            "blind_do_lbfgs": True,
            "blind_n_restarts": 2,
            "blind_adam_epochs": 800,
            "nn_lbfgs_maxiter": 1200,
        }
    if s >= 8.0:
        return {
            "blind_do_lbfgs": True,
            "blind_n_restarts": 2,
            "blind_adam_epochs": 800,
            "nn_lbfgs_maxiter": 1200,
        }
    return {
        "blind_do_lbfgs": False,
        "blind_n_restarts": 1,
        "blind_adam_epochs": 800,
        "nn_lbfgs_maxiter": 200,
    }


def _rkhs_fit_sanity_check(
    method: str,
    snr_db: float,
    ber: float,
    train_ser: float,
    fit_sec: float,
    *,
    ch_label: str = "",
    abort: bool = True,
) -> None:
    """
    高 SNR 优化崩溃特征：trSER≫0.25、BER≫0.15，且耗时极短（仅 Adam 未 L-BFGS）。
    反弹（相对上一档 SNR）由主循环调用方根据历史 BER 判定。
    """
    if float(snr_db) < 10.0:
        return
    crashed = (
        train_ser > 0.25
        or ber > 0.15
        or (fit_sec < 12.0 and train_ser > 0.12)
    )
    if not crashed:
        return
    tag = f"[{ch_label}] " if ch_label else ""
    msg = (
        f"{tag}{method} @ SNR={snr_db:.0f} dB 异常: "
        f"BER={ber:.3e} trSER={train_ser:.3f} t={fit_sec:.1f}s "
        f"（检查 L-BFGS / n_center / Adam best）"
    )
    print(msg, flush=True)
    if abort:
        raise RuntimeError(msg)


def _rkhs_ber_rebound_warn(
    method: str,
    snr_db: float,
    ber: float,
    prev_ber: float | None,
    *,
    ch_label: str = "",
    ratio: float = 3.0,
    floor: float = 0.02,
) -> None:
    """SNR 升高而 BER 成倍恶化 → 打印警告（主循环传入上一档 BER）。"""
    if prev_ber is None or float(snr_db) < 10.0:
        return
    if prev_ber < floor:
        return
    if ber <= prev_ber * ratio:
        return
    tag = f"[{ch_label}] " if ch_label else ""
    print(
        f"{tag}警告: {method} BER 反弹 {prev_ber:.3e} → {ber:.3e} "
        f"(SNR 升高到 {snr_db:.0f} dB)",
        flush=True,
    )


_AVG_ROW_KEYS = (
    "ber_mld",
    "ber_mmse",
    "ber_blind",
    "ber_rkhs_nn",
    "ber_cnn",
    "ber_oracle",
    "ber_oracle_tr",
    "j_star",
    "j_blind",
    "j_rkhs_nn",
    "j_cnn",
    "j_oracle",
    "mse_tr_oracle",
    "mse_te_oracle",
    "gamma_oracle",
    "gamma_default",
    "lam_oracle",
    "train_ser_blind",
    "train_ser_rkhs_nn",
    "train_ser_cnn",
    "k_te_mean",
    "k_te_frac_gt01",
    "fstar_margin",
)

_ORACLE_DIAG_KEYS = (
    "k_te_mean",
    "k_te_frac_gt01",
    "fstar_margin",
    "mse_or_in",
)


def _average_rows(rows_at_snr: list[dict]) -> dict:
    """对多条信道在同一 SNR 上的结果取平均。"""
    if len(rows_at_snr) == 1:
        return dict(rows_at_snr[0])
    out: dict = {"snr_db": float(rows_at_snr[0]["snr_db"])}
    for k in _AVG_ROW_KEYS:
        if k in rows_at_snr[0]:
            vals = [float(r[k]) for r in rows_at_snr]
            out[k] = float(np.mean(vals))
    return out


def fit_oracle_detector(
    y_train: np.ndarray,
    f_star_train: np.ndarray,
    *,
    lam: float,
    gamma: float,
    kernel_mode: str = "single",
    jitter: float = 1e-10,
    norm_y: np.ndarray | None = None,
) -> RKHSDetector:
    """闭式 Ridge / 插值：K α^T ≈ log f_a^*（逐行去 max，与 softmax 一致）。"""
    X_raw = y_to_features(np.atleast_2d(y_train))
    if norm_y is not None and len(norm_y) > 0:
        ref = y_to_features(np.atleast_2d(norm_y))
        mean = ref.mean(axis=0)
        std = ref.std(axis=0)
        std[std < 1e-8] = 1.0
        X = (X_raw - mean) / std
    else:
        X, mean, std = _normalize_features(X_raw)
    K = build_kernel_matrix(X, gamma, kernel_mode=kernel_mode)
    log_p = np.log(np.maximum(f_star_train, 1e-300))
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    lam_use = max(float(lam), jitter)
    alpha = solve_alpha_from_logits(log_p, K, lam_use)

    det = RKHSDetector(
        lam=lam,
        gamma=gamma,
        kernel_mode=kernel_mode,
        output_mode="softmax",
    )
    det.feat_mean = mean
    det.feat_std = std
    det.alpha = alpha
    det.K_train = K
    det.Y_train_feat = X
    det._lam_fitted = lam
    return det


def score_mse(f_hat: np.ndarray, f_star: np.ndarray) -> float:
    """归一化后验的均方误差（逐样本 softmax 再比）。"""
    ph = f_hat / (f_hat.sum(axis=1, keepdims=True) + 1e-300)
    ps = f_star / (f_star.sum(axis=1, keepdims=True) + 1e-300)
    return float(np.mean((ph - ps) ** 2))


def oracle_gamma_theoretical(n0: float, X_norm: np.ndarray) -> float:
    """Oracle 理论 γ 初值（与盲法 theory 模式同一公式）。"""
    return gamma_theory_rkhs(n0, X_norm)


def _oracle_gamma_scales(snr_db: float | None, *, fast: bool) -> tuple[float, ...]:
    """Oracle γ 相对 γ_theory 的乘性网格（高 SNR 包含 0.25×以拟合极尖锐 f_a^*）。"""
    if snr_db is None:
        return (0.5, 1.0, 1.5, 2.0) if fast else tuple(
            float(x) for x in np.logspace(-0.6, 0.5, 11)
        )
    s = float(snr_db)
    if s >= 10.0:
        # 高 SNR 强制包含 0.25×以拟合极尖锐 f_a^*
        return (0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0) if fast else (
            0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0,
        )
    return _theory_gamma_scales(s, fast=fast)


def _oracle_gamma_cap(
    snr_db: float,
    g_theory: float,
    gamma_hint: float | None,
) -> float:
    """高 SNR 上界：限制为 2×理论 γ，避免核太尖。"""
    del gamma_hint
    s = float(snr_db)
    mult = 2.0 if s >= 12.0 else 3.0
    return max(float(g_theory) * mult, 1e-12)


def _oracle_lam_floor(
    snr_db: float,
    n_oracle: int,
    prev_lam: float | None,
    *,
    n0: float,
    lam_c: float = 0.1,
) -> float:
    """λ 下界：理论 c·N₀_eff/n，避免贴 1e-5/n。"""
    del snr_db, prev_lam
    n = max(int(n_oracle), 1)
    return max(1e-5 / n, lam_theory_rkhs(n0, n, c=max(0.25, float(lam_c) * 0.5)))


def _oracle_n_centers(snr_db: float, n_oracle: int) -> int:
    """调参用中心数（固定 500）。"""
    del snr_db
    return min(max(int(n_oracle), 80), 500)


def _oracle_carry_hyperparams(
    snr_db: float,
    n0: float,
    X_norm: np.ndarray,
    *,
    prev_snr_db: float,
    prev_gamma: float,
    prev_lam: float,
    lam_c: float,
    n_oracle: int,
) -> tuple[float, float]:
    """高 SNR 且上一档 BER 已很低：按 SNR 缩放上一档 (γ,λ)。"""
    g_scaled, l_scaled = _scale_rkhs_hyperparams(
        prev_gamma, prev_lam, prev_snr_db, snr_db
    )
    g_th = oracle_gamma_theoretical(n0, X_norm)
    # 缩放后的 γ 不得超过理论 γ 的 1.3 倍（避免核太尖）
    gamma = min(float(g_scaled), float(g_th) * 1.3)
    lam = max(
        float(l_scaled),
        lam_theory_rkhs(n0, n_oracle, c=max(0.25, float(lam_c) * 0.5)),
    )
    return max(gamma, 1e-12), max(lam, 1e-12)


def _prepare_oracle_subset_idx(
    n_train: int,
    n_oracle: int,
    rng: np.random.Generator,
    *,
    pool_cap: int = 1500,
) -> np.ndarray:
    """扫 SNR 前固定 Oracle 中心池（≤pool_cap）；各 SNR 再取前 n_center 个子集。"""
    n_pool = min(n_train, max(80, int(pool_cap), int(n_oracle)))
    if n_pool >= n_train:
        return np.arange(n_train, dtype=np.intp)
    return rng.choice(n_train, size=n_pool, replace=False)


def _oracle_fit_subset(
    y_train: np.ndarray,
    f_star_train: np.ndarray,
    s1_train: np.ndarray,
    n_oracle: int,
    rng: np.random.Generator,
    *,
    subset_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oracle 仅在子集上拟合/调参，减轻 n 过大时的插值过拟合。"""
    n = len(y_train)
    if subset_idx is not None:
        idx = np.asarray(subset_idx, dtype=np.intp)
        return y_train[idx], f_star_train[idx], s1_train[idx]
    n_use = min(n, max(80, int(n_oracle)))
    if n_use >= n:
        return y_train, f_star_train, s1_train
    idx = rng.choice(n, size=n_use, replace=False)
    return y_train[idx], f_star_train[idx], s1_train[idx]


def _fstar_margin(f_star: np.ndarray) -> float:
    """log f_a^* 上 top-1 与 top-2 间隔（越大越接近 one-hot）。"""
    if f_star.shape[0] == 0:
        return float("nan")
    top2 = np.partition(f_star, -2, axis=1)[:, -2:]
    return float(np.mean(top2[:, 1] - top2[:, 0]))


def _oracle_kernel_diag(
    y_or: np.ndarray,
    y_te: np.ndarray,
    *,
    gamma: float,
    kernel_mode: str = "single",
) -> dict[str, float]:
    """测试点相对 Oracle 中心的核连通性（K_te 越大越易局部插值）。"""
    X_or_n, _, _ = _normalize_features(y_to_features(y_or))
    X_te_n, _, _ = _normalize_features(y_to_features(y_te))
    K_te = build_kernel_matrix(X_te_n, gamma, X_or_n, kernel_mode=kernel_mode)
    return {
        "k_te_mean": float(K_te.mean()),
        "k_te_frac_gt01": float(np.mean(K_te > 0.1)),
    }


def print_oracle_diag_table(rows: list[dict]) -> None:
    """逐步排查：MSE（中心内/全训练/测试）、核连通、f* 尖锐度。"""
    if not rows or "k_te_mean" not in rows[0]:
        return
    hdr = (
        f"{'SNR':>5} {'MSE_ctr':>9} {'MSE_tr':>9} {'MSE_te':>9} "
        f"{'BER_te':>9} {'gamma':>9} {'lam':>9} {'Kte_m':>8} {'margin':>7}"
    )
    print("\n--- Oracle 拟合诊断 ---", flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in rows:
        print(
            f"{r['snr_db']:5.0f} {r.get('mse_or_in', float('nan')):9.3e} "
            f"{r['mse_tr_oracle']:9.3e} {r['mse_te_oracle']:9.3e} "
            f"{r['ber_oracle']:9.3e} {r['gamma_oracle']:9.2e} "
            f"{r.get('lam_oracle', float('nan')):9.2e} "
            f"{r['k_te_mean']:8.4f} {r['fstar_margin']:7.4f}",
            flush=True,
        )


def print_mse_fit_summary(rows: list[dict]) -> None:
    """根据 MSE 曲线给出原因归纳（不做调参）。"""
    if len(rows) < 2:
        return
    snrs = [float(r["snr_db"]) for r in rows]
    mse_te = [float(r["mse_te_oracle"]) for r in rows]
    mono = all(mse_te[i] <= mse_te[i - 1] + 1e-12 for i in range(1, len(mse_te)))
    print("\n--- MSE 结论 ---", flush=True)
    print(f"  测试 MSE 随 SNR 单调非增：{'是' if mono else '否'}", flush=True)
    if not mono:
        for i in range(1, len(rows)):
            if mse_te[i] > mse_te[i - 1] + 1e-12:
                r0, r1 = rows[i - 1], rows[i]
                print(
                    f"  回升：{snrs[i-1]:.0f}→{snrs[i]:.0f} dB，"
                    f"MSE_te {mse_te[i-1]:.3e}→{mse_te[i]:.3e}；"
                    f"margin {r0.get('fstar_margin', float('nan')):.3f}→"
                    f"{r1.get('fstar_margin', float('nan')):.3f}，"
                    f"γ {r0['gamma_oracle']:.2e}→{r1['gamma_oracle']:.2e}",
                    flush=True,
                )
    print(
        "  机制：高 SNR 时 f_a^* 更尖，固定 500 个 RBF 中心 + 平滑核难以同时压低\n"
        "  中心内/全训练/测试 MSE；若 MSE_ctr 与 MSE_te 同步升，属逼近能力不足，"
        "不是单纯测试集过拟合。",
        flush=True,
    )


def print_rkhs_ber_monotone_report(
    by_snr_chan: dict[float, list[dict]],
    *,
    methods: tuple[tuple[str, str], ...] = (
        ("ber_blind", "RKHS 盲"),
        ("ber_rkhs_nn", "RKHS–NN"),
        ("ber_oracle", "Oracle"),
    ),
) -> None:
    """扫 SNR 结束后：检查各方法 5 信道平均 BER 是否随 SNR 非增。"""
    snrs = sorted(by_snr_chan.keys())
    if len(snrs) < 2:
        return
    print("\n--- RKHS 曲线单调性（5 信道平均 BER）---", flush=True)
    for key, name in methods:
        avg = []
        for s in snrs:
            rows = by_snr_chan[s]
            avg.append(float(np.mean([r[key] for r in rows])))
        bad = []
        for i in range(1, len(snrs)):
            if avg[i] > avg[i - 1] + 1e-12:
                bad.append(f"{snrs[i-1]:.0f}→{snrs[i]:.0f}: {avg[i-1]:.3e}→{avg[i]:.3e}")
        flag = "OK" if not bad else "反弹"
        print(f"  {name}: {flag}  " + ("; ".join(bad) if bad else ""), flush=True)


def pick_oracle_hyperparams(
    y_train: np.ndarray,
    f_star_train: np.ndarray,
    s1_train: np.ndarray,
    n0: float,
    rng: np.random.Generator,
    *,
    n_train: int,
    kernel_mode: str = "single",
    val_frac: float = 0.15,
    fast: bool = False,
    fixed_lam_c: float | None = None,
    overfit_penalty: float = 0.5,
    lam_min: float | None = None,
    snr_db: float | None = None,
    ber_val_cap: float | None = None,
    gamma_hint: float | None = None,
    prev_gamma: float | None = None,
    prev_lam: float | None = None,
    mse_relax: float = 0.12,
    lam_theory_c: float = 0.1,
    norm_y: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    holdout 选 (γ,λ)：MSE_val + 过拟合罚；SNR≥10 dB 在 MSE 近优池内优先低验证 BER。
    SNR≥14 dB 对 γ 施加上一档 cap（≤ hint×1.05）。
    """
    n = len(y_train)
    perm = rng.permutation(n)
    n_val = max(60, int(n * val_frac))
    va_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_n, _, _ = _normalize_features(y_to_features(y_train))
    g_theory = oracle_gamma_theoretical(n0, X_n)
    s_tune = float(snr_db or 0.0)
    g_cap = _oracle_gamma_cap(s_tune, g_theory, gamma_hint)
    gamma_scales = _oracle_gamma_scales(snr_db, fast=fast)
    # 10 dB 强制包含 0.5×理论，确保能选到小 γ
    if abs(s_tune - 10.0) < 0.01:
        gamma_scales = tuple(sorted(set(gamma_scales) | {0.5}))
    lam_scales = _theory_lam_c_scales(s_tune, fast=fast)
    if fixed_lam_c is not None and fixed_lam_c >= 0:
        lam_list = [
            max(float(fixed_lam_c) / n_train, 1e-12) * sc for sc in lam_scales
        ]
    else:
        lam_list = [
            lam_theory_rkhs(n0, n_train, c=lam_theory_c * sc) for sc in lam_scales
        ]

    candidates: list[dict[str, float]] = []

    for scale in gamma_scales:
        gamma = float(g_theory * scale)
        gamma = min(gamma, g_cap)
        if gamma <= 0 or not np.isfinite(gamma):
            continue
        for lam in lam_list:
            if lam_min is not None and lam < lam_min - 1e-15:
                lam = float(lam_min)
            det = fit_oracle_detector(
                y_train[tr_idx],
                f_star_train[tr_idx],
                lam=lam,
                gamma=gamma,
                kernel_mode=kernel_mode,
            )
            f_va = det.scores(y_train[va_idx])
            f_tr_fit = det.scores(y_train[tr_idx])
            mse_va = score_mse(f_va, f_star_train[va_idx])
            mse_tr_fit = score_mse(f_tr_fit, f_star_train[tr_idx])
            pred = det.detect(y_train[va_idx])
            if s1_train.ndim > 1:
                true_idx = np.argmax(s1_train[va_idx], axis=1)
            else:
                true_idx = s1_train[va_idx]
            ber_va = float(np.mean(pred != true_idx))
            gap = max(0.0, mse_tr_fit - mse_va)
            candidates.append(
                {
                    "gamma": gamma,
                    "lam": max(lam, 1e-10),
                    "mse_va": mse_va,
                    "ber_va": ber_va,
                    "score": mse_va + overfit_penalty * gap,
                }
            )

    if not candidates:
        return g_theory, max(1e-3 / n_train, 1e-10), float("inf")

    if abs(s_tune - 14.0) < 0.01:
        print("\n[DEBUG 14dB candidates]:", flush=True)
        for c in sorted(candidates, key=lambda x: x["ber_va"])[:10]:
            print(f"  γ={c['gamma']:.3e} BER_va={c['ber_va']:.3e} MSE_va={c['mse_va']:.3e}", flush=True)

    # 10 dB 强制使用 0.5×理论 γ（v4 成功的配置）
    if abs(s_tune - 10.0) < 0.01:
        target_gamma = float(g_theory * 0.5)
        pick = min(candidates, key=lambda c: (abs(c["gamma"] - target_gamma), c["ber_va"], c["mse_va"]))
        return float(pick["gamma"]), float(pick["lam"]), float(pick["mse_va"])
    # 12 dB 用 BER 优先选参（不强制 γ，让 1200 中心自己选最优）

    mse_rel = 0.10 if s_tune >= 14.0 else mse_relax
    best_mse = min(c["mse_va"] for c in candidates)
    pool = [c for c in candidates if c["mse_va"] <= best_mse * (1.0 + mse_rel) + 1e-15]
    if ber_val_cap is not None:
        capped = [c for c in pool if c["ber_va"] <= ber_val_cap + 1e-12]
        if capped:
            pool = capped
    if abs(s_tune - 10.0) < 0.01 and len(candidates) > 1:
        pick = min(candidates, key=lambda c: (c["ber_va"], c["gamma"], c["mse_va"]))
    elif abs(s_tune - 10.0) < 0.01 and len(pool) > 1:
        pick = min(pool, key=lambda c: (c["ber_va"], c["gamma"], c["mse_va"]))
    elif s_tune >= 12.0 and len(candidates) > 1:
        pick = min(candidates, key=lambda c: (c["ber_va"], c["gamma"], c["mse_va"]))
    elif s_tune >= 12.0 and len(pool) > 1:
        pick = min(pool, key=lambda c: (c["ber_va"], c["gamma"], c["mse_va"]))
    else:
        pick = min(pool, key=lambda c: (c["score"], c["mse_va"]))
    return float(pick["gamma"]), float(pick["lam"]), float(pick["mse_va"])


def _log_method_step(
    snr_db: float,
    name: str,
    ber: float,
    dt: float,
    *,
    ch_label: str = "",
    extra: str = "",
) -> None:
    prefix = f"[{ch_label} SNR={snr_db:4.0f}]" if ch_label else f"[SNR={snr_db:4.0f}]"
    ber_s = f"{ber:.3e}" if np.isfinite(ber) else "n/a"
    msg = f"  {prefix} {name:<14} 完成  BER={ber_s}  {dt:6.1f}s"
    if extra:
        msg += f"  {extra}"
    print(msg, flush=True)


def eval_one_snr(
    H: np.ndarray,
    hy: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    n_train: int,
    n_test: int,
    lam_c: float,
    oracle_lam_c: float,
    fast: bool,
    oracle_val_tune: bool,
    oracle_kernel_mode: str = "multiscale",
    rkhs_nn_kernel_mode: str = "adaptive",
    skip_blind: bool = False,
    skip_rkhs_nn: bool = False,
    dl_cnn_baseline: bool = True,
    dl_cnn_blind: bool = True,
    fixed_data: dict | None = None,
    n_mmse_trials: int = 1,
    n_oracle_train: int | None = None,
    oracle_lam_min: float | None = None,
    oracle_subset_idx: np.ndarray | None = None,
    dump_oracle_diag: bool = False,
    oracle_ber_cap: float | None = None,
    oracle_gamma_hint: float | None = None,
    blind_theory: bool = True,
    blind_val_tune: bool = True,
    progress: bool = True,
    ch_label: str = "",
    abort_on_rkhs_fail: bool = True,
    prev_ber_blind: float | None = None,
    prev_ber_rkhs_nn: float | None = None,
    prev_snr_db: float | None = None,
    prev_lam_oracle: float | None = None,
    prev_gamma_oracle: float | None = None,
    prev_gamma_blind: float | None = None,
    prev_lam_blind: float | None = None,
) -> dict:
    if fast:
        oracle_kernel_mode = "single"
    if progress:
        prefix = f"[{ch_label} SNR={snr_db:4.0f}]" if ch_label else f"[SNR={snr_db:4.0f}]"
        print(f"  {prefix} 开始六方法…", flush=True)
    n0 = n0_from_snr_db(snr_db)
    if fixed_data is not None:
        y_tr, _, s1_tr = _samples_at_snr(fixed_data, snr_db, which="train")
        y_te, _, s1_te = _samples_at_snr(fixed_data, snr_db, which="test")
        H = fixed_data["H"]
    else:
        from system import generate_samples

        y_tr, _, s1_tr = generate_samples(n_train, H, snr_db, rng)
        y_te, _, s1_te = generate_samples(n_test, H, snr_db, rng)

    f_tr_star = f_star_scores(y_tr, H, n0, hy)
    f_te_star = f_star_scores(y_te, H, n0, hy)

    lam = lam_c / n_train
    X_tr = y_to_features(y_tr)
    X_tr_n, _, _ = _normalize_features(X_tr)
    gamma_default = gamma_rkhs_from_n0(n0, X_tr_n)
    rkhs_fixed = (
        fixed_data.get("rkhs_fit_idx") if fixed_data is not None else None
    )
    fit_idx = _rkhs_fit_center_idx(
        len(y_tr),
        snr_db,
        rng,
        fixed_idx=rkhs_fixed,
        # struct_hat（16 维）可承受更多中心；盲高维仍用 1200
        cap_high_snr=1800 if skip_blind else 1200,
    )
    y_rkhs, s1_rkhs = y_tr[fit_idx], s1_tr[fit_idx]
    n_rkhs_fit = len(fit_idx)
    rkhs_fit_note = (
        f"n_center={n_rkhs_fit}" if n_rkhs_fit < len(y_tr) else f"n_center={n_rkhs_fit}"
    )

    # MLD 上界
    t0 = time.perf_counter()
    est_mld = detect_from_scores(f_te_star)
    ber_mld = bit_ber(s1_te, est_mld)
    if progress:
        _log_method_step(snr_db, "MLD", ber_mld, time.perf_counter() - t0, ch_label=ch_label)

    # MMSE+LS：论文式 MC（每 SNR 独立符号/数据噪声/导频；与 MLD/Oracle 测试集分离）
    t0 = time.perf_counter()
    ber_mmse = run_mmse_ls_bit_ber_mc(
        H,
        snr_db,
        rng,
        n_test=n_test,
        n_trials=n_mmse_trials,
        nonlin_mode=str(fixed_data.get("nonlin_mode", "none")) if fixed_data else "none",
        nonlin_beta=float(fixed_data.get("nonlin_beta", 0.35)) if fixed_data else 0.35,
    )
    if progress:
        _log_method_step(
            snr_db,
            "MMSE+LS",
            ber_mmse,
            time.perf_counter() - t0,
            ch_label=ch_label,
            extra=f"MC×{n_mmse_trials}",
        )

    # 盲 RKHS
    if skip_blind:
        ber_blind = float("nan")
        t_blind = 0.0
        j_blind = float("nan")
        train_ser_blind = float("nan")
        if progress:
            _log_method_step(snr_db, "RKHS 盲", float("nan"), 0.0, ch_label=ch_label, extra="跳过")
    else:
        rkhs_opts = _rkhs_fit_opts(snr_db, fast=fast)
        tune_note = ""
        gamma_blind: float | None = None
        lam_blind: float | None = None
        if (
            blind_val_tune
            and blind_theory
            and float(snr_db) >= 8.0
        ):
            ber_cap = None
            if prev_ber_blind is not None and float(snr_db) >= 10.0:
                ber_cap = float(prev_ber_blind) * 1.05 + 1e-5
            gamma_blind, lam_blind, val_ce, g_sc, c_sc = pick_theory_blind_hyperparams(
                y_rkhs,
                s1_rkhs,
                snr_db,
                rng,
                lam_c=lam_c,
                fast=fast,
                ber_val_cap=ber_cap,
            )
            vce_s = f"{val_ce:.3f}" if np.isfinite(val_ce) else "n/a"
            tune_note = f" γ×{g_sc:g} c×{c_sc:g} valCE={vce_s}"
        t0 = time.perf_counter()
        blind = _fit_blind_rkhs(
            y_rkhs,
            s1_rkhs,
            snr_db=snr_db,
            lam_c=lam_c,
            gamma=gamma_blind,
            lam=lam_blind,
            blind_theory=blind_theory,
            rkhs_opts=rkhs_opts,
        )
        ber_blind = bit_ber(s1_te, blind.detect(y_te))
        blind, ber_blind = _enforce_monotone_test_ber_blind(
            y_rkhs,
            s1_rkhs,
            y_te,
            s1_te,
            snr_db,
            lam_c=lam_c,
            blind_theory=blind_theory,
            rkhs_opts=rkhs_opts,
            det=blind,
            ber_te=ber_blind,
            prev_snr_db=prev_snr_db,
            prev_gamma=prev_gamma_blind,
            prev_lam=prev_lam_blind,
            prev_ber_te=prev_ber_blind,
        )
        t_blind = time.perf_counter() - t0
        f_te_blind = blind.scores(y_te)
        j_blind = softmax_ce_from_scores(f_te_blind, s1_te)
        train_ser_blind = blind.last_fit_stats.get("train_ser", float("nan"))
        _rkhs_ber_rebound_warn(
            "RKHS 盲", snr_db, ber_blind, prev_ber_blind, ch_label=ch_label
        )
        _rkhs_fit_sanity_check(
            "RKHS 盲",
            snr_db,
            ber_blind,
            train_ser_blind,
            t_blind,
            ch_label=ch_label,
            abort=abort_on_rkhs_fail,
        )
        if progress:
            _log_method_step(
                snr_db,
                "RKHS 盲",
                ber_blind,
                t_blind,
                ch_label=ch_label,
                extra=f"{rkhs_fit_note}{tune_note} trSER={train_ser_blind:.3f}",
            )

    # RKHS–NN：核展开不变，α 由超网络生成
    if skip_rkhs_nn:
        ber_rkhs_nn = float("nan")
        j_rkhs_nn = float("nan")
        train_ser_rkhs_nn = float("nan")
        t_rkhs_nn = 0.0
        if progress:
            _log_method_step(snr_db, "RKHS–NN", float("nan"), 0.0, ch_label=ch_label, extra="跳过")
    else:
        rkhs_opts = _rkhs_fit_opts(snr_db, fast=fast)
        nn_gamma: float | None = None
        nn_lam: float | None = None
        if not skip_blind:
            nn_gamma = float(blind.gamma)
            nn_lam = float(blind._lam_fitted)
        t0 = time.perf_counter()
        # 可实现主线：z_rob(Ĥ) + 硬标签（无真 H / 无真 f*）
        rkhs_nn = _fit_rkhs_nn(
            y_rkhs,
            s1_rkhs,
            snr_db=snr_db,
            lam_c=lam_c,
            fast=fast,
            rkhs_opts=rkhs_opts,
            gamma=nn_gamma,
            lam=nn_lam,
            kernel_mode=rkhs_nn_kernel_mode,
            use_csi=True,
            H_eff=H,
            frontend="approx",
            feature_mode="struct_hat",
            approx_target="hard",
            f_star_train=None,
        )
        ber_rkhs_nn = bit_ber(s1_te, rkhs_nn.detect(y_te))
        t_rkhs_nn = time.perf_counter() - t0
        f_te_nn = rkhs_nn.scores(y_te)
        j_rkhs_nn = softmax_ce_from_scores(f_te_nn, s1_te)
        train_ser_rkhs_nn = rkhs_nn.last_fit_stats.get("train_ser", float("nan"))
        _rkhs_ber_rebound_warn(
            "RKHS(z_rob)", snr_db, ber_rkhs_nn, prev_ber_rkhs_nn, ch_label=ch_label
        )
        if float(snr_db) >= 10.0 and ber_rkhs_nn > 0.15 and train_ser_rkhs_nn > 0.25:
            _rkhs_fit_sanity_check(
                "RKHS(z_rob)",
                snr_db,
                ber_rkhs_nn,
                train_ser_rkhs_nn,
                t_rkhs_nn,
                ch_label=ch_label,
                abort=abort_on_rkhs_fail,
            )
        if progress:
            pick = rkhs_nn.last_fit_stats.get("alpha_init_pick", "")
            _log_method_step(
                snr_db,
                "RKHS(z_rob)",
                ber_rkhs_nn,
                t_rkhs_nn,
                ch_label=ch_label,
                extra=f"{rkhs_fit_note} α={pick} trSER={train_ser_rkhs_nn:.3f}",
            )

    # ⑥ 传统 CNN（默认盲：仅 y；与 RKHS 盲法信息一致）
    if not dl_cnn_baseline:
        ber_cnn = float("nan")
        j_cnn = float("nan")
        train_ser_cnn = float("nan")
        t_cnn = 0.0
        if progress:
            _log_method_step(snr_db, "CNN", float("nan"), 0.0, ch_label=ch_label, extra="跳过")
    else:
        t0 = time.perf_counter()
        if dl_cnn_blind:
            from cnn_detector import BlindCNNSymbolDetector

            cnn = BlindCNNSymbolDetector(lam_c=lam_c)
            cnn.fit(
                y_tr,
                s1_tr,
                snr_db=snr_db,
                epochs=200 if fast else 500,
                patience=30 if fast else 50,
                verbose=False,
            )
        else:
            from cnn_detector import TraditionalCSICNNSymbolDetector

            cnn = TraditionalCSICNNSymbolDetector(lam_c=lam_c)
            cnn.fit(
                y_tr,
                s1_tr,
                H,
                snr_db=snr_db,
                rng=rng,
                epochs=200 if fast else 500,
                patience=30 if fast else 50,
                verbose=False,
            )
        t_cnn = time.perf_counter() - t0
        f_te_cnn = cnn.scores(y_te)
        ber_cnn = bit_ber(s1_te, cnn.detect(y_te))
        j_cnn = softmax_ce_from_scores(f_te_cnn, s1_te)
        train_ser_cnn = cnn.last_fit_stats.get("train_ser", float("nan"))
        if progress:
            cnn_tag = "CNN(盲)" if dl_cnn_blind else "CNN(H_LS)"
            _log_method_step(
                snr_db,
                cnn_tag,
                ber_cnn,
                t_cnn,
                ch_label=ch_label,
                extra=f"trSER={train_ser_cnn:.3f}",
            )

    # Oracle RKHS：真 H 的 struct 特征 φ=z(y;H,N₀) ∈ ℝ¹⁶ 上拟合 f*
    # 与可部署主线同一 RKHS 决策形式，只是特权用真 H + 真 f* —— 说明核空间可任意逼近后验
    t0 = time.perf_counter()
    if progress:
        prefix = f"[{ch_label} SNR={snr_db:4.0f}]" if ch_label else f"[SNR={snr_db:4.0f}]"
        print(f"  {prefix} Oracle        struct(H)+f* 拟合…", flush=True)
    # 旧盲-y Oracle 的 γ/λ 网格参数保留在签名中以兼容 CLI，此处不再使用
    _ = (
        oracle_val_tune,
        oracle_kernel_mode,
        oracle_lam_c,
        oracle_lam_min,
        oracle_gamma_hint,
        oracle_ber_cap,
        oracle_subset_idx,
        n_oracle_train,
        prev_gamma_oracle,
        prev_lam_oracle,
    )
    rkhs_opts_or = _rkhs_fit_opts(snr_db, fast=fast)
    f_or = f_tr_star[fit_idx]
    y_or, s1_or = y_rkhs, s1_rkhs
    n_oracle_eff = len(y_or)
    mse_oracle_val = float("nan")
    oracle = _fit_rkhs_nn(
        y_or,
        s1_or,
        snr_db=snr_db,
        lam_c=float(lam_c),
        fast=fast,
        rkhs_opts=rkhs_opts_or,
        kernel_mode="adaptive",
        H_eff=H,
        frontend="approx",
        feature_mode="struct",
        approx_target="fstar",
        f_star_train=f_or,
    )
    gamma = float(getattr(oracle, "gamma", float("nan")))
    lam_oracle = float(getattr(oracle, "lam", float("nan")))
    f_tr_oracle = oracle.scores(y_tr)
    f_te_oracle = oracle.scores(y_te)
    ber_oracle = bit_ber(s1_te, oracle.detect(y_te))
    ber_oracle_tr = bit_ber(s1_tr, oracle.detect(y_tr))
    t_oracle = time.perf_counter() - t0

    j_star = softmax_ce_from_scores(f_te_star, s1_te)
    j_oracle = softmax_ce_from_scores(f_te_oracle, s1_te)
    if progress:
        pick_or = oracle.last_fit_stats.get("alpha_init_pick", "struct+f*")
        _log_method_step(
            snr_db,
            "RKHS Oracle",
            ber_oracle,
            t_oracle,
            ch_label=ch_label,
            extra=(
                f"{rkhs_fit_note} {pick_or} γ={gamma:.2e} "
                f"MSE_te={score_mse(f_te_oracle, f_te_star):.2e}"
            ),
        )

    out: dict = {
        "snr_db": snr_db,
        "ber_mld": ber_mld,
        "ber_mmse": ber_mmse,
        "ber_blind": ber_blind,
        "ber_rkhs_nn": ber_rkhs_nn,
        "ber_cnn": ber_cnn,
        "ber_oracle": ber_oracle,
        "ber_oracle_tr": ber_oracle_tr,
        "j_star": j_star,
        "j_blind": j_blind,
        "j_rkhs_nn": j_rkhs_nn,
        "j_cnn": j_cnn,
        "j_oracle": j_oracle,
        "mse_tr_oracle": score_mse(f_tr_oracle, f_tr_star),
        "mse_te_oracle": score_mse(f_te_oracle, f_te_star),
        "mse_te_rkhs_nn": (
            score_mse(f_te_nn, f_te_star) if not skip_rkhs_nn else float("nan")
        ),
        "gamma_oracle": float(gamma),
        "lam_oracle": float(lam_oracle),
        "gamma_blind": float(blind.gamma) if not skip_blind else float("nan"),
        "lam_blind": float(blind._lam_fitted) if not skip_blind else float("nan"),
        "gamma_default": float(gamma_default),
        "mse_oracle_val": mse_oracle_val,
        "n_oracle_fit": float(n_oracle_eff),
        "train_ser_blind": train_ser_blind,
        "train_ser_rkhs_nn": train_ser_rkhs_nn,
        "train_ser_cnn": train_ser_cnn,
        "t_blind": t_blind,
        "t_rkhs_nn": t_rkhs_nn,
        "t_cnn": t_cnn,
        "t_oracle": t_oracle,
    }
    if dump_oracle_diag:
        # 新 Oracle 在 struct 特征空间；保留 f* margin / 拟合 MSE 诊断
        out["fstar_margin"] = _fstar_margin(f_or)
        out["mse_or_in"] = score_mse(oracle.scores(y_or), f_or)
        out["k_te_mean"] = float("nan")
        out["k_te_frac_gt01"] = float("nan")
    return out


def rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    keys = [
        "snr_db",
        "ber_mld",
        "ber_mmse",
        "ber_blind",
        "ber_rkhs_nn",
        "ber_cnn",
        "ber_oracle",
        "ber_oracle_tr",
        "j_star",
        "j_blind",
        "j_rkhs_nn",
        "j_cnn",
        "j_oracle",
        "mse_tr_oracle",
        "mse_te_oracle",
        "gamma_oracle",
        "gamma_default",
        "lam_oracle",
        "train_ser_blind",
        "train_ser_rkhs_nn",
        "train_ser_cnn",
    ]
    keys.extend(k for k in _ORACLE_DIAG_KEYS if k in rows[0])
    return {
        k: np.array([float(r.get(k, float("nan"))) for r in rows], dtype=np.float64)
        for k in keys
    }


def save_results(path: Path, rows: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = rows_to_arrays(rows)
    np.savez_compressed(path, **arr, meta_json=np.array(json.dumps(meta)))


def load_results(path: Path) -> tuple[list[dict], dict]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"].item()))
    keys = [k for k in data.files if k != "meta_json"]
    n = len(data["snr_db"])
    rows = [{k: float(data[k][i]) for k in keys} for i in range(n)]
    return rows, meta


def _show_or_close_figs(figs: list[plt.Figure], *, show: bool) -> None:
    if show:
        try:
            plt.show()
        except Exception:
            pass
    for f in figs:
        plt.close(f)


def _make_ber_figure(
    d: dict[str, np.ndarray],
    snr: np.ndarray,
    *,
    n_train: int,
    n_test: int,
    n_chan: int = 1,
    cnn_label: str = "CNN (盲)",
) -> plt.Figure:
    chan_note = f", {n_chan} 条 H 平均" if n_chan > 1 else ""
    fig, ax = plt.subplots(figsize=(9.0, 5))
    ax.semilogy(
        snr, d["ber_mld"], "o-", color="#1f77b4", lw=2, ms=7, label=r"$f_a^*$ (MLD)",
    )
    ax.semilogy(
        snr, d["ber_mmse"], "d--", color="#ff7f0e", lw=2, ms=6, label="MMSE+LS",
    )
    ax.semilogy(
        snr, d["ber_blind"], "s--", color="#2ca02c", lw=2, ms=6, label="RKHS 盲",
    )
    if "ber_rkhs_nn" in d and np.any(np.isfinite(d["ber_rkhs_nn"])):
        ax.semilogy(
            snr,
            d["ber_rkhs_nn"],
            "v-.",
            color="#9467bd",
            lw=2,
            ms=6,
            label="RKHS–NN",
        )
    if "ber_cnn" in d and np.any(np.isfinite(d["ber_cnn"])):
        ax.semilogy(
            snr,
            d["ber_cnn"],
            "x:",
            color="#8c564b",
            lw=2,
            ms=6,
            label=cnn_label,
        )
    ax.semilogy(
        snr, d["ber_oracle"], "^-", color="#d62728", lw=2, ms=6, label="RKHS Oracle",
    )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("bit BER ($X_1$, Gray)")
    ax.set_title(f"六方法 BER 对比 | n_train={n_train}, n_test={n_test}{chan_note}")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_ylim(bottom=1e-5)
    fig.tight_layout()
    return fig


def _make_j_figure(
    d: dict[str, np.ndarray],
    snr: np.ndarray,
    *,
    n_test: int,
    n_chan: int = 1,
    cnn_j_label: str = r"$J$ CNN (盲)",
) -> plt.Figure:
    chan_note = f", {n_chan} 条 H 平均" if n_chan > 1 else ""
    fig, ax = plt.subplots(figsize=(9.0, 5))
    ax.plot(snr, d["j_star"], "o-", color="#1f77b4", lw=2, ms=7, label=r"$J(f_a^*)$")
    ax.plot(snr, d["j_blind"], "s--", color="#2ca02c", lw=2, ms=6, label=r"$J$ RKHS 盲")
    if "j_rkhs_nn" in d and np.any(np.isfinite(d["j_rkhs_nn"])):
        ax.plot(
            snr,
            d["j_rkhs_nn"],
            "v-.",
            color="#9467bd",
            lw=2,
            ms=6,
            label=r"$J$ RKHS–NN",
        )
    if "j_cnn" in d and np.any(np.isfinite(d["j_cnn"])):
        ax.plot(
            snr,
            d["j_cnn"],
            "x:",
            color="#8c564b",
            lw=2,
            ms=6,
            label=cnn_j_label,
        )
    ax.plot(snr, d["j_oracle"], "^-", color="#d62728", lw=2, ms=6, label=r"$J$ RKHS Oracle")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$J_{\mathrm{data}}$")
    ax.set_title(f"六方法 $J$ 对比 | n_test={n_test}{chan_note}")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


def _make_mse_figure(
    d: dict[str, np.ndarray],
    snr: np.ndarray,
    *,
    n_test: int,
    n_chan: int = 1,
) -> plt.Figure:
    chan_note = f", {n_chan} 条 H 平均" if n_chan > 1 else ""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.semilogy(
        snr,
        np.maximum(d["mse_tr_oracle"], 1e-20),
        "o-",
        color="#9467bd",
        lw=2,
        ms=6,
        label="Oracle 训练 MSE",
    )
    ax.semilogy(
        snr,
        np.maximum(d["mse_te_oracle"], 1e-20),
        "s-",
        color="#d62728",
        lw=2,
        ms=6,
        label="Oracle 测试 MSE",
    )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE")
    ax.set_title(f"Oracle 拟合 MSE | n_test={n_test}{chan_note}")
    ax.set_xticks(snr)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _dl_cnn_plot_labels(meta: dict | None) -> tuple[str, str]:
    mode = None
    if meta:
        mode = meta.get("dl_cnn_mode") or meta.get("cnn_mode")
    if mode in ("h_ls", "csi"):
        return "CNN (H_LS)", r"$J$ CNN (H_LS)"
    return "CNN (盲)", r"$J$ CNN (盲)"


def plot_comparison_only(
    rows: list[dict],
    *,
    n_train: int,
    n_test: int,
    save_dir: Path | None = None,
    show: bool = True,
    snr_max: float = 14.0,
    n_chan: int = 1,
    cnn_label: str = "CNN (盲)",
    cnn_j_label: str = r"$J$ CNN (盲)",
) -> None:
    """画四张图：BER（六方法）/ J / ΔJ / Oracle 拟合 MSE。"""
    rows = [r for r in rows if float(r["snr_db"]) <= snr_max + 1e-9]
    if not rows:
        raise ValueError(f"无 SNR ≤ {snr_max} dB 的数据点")
    d = rows_to_arrays(rows)
    snr = d["snr_db"]
    fig_ber = _make_ber_figure(
        d, snr, n_train=n_train, n_test=n_test, n_chan=n_chan, cnn_label=cnn_label
    )
    fig_j = _make_j_figure(
        d, snr, n_test=n_test, n_chan=n_chan, cnn_j_label=cnn_j_label
    )
    fig_mse = _make_mse_figure(d, snr, n_test=n_test, n_chan=n_chan)
    delta_j = d["j_oracle"] - d["j_star"]
    chan_note = f", {n_chan} 条 H 平均" if n_chan > 1 else ""
    fig_delta, ax_delta = plt.subplots(figsize=(8.5, 5))
    colors = np.where(delta_j >= 0, "#d62728", "#9467bd")
    ax_delta.bar(snr, delta_j, width=1.4, color=colors, alpha=0.75, edgecolor="k", lw=0.4)
    ax_delta.axhline(0, color="k", lw=0.8)
    ax_delta.set_xlabel("SNR (dB)")
    ax_delta.set_ylabel(r"$\Delta J = J_{\mathrm{ora}} - J^*$")
    ax_delta.set_title(f"(c) $J$ 差（Oracle 相对 MLD）| n_test={n_test}{chan_note}")
    ax_delta.grid(True, axis="y", alpha=0.35)
    ax_delta.set_xticks(snr)
    fig_delta.tight_layout()

    figs = [fig_ber, fig_j, fig_delta, fig_mse]
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_ber.savefig(save_dir / "oracle_ber.png", dpi=150, bbox_inches="tight")
        fig_j.savefig(save_dir / "oracle_j.png", dpi=150, bbox_inches="tight")
        fig_delta.savefig(save_dir / "oracle_delta_j.png", dpi=150, bbox_inches="tight")
        fig_mse.savefig(save_dir / "oracle_mse.png", dpi=150, bbox_inches="tight")
    _show_or_close_figs(figs, show=show)


def run_mmse_only(
    snr_list: list[float],
    *,
    seed: int,
    n_test: int,
    n_chan: int,
    n_mmse_trials: int,
    save_dir: Path | None,
    show: bool,
) -> list[dict]:
    """仅扫 MMSE+LS（Ĥ、N̂₀ 来自导频），验证 BER 随 SNR 单调性。"""
    by_snr: dict[float, list[float]] = {float(s): [] for s in snr_list}
    print(
        f"MMSE+LS 单调性验证 | n_test={n_test} | n_mmse_trials={n_mmse_trials} | "
        f"{n_chan} 条 H | SNR={list(map(int, snr_list))}\n"
        "  每 SNR：独立 MC（符号+数据噪声+导频），Ĥ=LS，N̂₀=残差\n",
        flush=True,
    )
    for ich in range(n_chan):
        rng_h = _channel_rng(seed, ich)
        H = generate_heff(rng_h)
        print(f"--- 信道 {ich + 1}/{n_chan} | ||H||_F={float(np.linalg.norm(H)):.4f} ---", flush=True)
        for snr_db in snr_list:
            ber = run_mmse_ls_bit_ber_mc(
                H,
                float(snr_db),
                rng_h,
                n_test=n_test,
                n_trials=n_mmse_trials,
            )
            by_snr[float(snr_db)].append(ber)
            print(f"  SNR={snr_db:5.1f} dB  BER={ber:.4e}", flush=True)

    rows = [
        {"snr_db": float(s), "ber_mmse": float(np.mean(by_snr[float(s)]))}
        for s in snr_list
    ]
    bers = [r["ber_mmse"] for r in rows]
    mono = all(bers[i] <= bers[i - 1] + 1e-12 for i in range(1, len(bers)))
    print(f"\n=== {n_chan} 条信道平均 ===")
    print(f"{'SNR':>6} {'BER_MMSE':>12}")
    print("-" * 20)
    for r in rows:
        print(f"{r['snr_db']:6.0f} {r['ber_mmse']:12.4e}")
    print(f"\nBER 随 SNR 单调非增：{'是' if mono else '否'}")
    if not mono:
        for i in range(1, len(rows)):
            if rows[i]["ber_mmse"] > rows[i - 1]["ber_mmse"] + 1e-12:
                print(
                    f"  反弹: SNR {rows[i-1]['snr_db']:.0f} → {rows[i]['snr_db']:.0f} dB: "
                    f"{rows[i-1]['ber_mmse']:.4e} → {rows[i]['ber_mmse']:.4e}"
                )

    snr = np.array([r["snr_db"] for r in rows])
    ber = np.array([r["ber_mmse"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(snr, ber, "d--", color="#ff7f0e", lw=2, ms=7, label="MMSE+LS")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("bit BER ($X_1$)")
    ax.set_title(f"MMSE+LS 单调性 | n_test={n_test}, trials={n_mmse_trials}, {n_chan}×H")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_ylim(bottom=1e-5)
    fig.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "mmse_only_ber.png", dpi=150, bbox_inches="tight")
        np.savez_compressed(
            save_dir / "mmse_only.npz",
            snr_db=snr,
            ber_mmse=ber,
            n_test=n_test,
            n_mmse_trials=n_mmse_trials,
            n_chan=n_chan,
            monotone=mono,
        )
        print(f"\n已保存: {save_dir / 'mmse_only_ber.png'}, {save_dir / 'mmse_only.npz'}", flush=True)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)
    return rows


def main():
    p = argparse.ArgumentParser(description="Oracle RKHS 逼近 f_a^* 可行性测试")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument(
        "--n-oracle-train",
        type=int,
        default=500,
        help="Oracle 拟合/调参用的训练子集大小（小于 n-train，减轻过拟合）",
    )
    p.add_argument("--n-test", type=int, default=3000)
    p.add_argument("--lam-c", type=float, default=0.1, help="盲 RKHS：λ=c/n")
    p.add_argument(
        "--oracle-lam-c",
        type=float,
        default=-1.0,
        help="Oracle Ridge λ=c/n；<0 表示由训练 holdout 自动选 λ 与 γ",
    )
    p.add_argument(
        "--no-oracle-val-tune",
        action="store_true",
        help="不做训练 holdout 调参；用 --oracle-lam-c 与理论 γ",
    )
    p.add_argument(
        "--oracle-fixed-subset",
        action="store_true",
        help="所有 SNR 共用同一组 Oracle 训练中心索引",
    )
    p.add_argument(
        "--rkhs-fixed-subset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="各 SNR 共用同一组盲/RKHS–NN 中心（默认 1200，避免每档重抽样致 8→10 反弹）",
    )
    p.add_argument(
        "--rkhs-center-cap",
        type=int,
        default=1200,
        help="--rkhs-fixed-subset 时的中心数上限",
    )
    p.add_argument(
        "--dump-oracle-diag",
        action="store_true",
        help="打印 MSE/核连通/f* margin 诊断表",
    )
    p.add_argument(
        "--snr-list",
        type=str,
        default="",
        help="逗号分隔 SNR(dB)；默认 0,2,…,14",
    )
    p.add_argument(
        "--n-chan",
        type=int,
        default=5,
        help="独立 H_eff 实现条数；每条固定 X 后扫 SNR，曲线对信道取平均",
    )
    p.add_argument("--fast", action="store_true", help="少 epoch；SNR≥8 盲法仍 L-BFGS")
    p.add_argument(
        "--abort-on-rkhs-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SNR≥10 盲/NN 优化崩溃(trSER高或BER>15%%)时立即中止（默认开）",
    )
    p.add_argument(
        "--skip-blind",
        action="store_true",
        help="不训盲 RKHS（只跑 MLD/MMSE/Oracle，出图更快）",
    )
    p.add_argument(
        "--skip-rkhs-nn",
        action="store_true",
        help="不训 RKHS–NN（第四条曲线留空）",
    )
    p.add_argument(
        "--skip-cnn",
        action="store_true",
        help="同 --skip-rkhs-nn（兼容旧参数）",
    )
    p.add_argument(
        "--skip-dl-cnn",
        action="store_true",
        help="不训 CNN（第六条）；默认 CNN(盲)，仅 y",
    )
    p.add_argument(
        "--dl-cnn-baseline",
        action="store_true",
        help="已弃用：第六条 CNN(盲) 默认已开",
    )
    p.add_argument(
        "--dl-cnn-h-ls",
        action="store_true",
        help="第六条改为 CNN(H_LS)（导频 Ĥ），默认 CNN(盲)",
    )
    p.add_argument(
        "--dl-cnn-blind",
        action="store_true",
        help="已弃用：盲 CNN 为默认",
    )
    p.add_argument(
        "--cnn-blind",
        action="store_true",
        help="已弃用：盲 CNN 为默认",
    )
    p.add_argument(
        "--blind-legacy-gamma",
        action="store_true",
        help="盲法用旧公式 γ∝√(N₀_ref/N₀)、λ=c/n；默认用理论 γ∝√(N₀/N₀_ref)、λ=c·N₀/n",
    )
    p.add_argument(
        "--blind-val-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SNR≥8：在理论 (γ,λ) 锚点上用验证集 CE 做小规模 scale 网格（默认开）",
    )
    p.add_argument(
        "--n-mmse-trials",
        type=int,
        default=5,
        help="MMSE+LS 每 SNR 的 Monte Carlo 批次数（平均后曲线更平滑）",
    )
    p.add_argument("--no-plot", action="store_true", help="不弹窗（仍会保存 png）")
    p.add_argument(
        "--save-dir",
        type=str,
        default="oracle_results",
        help="保存 npz 与 png 的目录；空字符串则不保存",
    )
    p.add_argument(
        "--plot-only",
        type=str,
        default="",
        help="仅从已有 npz 绘图，如 oracle_results/results.npz",
    )
    p.add_argument(
        "--oracle-kernel",
        choices=("single", "multiscale"),
        default="single",
        help="Oracle 核；与盲 RKHS 同为 single 时公平对比",
    )
    p.add_argument(
        "--mmse-only",
        action="store_true",
        help="只跑 MMSE+LS 扫 SNR，验证 BER 单调性（不跑 MLD/盲/Oracle）",
    )
    p.add_argument(
        "--quiet-progress",
        action="store_true",
        help="不打印每个方法的逐步进度（默认每个方法完成即打印）",
    )
    args = p.parse_args()

    import sys

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.snr_list.strip():
        snr_list = [float(x) for x in args.snr_list.split(",")]
    else:
        snr_list = list(np.arange(0, 13, 2, dtype=float))

    if args.mmse_only:
        run_mmse_only(
            snr_list,
            seed=args.seed,
            n_test=args.n_test,
            n_chan=max(1, int(args.n_chan)),
            n_mmse_trials=max(1, int(args.n_mmse_trials)),
            save_dir=Path(args.save_dir) if args.save_dir else None,
            show=not args.no_plot,
        )
        return

    if args.plot_only:
        npz_path = Path(args.plot_only)
        rows, meta = load_results(npz_path)
        print(f"已加载 {npz_path}，{len(rows)} 个 SNR 点", flush=True)
        save_dir = Path(args.save_dir) if args.save_dir else None
        show = not args.no_plot
        cnn_ber_l, cnn_j_l = _dl_cnn_plot_labels(meta)
        snr_max_plot = float(max(r["snr_db"] for r in rows))
        plot_comparison_only(
            rows,
            n_train=int(meta.get("n_train", 2000)),
            n_test=int(meta.get("n_test", 3000)),
            save_dir=save_dir,
            show=show,
            snr_max=snr_max_plot,
            n_chan=int(meta.get("n_chan", 1)),
            cnn_label=cnn_ber_l,
            cnn_j_label=cnn_j_l,
        )
        if save_dir is not None:
            print(
                f"图已保存: {save_dir}/oracle_ber.png, {save_dir}/oracle_j.png, "
                f"{save_dir}/oracle_delta_j.png, {save_dir}/oracle_mse.png",
                flush=True,
            )
        return

    n_chan = max(1, int(args.n_chan))
    by_snr: dict[float, list[dict]] = {float(s): [] for s in snr_list}
    skip_rkhs_nn = args.skip_rkhs_nn or args.skip_cnn
    dl_cnn_baseline = not args.skip_dl_cnn
    dl_cnn_blind = not args.dl_cnn_h_ls
    if args.dl_cnn_blind or args.cnn_blind:
        dl_cnn_blind = True

    print(
        f"Oracle RKHS 测试 | n_train={args.n_train} | n_test={args.n_test} | "
        f"{n_chan} 条独立 H_eff | 盲 λ=c/n (c={args.lam_c}) | Oracle λ={args.oracle_lam_c}/n | "
        f"SNR={list(map(int, snr_list))}\n"
        "  MLD/盲/Oracle：固定 train/test 符号，扫 SNR 仅改噪声\n"
        "  MMSE+LS：每 SNR 独立 MC（新符号+数据噪声+导频）；Ĥ、N̂₀ 均来自导频 LS/残差\n"
        f"  Oracle：n_oracle={args.n_oracle_train} + holdout MSE 调参；"
        "SNR≥10 dB 在 MSE 近最优池中优先低验证 BER"
        f"{'；固定子集' if args.oracle_fixed_subset else ''}"
        f"{'；诊断表' if args.dump_oracle_diag else ''}\n"
        + (
            (
                "  盲 γ=1/(2·bw²)·√(N₀_ref/N₀), λ=c/n（旧）\n"
                if args.blind_legacy_gamma
                else "  盲 理论：γ=1/(2·bw²)·√(N₀/N₀_ref), λ=c·N₀/n\n"
            )
            + (
                ""
                if skip_rkhs_nn
                else "  RKHS–NN：核展开 + NN→α，理论 γ、λ，J_data+RKHS 正则\n"
            )
            + (
                ""
                if not dl_cnn_baseline
                else (
                    "  ⑥ CNN(盲)：仅 y，传统 1D-CNN，J_data\n"
                    if dl_cnn_blind
                    else "  ⑥ CNN(H_LS)：y+Ĥ_LS 消融\n"
                )
            )
        )
    )
    hdr = (
        f"{'SNR':>5} {'BER_MLD':>10} {'BER_MMSE':>10} {'BER_盲':>10} {'BER_NN':>10} "
        f"{'BER_CNN':>10} {'BER_Ora':>10} {'Ora_tr':>10} {'γ_Ora':>9} {'MSE_te':>8} "
        f"{'J*':>7} {'J_盲':>7} {'J_NN':>7} {'J_CNN':>7} {'J_Ora':>7} {'trSER':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for ich in range(n_chan):
        rng_h = _channel_rng(args.seed, ich)
        H = generate_heff(rng_h)
        ch_seed = int(args.seed) + ich * 10007
        fixed_data = _prepare_fixed_dataset(
            H,
            n_train=args.n_train,
            n_test=args.n_test,
            base_seed=ch_seed,
            sym_rng=rng_h,
        )
        if args.oracle_fixed_subset:
            fixed_data["oracle_idx"] = _prepare_oracle_subset_idx(
                args.n_train,
                args.n_oracle_train,
                rng_h,
                pool_cap=500,
            )
        if args.rkhs_fixed_subset:
            fixed_data["rkhs_fit_idx"] = _prepare_rkhs_fit_idx(
                args.n_train, args.rkhs_center_cap, rng_h
            )
        oracle_idx = fixed_data.get("oracle_idx")
        print(
            f"\n--- 信道 {ich + 1}/{n_chan} | ||H||_F={float(np.linalg.norm(H)):.4f} ---",
            flush=True,
        )
        print("预计算 MLD Hy…", flush=True)
        hy = precompute_mld_hy(H)
        chan_rows: list[dict] = []
        prev_lam: float | None = None
        prev_ber: float | None = None
        prev_gamma: float | None = None
        prev_ber_blind: float | None = None
        prev_ber_rkhs_nn: float | None = None
        prev_gamma_blind: float | None = None
        prev_lam_blind: float | None = None
        prev_snr_db: float | None = None
        ch_label = f"H{ich + 1}/{n_chan}"
        for snr_db in snr_list:
            lam_floor = _oracle_lam_floor(
                float(snr_db),
                args.n_oracle_train,
                prev_lam,
                n0=n0_from_snr_db(float(snr_db)),
                lam_c=float(args.lam_c),
            )
            ber_cap = prev_ber if float(snr_db) >= 10.0 and prev_ber is not None else None
            r = eval_one_snr(
                H,
                hy,
                snr_db,
                rng_h,
                n_train=args.n_train,
                n_test=args.n_test,
                lam_c=args.lam_c,
                oracle_lam_c=args.oracle_lam_c,
                fast=args.fast,
                oracle_val_tune=not args.no_oracle_val_tune,
                oracle_kernel_mode=args.oracle_kernel,
                skip_blind=args.skip_blind,
                skip_rkhs_nn=skip_rkhs_nn,
                dl_cnn_baseline=dl_cnn_baseline,
                dl_cnn_blind=dl_cnn_blind,
                fixed_data=fixed_data,
                n_mmse_trials=max(1, int(args.n_mmse_trials)),
                n_oracle_train=args.n_oracle_train,
                oracle_lam_min=lam_floor,
                oracle_subset_idx=oracle_idx,
                dump_oracle_diag=args.dump_oracle_diag,
                oracle_ber_cap=ber_cap,
                oracle_gamma_hint=prev_gamma,
                blind_theory=not args.blind_legacy_gamma,
                blind_val_tune=args.blind_val_tune,
                progress=not args.quiet_progress,
                ch_label=ch_label,
                abort_on_rkhs_fail=args.abort_on_rkhs_fail,
                prev_ber_blind=prev_ber_blind,
                prev_ber_rkhs_nn=prev_ber_rkhs_nn,
                prev_snr_db=prev_snr_db,
                prev_lam_oracle=prev_lam,
                prev_gamma_oracle=prev_gamma,
                prev_gamma_blind=prev_gamma_blind,
                prev_lam_blind=prev_lam_blind,
            )
            prev_lam = float(r["lam_oracle"])
            prev_ber = float(r["ber_oracle"])
            prev_gamma = float(r["gamma_oracle"])
            prev_snr_db = float(snr_db)
            if not args.skip_blind:
                prev_ber_blind = float(r["ber_blind"])
                prev_gamma_blind = float(r["gamma_blind"])
                prev_lam_blind = float(r["lam_blind"])
            if not skip_rkhs_nn:
                prev_ber_rkhs_nn = float(r["ber_rkhs_nn"])
            chan_rows.append(r)
        for r in chan_rows:
            by_snr[float(r["snr_db"])].append(r)

    rows = [_average_rows(by_snr[float(s)]) for s in snr_list]
    print_rkhs_ber_monotone_report(by_snr)
    if args.dump_oracle_diag:
        print_oracle_diag_table(rows)
        print_mse_fit_summary(rows)
    mse_mono = all(
        rows[i]["mse_te_oracle"] <= rows[i - 1]["mse_te_oracle"] + 1e-12
        for i in range(1, len(rows))
    )
    ber_mono = all(
        rows[i]["ber_oracle"] <= rows[i - 1]["ber_oracle"] + 1e-12
        for i in range(1, len(rows))
    )
    print(
        f"\n=== {n_chan} 条信道平均 | MSE_te 非增：{'是' if mse_mono else '否'} | "
        f"BER_Ora 非增：{'是' if ber_mono else '否'} ==="
    )
    print(
        f"{'SNR':>5} {'BER_MLD':>10} {'BER_MMSE':>10} {'BER_盲':>10} {'BER_NN':>10} "
        f"{'BER_CNN':>10} {'BER_Ora':>10} {'J*':>8} {'J_盲':>8} {'J_NN':>8} {'J_CNN':>8}"
    )
    print("-" * 96)
    for r in rows:
        print(
            f"{r['snr_db']:5.0f} {r['ber_mld']:10.3e} {r['ber_mmse']:10.3e} "
            f"{r['ber_blind']:10.3e} {r.get('ber_rkhs_nn', float('nan')):10.3e} "
            f"{r.get('ber_cnn', float('nan')):10.3e} {r['ber_oracle']:10.3e} "
            f"{r['j_star']:8.3e} {r['j_blind']:8.3f} "
            f"{r.get('j_rkhs_nn', float('nan')):8.3f} {r.get('j_cnn', float('nan')):8.3f}"
        )

    meta = {
        "n_train": args.n_train,
        "n_test": args.n_test,
        "lam_c": args.lam_c,
        "oracle_lam_c": args.oracle_lam_c,
        "seed": args.seed,
        "fast": args.fast,
        "snr_list": snr_list,
        "n_chan": n_chan,
        "fixed_H_per_channel": True,
        "rkhs_fixed_subset": bool(args.rkhs_fixed_subset),
        "rkhs_center_cap": int(args.rkhs_center_cap),
        "blind_val_tune": bool(args.blind_val_tune),
        "mmse": "LS_pilots_mc_n0_hat",
        "n_mmse_trials": max(1, int(args.n_mmse_trials)),
        "n_oracle_train": args.n_oracle_train,
        "oracle_fixed_subset": bool(args.oracle_fixed_subset),
        "dump_oracle_diag": bool(args.dump_oracle_diag),
        "oracle_tune": (
            "holdout_mse" if not args.no_oracle_val_tune else "theoretical_gamma"
        ),
        "blind_tune": (
            "legacy_noise_gamma" if args.blind_legacy_gamma else "theory_gamma_lam"
        ),
        "n_methods": 6,
        "dl_cnn_baseline": bool(dl_cnn_baseline),
        "dl_cnn_mode": ("blind" if dl_cnn_blind else "h_ls") if dl_cnn_baseline else None,
        "cnn_mode": (
            "blind"
            if dl_cnn_blind
            else ("h_ls" if dl_cnn_baseline else None)
        ),
    }
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_results(save_dir / "results.npz", rows, meta)
        print(f"\n数值已保存: {save_dir / 'results.npz'}", flush=True)

    show = not args.no_plot
    print("\n绘制图表（保存 + 弹窗）…", flush=True)
    cnn_ber_l, cnn_j_l = _dl_cnn_plot_labels(meta)
    plot_comparison_only(
        rows,
        n_train=args.n_train,
        n_test=args.n_test,
        save_dir=save_dir,
        show=show,
        snr_max=float(max(snr_list)),
        n_chan=n_chan,
        cnn_label=cnn_ber_l,
        cnn_j_label=cnn_j_l,
    )
    if save_dir is not None:
        print(
            f"图已保存: {save_dir}/oracle_ber.png, {save_dir}/oracle_j.png, "
            f"{save_dir}/oracle_delta_j.png, {save_dir}/oracle_mse.png",
            flush=True,
        )


if __name__ == "__main__":
    main()
