"""
16×5 MU-MIMO（方案 A）：MMSE+LS vs 边际 MLD vs 盲 RKHS。

每个 (H, SNR)：
  - MMSE+LS：一次导频估信道
  - MLD：真 H_eff（上界，仅评估）
  - RKHS：仅 (y, s1) 训练，γ=1/N₀（与 f_a^* 高斯核一致），λ=c/n

输出：测试集 J_data(f_a^*)、J_data(RKHS)、ΔJ，以及三者 bit BER。
  MLD / J(f_a^*)：n_test_mld（默认 500）；MMSE+LS / RKHS：n_test（默认 5000）。
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from kernel_rkhs import RKHSDetector
from mld import marginal_mld_detect, marginal_scores, precompute_mld_hy
from mmse import (
    estimate_n0_from_residual,
    generate_pilots,
    ls_estimate_heff,
    mmse_detect_x1,
)
from objective import softmax_ce_from_scores
from system import M, K, bit_ber, generate_heff, generate_samples, n0_from_snr_db

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def f_star_scores(
    y: np.ndarray,
    H_eff: np.ndarray,
    n0: float,
    hy_cache,
) -> np.ndarray:
    """MLD 边际打分 → 正数 f_a^*(y)（仅评估用；大 K 为高斯干扰代理）。"""
    log_f = marginal_scores(y, H_eff, n0, log_domain=True, hy_cache=hy_cache)
    log_f -= log_f.max(axis=1, keepdims=True)
    return np.exp(log_f)


def run_mmse_genie_bit_ber(
    y_test: np.ndarray,
    x1_test: np.ndarray,
    H_eff: np.ndarray,
    snr_db: float,
    mapping: str = "gray",
) -> float:
    """MMSE 用真 H_eff、真 N₀（与 MLD 同信道知识），扫 SNR 时单调。"""
    from mmse import mmse_detect_x1

    n0 = n0_from_snr_db(snr_db)
    est = mmse_detect_x1(y_test, H_eff, n0)
    return bit_ber(x1_test, est, mapping=mapping)


def run_mmse_ls_bit_ber(
    y_test: np.ndarray,
    x1_test: np.ndarray,
    H_eff: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    mapping: str = "gray",
    *,
    pilot_rng: np.random.Generator | None = None,
    use_true_n0: bool = False,
) -> float:
    """导频 LS 得 Ĥ；MMSE 均衡用 N̂₀（默认）或真 N₀（SNR 扫频仿真）。"""
    n0 = n0_from_snr_db(snr_db)
    std = np.sqrt(n0 / 2)
    X_p = generate_pilots()
    T_p = X_p.shape[1]
    prng = pilot_rng if pilot_rng is not None else rng
    noise_p = std * (prng.standard_normal((M, T_p)) + 1j * prng.standard_normal((M, T_p)))
    Y_p = H_eff @ X_p + noise_p
    H_hat = ls_estimate_heff(Y_p, X_p)
    n0_hat = estimate_n0_from_residual(Y_p, H_hat, X_p)
    n0_mmse = n0 if use_true_n0 else n0_hat
    est = mmse_detect_x1(y_test, H_hat, n0_mmse)
    return bit_ber(x1_test, est, mapping=mapping)


def run_mmse_ls_bit_ber_mc(
    H_eff: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    n_test: int,
    n_trials: int = 1,
    mapping: str = "gray",
    use_true_n0: bool = False,
    nonlin_mode: str = "none",
    nonlin_beta: float = 0.35,
) -> float:
    """
    论文式 Monte Carlo（与 run_experiment 主循环一致）：
    每个 trial 独立生成 (符号, 数据噪声, 导频)；Ĥ、N̂₀ 均来自导频 LS 与残差。
    """
    if n_trials < 1:
        raise ValueError("n_trials 须 >= 1")
    acc = 0.0
    for _ in range(n_trials):
        y_te, _, s1_te = generate_samples(
            n_test,
            H_eff,
            snr_db,
            rng,
            nonlin_mode=nonlin_mode,
            nonlin_beta=nonlin_beta,
        )
        acc += run_mmse_ls_bit_ber(
            y_te,
            s1_te,
            H_eff,
            snr_db,
            rng,
            mapping=mapping,
            use_true_n0=use_true_n0,
        )
    return acc / n_trials


def run_mld_bit_ber(
    y_test: np.ndarray,
    x1_test: np.ndarray,
    H_eff: np.ndarray,
    snr_db: float,
    hy_cache: np.ndarray,
    mapping: str = "gray",
) -> float:
    n0 = n0_from_snr_db(snr_db)
    est = marginal_mld_detect(y_test, H_eff, n0, hy_cache=hy_cache)
    return bit_ber(x1_test, est, mapping=mapping)


def run_rkhs(
    y_train: np.ndarray,
    s1_train: np.ndarray,
    y_test: np.ndarray,
    s1_test: np.ndarray,
    y_test_mld: np.ndarray,
    s1_test_mld: np.ndarray,
    H_eff: np.ndarray,
    snr_db: float,
    hy_cache: np.ndarray,
    *,
    n_restarts: int,
    lam_c: float,
    output_mode: str = "softmax",
    kernel_mode: str = "single",
    use_apspm: bool = False,
    margin_tau: float = 0.7,
    margin_tau_start: float = 0.3,
    margin_mu: float = 0.4,
    use_margin_loss: bool | None = None,
    apsm_mode: str = "local",
    mapping: str = "gray",
    verbose: bool = False,
) -> tuple[float, float, float, float, dict[str, float]]:
    """返回 (J_star, J_rkhs, ΔJ, bit_ber_rkhs, fit_stats)。

    J(f_a^*) 在 n_test_mld 上算（省 16^4 边际化）；J(RKHS)/BER 在 n_test 上算。
    ΔJ 在共用子集 n_test_mld 上算，便于同集比较。
    """
    n0 = n0_from_snr_db(snr_db)
    det = RKHSDetector(
        lam_c=lam_c,
        kernel_mode=kernel_mode,
        output_mode=output_mode,
        use_apspm=use_apspm,
        margin_tau=margin_tau,
        margin_tau_start=margin_tau_start,
        margin_mu=margin_mu,
        use_margin_loss=use_margin_loss,
        apsm_mode=apsm_mode,
        gamma_mode="noise",
        tune_hyperparams=False,
        n_restarts=n_restarts,
    )
    det.fit(y_train, s1_train, verbose=verbose, snr_db=float(snr_db))

    f_star_mld = f_star_scores(y_test_mld, H_eff, n0, hy_cache)
    f_rkhs_mld = det.scores(y_test_mld)
    f_rkhs = det.scores(y_test)

    j_star = softmax_ce_from_scores(f_star_mld, s1_test_mld)
    j_rkhs = softmax_ce_from_scores(f_rkhs, s1_test)
    j_rkhs_mld = softmax_ce_from_scores(f_rkhs_mld, s1_test_mld)
    delta_j = j_rkhs_mld - j_star

    est = det.detect(y_test)
    ber_rkhs = bit_ber(s1_test, est, mapping=mapping)

    stats = dict(det.last_fit_stats)
    stats["test_j_data"] = float(j_rkhs)
    stats["test_j_star"] = float(j_star)
    stats["test_delta_j"] = float(delta_j)
    stats["test_ber"] = float(ber_rkhs)
    return j_star, j_rkhs, delta_j, ber_rkhs, stats


def plot_results(
    snr_db_list: np.ndarray,
    ber_mld: np.ndarray,
    ber_mmse: np.ndarray,
    ber_rkhs: np.ndarray,
    j_star: np.ndarray,
    j_rkhs: np.ndarray,
    delta_j: np.ndarray,
    *,
    n_exp: int,
    n_train: int,
    n_test: int,
    n_test_mld: int,
    lam_c: float,
    output_mode: str,
    kernel_mode: str,
    use_apspm: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax0 = axes[0]
    ax0.semilogy(
        snr_db_list, ber_mld, "o-", color="#1f77b4", lw=2, ms=7,
        label=r"MLD $f_a^*$（真 $H_{\mathrm{eff}}$）",
    )
    ax0.semilogy(
        snr_db_list, ber_mmse, "s--", color="#ff7f0e", lw=2, ms=6,
        label=r"MMSE+LS（导频 $\hat H$）",
    )
    ax0.semilogy(
        snr_db_list, ber_rkhs, "^-", color="#2ca02c", lw=2, ms=6,
        label=r"RKHS 盲检测（仅 $y,s_1$ 训练）",
    )
    ax0.set_xlabel("SNR (dB)")
    ax0.set_ylabel("bit BER ($X_1$, Gray)")
    ax0.set_title(
        f"bit BER（{n_exp} 信道平均；MLD n={n_test_mld}，MMSE/RKHS n={n_test}）"
    )
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, which="both", alpha=0.35)
    ax0.set_ylim(bottom=1e-5)

    ax1 = axes[1]
    ax1.plot(
        snr_db_list, j_star, "o-", color="#1f77b4", lw=2, ms=7,
        label=r"$J_{\mathrm{data}}(f_a^*)$",
    )
    ax1.plot(
        snr_db_list, j_rkhs, "s--", color="#2ca02c", lw=2, ms=6,
        label=r"$J_{\mathrm{data}}(\hat f_{\mathrm{RKHS}})$",
    )
    ax1b = ax1.twinx()
    ax1b.bar(
        snr_db_list,
        delta_j,
        width=1.0,
        alpha=0.25,
        color="#d62728",
        label=r"$\Delta J$",
    )
    ax1b.set_ylabel(r"$\Delta J = J(\hat f) - J(f_a^*)$", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel(r"$J_{\mathrm{data}}$（越小越好）")
    ax1.set_title(f"目标函数 vs SNR（J* 在 n={n_test_mld}，J_RKHS 在 n={n_test}）")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.35)

    rkhs_tag = "RKHS+APSM v2" if use_apspm else "RKHS softmax"
    fig.suptitle(
        f"方案 A {M}×{K} | n_train={n_train} | {rkhs_tag} | kernel={kernel_mode} | "
        rf"output={output_mode} | $\lambda$=c/n (c={lam_c})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description="MLD vs MMSE+LS vs 盲 RKHS")
    p.add_argument("--n-train", type=int, default=2000, help="每 (H,SNR) RKHS 训练样本")
    p.add_argument(
        "--n-test",
        type=int,
        default=5000,
        help="MMSE+LS / RKHS 测试样本数",
    )
    p.add_argument(
        "--n-test-mld",
        type=int,
        default=500,
        help="MLD 及 J(f_a^*) 测试样本数（省 16^4 边际化）",
    )
    p.add_argument("--n-exp", type=int, default=3, help="独立信道实验次数")
    p.add_argument("--n-restarts", type=int, default=3, help="RKHS 多起点优化次数")
    p.add_argument("--lam-c", type=float, default=0.1, help="λ = c / n_train")
    p.add_argument(
        "--use-apspm",
        action="store_true",
        help="τ 退火 + margin 软损失 + 局部 APSM（默认 local）",
    )
    p.add_argument(
        "--margin-tau",
        type=float,
        default=0.7,
        help="APSM 终态 τ（f_{s1}/∑f_b ≥ τ）",
    )
    p.add_argument(
        "--margin-tau-start",
        type=float,
        default=0.3,
        help="τ 退火起点",
    )
    p.add_argument(
        "--margin-mu",
        type=float,
        default=0.4,
        help="margin 软惩罚权重（需 --use-apspm 或 --use-margin-loss）",
    )
    p.add_argument(
        "--use-margin-loss",
        action="store_true",
        help="仅 CE+margin，不做 APSM 投影",
    )
    p.add_argument(
        "--apsm-mode",
        choices=("local", "global"),
        default="local",
        help="APSM：local 只更新锚点列 α[:,n]；global 全局解 α",
    )
    p.add_argument(
        "--apsm-global",
        action="store_true",
        help="等价 --apsm-mode global（旧版全局 APSM）",
    )
    p.add_argument(
        "--kernel-mode",
        choices=("single", "multiscale"),
        default="single",
        help="RBF 核：single 或 multiscale（0.25/1/4×γ_base 等权）",
    )
    p.add_argument(
        "--output-mode",
        choices=("softmax", "softplus"),
        default="softmax",
        help="RKHS 输出：softmax(logits) 或 16 路 softplus",
    )
    p.add_argument("--verbose-rkhs", action="store_true", help="打印每个 SNR 的 RKHS 训练细节")
    p.add_argument(
        "--save-fig",
        type=str,
        default="",
        help="保存对比图路径（如 results.png）；不填则 plt.show()",
    )
    args = p.parse_args()
    if args.apsm_global:
        args.apsm_mode = "global"
    use_apspm = args.use_apspm
    use_margin_loss = args.use_margin_loss or use_apspm
    if args.use_margin_loss and not use_apspm:
        use_margin_loss = True

    rng = np.random.default_rng(42)
    mapping = "gray"
    n_train = args.n_train
    n_test = args.n_test
    n_test_mld = args.n_test_mld
    n_exp = args.n_exp
    snr_db_list = np.arange(0, 18, 2, dtype=float)
    n_snr = len(snr_db_list)

    print(
        f"结构：{n_exp} 次实验 × {n_snr} 个 SNR\n"
        f"  n_train={n_train}, n_test={n_test} (MMSE/RKHS), "
        f"n_test_mld={n_test_mld} (MLD/J*), λ=c/n (c={args.lam_c})\n"
        f"  kernel={args.kernel_mode}, output={args.output_mode}, "
        f"γ = 1/N₀（与 f_a^* 一致），盲训练 (y,s₁)\n"
    )
    if use_apspm or args.use_margin_loss:
        print(
            f"  margin: μ={args.margin_mu}, τ {args.margin_tau_start}→{args.margin_tau}, "
            f"APSM={args.apsm_mode if use_apspm else 'off'}\n"
        )

    acc_mmse = np.zeros(n_snr)
    acc_mld = np.zeros(n_snr)
    acc_rkhs = np.zeros(n_snr)
    acc_j_star = np.zeros(n_snr)
    acc_j_rkhs = np.zeros(n_snr)
    acc_delta_j = np.zeros(n_snr)
    acc_alpha_fro = np.zeros(n_snr)

    for exp in range(n_exp):
        t_exp = time.perf_counter()
        H_eff = generate_heff(rng)
        print(f"实验 {exp + 1}/{n_exp}：预计算 MLD Hy…", flush=True)
        hy_cache = precompute_mld_hy(H_eff)

        for j, snr_db in enumerate(snr_db_list):
            t0 = time.perf_counter()
            snr_f = float(snr_db)
            y_tr, _, s1_tr = generate_samples(n_train, H_eff, snr_f, rng)
            y_te, _, s1_te = generate_samples(n_test, H_eff, snr_f, rng)
            y_te_mld, _, s1_te_mld = generate_samples(n_test_mld, H_eff, snr_f, rng)

            ber_mmse_pt = run_mmse_ls_bit_ber(
                y_te, s1_te, H_eff, snr_f, rng, mapping
            )
            acc_mmse[j] += ber_mmse_pt
            ber_mld = run_mld_bit_ber(
                y_te_mld, s1_te_mld, H_eff, snr_f, hy_cache, mapping
            )
            acc_mld[j] += ber_mld

            j_star, j_rkhs, delta_j, ber_rkhs, stats = run_rkhs(
                y_tr,
                s1_tr,
                y_te,
                s1_te,
                y_te_mld,
                s1_te_mld,
                H_eff,
                snr_f,
                hy_cache,
                n_restarts=args.n_restarts,
                lam_c=args.lam_c,
                output_mode=args.output_mode,
                kernel_mode=args.kernel_mode,
                use_apspm=use_apspm,
                margin_tau=args.margin_tau,
                margin_tau_start=args.margin_tau_start,
                margin_mu=args.margin_mu,
                use_margin_loss=use_margin_loss if not use_apspm else None,
                apsm_mode=args.apsm_mode,
                mapping=mapping,
                verbose=args.verbose_rkhs,
            )
            acc_rkhs[j] += ber_rkhs
            acc_j_star[j] += j_star
            acc_j_rkhs[j] += j_rkhs
            acc_delta_j[j] += delta_j
            acc_alpha_fro[j] += stats["alpha_fro"]

            dt = time.perf_counter() - t0
            print(
                f"  SNR={snr_f:5.1f} dB | "
                f"J*={j_star:.3f} J_RKHS={j_rkhs:.3f} ΔJ={delta_j:+.3f} | "
                f"BER MLD={ber_mld:.2e} MMSE={ber_mmse_pt:.2e} RKHS={ber_rkhs:.2e} | "
                f"||α||={stats['alpha_fro']:.2f} train_SER={stats['train_ser']:.3f} | {dt:.1f}s",
                flush=True,
            )

        print(f"  实验 {exp + 1} 完成，耗时 {time.perf_counter() - t_exp:.1f}s", flush=True)

    inv = 1.0 / n_exp
    ber_mmse = acc_mmse * inv
    ber_mld = acc_mld * inv
    ber_rkhs = acc_rkhs * inv
    j_star = acc_j_star * inv
    j_rkhs = acc_j_rkhs * inv
    delta_j = acc_delta_j * inv
    alpha_fro = acc_alpha_fro * inv

    print(f"\n平均结果（{n_exp} 次实验，{mapping}）")
    print(
        f"{'SNR':>6} {'J(f*)':>8} {'J(RKHS)':>8} {'ΔJ':>8} "
        f"{'BER_MLD':>10} {'BER_MMSE':>10} {'BER_RKHS':>10} {'||α||':>8}"
    )
    print("-" * 78)
    for snr_db, js, jr, dj, bl, bm, br, af in zip(
        snr_db_list, j_star, j_rkhs, delta_j, ber_mld, ber_mmse, ber_rkhs, alpha_fro
    ):
        print(
            f"{snr_db:6.0f} {js:8.4f} {jr:8.4f} {dj:+8.4f} "
            f"{bl:10.4e} {bm:10.4e} {br:10.4e} {af:8.2f}"
        )

    print(
        "\n说明：J* 在 n_test_mld 上算；J(RKHS) 在 n_test 上算；"
        "ΔJ 在同集 n_test_mld 上算。"
        "||α||>0 且 train_SER≪0.94 表示已学到非平凡解。"
    )
    fig = plot_results(
        snr_db_list,
        ber_mld,
        ber_mmse,
        ber_rkhs,
        j_star,
        j_rkhs,
        delta_j,
        n_exp=n_exp,
        n_train=n_train,
        n_test=n_test,
        n_test_mld=n_test_mld,
        lam_c=args.lam_c,
        output_mode=args.output_mode,
        kernel_mode=args.kernel_mode,
        use_apspm=use_apspm,
    )
    if args.save_fig:
        fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
        print(f"\n图已保存: {args.save_fig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
