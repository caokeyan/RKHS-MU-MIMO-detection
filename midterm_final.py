#!/usr/bin/env python
"""
中期报告最终版
- 只到 10 dB
- 3 信道平均
- 方法：MLD + MMSE+LS（只 BER） + Oracle + RKHS–NN（λ 减半） + CNN（只 BER）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from test_oracle_rkhs import (
    _channel_rng,
    _prepare_fixed_dataset,
    eval_one_snr,
    generate_heff,
    precompute_mld_hy,
)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def run_final_experiment(
    snr_list: list[float],
    n_train: int = 2000,
    n_test: int = 3000,
    n_chan: int = 3,
    seed: int = 42,
    save_dir: str = "midterm_results",
) -> dict:
    """运行最终实验"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    snr_arr = np.array(snr_list, dtype=float)

    results_keys = [
        "ber_mld", "ber_mmse",
        "ber_oracle", "j_star", "j_oracle", "mse_te_oracle",
        "ber_rkhs_nn", "j_rkhs_nn", "mse_te_rkhs_nn",
        "ber_cnn",
    ]
    results: dict[str, list[np.ndarray]] = {k: [] for k in results_keys}

    for ich in range(n_chan):
        rng_h = _channel_rng(seed, ich)
        H = generate_heff(rng_h)
        ch_seed = int(seed) + ich * 10007

        fixed_data = _prepare_fixed_dataset(
            H, n_train=n_train, n_test=n_test, base_seed=ch_seed, sym_rng=rng_h,
        )
        hy = precompute_mld_hy(H)

        chan_results: dict[str, list[float]] = {k: [] for k in results_keys}

        for snr_db in snr_list:
            # 10/12 dB λ 减半
            lam_c_use = 0.05 if snr_db >= 10.0 else 0.1

            r = eval_one_snr(
                H, hy, snr_db, rng_h,
                n_train=n_train, n_test=n_test,
                lam_c=lam_c_use,
                oracle_lam_c=-1.0, fast=True,
                oracle_val_tune=True, oracle_kernel_mode="single",
                rkhs_nn_kernel_mode="adaptive",
                skip_blind=True, skip_rkhs_nn=False,
                dl_cnn_baseline=True, dl_cnn_blind=True,
                fixed_data=fixed_data, n_mmse_trials=5, n_oracle_train=1200,
                progress=True, ch_label=f"H{ich + 1}/{n_chan}",
                abort_on_rkhs_fail=False,
            )

            for k in results_keys:
                if k in r:
                    chan_results[k].append(float(r[k]))

            print(
                f"    J*={r.get('j_star', float('nan')):.4f} "
                f"J_oracle={r.get('j_oracle', float('nan')):.4f} "
                f"J_rkhs_nn={r.get('j_rkhs_nn', float('nan')):.4f}",
                flush=True,
            )

        for k in results_keys:
            if chan_results[k]:
                results[k].append(np.array(chan_results[k]))

    agg = {k: np.mean(v, axis=0) for k, v in results.items() if v}
    return agg


def plot_final_figures(agg: dict[str, np.ndarray], snr: np.ndarray, save_dir: str) -> None:
    """绘制最终三张图（只到 10 dB）"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 只到 10 dB
    snr = snr[:6]
    for k in agg:
        agg[k] = agg[k][:6]

    # 图1：MLD 基准
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(snr, agg["j_star"], "o-", color="#1f77b4", lw=2, ms=7, label=r"$J(f_a^*)$")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$J_{\mathrm{data}}$")
    ax.set_title("MLD 对数损失")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)

    ax = axes[1]
    ax.semilogy(snr, agg["ber_mld"], "o-", color="#1f77b4", lw=2, ms=7, label=r"$f_a^*$ (MLD)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("bit BER ($X_1$, Gray)")
    ax.set_title("MLD BER")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_ylim(bottom=1e-5)
    fig.tight_layout()
    fig.savefig(save_path / "exp1_mld_baseline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图1已保存: {save_path / 'exp1_mld_baseline.png'}")

    # 图2：RKHS 闭式解
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.plot(snr, agg["mse_te_oracle"], "o-", color="#d62728", lw=2, ms=7, label="RKHS Oracle")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE (test)")
    ax.set_title("RKHS 逼近 MLD: MSE")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)

    ax = axes[1]
    ax.semilogy(snr, agg["ber_mld"], "o-", color="#1f77b4", lw=2, ms=7, label=r"$f_a^*$ (MLD)")
    ax.semilogy(snr, agg["ber_oracle"], "^-", color="#d62728", lw=2, ms=6, label="RKHS Oracle")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("bit BER ($X_1$, Gray)")
    ax.set_title("RKHS 逼近 MLD: BER")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_ylim(bottom=1e-5)

    ax = axes[2]
    ax.plot(snr, agg["j_star"], "o-", color="#1f77b4", lw=2, ms=7, label=r"$J(f_a^*)$")
    ax.plot(snr, agg["j_oracle"], "^-", color="#d62728", lw=2, ms=6, label=r"$J$ RKHS Oracle")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$J_{\mathrm{data}}$")
    ax.set_title("RKHS 逼近 MLD: 对数损失")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path / "exp2_rkhs_approximation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图2已保存: {save_path / 'exp2_rkhs_approximation.png'}")

    # 图3：NN 优化 + CNN + MMSE
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.plot(snr, agg["mse_te_oracle"], "o-", color="#d62728", lw=2, ms=7, label="RKHS Oracle")
    if "mse_te_rkhs_nn" in agg and np.any(np.isfinite(agg["mse_te_rkhs_nn"])):
        ax.plot(snr, agg["mse_te_rkhs_nn"], "v-.", color="#9467bd", lw=2, ms=6, label=r"RKHS $z_{\mathrm{rob}}(\hat H)$")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE (test)")
    ax.set_title(r"稳健 CSI 逼近 $f^*$: MSE")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)

    ax = axes[1]
    ax.semilogy(snr, agg["ber_mld"], "o-", color="#1f77b4", lw=2, ms=7, label=r"$f_a^*$ (MLD)")
    ax.semilogy(snr, agg["ber_oracle"], "^-", color="#d62728", lw=2, ms=6, label="RKHS Oracle")
    if "ber_rkhs_nn" in agg and np.any(np.isfinite(agg["ber_rkhs_nn"])):
        ax.semilogy(snr, agg["ber_rkhs_nn"], "v-.", color="#9467bd", lw=2, ms=6, label=r"RKHS $z_{\mathrm{rob}}(\hat H)$")
    if "ber_cnn" in agg and np.any(np.isfinite(agg["ber_cnn"])):
        ax.semilogy(snr, agg["ber_cnn"], "x:", color="#8c564b", lw=2, ms=6, label="CNN")
    if "ber_mmse" in agg and np.any(np.isfinite(agg["ber_mmse"])):
        ax.semilogy(snr, agg["ber_mmse"], "d--", color="#ff7f0e", lw=2, ms=6, label="MMSE+LS")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("bit BER ($X_1$, Gray)")
    ax.set_title(r"RKHS 堆叠: $z_{\mathrm{rob}}\!\to\!L_1\!\to\!\phi_2$")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_ylim(bottom=1e-5)

    ax = axes[2]
    ax.plot(snr, agg["j_star"], "o-", color="#1f77b4", lw=2, ms=7, label=r"$J(f_a^*)$")
    ax.plot(snr, agg["j_oracle"], "^-", color="#d62728", lw=2, ms=6, label=r"$J$ RKHS Oracle")
    if "j_rkhs_nn" in agg and np.any(np.isfinite(agg["j_rkhs_nn"])):
        ax.plot(snr, agg["j_rkhs_nn"], "v-.", color="#9467bd", lw=2, ms=6, label=r"$J$ RKHS $z_{\mathrm{rob}}$")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$J_{\mathrm{data}}$")
    ax.set_title("端到端性能: 对数损失")
    ax.set_xticks(snr)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path / "exp3_end_to_end.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图3已保存: {save_path / 'exp3_end_to_end.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="中期报告最终版")
    parser.add_argument("--snr-list", type=str, default="0,2,4,6,8,10", help="SNR 列表 (dB)")
    parser.add_argument("--n-train", type=int, default=2000, help="训练样本数")
    parser.add_argument("--n-test", type=int, default=3000, help="测试样本数")
    parser.add_argument("--n-chan", type=int, default=3, help="信道数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save-dir", type=str, default="midterm_results", help="保存目录")
    args = parser.parse_args()

    snr_list = [float(x) for x in args.snr_list.split(",")]

    from system import M, K, MOD_ORDER

    print(
        f"运行最终实验: {M}×{K}, {MOD_ORDER}-QAM | "
        f"SNR={snr_list}, n_train={args.n_train}, n_test={args.n_test}, n_chan={args.n_chan}",
        flush=True,
    )

    agg = run_final_experiment(
        snr_list=snr_list,
        n_train=args.n_train,
        n_test=args.n_test,
        n_chan=args.n_chan,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    plot_final_figures(agg, np.array(snr_list), args.save_dir)

    print(f"\n实验完成！结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()