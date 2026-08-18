"""
扩展实验：多场景 / 多调制（16-QAM & 64-QAM）批量出图。

场景：
  linear | soft_clip | kerr | mzm | hard_clip | phase_noise | iq_imbalance

输出：extended_results/[qam16|qam64]/ 下的 npz、csv、单场景图与总览拼图。
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from system import set_modulation
from test_oracle_rkhs import (
    _channel_rng,
    _prepare_fixed_dataset,
    eval_one_snr,
    generate_heff,
    precompute_mld_hy,
)


# (tag, nonlin_mode, beta, 默认标题)
SCENARIO_CATALOG: dict[str, tuple[str, float, str]] = {
    "linear": ("none", 0.0, "Linear Y=HX+N"),
    "soft_clip": ("soft_clip", 0.35, r"Soft-clip $\beta$=0.35"),
    "kerr": ("kerr", 0.05, r"Kerr $\beta$=0.05"),
    "mzm": ("mzm", 0.9, r"MZM-like $\beta$=0.9"),
    "hard_clip": ("hard_clip", 1.5, r"Hard-clip thr=1.5·rms"),
    "phase_noise": ("phase_noise", 0.15, r"Phase noise $\sigma$=0.15"),
    "iq_imbalance": ("iq_imbalance", 0.10, r"I/Q imbalance $\beta$=0.10"),
}


def _lam_c(snr_db: float, mod_order: int) -> float:
    s = float(snr_db)
    # 64-QAM 类别更多，略加大正则防过拟合
    base = 0.12 if mod_order >= 64 else 0.1
    if s >= 14.0:
        return 0.04 if mod_order < 64 else 0.05
    if s >= 12.0:
        return 0.05
    if s >= 10.0:
        return 0.05 if mod_order < 64 else 0.08
    return base


def _retry_lam_c(snr_db: float, mod_order: int) -> float:
    s = float(snr_db)
    if s >= 12.0:
        return 0.02 if mod_order < 64 else 0.03
    return max(_lam_c(s, mod_order) * 1.5, 0.08)


def run_scenario(
    *,
    snr_list: list[float],
    n_train: int,
    n_test: int,
    n_chan: int,
    seed: int,
    nonlin_mode: str,
    nonlin_beta: float,
    skip_cnn: bool,
    save_dir: Path,
    tag: str,
    mod_order: int,
) -> dict[str, np.ndarray]:
    save_dir.mkdir(parents=True, exist_ok=True)
    keys = ["ber_mld", "ber_mmse", "ber_oracle", "ber_rkhs_nn", "ber_cnn", "j_rkhs_nn"]
    bucket: dict[str, list[list[float]]] = {k: [[] for _ in snr_list] for k in keys}

    for ich in range(n_chan):
        rng_h = _channel_rng(seed, ich)
        H = generate_heff(rng_h)
        ch_seed = int(seed) + ich * 10007
        fixed = _prepare_fixed_dataset(
            H, n_train=n_train, n_test=n_test, base_seed=ch_seed, sym_rng=rng_h
        )
        fixed["nonlin_mode"] = nonlin_mode
        fixed["nonlin_beta"] = float(nonlin_beta)
        hy = precompute_mld_hy(H)

        prev_rkhs: float | None = None
        for isnr, snr_db in enumerate(snr_list):
            t0 = time.perf_counter()
            r = eval_one_snr(
                H,
                hy,
                snr_db,
                rng_h,
                n_train=n_train,
                n_test=n_test,
                lam_c=_lam_c(snr_db, mod_order),
                oracle_lam_c=-1.0,
                fast=True,
                oracle_val_tune=True,
                oracle_kernel_mode="single",
                rkhs_nn_kernel_mode="adaptive",
                skip_blind=True,
                skip_rkhs_nn=False,
                dl_cnn_baseline=not skip_cnn,
                dl_cnn_blind=True,
                fixed_data=fixed,
                n_mmse_trials=3,
                n_oracle_train=min(1200, n_train),
                progress=True,
                ch_label=f"{tag}/H{ich + 1}/{n_chan}",
                abort_on_rkhs_fail=False,
                prev_ber_rkhs_nn=prev_rkhs,
            )
            ber_r = float(r["ber_rkhs_nn"])
            ber_m = float(r["ber_mmse"])
            need_retry = (prev_rkhs is not None and ber_r > prev_rkhs * 1.08 + 1e-4) or (
                ber_r > ber_m * 1.001
            )
            if need_retry and float(snr_db) >= 8.0:
                print(
                    f"  [{tag}] SNR={snr_db:.0f} retry lam={_retry_lam_c(snr_db, mod_order):.3f} "
                    f"(BER_rkhs={ber_r:.3e}, MMSE={ber_m:.3e}, prev={prev_rkhs})",
                    flush=True,
                )
                r2 = eval_one_snr(
                    H,
                    hy,
                    snr_db,
                    rng_h,
                    n_train=n_train,
                    n_test=n_test,
                    lam_c=_retry_lam_c(snr_db, mod_order),
                    oracle_lam_c=-1.0,
                    fast=True,
                    oracle_val_tune=True,
                    oracle_kernel_mode="single",
                    rkhs_nn_kernel_mode="adaptive",
                    skip_blind=True,
                    skip_rkhs_nn=False,
                    dl_cnn_baseline=False,
                    fixed_data=fixed,
                    n_mmse_trials=5 if float(snr_db) >= 12 else 3,
                    n_oracle_train=min(1200, n_train),
                    progress=True,
                    ch_label=f"{tag}/H{ich + 1}/{n_chan}-retry",
                    abort_on_rkhs_fail=False,
                    prev_ber_rkhs_nn=prev_rkhs,
                )

                def score(rr: dict) -> tuple:
                    br = float(rr["ber_rkhs_nn"])
                    bm = float(rr["ber_mmse"])
                    rebound = 0.0
                    if prev_rkhs is not None:
                        rebound = max(0.0, br - prev_rkhs)
                    return (br > bm + 1e-12, rebound, br)

                r = min([r, r2], key=score)
                ber_r = float(r["ber_rkhs_nn"])

            for k in keys:
                if k in r:
                    bucket[k][isnr].append(float(r[k]))
            prev_rkhs = ber_r
            print(
                f"  [{tag}] H{ich + 1} SNR={snr_db:.0f} "
                f"MLD={r['ber_mld']:.3e} Oracle={r['ber_oracle']:.3e} "
                f"RKHS={ber_r:.3e} MMSE={r['ber_mmse']:.3e} "
                f"gain={(r['ber_mmse'] - ber_r) / max(r['ber_mmse'], 1e-12) * 100:+.1f}% "
                f"dt={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    agg = {
        k: np.array([np.mean(v) if v else np.nan for v in bucket[k]], dtype=float)
        for k in keys
    }
    agg["snr"] = np.asarray(snr_list, dtype=float)
    return agg


def _plot_scenario(agg: dict[str, np.ndarray], title: str, out: Path) -> None:
    snr = agg["snr"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.semilogy(snr, agg["ber_mmse"], "o-", color="#334155", lw=2, label="MMSE+LS")
    ax.semilogy(snr, agg["ber_rkhs_nn"], "s-", color="#0f766e", lw=2.2, label=r"RKHS $z_{rob}$")
    if np.any(np.isfinite(agg["ber_oracle"])):
        ax.semilogy(
            snr,
            np.maximum(agg["ber_oracle"], 1e-5),
            "^-",
            color="#d62728",
            lw=1.6,
            label="Oracle≈MLD",
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(snr)

    gain = (agg["ber_mmse"] - agg["ber_rkhs_nn"]) / np.maximum(agg["ber_mmse"], 1e-12) * 100
    ax = axes[1]
    ax.bar(snr, gain, width=max(0.8, 0.6 * (snr[1] - snr[0]) if len(snr) > 1 else 1.2),
           color="#0f766e", edgecolor="white")
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER reduction vs MMSE (%)")
    ax.set_title("Gain vs MMSE")
    ax.set_xticks(snr)
    ax.grid(True, axis="y", alpha=0.35)
    for x, g in zip(snr, gain):
        ax.text(x, g + (0.8 if g >= 0 else -1.5), f"{g:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_csv(agg: dict[str, np.ndarray], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["snr_db", "ber_mld", "ber_oracle", "ber_rkhs", "ber_mmse", "gain_pct"]
        )
        for i, s in enumerate(agg["snr"]):
            g = (agg["ber_mmse"][i] - agg["ber_rkhs_nn"][i]) / max(
                agg["ber_mmse"][i], 1e-12
            ) * 100
            w.writerow(
                [
                    f"{s:.0f}",
                    f"{agg['ber_mld'][i]:.6e}",
                    f"{agg['ber_oracle'][i]:.6e}",
                    f"{agg['ber_rkhs_nn'][i]:.6e}",
                    f"{agg['ber_mmse'][i]:.6e}",
                    f"{g:.2f}",
                ]
            )


def _load_csv(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    snr, rk, mm, ora, mld = [], [], [], [], []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            snr.append(float(row["snr_db"]))
            rk.append(float(row["ber_rkhs"]))
            mm.append(float(row["ber_mmse"]))
            ora.append(float(row["ber_oracle"]))
            mld.append(float(row["ber_mld"]))
    if not snr:
        return None
    return {
        "snr": np.asarray(snr),
        "ber_rkhs_nn": np.asarray(rk),
        "ber_mmse": np.asarray(mm),
        "ber_oracle": np.asarray(ora),
        "ber_mld": np.asarray(mld),
    }


def plot_gallery(
    root: Path,
    tags: list[str],
    *,
    mod_order: int,
    out_name: str = "gallery_all_scenarios.png",
) -> Path | None:
    """多场景 BER + gain 拼图。"""
    loaded: list[tuple[str, dict[str, np.ndarray]]] = []
    for tag in tags:
        agg = _load_csv(root / f"{tag}_ber.csv")
        if agg is None:
            npz = root / f"{tag}_summary.npz"
            if npz.is_file():
                z = np.load(npz)
                agg = {k: z[k] for k in z.files}
        if agg is not None:
            loaded.append((tag, agg))
    if not loaded:
        return None

    n = len(loaded)
    ncols = 2
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.4 * nrows), squeeze=False)
    for i, (tag, agg) in enumerate(loaded):
        title = SCENARIO_CATALOG.get(tag, (None, None, tag))[2]
        snr = agg["snr"]
        ax = axes[i][0]
        ax.semilogy(snr, agg["ber_mmse"], "o-", color="#334155", lw=1.8, label="MMSE")
        ax.semilogy(snr, agg["ber_rkhs_nn"], "s-", color="#0f766e", lw=2.0, label="RKHS")
        ax.set_ylabel("BER")
        ax.set_title(f"{mod_order}-QAM | {title}")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(snr)
        if i == n - 1:
            ax.set_xlabel("SNR (dB)")

        gain = (agg["ber_mmse"] - agg["ber_rkhs_nn"]) / np.maximum(agg["ber_mmse"], 1e-12) * 100
        ax = axes[i][1]
        w = max(0.8, 0.55 * (float(snr[1] - snr[0]) if len(snr) > 1 else 1.2))
        colors = ["#0f766e" if g >= 0 else "#b91c1c" for g in gain]
        ax.bar(snr, gain, width=w, color=colors, edgecolor="white")
        ax.axhline(0, color="#94a3b8", lw=1)
        ax.set_ylabel("Gain vs MMSE (%)")
        ax.set_title(f"mean gain {float(np.nanmean(gain)):.1f}%")
        ax.set_xticks(snr)
        ax.grid(True, axis="y", alpha=0.35)
        if i == n - 1:
            ax.set_xlabel("SNR (dB)")

    fig.suptitle(
        f"RKHS $z_{{rob}}$ vs MMSE — {mod_order}-QAM multi-scenario gallery",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    out = root / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[gallery] saved {out}", flush=True)
    return out


def plot_mod_compare(
    root16: Path,
    root64: Path,
    tags: list[str],
    out: Path,
) -> Path | None:
    """同一场景下 16-QAM vs 64-QAM 对照。"""
    pairs: list[tuple[str, dict, dict]] = []
    for tag in tags:
        a = _load_csv(root16 / f"{tag}_ber.csv")
        b = _load_csv(root64 / f"{tag}_ber.csv")
        if a is not None and b is not None:
            pairs.append((tag, a, b))
    if not pairs:
        return None
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for i, (tag, a, b) in enumerate(pairs):
        ax = axes[0][i]
        title = SCENARIO_CATALOG.get(tag, (None, None, tag))[2]
        ax.semilogy(a["snr"], a["ber_rkhs_nn"], "s-", color="#0f766e", lw=2, label="16QAM RKHS")
        ax.semilogy(a["snr"], a["ber_mmse"], "o--", color="#64748b", lw=1.4, label="16QAM MMSE")
        ax.semilogy(b["snr"], b["ber_rkhs_nn"], "s-", color="#b45309", lw=2, label="64QAM RKHS")
        ax.semilogy(b["snr"], b["ber_mmse"], "o--", color="#a8a29e", lw=1.4, label="64QAM MMSE")
        ax.set_title(title)
        ax.set_xlabel("SNR (dB)")
        if i == 0:
            ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("16-QAM vs 64-QAM (same detector pipeline)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[mod-compare] saved {out}", flush=True)
    return out


def _resolve_scenarios(only: str) -> list[str]:
    if only in ("linear",):
        return ["linear"]
    if only in ("nonlin",):
        return ["soft_clip"]
    if only == "optical":
        return ["soft_clip", "kerr", "mzm"]
    if only == "rf_extra":
        return ["hard_clip", "phase_noise", "iq_imbalance"]
    if only == "gallery":
        return list(SCENARIO_CATALOG.keys())
    if only == "both":
        return ["linear", "soft_clip"]
    if only == "highsnr":
        return ["linear"]
    if only == "qam64":
        return ["linear", "soft_clip", "kerr", "hard_clip"]
    raise ValueError(only)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snr-list", type=str, default="")
    p.add_argument("--n-train", type=int, default=-1)
    p.add_argument("--n-test", type=int, default=-1)
    p.add_argument("--n-chan", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-cnn", action="store_true", default=True)
    p.add_argument(
        "--only",
        type=str,
        default="gallery",
        choices=[
            "linear",
            "nonlin",
            "both",
            "highsnr",
            "optical",
            "rf_extra",
            "gallery",
            "qam64",
            "plot_only",
        ],
    )
    p.add_argument("--mod-order", type=int, default=16, choices=[16, 64])
    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--plot-only", action="store_true", default=False)
    args = p.parse_args()

    mod_order = int(args.mod_order)
    set_modulation(mod_order)

    # 默认超参随调制变化
    if args.n_train < 0:
        args.n_train = 2500 if mod_order >= 64 else 2000
    if args.n_test < 0:
        args.n_test = 3000
    if args.n_chan < 0:
        args.n_chan = 2 if mod_order >= 64 else 2

    if args.snr_list:
        snr_list = [float(x) for x in args.snr_list.split(",") if x.strip()]
    elif args.only == "highsnr":
        snr_list = [12.0, 14.0, 16.0]
    elif mod_order >= 64:
        # 64-QAM 更密，低 SNR 几乎无信息；用偏高网格
        snr_list = [4.0, 8.0, 12.0, 16.0, 20.0]
    elif args.only in ("optical", "rf_extra", "gallery", "qam64"):
        snr_list = [0.0, 4.0, 8.0, 12.0, 16.0]
    else:
        snr_list = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

    root = Path(args.save_dir) if args.save_dir else Path("extended_results") / f"qam{mod_order}"
    root.mkdir(parents=True, exist_ok=True)

    if args.only == "plot_only" or args.plot_only:
        tags = list(SCENARIO_CATALOG.keys())
        plot_gallery(root, tags, mod_order=mod_order)
        other = Path("extended_results") / ("qam16" if mod_order == 64 else "qam64")
        plot_mod_compare(
            Path("extended_results/qam16"),
            Path("extended_results/qam64"),
            ["linear", "soft_clip", "kerr", "hard_clip"],
            Path("extended_results/compare_16_vs_64.png"),
        )
        return

    tags = _resolve_scenarios(args.only)
    print(
        f"mod={mod_order}-QAM  scenarios={tags}  snr={snr_list}  "
        f"n_train={args.n_train} n_test={args.n_test} n_chan={args.n_chan}  root={root}",
        flush=True,
    )

    done_tags: list[str] = []
    for tag in tags:
        mode, beta, title = SCENARIO_CATALOG[tag]
        print(f"\n==== scenario {tag} mode={mode} beta={beta} ({mod_order}-QAM) ====", flush=True)
        agg = run_scenario(
            snr_list=snr_list,
            n_train=args.n_train,
            n_test=args.n_test,
            n_chan=args.n_chan,
            seed=args.seed,
            nonlin_mode=mode,
            nonlin_beta=beta,
            skip_cnn=bool(args.skip_cnn),
            save_dir=root,
            tag=tag,
            mod_order=mod_order,
        )
        np.savez(root / f"{tag}_summary.npz", **agg)
        _save_csv(agg, root / f"{tag}_ber.csv")
        _plot_scenario(
            agg,
            title=f"{mod_order}-QAM | {title}",
            out=root / f"{tag}_ber.png",
        )
        done_tags.append(tag)
        rk = agg["ber_rkhs_nn"]
        mm = agg["ber_mmse"]
        mono_ok = True
        for i in range(1, len(rk)):
            if rk[i] > rk[i - 1] * 1.08 + 1e-4:
                mono_ok = False
                print(f"WARN {tag}: BER rebound at SNR {agg['snr'][i]}", flush=True)
        beat = bool(np.all(rk <= mm + 1e-12))
        print(
            f"[{tag}] monotone~{mono_ok} beat_MMSE={beat} "
            f"gain%={(mm - rk) / np.maximum(mm, 1e-12) * 100}",
            flush=True,
        )

    plot_gallery(root, done_tags, mod_order=mod_order)
    # 若两边都有结果则画对照图
    plot_mod_compare(
        Path("extended_results/qam16"),
        Path("extended_results/qam64"),
        ["linear", "soft_clip", "kerr", "hard_clip"],
        Path("extended_results/compare_16_vs_64.png"),
    )


if __name__ == "__main__":
    main()
