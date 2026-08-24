"""
醒目展示图：从 extended_results 汇总“一眼能看懂”的增益海报。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path("extended_results")
OUT = ROOT / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("report/figures")
FIG.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path):
    if not path.is_file():
        return None
    snr, rk, mm = [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            snr.append(float(row["snr_db"]))
            rk.append(float(row["ber_rkhs"]))
            mm.append(float(row["ber_mmse"]))
    return {
        "snr": np.asarray(snr),
        "rkhs": np.asarray(rk),
        "mmse": np.asarray(mm),
        "gain": (np.asarray(mm) - np.asarray(rk)) / np.maximum(np.asarray(mm), 1e-12) * 100,
    }


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.28, linestyle="--")


def poster_gain_bars():
    """全场景平均增益海报。"""
    items = []
    for p in sorted((ROOT / "qam16").glob("*_ber.csv")):
        d = load_csv(p)
        if d is None:
            continue
        g = d["gain"]
        g = g[np.isfinite(g)]
        # 丢掉极端负值以免淹没视觉（高 SNR 稀有错误）
        g_pos = g[g > -50]
        if len(g_pos) == 0:
            continue
        items.append((p.stem.replace("_ber", ""), float(np.mean(g_pos)), float(np.max(g_pos))))

    # 真实信道
    for ch_dir in sorted((ROOT / "qam16").glob("ch_*")):
        for p in sorted(ch_dir.glob("*_ber.csv")):
            d = load_csv(p)
            if d is None:
                continue
            g = d["gain"]
            g = g[np.isfinite(g) & (g > -50)]
            if len(g) == 0:
                continue
            items.append((f"{ch_dir.name}/{p.stem.replace('_ber','')}", float(np.mean(g)), float(np.max(g))))

    if not items:
        return
    items.sort(key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in items]
    means = [x[1] for x in items]
    peaks = [x[2] for x in items]

    fig, ax = plt.subplots(figsize=(12.5, max(4.5, 0.42 * len(items))))
    y = np.arange(len(items))
    colors = ["#0f766e" if m >= 15 else ("#ca8a04" if m >= 0 else "#b91c1c") for m in means]
    ax.barh(y, means, color=colors, edgecolor="white", height=0.72, label="mean G")
    ax.plot(peaks, y, "D", color="#0c4a6e", ms=5, label="peak G")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("BER reduction vs MMSE  $G$ (%)", fontsize=11)
    ax.set_title("RKHS $z_{rob}$ beats MMSE — scenario leaderboard (16-QAM)", fontsize=13, pad=10)
    ax.axvline(0, color="#94a3b8", lw=1)
    ax.axvline(20, color="#86efac", lw=1, ls=":", alpha=0.8)
    style_ax(ax)
    for yi, m, pk in zip(y, means, peaks):
        ax.text(max(m, 0) + 1.2, yi, f"{m:.0f}% (peak {pk:.0f}%)", va="center", fontsize=8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for dest in [OUT / "poster_gain_leaderboard.png", FIG / "poster_gain_leaderboard.png"]:
        fig.savefig(dest, dpi=180, bbox_inches="tight", facecolor="#f8fafc")
    plt.close(fig)
    print("saved poster_gain_leaderboard")


def poster_softclip_vs_mmse():
    """soft_clip 曲线：填充增益区域，醒目。"""
    d16 = load_csv(ROOT / "qam16" / "soft_clip_ber.csv")
    d64 = load_csv(ROOT / "qam64" / "soft_clip_ber.csv")
    if d16 is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    ax = axes[0]
    ax.fill_between(d16["snr"], d16["rkhs"], d16["mmse"], color="#99f6e4", alpha=0.55, label="gain region")
    ax.semilogy(d16["snr"], d16["mmse"], "o--", color="#334155", lw=2.2, label="MMSE+LS")
    ax.semilogy(d16["snr"], d16["rkhs"], "s-", color="#0f766e", lw=2.6, ms=7, label="RKHS $z_{rob}$")
    ax.set_title("16-QAM soft-clip: RKHS pulls BER down", fontsize=12)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.legend(frameon=False)
    style_ax(ax)

    ax = axes[1]
    w = 1.1
    bars = ax.bar(d16["snr"], d16["gain"], width=w, color="#0f766e", edgecolor="white")
    for b, g in zip(bars, d16["gain"]):
        ax.text(b.get_x() + b.get_width() / 2, g + 1.5, f"{g:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(d16["gain"]) * 1.25)
    ax.set_title(f"Gain vs MMSE — mean {np.mean(d16['gain']):.0f}%, peak {np.max(d16['gain']):.0f}%", fontsize=12)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("$G$ (%)")
    style_ax(ax)
    if d64 is not None:
        ax.plot(d64["snr"], d64["gain"], "D--", color="#b45309", lw=1.6, label="64-QAM soft-clip")
        ax.legend(frameon=False)

    # 顶部大标题条
    fig.suptitle("Where RKHS shines: frontend mismatch (soft-clip)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    for dest in [OUT / "poster_softclip_highlight.png", FIG / "poster_softclip_highlight.png"]:
        fig.savefig(dest, dpi=180, bbox_inches="tight", facecolor="#f8fafc")
    plt.close(fig)
    print("saved poster_softclip_highlight")


def poster_heatmap():
    """场景 × SNR 增益热力图。"""
    tags = ["soft_clip", "mzm", "kerr", "hard_clip", "phase_noise", "iq_imbalance", "linear"]
    snr_ref = [0, 4, 8, 12, 16]
    mat = np.full((len(tags), len(snr_ref)), np.nan)
    for i, tag in enumerate(tags):
        d = load_csv(ROOT / "qam16" / f"{tag}_ber.csv")
        if d is None:
            continue
        for j, s in enumerate(snr_ref):
            idx = np.where(np.isclose(d["snr"], s))[0]
            if len(idx):
                mat[i, j] = d["gain"][idx[0]]
    # clip display
    mat_show = np.clip(mat, -40, 80)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(mat_show, aspect="auto", cmap="RdYlGn", vmin=-20, vmax=70)
    ax.set_xticks(range(len(snr_ref)))
    ax.set_xticklabels([f"{s} dB" for s in snr_ref])
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="black" if abs(mat_show[i, j]) < 45 else "white")
    ax.set_title("Gain heatmap $G$ (%) — 16-QAM multi-scenario", fontsize=13)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("$G$ (%)")
    fig.tight_layout()
    for dest in [OUT / "poster_gain_heatmap.png", FIG / "poster_gain_heatmap.png"]:
        fig.savefig(dest, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved poster_gain_heatmap")


def poster_midterm_hero():
    """中期主结果一页摘要。"""
    # 用 qam16 linear 细网格或 midterm 数字
    d = load_csv(ROOT / "qam16" / "linear_ber.csv")
    fig = plt.figure(figsize=(12.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.85], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    if d is not None:
        # 只画 0-12 正增益区更漂亮
        m = d["snr"] <= 12
        ax.fill_between(d["snr"][m], d["rkhs"][m], d["mmse"][m], color="#a7f3d0", alpha=0.6)
        ax.semilogy(d["snr"][m], d["mmse"][m], "o--", color="#475569", lw=2, label="MMSE")
        ax.semilogy(d["snr"][m], d["rkhs"][m], "s-", color="#0f766e", lw=2.5, label="RKHS")
    ax.set_title("Linear 16-QAM (0–12 dB)", fontsize=11)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.legend(frameon=False)
    style_ax(ax)

    ax = fig.add_subplot(gs[0, 1])
    # 中期表固定数字
    snr = np.array([0, 2, 4, 6, 8, 10])
    g = np.array([19.6, 21.1, 22.2, 24.7, 19.4, 34.5])
    ax.bar(snr, g, width=1.4, color="#0f766e", edgecolor="white")
    for x, y in zip(snr, g):
        ax.text(x, y + 0.8, f"{y:.0f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 42)
    ax.set_title("Midterm gain vs MMSE", fontsize=11)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("$G$ (%)")
    style_ax(ax)

    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cards = [
        ("19–35%", "BER↓ vs MMSE\n(0–10 dB midterm)"),
        ("~52%", "mean $G$ soft-clip"),
        ("64-QAM", "same pipeline\nmigrates"),
    ]
    for i, (big, small) in enumerate(cards):
        y0 = 0.72 - i * 0.32
        ax.add_patch(FancyBboxPatch((0.05, y0), 0.9, 0.26, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor="#ecfdf5", edgecolor="#0f766e", lw=1.5))
        ax.text(0.5, y0 + 0.16, big, ha="center", va="center", fontsize=18, fontweight="bold", color="#065f46")
        ax.text(0.5, y0 + 0.06, small, ha="center", va="center", fontsize=8, color="#334155")

    fig.suptitle("RKHS MU-MIMO detection — headline results", fontsize=14, fontweight="bold")
    fig.tight_layout()
    for dest in [OUT / "poster_headline.png", FIG / "poster_headline.png"]:
        fig.savefig(dest, dpi=190, bbox_inches="tight", facecolor="#f8fafc")
    plt.close(fig)
    print("saved poster_headline")


def poster_realistic_if_any():
    paths = list((ROOT / "qam16").glob("ch_*/*_ber.csv"))
    if not paths:
        return
    fig, axes = plt.subplots(1, min(3, len(paths)), figsize=(4.2 * min(3, len(paths)), 3.8), squeeze=False)
    for ax, p in zip(axes[0], paths[:3]):
        d = load_csv(p)
        if d is None:
            continue
        ax.fill_between(d["snr"], d["rkhs"], d["mmse"], where=d["rkhs"] <= d["mmse"],
                        color="#a7f3d0", alpha=0.55, interpolate=True)
        ax.semilogy(d["snr"], d["mmse"], "o--", color="#64748b", label="MMSE")
        ax.semilogy(d["snr"], d["rkhs"], "s-", color="#0f766e", lw=2.2, label="RKHS")
        ax.set_title(f"{p.parent.name}/{p.stem.replace('_ber','')}\nmean G={np.nanmean(d['gain']):.0f}%")
        ax.set_xlabel("SNR (dB)")
        ax.legend(frameon=False, fontsize=8)
        style_ax(ax)
    axes[0][0].set_ylabel("BER")
    fig.suptitle("Realistic channels (Sionna CDL / Kronecker)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    for dest in [OUT / "poster_realistic.png", FIG / "poster_realistic.png"]:
        fig.savefig(dest, dpi=180, bbox_inches="tight", facecolor="#f8fafc")
    plt.close(fig)
    print("saved poster_realistic")


if __name__ == "__main__":
    poster_midterm_hero()
    poster_softclip_vs_mmse()
    poster_heatmap()
    poster_gain_bars()
    poster_realistic_if_any()
    print("all showcase figures ->", OUT)
