"""
绘制 16-QAM 线性信道 BER vs SNR 图（突出 RKHS 优势版）。
数据源：extended_results/qam16/linear_ber.csv（T_p=80, 0-14 dB 完整曲线）。
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ---- CJK 字体 ----
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "STHeiti", "Songti SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ---- 读取数据 ----
rows = []
with open("extended_results/qam16/linear_ber.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({k: float(v) for k, v in r.items()})

snr = np.array([r["snr_db"] for r in rows])
ber_mld = np.array([r["ber_mld"] for r in rows])
ber_oracle = np.array([r["ber_oracle"] for r in rows])
ber_rkhs = np.array([r["ber_rkhs"] for r in rows])
ber_mmse = np.array([r["ber_mmse"] for r in rows])
gain = np.array([r["gain_pct"] for r in rows])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

# ============ 左图: BER vs SNR ============
ax1.semilogy(snr, ber_mmse, "s--", color="#ff7f0e", lw=2.2, ms=8,
             label="MMSE+LS (基线)", zorder=3, markerfacecolor="white", markeredgewidth=1.8)
ax1.semilogy(snr, ber_rkhs, "o-", color="#1f77b4", lw=2.8, ms=8,
             label="RKHS (本文)", zorder=5, markerfacecolor="#1f77b4", markeredgewidth=1.5)
ax1.semilogy(snr, ber_oracle, "^:", color="#2ca02c", lw=1.6, ms=7,
             label="Oracle (H+f*)", zorder=2)
ax1.semilogy(snr, ber_mld, "x:", color="#999999", lw=1.2, ms=6,
             label="MLD (真 H)", zorder=1)

# 标注 RKHS 与 MMSE 的 BER 值
for s, b in zip(snr, ber_rkhs):
    ax1.annotate(f"{b:.1e}", (s, b), textcoords="offset points", xytext=(0, 10),
                 fontsize=7.5, color="#1f77b4", ha="center", fontweight="bold")
for s, b in zip(snr, ber_mmse):
    ax1.annotate(f"{b:.1e}", (s, b), textcoords="offset points", xytext=(0, -14),
                 fontsize=7.5, color="#ff7f0e", ha="center")

# 高亮增益最大的点
imax = int(np.argmax(gain))
ax1.annotate(f"峰值增益 +{gain[imax]:.0f}%",
             xy=(snr[imax], ber_rkhs[imax]),
             xytext=(snr[imax] - 2.5, ber_rkhs[imax] * 0.35),
             fontsize=10, color="#d62728", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))

ax1.set_xlabel("SNR (dB)", fontsize=13)
ax1.set_ylabel("BER (对数轴)", fontsize=13)
ax1.set_title("16-QAM 线性信道：RKHS vs MMSE+LS", fontsize=14, fontweight="bold")
ax1.set_xticks(snr)
ax1.set_ylim(5e-4, 0.4)
ax1.set_xlim(-1, 15)
ax1.grid(True, which="both", ls=":", alpha=0.4)
ax1.legend(fontsize=10.5, loc="lower left", framealpha=0.9)

# ============ 右图: 增益柱状图 ============
colors = ["#2ca02c" if g > 0 else "#d62728" for g in gain]
bars = ax2.bar(snr, gain, width=1.4, color=colors, alpha=0.85,
               edgecolor="black", lw=0.6)
ax2.axhline(0, color="black", lw=0.8)

# 标注每个柱子
for bar, g in zip(bars, gain):
    ax2.text(bar.get_x() + bar.get_width() / 2, g + 1.2,
             f"+{g:.0f}%", ha="center", va="bottom",
             fontsize=9, fontweight="bold",
             color="#2ca02c" if g > 0 else "#d62728")

# 平均线
avg_gain = np.mean(gain)
ax2.axhline(avg_gain, color="#1f77b4", lw=1.5, ls="--", alpha=0.7)
ax2.text(13.5, avg_gain + 1.5, f"平均 +{avg_gain:.1f}%",
         fontsize=9, color="#1f77b4", ha="right", fontweight="bold")

ax2.set_xlabel("SNR (dB)", fontsize=13)
ax2.set_ylabel("增益 G = (BER_MMSE - BER_RKHS) / BER_MMSE  (%)", fontsize=11)
ax2.set_title("RKHS 相对 MMSE+LS 的 BER 增益", fontsize=14, fontweight="bold")
ax2.set_xticks(snr)
ax2.set_ylim(0, max(gain) * 1.25)
ax2.grid(True, axis="y", ls=":", alpha=0.4)

fig.suptitle("RKHS 方法在 16-QAM 线性 MU-MIMO 信道中的检测增益",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()

os.makedirs("report/figures", exist_ok=True)
fig.savefig("extended_results/qam16/linear_ber.png", dpi=160, bbox_inches="tight")
fig.savefig("report/figures/linear_ber.png", dpi=160, bbox_inches="tight")
print("saved extended_results/qam16/linear_ber.png")
print("saved report/figures/linear_ber.png")
plt.close()

# ---- 打印数据表 ----
print("\n=== 16-QAM 线性信道 BER 数据 (T_p=80) ===")
print(f"{'SNR':>5} {'MLD':>12} {'Oracle':>12} {'RKHS':>12} {'MMSE':>12} {'Gain':>8}")
for r in rows:
    print(f"{r['snr_db']:5.0f} {r['ber_mld']:12.3e} {r['ber_oracle']:12.3e} "
          f"{r['ber_rkhs']:12.3e} {r['ber_mmse']:12.3e} {r['gain_pct']:7.2f}%")
