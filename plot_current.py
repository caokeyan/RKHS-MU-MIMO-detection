"""画 linear BER 曲线（16-QAM），清晰展示 RKHS 优势。"""
import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "STHeiti", "Songti SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# --- 左图: BER vs SNR ---
ax1.semilogy(snr, ber_mmse, "d--", color="#ff7f0e", lw=2, ms=7, label="MMSE+LS", zorder=3)
ax1.semilogy(snr, ber_rkhs, "o-", color="#1f77b4", lw=2.5, ms=7, label="RKHS (本文)", zorder=4)
ax1.semilogy(snr, ber_oracle, "^:", color="#2ca02c", lw=1.5, ms=6, label="Oracle (H+f*)", zorder=2)
ax1.semilogy(snr, ber_mld, "x:", color="#999999", lw=1, ms=5, label="MLD (真H)", zorder=1)

# 标注 RKHS BER 值
for s, b in zip(snr, ber_rkhs):
    ax1.annotate(f"{b:.1e}", (s, b), textcoords="offset points", xytext=(0, 8),
                 fontsize=7, color="#1f77b4", ha="center")

ax1.set_xlabel("SNR (dB)", fontsize=12)
ax1.set_ylabel("BER", fontsize=12)
ax1.set_title("16-QAM 线性信道：RKHS vs MMSE", fontsize=13)
ax1.set_xticks(snr)
ax1.set_ylim(5e-4, 0.4)
ax1.set_xlim(-1, 15)
ax1.grid(True, which="both", ls=":", alpha=0.4)
ax1.legend(fontsize=10, loc="lower left")

# --- 右图: 增益 ---
colors = ["#2ca02c" if g > 0 else "#d62728" for g in gain]
bars = ax2.bar(snr, gain, width=1.5, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xlabel("SNR (dB)", fontsize=12)
ax2.set_ylabel("增益 G (%)", fontsize=12)
ax2.set_title("RKHS 相对 MMSE+LS 的增益", fontsize=13)
ax2.set_xticks(snr)
ax2.grid(True, axis="y", ls=":", alpha=0.4)

for bar, g in zip(bars, gain):
    ax2.text(bar.get_x() + bar.get_width()/2, g + 1.5, f"+{g:.0f}%",
             ha="center", va="bottom", fontsize=8, fontweight="bold")

fig.tight_layout()
fig.savefig("extended_results/qam16/linear_ber.png", dpi=150, bbox_inches="tight")
fig.savefig("report/figures/linear_ber.png", dpi=150, bbox_inches="tight")
print("saved extended_results/qam16/linear_ber.png")
print("saved report/figures/linear_ber.png")
plt.close()
