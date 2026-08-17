# RKHS MU-MIMO 检测（方案 A：只检 $X_1$）

可部署设定下，用 RKHS / Adaptive-MKL 最小化贝叶斯后验损失，在 **$128\times 40$、16-QAM** 上尽量超过 MMSE+LS。

## 问题设定

- 模型：$Y = HX + N$，用户数 $K=40$，天线 $M=128$，调制 16-QAM
- 只检测期望用户符号 $X_1\in\mathcal{A}$（$|\mathcal{A}|=16$）
- **部署约束**：不用真 $H$、不用真后验 $f^*$ 作监督；可用导频 LS 得到的 $\hat H$

目标函数：

$$
J(f)=\mathbb{E}\Big[-\log\frac{f_{X_1}(Y)}{\sum_b f_b(Y)}\Big]+\lambda\|f\|_{\mathcal{H}}^2
$$

决策始终在 RKHS 中：

$$
L(y)=K_\eta\big(\varphi(y),C\big)\,\alpha^\top,\qquad
\hat X_1=\arg\max_a L_a(y)
$$

## 可部署主线（当前默认）

$$
(Y,\hat H)
\;\xrightarrow{R_{\mathrm{rob}}}\;
z_{\mathrm{rob}}(Y;\hat H)\in\mathbb{R}^{16}
\;\xrightarrow{\text{Adaptive-MKL / NN / 聚合 / 可选堆叠}}\;
\hat X_1
$$

| 模块 | 做法 |
|------|------|
| 特征 `struct_hat` | 用 $\hat H$ 的稳健充分统计 $z_{\mathrm{rob}}$：$R_{\mathrm{rob}}=\hat N_0 I+E_s\hat H_I\hat H_I^H+(K-1)\sigma_e^2 I$ |
| 损失 `hard` | 用真实 $X_1$ 的交叉熵（经验 $J$）；不默认加重 plugin |
| 核 / 系数 | Adaptive-MKL（多核 $\eta$ + 多 $\alpha_m$）；可选 NN 生成 $\alpha$；验证集挑模型 |
| 聚合 / 堆叠 | logits 聚合后投影回 $\alpha$；高 SNR 可堆叠 $\phi_2=[z_{\mathrm{rob}},\mathrm{softmax}(L_1)]$ |

## RKHS Oracle（可逼近性上界）

同一 RKHS 决策形式，特权信息：

$$
\varphi(y)=z(y;H,N_0)\in\mathbb{R}^{16}
\quad\text{（真 $H$ 的 struct 充分统计）},\qquad
\text{目标}=f^*
$$

在合适核空间中可任意逼近后验。中期复现中 **Oracle BER / $J$ / MSE 均贴合 MLD**（不再使用旧的盲 $y$ 拟合）。

## 中期结果（3 信道平均，0–10 dB）

相对 MMSE+LS，可部署 RKHS（$z_{\mathrm{rob}}$）**全程更优**，BER 约降低 **19%–35%**：

| SNR (dB) | MMSE | RKHS $z_{\mathrm{rob}}$ | vs MMSE | Oracle≈MLD |
|----------|------|-------------------------|---------|------------|
| 0 | $2.76\times10^{-1}$ | $2.22\times10^{-1}$ | **−19.6%** | $\approx$ MLD |
| 2 | $2.15\times10^{-1}$ | $1.70\times10^{-1}$ | **−21.0%** | $\approx$ MLD |
| 4 | $1.60\times10^{-1}$ | $1.24\times10^{-1}$ | **−22.2%** | $\approx$ MLD |
| 6 | $1.05\times10^{-1}$ | $7.93\times10^{-2}$ | **−24.7%** | $\approx$ MLD |
| 8 | $5.15\times10^{-2}$ | $4.15\times10^{-2}$ | **−19.4%** | $\approx$ MLD |
| 10 | $2.40\times10^{-2}$ | $1.57\times10^{-2}$ | **−34.5%** | $\approx$ MLD |

图与表：`midterm_results/`（`exp1/2/3_*.png`、`ppt_ber_gain.png`、`ber_table.txt`）。

## 文件结构

### 核心库

| 文件 | 作用 |
|------|------|
| `system.py` | 系统参数、$H$ 生成、采样、比特 BER |
| `mld.py` | MLD 边际 / 高斯干扰代理软分 |
| `mmse.py` | 导频、LS 信道估计、MMSE+LS 检测 |
| `objective.py` | Softmax CE、$J$ 相关项、RKHS 惩罚 |
| `kernel_rkhs.py` | 核矩阵、Adaptive-MKL、$\alpha$ 求解、理论 $\gamma/\lambda$ |
| `rkhs_mld_approx.py` | **主检测器**：$z_{\mathrm{rob}}$ + MKL/NN/聚合/堆叠；Oracle 亦用此（`struct`+$f^*$） |

### 对照与可选前端

| 文件 | 作用 |
|------|------|
| `cnn_detector.py` | CNN 基线 |
| `rkhs_nn_detector.py` | 早期 RKHS–NN（回退） |
| `residual_mkl.py` | 旧 residual/PIC 前端（非主线） |
| `rkhs_cond_detector.py` | 条件核尝试（弱于 $z_{\mathrm{rob}}$） |

### 实验入口

| 文件 | 作用 |
|------|------|
| `midterm_final.py` | **推荐**：中期最终实验（0–10 dB，3 信道，出图） |
| `test_oracle_rkhs.py` | 完整评测（MLD / MMSE / Oracle=`struct`+$f^*$ / $z_{\mathrm{rob}}$ / CNN） |
| `run_experiment.py` | 早期盲 RKHS 入口；也被评测脚本复用 |

### 结果目录 `midterm_results/`

| 文件 | 内容 |
|------|------|
| `exp1_mld_baseline.png` | MLD 基准 |
| `exp2_rkhs_approximation.png` | Oracle 逼近 MLD（MSE / BER / $J$） |
| `exp3_end_to_end.png` | 端到端：$z_{\mathrm{rob}}$ vs MMSE/CNN/Oracle |
| `ppt_ber_gain.png` | 相对 MMSE 增益 |
| `ber_table.txt` | 数值表 |
| `run_final_oracle_struct.log` | 本次复现日志 |
| `final_summary.npz` | 汇总数组 |

## 环境

```bash
pip install -r requirements.txt
```

依赖：`numpy`、`scipy`、`matplotlib`、`torch`。

## 快速复现

```bash
cd "/Users/a1111/Desktop/RKHS优化问题"

PYTHONUNBUFFERED=1 python -u midterm_final.py \
  --snr-list "0,2,4,6,8,10" \
  --n-train 2000 --n-test 3000 --n-chan 3 \
  --save-dir midterm_results
```

默认：MLD、MMSE+LS、Oracle（`struct`+$f^*$）、可部署 RKHS（$z_{\mathrm{rob}}$）、CNN → 写出 `exp1/2/3_*.png`。

## 设计取舍

- **做**：$z_{\mathrm{rob}}$ + Adaptive-MKL；Oracle 用真 $H$ 证明核空间可逼近 $f^*$；打过 MMSE
- **不做主线**：加长导频死磕 CSI、PIC/MMSE residual 当主检测器、真 $H$/$f^*$ 监督上线

## 许可与用途

课程 / 研究代码提交用。中期 3 信道全 SNR 较耗时。
