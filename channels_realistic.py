"""
更真实信道：Sionna 3GPP TR 38.901 CDL，以及 Kronecker 空间相关 Rayleigh。

输出 H_eff ∈ C^{M×K}，与 system.generate_heff 接口一致。
"""
from __future__ import annotations

import numpy as np

from system import K, M


def generate_heff_kronecker(
    rng: np.random.Generator,
    *,
    rho_rx: float = 0.5,
    rho_tx: float = 0.3,
) -> np.ndarray:
    """
    Kronecker 相关模型：H = R_rx^{1/2} G R_tx^{1/2}，G i.i.d. CN(0,1)。
    指数相关：R_{ij}=ρ^{|i-j|}。
    """
    rx = rho_rx ** np.abs(np.arange(M)[:, None] - np.arange(M)[None, :])
    tx = rho_tx ** np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])
    # 对称方根
    wr, vr = np.linalg.eigh(rx)
    wt, vt = np.linalg.eigh(tx)
    Rrx_h = (vr * np.sqrt(np.maximum(wr, 0.0))) @ vr.conj().T
    Rtx_h = (vt * np.sqrt(np.maximum(wt, 0.0))) @ vt.conj().T
    G = (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) / np.sqrt(2.0)
    H = Rrx_h @ G @ Rtx_h
    # 列功率归一到均值 1（与 i.i.d. Rayleigh 约定一致）
    col = np.mean(np.abs(H) ** 2, axis=0, keepdims=True)
    H = H / np.sqrt(col + 1e-12)
    return H.astype(np.complex128)


def generate_heff_cdl(
    rng: np.random.Generator,
    *,
    model: str = "C",
    carrier_frequency: float = 3.5e9,
    delay_spread: float = 300e-9,
) -> np.ndarray:
    """
    用 NVIDIA Sionna 的 3GPP CDL 生成上行等效平坦信道。

    CDL 本身是单链路模型：簇角度相对阵列固定。若对 K 个用户共用同一
    ``bs_orientation``，则各用户落在几乎相同的角度子空间，条件数可达
    10^2–10^3，线性 MLD/MMSE 都会崩（BER 地板 ~0.2–0.4）。

    修复：为每个用户采样独立的基站方位角（``bs_orientation`` batch 维），
    等效于用户分布在不同到达角；路径域系数对 path 求和得到窄带 H_eff。
    需要 sionna>=2.0。
    """
    try:
        import torch
        from sionna.phy.channel.tr38901 import AntennaArray, CDL
    except Exception as e:  # pragma: no cover
        raise ImportError(f"需要安装 sionna 以使用 CDL 信道: {e}") from e

    # 128 天线 ≈ 8×16 面板；用户单天线
    n_rows, n_cols = 8, 16
    assert n_rows * n_cols == M

    # 固定种子到 torch，便于复现
    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)

    bs = AntennaArray(
        num_rows=n_rows,
        num_cols=n_cols,
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=carrier_frequency,
    )
    ut = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )
    # 每用户独立方位角 → 空间可分；否则 MU 信道近奇异
    bs_az = rng.uniform(-np.pi, np.pi, size=K)
    bs_orientation = torch.tensor(
        np.stack([bs_az, np.zeros(K), np.zeros(K)], axis=-1),
        dtype=torch.float32,
    )
    cdl = CDL(
        model=str(model).upper(),
        delay_spread=float(delay_spread),
        carrier_frequency=float(carrier_frequency),
        ut_array=ut,
        bs_array=bs,
        direction="uplink",
        min_speed=0.0,
        bs_orientation=bs_orientation,
    )
    # batch=K：K 条用户链路（角度已打散）
    # sampling_frequency：取略高于 delay resolution
    fs = 1.0 / max(delay_spread / 20.0, 10e-9)
    a, _tau = cdl(batch_size=K, num_time_steps=1, sampling_frequency=fs)
    # a: (K, 1, M, 1, 1, n_paths, 1)
    if hasattr(a, "numpy"):
        a_np = a.detach().cpu().numpy()
    else:
        a_np = np.asarray(a)
    h = np.sum(a_np, axis=-2)  # sum paths
    h = np.squeeze(h)  # (K, M)
    H = h.T.astype(np.complex128)  # (M, K)
    col = np.mean(np.abs(H) ** 2, axis=0, keepdims=True)
    H = H / np.sqrt(col + 1e-12)
    return H


def generate_heff_by_name(
    name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    name = str(name).lower()
    if name in ("iid", "rayleigh", "none", ""):
        from system import generate_heff

        return generate_heff(rng)
    if name in ("kronecker", "corr", "correlated"):
        return generate_heff_kronecker(rng, rho_rx=0.5, rho_tx=0.3)
    if name.startswith("cdl"):
        # cdl_a / cdl_c / cdl-a
        model = name.replace("cdl_", "").replace("cdl-", "").replace("cdl", "C")
        if model in ("", "cdl"):
            model = "C"
        return generate_heff_cdl(rng, model=model.upper())
    raise ValueError(f"未知信道类型: {name}")
