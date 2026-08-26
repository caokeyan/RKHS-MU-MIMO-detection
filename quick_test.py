"""快速测试：SNR=12, 1信道, 看原始y特征是否帮助。"""
import numpy as np
import time
import sys
sys.path.insert(0, ".")

from system import set_modulation, generate_samples, M, K
from mld import precompute_mld_hy, marginal_mld_detect
from mmse import mmse_detect_x1
from test_oracle_rkhs import eval_one_snr

set_modulation(16)

rng = np.random.default_rng(42)
snr_db = 12.0

# 生成信道
from system import generate_heff
H = generate_heff(rng)
hy = precompute_mld_hy(H)

t0 = time.time()
r = eval_one_snr(
    H, hy, snr_db, rng,
    n_train=2500, n_test=20000,
    lam_c=0.02,
    oracle_lam_c=-1.0,
    fast=True,
    oracle_val_tune=True,
    oracle_kernel_mode="single",
    rkhs_nn_kernel_mode="adaptive",
    skip_blind=True,
    skip_rkhs_nn=False,
    dl_cnn_baseline=False,
    n_mmse_trials=3,
    progress=True,
    ch_label="test",
)
dt = time.time() - t0
print(f"\n=== RESULT SNR=12 ===")
print(f"RKHS={r['ber_rkhs_nn']:.4e} MMSE={r['ber_mmse']:.4e} "
      f"MLD={r['ber_mld']:.4e} Oracle={r['ber_oracle']:.4e} "
      f"gain={ (r['ber_mmse']-r['ber_rkhs_nn'])/max(r['ber_mmse'],1e-12)*100:+.1f}% "
      f"dt={dt:.1f}s")
