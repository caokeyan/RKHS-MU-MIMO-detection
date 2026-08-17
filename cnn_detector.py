"""
CNN 检测（损失均为 J_data = softmax CE）。

1. BlindCNNSymbolDetector — 默认：仅 I/Q 接收向量，不用 H（与 RKHS 盲法信息一致）
2. TraditionalCSICNNSymbolDetector — 消融：输入 [y, vec(Ĥ_eff)]，Ĥ 由导频 LS
"""
from __future__ import annotations

import numpy as np

from mmse import batch_pilot_estimates, generate_pilots
from objective import softmax_ce_from_scores
from system import K, M, MOD_ORDER, n0_from_snr_db, y_to_features

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError("cnn_detector 需要 PyTorch") from e


def _y_to_cnn_input(y: np.ndarray) -> np.ndarray:
    """(n, M) complex -> (n, 2, M) float32。"""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[None, :]
    return np.stack([y.real, y.imag], axis=1).astype(np.float32)


def _heff_features(H: np.ndarray, n: int) -> np.ndarray:
    """H (M,K) 或 (n,M,K) -> (n, 2*M*K)。"""
    H = np.asarray(H, dtype=np.complex128)
    if H.ndim == 2:
        hf = np.concatenate([H.real.ravel(), H.imag.ravel()]).astype(np.float32)
        return np.tile(hf[None, :], (n, 1))
    if H.shape[0] != n:
        raise ValueError(f"H 批大小 {H.shape[0]} 与 n={n} 不一致")
    return np.concatenate(
        [H.real.reshape(n, -1), H.imag.reshape(n, -1)], axis=1
    ).astype(np.float32)


def estimate_heff_ls_per_sample(
    H_eff: np.ndarray,
    n: int,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """各样本独立导频块 → Ĥ_eff (n,M,K)，LS + 残差估 N₀（与 mmse.batch_pilot_estimates 一致）。"""
    n0 = n0_from_snr_db(float(snr_db))
    X_p = generate_pilots()
    H_hat, _ = batch_pilot_estimates(H_eff, X_p, n0, rng, n)
    return H_hat


def _csi_vector_features(y: np.ndarray, H: np.ndarray) -> np.ndarray:
    """拼接 [φ(y), vec(H)]；H 可为真信道或 Ĥ_LS。"""
    yf = y_to_features(y).astype(np.float32)
    hf = _heff_features(H, yf.shape[0])
    return np.concatenate([yf, hf], axis=1)


class _BlindConvNet(nn.Module):
    """1D-CNN：仅接收信号，2 通道 Re/Im。"""

    def __init__(self, n_ant: int = M, n_class: int = MOD_ORDER) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * n_ant, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class _TraditionalCSINet(nn.Module):
    """
    传统 MIMO-CNN：Conv 提取 y，再与 vec(Ĥ_eff) 融合（Ĥ 来自导频 LS，非真 H）。
    """

    def __init__(
        self,
        n_ant: int = M,
        h_dim: int = 2 * M * K,
        n_class: int = MOD_ORDER,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * n_ant + h_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_class),
        )

    def forward(self, y_ch: torch.Tensor, h_feat: torch.Tensor) -> torch.Tensor:
        z = self.conv(y_ch)
        z = z.flatten(1)
        return self.head(torch.cat([z, h_feat], dim=1))


def _train_loop(
    model: nn.Module,
    *,
    train_batches: tuple[torch.Tensor, ...],
    val_batches: tuple[torch.Tensor, ...],
    labels_tr: torch.Tensor,
    labels_va: torch.Tensor,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    loss_fn,
) -> tuple[nn.Module, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(*train_batches, labels_tr)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(labels_tr)),
        shuffle=True,
    )
    X_va = [t.to(device) for t in val_batches]
    y_va = labels_va.to(device)

    best_state = None
    best_val = float("inf")
    stale = 0

    for ep in range(epochs):
        model.train()
        for batch in loader:
            *xb, yb = batch
            xb = [t.to(device) for t in xb]
            yb = yb.to(device)
            opt.zero_grad()
            if len(xb) == 1:
                logits = model(xb[0])
            else:
                logits = model(xb[0], xb[1])
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            if len(X_va) == 1:
                val_loss = float(loss_fn(model(X_va[0]), y_va))
            else:
                val_loss = float(loss_fn(model(X_va[0], X_va[1]), y_va))
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


class TraditionalCSICNNSymbolDetector:
    """传统 CNN：y + 导频 LS 的 Ĥ_eff（仿真用真 H 仅生成导频观测），最小化 J_data。"""

    def __init__(
        self,
        *,
        lam_c: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> None:
        self.lam_c = lam_c
        self.lr = lr
        self.batch_size = batch_size
        self.H_eff: np.ndarray | None = None
        self.snr_db: float | None = None
        self._rng: np.random.Generator | None = None
        self.model: _TraditionalCSINet | None = None
        self.device = torch.device("cpu")
        self.last_fit_stats: dict[str, float] = {}

    def _h_ls_features(self, y: np.ndarray) -> np.ndarray:
        if self.H_eff is None or self.snr_db is None or self._rng is None:
            raise RuntimeError("先 fit(y, s1, H_eff, snr_db=..., rng=...)")
        n = len(np.atleast_2d(y))
        H_hat = estimate_heff_ls_per_sample(self.H_eff, n, self.snr_db, self._rng)
        return _heff_features(H_hat, n)

    def fit(
        self,
        y_train: np.ndarray,
        s1_train: np.ndarray,
        H_eff: np.ndarray,
        *,
        snr_db: float | None = None,
        rng: np.random.Generator | None = None,
        val_frac: float = 0.15,
        epochs: int = 400,
        patience: int = 40,
        verbose: bool = False,
    ) -> float:
        if snr_db is None:
            raise ValueError("CNN(H_LS) 需要 snr_db 以生成导频噪声")
        self.H_eff = np.asarray(H_eff)
        self.snr_db = float(snr_db)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        labels = np.asarray(s1_train, dtype=np.int64)
        n = len(labels)
        n0 = n0_from_snr_db(self.snr_db)
        wd = self.lam_c * n0 / max(n, 1)

        Y = _y_to_cnn_input(y_train)
        Hf = self._h_ls_features(y_train)

        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        n_val = max(32, int(n * val_frac))
        va, tr = perm[:n_val], perm[n_val:]

        model = _TraditionalCSINet().to(self.device)
        model, best_val = _train_loop(
            model,
            train_batches=(torch.from_numpy(Y[tr]), torch.from_numpy(Hf[tr])),
            val_batches=(torch.from_numpy(Y[va]), torch.from_numpy(Hf[va])),
            labels_tr=torch.from_numpy(labels[tr]),
            labels_va=torch.from_numpy(labels[va]),
            lr=self.lr,
            weight_decay=wd,
            epochs=epochs,
            patience=patience,
            batch_size=self.batch_size,
            device=self.device,
            loss_fn=nn.functional.cross_entropy,
        )
        self.model = model

        f_tr = self.scores(y_train)
        train_ce = softmax_ce_from_scores(f_tr, s1_train)
        self.last_fit_stats = {
            "train_j_data": float(train_ce),
            "train_ser": float(np.mean(np.argmax(f_tr, 1) != s1_train)),
            "val_j_data": float(best_val),
            "weight_decay": float(wd),
            "mode": "h_ls",
        }
        if verbose:
            print(
                f"  CNN(H_LS): J_data={train_ce:.4f}, SER={self.last_fit_stats['train_ser']:.4f}, "
                f"val={best_val:.4f}",
                flush=True,
            )
        return train_ce

    def _logits(self, y: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("先 fit(y, s1, H_eff)")
        Y = _y_to_cnn_input(y)
        Hf = self._h_ls_features(y)
        self.model.eval()
        with torch.no_grad():
            g = self.model(
                torch.from_numpy(Y).to(self.device),
                torch.from_numpy(Hf).to(self.device),
            ).cpu().numpy()
        return g

    def scores(self, y: np.ndarray) -> np.ndarray:
        g = np.clip(self._logits(y), -40.0, 40.0)
        z = g.max(axis=1, keepdims=True)
        e = np.exp(g - z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-300)

    def detect(self, y: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(y), axis=1)


class BlindCNNSymbolDetector:
    """盲 CNN：仅 y，最小化 J_data（六方法默认 DL 基线）。"""

    def __init__(
        self,
        *,
        lam_c: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> None:
        self.lam_c = lam_c
        self.lr = lr
        self.batch_size = batch_size
        self.model: _BlindConvNet | None = None
        self.device = torch.device("cpu")
        self.last_fit_stats: dict[str, float] = {}

    def fit(
        self,
        y_train: np.ndarray,
        s1_train: np.ndarray,
        *,
        snr_db: float | None = None,
        val_frac: float = 0.15,
        epochs: int = 400,
        patience: int = 40,
        verbose: bool = False,
        H_eff: np.ndarray | None = None,
    ) -> float:
        del H_eff
        X = _y_to_cnn_input(y_train)
        labels = np.asarray(s1_train, dtype=np.int64)
        n = len(labels)
        n0 = n0_from_snr_db(float(snr_db)) if snr_db is not None else 1.0
        wd = self.lam_c * n0 / max(n, 1)

        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        n_val = max(32, int(n * val_frac))
        va, tr = perm[:n_val], perm[n_val:]

        model = _BlindConvNet().to(self.device)
        model, best_val = _train_loop(
            model,
            train_batches=(torch.from_numpy(X[tr]),),
            val_batches=(torch.from_numpy(X[va]),),
            labels_tr=torch.from_numpy(labels[tr]),
            labels_va=torch.from_numpy(labels[va]),
            lr=self.lr,
            weight_decay=wd,
            epochs=epochs,
            patience=patience,
            batch_size=self.batch_size,
            device=self.device,
            loss_fn=nn.functional.cross_entropy,
        )
        self.model = model

        f_tr = self.scores(y_train)
        train_ce = softmax_ce_from_scores(f_tr, s1_train)
        self.last_fit_stats = {
            "train_j_data": float(train_ce),
            "train_ser": float(np.mean(np.argmax(f_tr, 1) != s1_train)),
            "val_j_data": float(best_val),
            "weight_decay": float(wd),
            "mode": "blind",
        }
        if verbose:
            print(
                f"  CNN(盲): J_data={train_ce:.4f}, SER={self.last_fit_stats['train_ser']:.4f}",
                flush=True,
            )
        return train_ce

    def _logits(self, y: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("先调用 fit()")
        xt = torch.from_numpy(_y_to_cnn_input(y)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(xt).cpu().numpy()

    def scores(self, y: np.ndarray) -> np.ndarray:
        g = np.clip(self._logits(y), -40.0, 40.0)
        z = g.max(axis=1, keepdims=True)
        e = np.exp(g - z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-300)

    def detect(self, y: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(y), axis=1)
