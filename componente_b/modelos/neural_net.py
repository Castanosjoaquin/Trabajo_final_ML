"""Red neuronal (MLP) para predecir rinde.

MLP feed-forward entrenado con Adam y MSE. Pensado para tunear arquitectura y
regularización de forma explícita:

- Arquitectura   : `hidden_dims`, `activation`, `use_batch_norm`.
- Regularización : `dropout`, `weight_decay` (L2 vía el optimizador),
                   `l1_lambda` (L1 sobre los pesos, sumada a la loss),
                   y early stopping sobre un split interno de validación.
- Optimización   : `lr`, `batch_size`, `max_epochs`, `patience`, `lr_schedule`.

El target (`rinde_kgha`) se estandariza internamente (se guarda media/desvío de
train) para estabilizar el entrenamiento; `predict()` deshace la transformación,
así que devuelve kg/ha directamente. Las features X se asumen ya escaladas por el
pipeline de datos (igual que en el Componente A).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import Regressor

_ACTIVATIONS = {
    "relu":       nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "elu":        nn.ELU,
    "gelu":       nn.GELU,
    "tanh":       nn.Tanh,
}


def _build_mlp(sizes: List[int], dropout: float, use_batch_norm: bool,
               activation: str) -> nn.Sequential:
    """MLP de regresión: Linear → BN → Activación → Dropout en las capas ocultas,
    y una Linear final sin activación (salida escalar continua)."""
    act_fn = _ACTIVATIONS.get(activation)
    if act_fn is None:
        raise ValueError(
            f"Activación desconocida: {activation!r}. Opciones: {list(_ACTIVATIONS)}")
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:            # todas menos la capa de salida
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class NeuralNetRegressor(Regressor):
    """MLP de PyTorch con early stopping y regularización L1/L2/dropout."""

    model_type = "neural_net"

    def __init__(self, hidden_dims: Tuple[int, ...] = (64, 32),
                 activation: str = "relu", dropout: float = 0.0,
                 use_batch_norm: bool = False, lr: float = 1e-3,
                 weight_decay: float = 0.0, l1_lambda: float = 0.0,
                 batch_size: int = 64, max_epochs: int = 300, patience: int = 20,
                 lr_schedule: Optional[str] = None, val_frac: float = 0.15,
                 random_state: int = 42):
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_schedule = lr_schedule
        self.val_frac = val_frac
        self.random_state = random_state
        self._net: Optional[nn.Sequential] = None

    def _make_scheduler(self, optimizer):
        if self.lr_schedule is None:
            return None
        if self.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs)
        if self.lr_schedule == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=max(1, self.patience // 3))
        raise ValueError(f"lr_schedule desconocido: {self.lr_schedule!r}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetRegressor":
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        # --- Estandarización del target (se guarda para invertir en predict) ---
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        y_std = (y - self.y_mean_) / self.y_std_

        # --- Split interno train/val para early stopping ---
        n = len(X)
        idx = rng.permutation(n)
        n_val = max(1, int(round(n * self.val_frac))) if n > 1 else 0
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr = torch.tensor(X[tr_idx]); ytr = torch.tensor(y_std[tr_idx])
        has_val = n_val > 0
        if has_val:
            Xva = torch.tensor(X[val_idx]); yva = torch.tensor(y_std[val_idx])

        self._net = _build_mlp([X.shape[1], *self.hidden_dims, 1],
                               self.dropout, self.use_batch_norm, self.activation)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = self._make_scheduler(optimizer)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0
        self.history_ = {"train_loss": [], "val_loss": []}

        for _ in range(self.max_epochs):
            self._net.train()
            perm = torch.randperm(len(Xtr))
            epoch_loss = 0.0
            for start in range(0, len(Xtr), self.batch_size):
                b = perm[start:start + self.batch_size]
                xb, yb = Xtr[b], ytr[b]
                optimizer.zero_grad()
                pred = self._net(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                if self.l1_lambda > 0:
                    l1 = sum(p.abs().sum() for p in self._net.parameters())
                    loss = loss + self.l1_lambda * l1
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(b)
            epoch_loss /= max(1, len(Xtr))
            self.history_["train_loss"].append(epoch_loss)

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    vpred = self._net(Xva).squeeze(-1)
                    val_loss = loss_fn(vpred, yva).item()
                self.history_["val_loss"].append(val_loss)
                if scheduler is not None:
                    scheduler.step(val_loss) if self.lr_schedule == "plateau" else scheduler.step()
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            else:
                if scheduler is not None and self.lr_schedule != "plateau":
                    scheduler.step()

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Llamá fit() primero.")
        self._net.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            out = self._net(X_t).squeeze(-1).numpy()
        return out * self.y_std_ + self.y_mean_       # deshace la estandarización

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "l1_lambda": self.l1_lambda,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "lr_schedule": self.lr_schedule,
            "val_frac": self.val_frac,
            "random_state": self.random_state,
        }
