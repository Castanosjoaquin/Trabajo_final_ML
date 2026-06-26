"""Autoencoder y Denoising Autoencoder para detección de anomalías.

Referencia: Sakurada & Yairi (2014) — error de reconstrucción como score.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AnomalyDetector
from .trainer import train_ae


def _build_mlp(sizes: List[int], dropout: float = 0.0,
               use_batch_norm: bool = False) -> nn.Sequential:
    """MLP con orden Linear → BN → ReLU → Dropout entre capas ocultas."""
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _AENet(nn.Module):
    def __init__(self, n_features: int, hidden_dims: Tuple[int, ...], latent_dim: int,
                 dropout: float = 0.0, use_batch_norm: bool = False):
        super().__init__()
        enc_sizes = [n_features] + list(hidden_dims) + [latent_dim]
        dec_sizes = [latent_dim] + list(reversed(hidden_dims)) + [n_features]
        self.encoder = _build_mlp(enc_sizes, dropout, use_batch_norm)
        self.decoder = _build_mlp(dec_sizes, dropout, use_batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def loss(self, x_in: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(x_in), x_target)


def _salt_pepper_noise(x: torch.Tensor, corruption: float) -> torch.Tensor:
    """Salt-and-pepper noise para datos z-scored.

    pepper (prob=corruption/2): feature → 0 (media distribución normal).
    salt   (prob=corruption/2): feature → signo(x) × 3 (valor extremo en σ).
    """
    r = torch.rand_like(x)
    out = x.clone()
    pepper = r < (corruption / 2)
    salt = r > (1.0 - corruption / 2)
    out[pepper] = 0.0
    out[salt] = x[salt].sign() * 3.0
    return out


def _gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + torch.randn_like(x) * std


class AEDetector(AnomalyDetector):
    """Autoencoder: score = MSE de reconstrucción (Sakurada & Yairi 2014)."""

    model_type = "ae"

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        latent_dim: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        grad_clip_norm: float = 0.0,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.grad_clip_norm = grad_clip_norm
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self._net: _AENet | None = None

    def fit(self, X: np.ndarray, wandb_run=None) -> "AEDetector":
        torch.manual_seed(self.random_state)
        self._net = _AENet(X.shape[1], self.hidden_dims, self.latent_dim,
                           self.dropout, self.use_batch_norm)
        X_t = torch.tensor(X, dtype=torch.float32)
        self.history_ = train_ae(self._net, X_t, lr=self.lr,
                                 weight_decay=self.weight_decay,
                                 grad_clip_norm=self.grad_clip_norm,
                                 max_epochs=self.max_epochs, patience=self.patience,
                                 batch_size=self.batch_size, wandb_run=wandb_run)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Llamá fit() primero.")
        self._net.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            recon = self._net(X_t)
            scores = F.mse_loss(recon, X_t, reduction="none").mean(dim=1)
        return scores.numpy()

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "grad_clip_norm": self.grad_clip_norm,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }


class DenoisingAEDetector(AnomalyDetector):
    """Denoising AE: se entrena con input corrupto, score sobre input limpio.

    noise_type ∈ {'salt_pepper', 'gaussian'}.
    corruption: fracción de features corruptos (salt_pepper) o std del ruido (gaussian).
    """

    model_type = "dae"

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        latent_dim: int = 8,
        corruption: float = 0.1,
        noise_type: str = "salt_pepper",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        grad_clip_norm: float = 0.0,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = latent_dim
        self.corruption = corruption
        self.noise_type = noise_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.grad_clip_norm = grad_clip_norm
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self._net: _AENet | None = None

    def _noise_fn(self, batch: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "salt_pepper":
            return _salt_pepper_noise(batch, self.corruption)
        if self.noise_type == "gaussian":
            return _gaussian_noise(batch, self.corruption)
        raise ValueError(f"noise_type desconocido: {self.noise_type!r}")

    def fit(self, X: np.ndarray, wandb_run=None) -> "DenoisingAEDetector":
        torch.manual_seed(self.random_state)
        self._net = _AENet(X.shape[1], self.hidden_dims, self.latent_dim,
                           self.dropout, self.use_batch_norm)
        X_t = torch.tensor(X, dtype=torch.float32)
        self.history_ = train_ae(self._net, X_t, lr=self.lr,
                                 weight_decay=self.weight_decay,
                                 grad_clip_norm=self.grad_clip_norm,
                                 max_epochs=self.max_epochs, patience=self.patience,
                                 batch_size=self.batch_size, noise_fn=self._noise_fn,
                                 wandb_run=wandb_run)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score en input limpio (sin ruido)."""
        if self._net is None:
            raise RuntimeError("Llamá fit() primero.")
        self._net.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            recon = self._net(X_t)
            scores = F.mse_loss(recon, X_t, reduction="none").mean(dim=1)
        return scores.numpy()

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "corruption": self.corruption,
            "noise_type": self.noise_type,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "grad_clip_norm": self.grad_clip_norm,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }
