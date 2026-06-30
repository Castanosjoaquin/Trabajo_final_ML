"""Autoencoder y Denoising Autoencoder para detección de anomalías.

Referencia: Sakurada & Yairi (2014) — error de reconstrucción como score.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AnomalyDetector
from .trainer import train_ae


_ACTIVATIONS = {
    "relu":       nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "elu":        nn.ELU,
    "gelu":       nn.GELU,
    "tanh":       nn.Tanh,
}


def _build_mlp(sizes: List[int], dropout: float = 0.0,
               use_batch_norm: bool = False,
               activation: str = "relu") -> nn.Sequential:
    """MLP con orden Linear → BN → Activation → Dropout entre capas ocultas."""
    act_fn = _ACTIVATIONS.get(activation)
    if act_fn is None:
        raise ValueError(f"Activación desconocida: {activation!r}. Opciones: {list(_ACTIVATIONS)}")
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _AENet(nn.Module):
    def __init__(self, n_features: int, hidden_dims: Tuple[int, ...], latent_dim: int,
                 dropout: float = 0.0, use_batch_norm: bool = False,
                 activation: str = "relu"):
        super().__init__()
        enc_sizes = [n_features] + list(hidden_dims) + [latent_dim]
        dec_sizes = [latent_dim] + list(reversed(hidden_dims)) + [n_features]
        self.encoder = _build_mlp(enc_sizes, dropout, use_batch_norm, activation)
        self.decoder = _build_mlp(dec_sizes, dropout, use_batch_norm, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def loss(self, x_in: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(x_in), x_target)

    def per_sample_error(self, x: torch.Tensor) -> torch.Tensor:
        """Error de reconstrucción por muestra (MSE por fila). Lo usa el tracking
        de dinámica de entrenamiento (data map)."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)


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


def _aggregate_recon_error(sq_err: torch.Tensor, score_mode: str,
                           top_k: int) -> torch.Tensor:
    """Agrega el error de reconstrucción por-feature a un score por-muestra.

    `sq_err`: (n, d) errores cuadráticos por feature.
    - 'mse'  : media sobre features (Sakurada & Yairi 2014). Diluye la señal si
               pocas features reconstruyen muy mal entre muchas que reconstruyen
               bien.
    - 'max'  : máximo error por feature — sensible a una sola feature muy mal
               reconstruida (ranking idéntico al máximo error absoluto).
    - 'topk' : suma de los top_k errores más altos — señal concentrada, sin
               diluirse en las features bien reconstruidas.
    """
    if score_mode == "mse":
        return sq_err.mean(dim=1)
    if score_mode == "max":
        return sq_err.max(dim=1).values
    if score_mode == "topk":
        k = min(top_k, sq_err.shape[1])
        return sq_err.topk(k, dim=1).values.sum(dim=1)
    raise ValueError(
        f"score_mode desconocido: {score_mode!r}. Opciones: 'mse', 'max', 'topk'")


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
        activation: str = "relu",
        lr_schedule: Optional[str] = None,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 64,
        score_mode: str = "mse",
        top_k: int = 5,
        random_state: int = 42,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.grad_clip_norm = grad_clip_norm
        self.activation = activation
        self.lr_schedule = lr_schedule
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.score_mode = score_mode
        self.top_k = top_k
        self.random_state = random_state
        self._net: _AENet | None = None

    def _make_noise_fn(self) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Hook de corrupción de entrada. El AE base entrena sin ruido;
        el DenoisingAE lo sobreescribe."""
        return None

    def fit(self, X: np.ndarray) -> "AEDetector":
        torch.manual_seed(self.random_state)
        self._net = _AENet(X.shape[1], self.hidden_dims, self.latent_dim,
                           self.dropout, self.use_batch_norm, self.activation)
        X_t = torch.tensor(X, dtype=torch.float32)
        self.history_ = train_ae(self._net, X_t, lr=self.lr,
                                 weight_decay=self.weight_decay,
                                 grad_clip_norm=self.grad_clip_norm,
                                 lr_schedule=self.lr_schedule,
                                 max_epochs=self.max_epochs, patience=self.patience,
                                 batch_size=self.batch_size,
                                 noise_fn=self._make_noise_fn())
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Llamá fit() primero.")
        self._net.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            recon = self._net(X_t)
            sq_err = (recon - X_t) ** 2
            scores = _aggregate_recon_error(sq_err, self.score_mode, self.top_k)
        return scores.numpy()

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Representación latente (salida del encoder) de cada muestra.

        La usa el detector híbrido AE+IForest para correr el bosque sobre el
        espacio latente en vez del espacio original de features."""
        if self._net is None:
            raise RuntimeError("Llamá fit() primero.")
        self._net.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            z = self._net.encoder(X_t)
        return z.numpy()

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
            "activation": self.activation,
            "lr_schedule": self.lr_schedule,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "score_mode": self.score_mode,
            "top_k": self.top_k,
            "random_state": self.random_state,
        }


class DenoisingAEDetector(AEDetector):
    """Denoising AE: idéntico al AE salvo que entrena con input corrupto y
    scorea sobre el input limpio (Vincent et al. 2008).

    noise_type ∈ {'salt_pepper', 'gaussian'}.
    corruption: fracción de features corruptos (salt_pepper) o std del ruido (gaussian).
    """

    model_type = "dae"

    def __init__(self, *, corruption: float = 0.1,
                 noise_type: str = "salt_pepper", **ae_kwargs):
        super().__init__(**ae_kwargs)
        self.corruption = corruption
        self.noise_type = noise_type

    def _make_noise_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.noise_type == "salt_pepper":
            return lambda b: _salt_pepper_noise(b, self.corruption)
        if self.noise_type == "gaussian":
            return lambda b: _gaussian_noise(b, self.corruption)
        raise ValueError(f"noise_type desconocido: {self.noise_type!r}")

    def get_config(self) -> Dict:
        cfg = super().get_config()  # model_type ya resuelve a 'dae' (atributo de clase)
        cfg["corruption"] = self.corruption
        cfg["noise_type"] = self.noise_type
        return cfg
