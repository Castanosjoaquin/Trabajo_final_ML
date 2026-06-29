"""Variational Autoencoder para detección de anomalías.

Referencia: Kingma & Welling (2013) para arquitectura y ELBO.
            An & Cho (2015) para score_mode='recon_prob' (Algorithm 4).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AnomalyDetector
from .trainer import train_ae

_LOG2PI = math.log(2 * math.pi)


class _VAENet(nn.Module):
    """Red VAE con decoder Gaussiano (mu_x, logvar_x) y beta-KL."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: Tuple[int, ...],
        latent_dim: int,
        beta: float = 1.0,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        decoder_dist: str = "gaussian",
        student_t_df: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.decoder_dist = decoder_dist
        self.nu = float(student_t_df)

        def _block(in_dim: int, out_dim: int) -> List[nn.Module]:
            layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return layers

        # --- Encoder body ---
        enc_sizes = [n_features] + list(hidden_dims)
        enc_layers: List[nn.Module] = []
        for i in range(len(enc_sizes) - 1):
            enc_layers += _block(enc_sizes[i], enc_sizes[i + 1])
        self.encoder_body = nn.Sequential(*enc_layers)
        last_enc = enc_sizes[-1]

        self.fc_mu_z = nn.Linear(last_enc, latent_dim)
        self.fc_logvar_z = nn.Linear(last_enc, latent_dim)

        # --- Decoder body ---
        dec_sizes = [latent_dim] + list(reversed(hidden_dims))
        dec_layers: List[nn.Module] = []
        for i in range(len(dec_sizes) - 1):
            dec_layers += _block(dec_sizes[i], dec_sizes[i + 1])
        self.decoder_body = nn.Sequential(*dec_layers)
        last_dec = dec_sizes[-1]

        self.fc_mu_x = nn.Linear(last_dec, n_features)
        self.fc_logvar_x = nn.Linear(last_dec, n_features)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mu_z(h), self.fc_logvar_z(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder_body(z)
        mu_x = self.fc_mu_x(h)
        logvar_x = self.fc_logvar_x(h).clamp(-10.0, 10.0)
        return mu_x, logvar_x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decode(z)
        return mu_x, logvar_x, mu_z, logvar_z

    def per_sample_error(self, x: torch.Tensor) -> torch.Tensor:
        """Error de reconstrucción por muestra (MSE de mu_x, determinista en
        eval). Lo usa el tracking de dinámica de entrenamiento (data map)."""
        mu_x, _, _, _ = self.forward(x)
        return ((mu_x - x) ** 2).mean(dim=1)

    def log_prob(self, x: torch.Tensor, mu_x: torch.Tensor,
                 logvar_x: torch.Tensor) -> torch.Tensor:
        """log densidad p(x|z) por elemento (n, d) según el decoder.

        - gaussian: log N(x; mu_x, σ²) con σ² = exp(logvar_x).
        - student_t: log t_ν(x; mu_x, σ) — colas pesadas, robusto a outliers.
          σ = exp(0.5·logvar_x); se usa log1p para estabilidad numérica.
        """
        log_sigma = 0.5 * logvar_x
        if self.decoder_dist == "gaussian":
            return -0.5 * (logvar_x + (x - mu_x).pow(2) / logvar_x.exp() + _LOG2PI)
        if self.decoder_dist == "student_t":
            nu = self.nu
            z2 = ((x - mu_x) * torch.exp(-log_sigma)).pow(2)
            c = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                 - 0.5 * math.log(nu * math.pi))
            return c - log_sigma - 0.5 * (nu + 1) * torch.log1p(z2 / nu)
        raise ValueError(f"decoder_dist desconocido: {self.decoder_dist!r}")

    def loss(self, x_in: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """ELBO negativo (loss a minimizar): recon_loss + beta * KL."""
        mu_x, logvar_x, mu_z, logvar_z = self.forward(x_in)
        recon = -self.log_prob(x_target, mu_x, logvar_x).sum(dim=1).mean()
        kl = 0.5 * (mu_z.pow(2) + logvar_z.exp() - 1.0 - logvar_z).sum(dim=1).mean()
        return recon + self.beta * kl


def _mc_recon_prob(net: _VAENet, x: torch.Tensor, n_samples: int) -> torch.Tensor:
    """An & Cho (2015) Algorithm 4: -E_z~q[log p(x|z)] via MC sampling."""
    net.train()  # activa reparameterize con ruido (muestreo de z)
    for m in net.modules():  # apaga dropout y batch norm durante el scoring
        if isinstance(m, (nn.Dropout, nn.BatchNorm1d)):
            m.eval()
    log_px_z_acc = torch.zeros(len(x), device=x.device)
    with torch.no_grad():
        for _ in range(n_samples):
            mu_x, logvar_x, _, _ = net(x)
            log_px_z_acc += net.log_prob(x, mu_x, logvar_x).sum(dim=1)
    net.eval()
    return -(log_px_z_acc / n_samples)


class VAEDetector(AnomalyDetector):
    """VAE con tres modos de score:

    - 'recon_error': MSE(mu_x, x) — más simple, robusto.
    - 'recon_prob' : -E[log p(x|z)] vía MC (An & Cho 2015 Algorithm 4).
    - 'neg_elbo'   : -ELBO(x) = recon_loss + KL por fila.

    beta > 1 → β-VAE (mayor regularización del espacio latente).
    """

    model_type = "vae"

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        latent_dim: int = 8,
        beta: float = 1.0,
        score_mode: str = "recon_error",
        n_mc_samples: int = 20,
        decoder_dist: str = "gaussian",
        student_t_df: float = 4.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        grad_clip_norm: float = 0.0,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 64,
        track_datamap: bool = False,
        random_state: int = 42,
    ):
        assert score_mode in ("recon_error", "recon_prob", "neg_elbo"), \
            f"score_mode debe ser 'recon_error', 'recon_prob' o 'neg_elbo'; got {score_mode!r}"
        assert decoder_dist in ("gaussian", "student_t"), \
            f"decoder_dist debe ser 'gaussian' o 'student_t'; got {decoder_dist!r}"
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = latent_dim
        self.beta = beta
        self.score_mode = score_mode
        self.n_mc_samples = n_mc_samples
        self.decoder_dist = decoder_dist
        self.student_t_df = student_t_df
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.grad_clip_norm = grad_clip_norm
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.track_datamap = track_datamap
        self.random_state = random_state
        self._net: _VAENet | None = None
        self.per_sample_error_: np.ndarray | None = None  # (n_epochs, n) si track_datamap

    def fit(self, X: np.ndarray, wandb_run=None) -> "VAEDetector":
        torch.manual_seed(self.random_state)
        self._net = _VAENet(X.shape[1], self.hidden_dims, self.latent_dim,
                            self.beta, self.dropout, self.use_batch_norm,
                            self.decoder_dist, self.student_t_df)
        X_t = torch.tensor(X, dtype=torch.float32)
        self.history_ = train_ae(self._net, X_t, lr=self.lr,
                                 weight_decay=self.weight_decay,
                                 grad_clip_norm=self.grad_clip_norm,
                                 max_epochs=self.max_epochs, patience=self.patience,
                                 batch_size=self.batch_size,
                                 track_per_sample=self.track_datamap,
                                 wandb_run=wandb_run)
        self.per_sample_error_ = getattr(self._net, "per_sample_history_", None)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Llamá fit() primero.")
        self._net.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            if self.score_mode == "recon_error":
                mu_x, _, _, _ = self._net(X_t)
                scores = F.mse_loss(mu_x, X_t, reduction="none").mean(dim=1)
            elif self.score_mode == "recon_prob":
                scores = _mc_recon_prob(self._net, X_t, self.n_mc_samples)
            elif self.score_mode == "neg_elbo":
                mu_x, logvar_x, mu_z, logvar_z = self._net(X_t)
                recon = -self._net.log_prob(X_t, mu_x, logvar_x).sum(dim=1)
                kl = 0.5 * (
                    mu_z.pow(2) + logvar_z.exp() - 1.0 - logvar_z
                ).sum(dim=1)
                scores = recon + self.beta * kl
        return scores.numpy()

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "beta": self.beta,
            "score_mode": self.score_mode,
            "n_mc_samples": self.n_mc_samples,
            "decoder_dist": self.decoder_dist,
            "student_t_df": self.student_t_df,
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
