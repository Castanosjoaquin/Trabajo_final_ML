"""Detector híbrido: Autoencoder (reducción dimensional) + Isolation Forest
sobre el espacio latente.

Idea (deep representation + shallow detector): el AE aprende una representación
compacta de los datos normales y el Isolation Forest aísla anomalías en ese
espacio latente de baja dimensión, donde el ruido de features redundantes ya
fue descartado. En la literatura esta combinación supera sistemáticamente a
cada parte por separado (p. ej. Erfani et al. 2016, "High-dimensional and
large-scale anomaly detection using a linear one-class SVM with deep learning").

El score final es el del Isolation Forest sobre el latente; el AE NO contribuye
con su error de reconstrucción. La curva de loss del AE se expone vía
`history_` para que el runner la grafique igual que en un AE puro.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .base import AnomalyDetector
from .ae import AEDetector
from .baselines import IsolationForestDetector


class AEIForestDetector(AnomalyDetector):
    """AE para proyectar al latente + IsolationForest como scorer sobre el latente."""

    model_type = "ae_iforest"

    def __init__(
        self,
        # --- Autoencoder ---
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
        # --- Isolation Forest sobre el latente ---
        n_estimators: int = 100,
        max_samples="auto",
        max_features: float = 1.0,
        contamination="auto",
        random_state: int = 42,
    ):
        self._ae = AEDetector(
            hidden_dims=hidden_dims, latent_dim=latent_dim, lr=lr,
            weight_decay=weight_decay, dropout=dropout,
            use_batch_norm=use_batch_norm, grad_clip_norm=grad_clip_norm,
            activation=activation, lr_schedule=lr_schedule,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            random_state=random_state,
        )
        self._iforest = IsolationForestDetector(
            n_estimators=n_estimators, max_samples=max_samples,
            max_features=max_features, contamination=contamination,
            random_state=random_state,
        )
        self.random_state = random_state

    def fit(self, X: np.ndarray, wandb_run=None) -> "AEIForestDetector":
        self._ae.fit(X, wandb_run=wandb_run)
        Z = self._ae.encode(X)
        self._iforest.fit(Z)
        # Expone la curva de loss del AE al runner (mismo formato que un AE puro).
        self.history_ = getattr(self._ae, "history_", None)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Z = self._ae.encode(X)
        return self._iforest.score_samples(Z)

    def get_config(self) -> Dict:
        cfg: Dict = {"model_type": self.model_type}
        cfg.update({f"ae_{k}": v for k, v in self._ae.get_config().items()
                    if k != "model_type"})
        cfg.update({f"if_{k}": v for k, v in self._iforest.get_config().items()
                    if k != "model_type"})
        return cfg
