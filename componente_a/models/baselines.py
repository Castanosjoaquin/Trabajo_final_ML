"""Detectores baseline: IsolationForest y PCA Reconstruction Error."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """Baseline sklearn IsolationForest. Score invertido: mayor = más anómalo."""

    model_type = "isolation_forest"

    def __init__(self, n_estimators: int = 100, max_samples="auto",
                 max_features: float = 1.0, contamination="auto",
                 random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._clf: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        max_s = self.max_samples
        if max_s == "auto":
            max_s = min(256, len(X))
        elif isinstance(max_s, int):
            max_s = min(max_s, len(X))
        self._clf = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=max_s,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._clf.fit(X)
        self._resolved_max_samples = max_s
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Llamá fit() primero.")
        return -self._clf.score_samples(X)

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "resolved_max_samples": getattr(self, "_resolved_max_samples", None),
            "max_features": self.max_features,
            "contamination": str(self.contamination),
            "random_state": self.random_state,
        }


class PCAReconDetector(AnomalyDetector):
    """Baseline lineal: error de reconstrucción PCA (MSE por fila).

    Referencia: equivalente lineal del autoencoder (Sakurada & Yairi 2014 §2.1).
    n_components: int = nro de componentes exacto, float ∈ (0,1) = varianza explicada.
    """

    model_type = "pca_recon"

    def __init__(self, n_components=0.95, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCAReconDetector":
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._pca.fit(X)
        self._resolved_n_components = self._pca.n_components_
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Llamá fit() primero.")
        X_recon = self._pca.inverse_transform(self._pca.transform(X))
        return np.mean((X - X_recon) ** 2, axis=1)

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "n_components": self.n_components,
            "resolved_n_components": getattr(self, "_resolved_n_components", None),
            "random_state": self.random_state,
        }
