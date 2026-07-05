"""Detectores baseline: IsolationForest y OneClassSVM."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

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


class OneClassSVMDetector(AnomalyDetector):
    """Baseline sklearn OneClassSVM: aprende una frontera que envuelve a los datos
    normales; lo que cae afuera es anómalo. Con kernel RBF (gaussiano) la frontera
    es no lineal. Score = distancia al hiperplano invertida (mayor = más anómalo).

    kernel: 'rbf' (gaussiano, no lineal — default) | 'linear' | 'poly' | 'sigmoid'.
    nu:     cota superior de la fracción de outliers en train (~contaminación).
    gamma:  ancho del kernel RBF ('scale' = 1/(n_features·var), default de sklearn).
    """

    model_type = "ocsvm"

    def __init__(self, kernel: str = "rbf", nu: float = 0.1,
                 gamma="scale", random_state: int = 42):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.random_state = random_state  # OneClassSVM es determinístico; se guarda por consistencia
        self._clf: OneClassSVM | None = None

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        self._clf = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self._clf.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Llamá fit() primero.")
        # decision_function > 0 = dentro de la frontera (normal); invertimos el signo
        # para que mayor = más anómalo, como el resto de los detectores.
        return -self._clf.decision_function(X)

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "kernel": self.kernel,
            "nu": self.nu,
            "gamma": str(self.gamma),
            "random_state": self.random_state,
        }
