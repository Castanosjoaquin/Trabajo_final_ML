"""Regresión lineal con regularización configurable.

Un solo detector que cubre las cuatro variantes clásicas según `penalty`:

- 'none'       : mínimos cuadrados ordinarios (OLS, sin regularización).
- 'l2' / ridge : penaliza la norma L2 de los coeficientes → los encoge, útil
                 contra colinealidad (las features mensuales están muy correladas).
- 'l1' / lasso : penaliza la norma L1 → hace selección de features (coefs a 0).
- 'elasticnet' : combinación L1+L2, controlada por `l1_ratio`.

`alpha` es la fuerza de la regularización (0 = OLS). Se tunea junto con
`l1_ratio` cuando `penalty='elasticnet'`.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .base import Regressor


class LinearRegressor(Regressor):
    """Regresión lineal (OLS / Ridge / Lasso / ElasticNet) en una sola clase."""

    model_type = "linear"

    _ALIASES = {"ridge": "l2", "lasso": "l1"}

    def __init__(self, penalty: str = "none", alpha: float = 1.0,
                 l1_ratio: float = 0.5, max_iter: int = 20000,
                 random_state: int = 42):
        self.penalty = self._ALIASES.get(penalty, penalty)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None

    def _build(self):
        if self.penalty == "none":
            return LinearRegression()
        if self.penalty == "l2":
            return Ridge(alpha=self.alpha, random_state=self.random_state)
        if self.penalty == "l1":
            return Lasso(alpha=self.alpha, max_iter=self.max_iter,
                         random_state=self.random_state)
        if self.penalty == "elasticnet":
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                              max_iter=self.max_iter, random_state=self.random_state)
        raise ValueError(
            f"penalty desconocido: {self.penalty!r}. "
            f"Opciones: 'none', 'l2'/'ridge', 'l1'/'lasso', 'elasticnet'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        self._model = self._build()
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Llamá fit() primero.")
        return self._model.predict(X)

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "penalty": self.penalty,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
