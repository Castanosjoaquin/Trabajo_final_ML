"""Gradient boosting nativo de sklearn (histogram-based) para predecir rinde.

`HistGradientBoostingRegressor` implementa boosting de árboles con el mismo
truco de binning por histogramas que XGBoost/LightGBM, pero sin depender del
paquete `xgboost`: viene con scikit-learn. Es fuerte en datos tabulares y
suele entrenar más rápido que un GBM clásico en datasets medianos/grandes.
La regularización se controla por:

- `learning_rate` + `max_iter` : shrinkage vs. nº de árboles (como en XGBoost).
- `max_depth` / `max_leaf_nodes` : complejidad de cada árbol.
- `l2_regularization` : penaliza los pesos de las hojas.
- `early_stopping` : corta el entrenamiento si la validación interna deja de
  mejorar (usa un split interno de sklearn, no requiere pasar un eval_set).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .base import Regressor


class HistGBMRegressor(Regressor):
    """Wrapper de `HistGradientBoostingRegressor` (sklearn) con la interfaz común."""

    model_type = "hist_gbm"

    def __init__(self, learning_rate: float = 0.05, max_depth: Optional[int] = None,
                 max_leaf_nodes: int = 31, l2_regularization: float = 0.0,
                 max_iter: int = 300, early_stopping: bool = True,
                 random_state: int = 42):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self._model: Optional[HistGradientBoostingRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGBMRegressor":
        self._model = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Llamá fit() primero.")
        return self._model.predict(X)

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "l2_regularization": self.l2_regularization,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping,
            "random_state": self.random_state,
        }
