"""Bosque aleatorio (Random Forest) para predecir rinde.

Ensamble de árboles entrenados sobre bootstraps de las filas y subconjuntos
aleatorios de features (`max_features`), promediando sus predicciones. A
diferencia del boosting (XGBoost), los árboles se entrenan en paralelo e
independientes entre sí: la regularización viene de la profundidad
(`max_depth`), el tamaño mínimo de hoja (`min_samples_leaf`) y la
aleatorización (`max_features`), no de un shrinkage secuencial. Suele ser más
robusto a hiperparámetros default que el boosting, aunque típicamente algo
menos preciso en el techo de performance.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import Regressor


class RandomForestRegressorModel(Regressor):
    """Wrapper de `RandomForestRegressor` (sklearn) con la interfaz común."""

    model_type = "random_forest"

    def __init__(self, n_estimators: int = 400, max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1, max_features="sqrt",
                 n_jobs: int = -1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: Optional[RandomForestRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorModel":
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
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
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }
