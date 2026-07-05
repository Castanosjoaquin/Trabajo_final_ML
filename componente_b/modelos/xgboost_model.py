"""Gradient boosting de árboles (XGBoost) para predecir rinde.

Modelo no lineal fuerte para datos tabulares. La regularización se controla por
varios frentes, todos tuneables:

- `reg_lambda` : regularización L2 sobre los pesos de las hojas.
- `reg_alpha`  : regularización L1 sobre los pesos de las hojas (esparsidad).
- `gamma`      : ganancia mínima para partir un nodo (poda).
- `max_depth`, `min_child_weight` : complejidad de cada árbol.
- `subsample`, `colsample_bytree` : submuestreo de filas/columnas (regulariza
                 por randomización, estilo bagging dentro del boosting).
- `learning_rate` + `n_estimators` : el clásico trade-off shrinkage vs. nº árboles.

Soporta early stopping opcional: si a `fit()` se le pasa `eval_set`, corta cuando
la métrica de validación deja de mejorar durante `early_stopping_rounds` rondas.
"""
from __future__ import annotations

import os

# Salvaguarda contra el doble runtime de OpenMP en macOS (xgboost + torch).
# El fix principal es el ORDEN de import (ver modelos/__init__.py: xgboost antes
# que torch); este flag es un cinturón extra y debe setearse antes de importar
# xgboost.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Dict, Optional, Tuple

import numpy as np
from xgboost import XGBRegressor

from .base import Regressor


class XGBoostRegressor(Regressor):
    """Wrapper de XGBRegressor con la interfaz común del Componente B."""

    model_type = "xgboost"

    def __init__(self, n_estimators: int = 400, max_depth: int = 4,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, min_child_weight: float = 1.0,
                 gamma: float = 0.0, reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0, early_stopping_rounds: Optional[int] = None,
                 random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model: Optional[XGBRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> "XGBoostRegressor":
        # early stopping solo tiene sentido con un set de validación explícito.
        es_rounds = self.early_stopping_rounds if eval_set is not None else None
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            early_stopping_rounds=es_rounds,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = False
        self._model.fit(X, y, **fit_kwargs)
        self.best_iteration_ = getattr(self._model, "best_iteration", None)
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
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "best_iteration": getattr(self, "best_iteration_", None),
            "random_state": self.random_state,
        }
