"""Stacking (XGBoost + MLP) para predecir rinde — extensión opcional de la propuesta.

Combina dos modelos base de naturaleza distinta (árboles de boosting vs. red
neuronal) mediante un meta-modelo lineal (Ridge) que aprende a pesar sus
predicciones. La idea: si los errores de XGBoost y del MLP no están
perfectamente correlacionados, el promedio ponderado por Ridge puede superar a
cada uno por separado.

Para evitar que el meta-modelo aprenda sobre predicciones "de memoria" (cada
base model prediciendo sobre filas que ya vio en su propio train), las
meta-features son predicciones OUT-OF-FOLD: se arman con `TimeSeriesSplit`
sobre el ORDEN de las filas, igual que `evaluacion.buscar` — ventana expansiva,
nunca se valida con datos anteriores al train del fold. El `RegDataset` del
repo (`datos.py::crop_frame`) viene ordenado por (departamento, año), así que
ese orden de filas ya es temporalmente coherente DENTRO de cada departamento;
si se pasa `years` a `fit()`, las filas se reordenan primero por año (orden
estable) antes de armar los folds, para más robustez frente a datasets que no
vengan pre-ordenados así.

Una vez ajustado el Ridge sobre las OOF preds, los modelos base se
RE-entrenan con todo el train (más datos = mejor modelo base para predecir en
producción/test); el Ridge ya aprendido se deja fijo.

El import de `xgboost` es perezoso (dentro de `fit`): si el paquete no está
instalado, el stacking cae a usar solo el MLP como base model (con un aviso
por `warnings.warn`), en vez de romper la importación del módulo.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .base import Regressor
from .neural_net import NeuralNetRegressor


class StackingRegressorModel(Regressor):
    """Stacking de XGBoost + MLP con meta-modelo Ridge sobre OOF preds temporales."""

    model_type = "stacking"

    def __init__(self, xgb_params: Optional[Dict] = None,
                 nn_params: Optional[Dict] = None, ridge_alpha: float = 1.0,
                 n_folds: int = 4, random_state: int = 42):
        self.xgb_params = dict(xgb_params) if xgb_params else {}
        self.nn_params = dict(nn_params) if nn_params else {}
        self.ridge_alpha = ridge_alpha
        self.n_folds = n_folds
        self.random_state = random_state
        self._models: List[Regressor] = []
        self._ridge: Optional[Ridge] = None
        self._use_xgb: Optional[bool] = None

    def _make_xgb(self):
        from .xgboost_model import XGBoostRegressor
        params = {"random_state": self.random_state, **self.xgb_params}
        return XGBoostRegressor(**params)

    def _make_nn(self):
        params = {"random_state": self.random_state, **self.nn_params}
        return NeuralNetRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            years: Optional[np.ndarray] = None) -> "StackingRegressorModel":
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n = len(X)

        # Intento perezoso de xgboost: si falta, el stacking usa solo el MLP.
        try:
            from .xgboost_model import XGBoostRegressor  # noqa: F401
            self._use_xgb = True
        except ImportError as e:
            self._use_xgb = False
            warnings.warn(
                "xgboost no está disponible; StackingRegressorModel usa "
                f"solo el MLP como base model. Error original: {e}")

        # Orden temporal de las filas para armar los folds (ventana expansiva).
        order = np.argsort(years, kind="stable") if years is not None else np.arange(n)

        n_base = 2 if self._use_xgb else 1
        oof_preds = np.full((n, n_base), np.nan, dtype=float)

        tss = TimeSeriesSplit(n_splits=self.n_folds)
        for tr_pos, va_pos in tss.split(order):
            tr_idx, va_idx = order[tr_pos], order[va_pos]
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xva = X[va_idx]

            col = 0
            if self._use_xgb:
                m_xgb = self._make_xgb().fit(Xtr, ytr)
                oof_preds[va_idx, col] = m_xgb.predict(Xva)
                col += 1
            m_nn = self._make_nn().fit(Xtr, ytr)
            oof_preds[va_idx, col] = m_nn.predict(Xva)

        # Filas nunca validadas (bloque inicial de TimeSeriesSplit): se descartan
        # del ajuste del meta-modelo, no del entrenamiento final de los base models.
        mask = ~np.isnan(oof_preds).any(axis=1)
        self._ridge = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        self._ridge.fit(oof_preds[mask], y[mask])

        # Re-entrenar los base models con TODO el train (ya con el Ridge fijo).
        self._models = []
        if self._use_xgb:
            self._models.append(self._make_xgb().fit(X, y))
        self._models.append(self._make_nn().fit(X, y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models or self._ridge is None:
            raise RuntimeError("Llamá fit() primero.")
        meta_X = np.column_stack([m.predict(X) for m in self._models])
        return self._ridge.predict(meta_X)

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "xgb_params": self.xgb_params,
            "nn_params": self.nn_params,
            "ridge_alpha": self.ridge_alpha,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "use_xgb": self._use_xgb,
        }
