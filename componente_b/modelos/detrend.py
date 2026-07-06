"""Meta-regresor con tendencia lineal explícita (detrend) — fix de extrapolación.

Problema que resuelve: el rinde tiene tendencia tecnológica creciente y el test
(2021–2024) queda FUERA del rango de años de train (≤2020). Los árboles (XGBoost,
RF) no extrapolan: para años nuevos predicen la hoja del último año visto, así
que la tendencia se les "aplana" justo en el test. Pasarles `year` crudo no
alcanza.

Solución clásica: descomponer rinde = tendencia_lineal(año) + residuo.
1. Se ajusta en train una recta y ~ año (la tendencia secular, que SÍ extrapola
   por ser lineal).
2. El modelo interno (cualquier `Regressor`) aprende el RESIDUO, donde ya no hay
   tendencia que extrapolar.
3. `predict` = tendencia(año) + residuo_predicho.

`year_idx` es el índice de la columna de año dentro de X. En el RegDataset del
repo `year` es la última feature cuando `use_year=True`, y viene ESCALADA
(StandardScaler de train); como el escalado es afín, la recta sobre el año
escalado es equivalente a la recta sobre el año crudo.
"""
from __future__ import annotations

from typing import Dict, Optional, Type

import numpy as np

from .base import Regressor


class DetrendedRegressor(Regressor):
    """Envuelve cualquier Regressor: tendencia lineal por año + modelo del residuo."""

    model_type = "detrended"

    def __init__(self, inner_cls: Type[Regressor], inner_params: Optional[Dict] = None,
                 year_idx: int = -1):
        self.inner_cls = inner_cls
        self.inner_params = dict(inner_params or {})
        self.year_idx = year_idx
        self._inner: Optional[Regressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DetrendedRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float).reshape(-1)
        yr = X[:, self.year_idx].astype(float)
        # Tendencia lineal de train (polyfit grado 1). Con año constante en
        # train la pendiente se fuerza a 0 (no hay tendencia identificable).
        if np.ptp(yr) > 0:
            self.slope_, self.intercept_ = np.polyfit(yr, y, deg=1)
        else:
            self.slope_, self.intercept_ = 0.0, float(y.mean())
        resid = y - (self.slope_ * yr + self.intercept_)
        self._inner = self.inner_cls(**self.inner_params).fit(X, resid)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._inner is None:
            raise RuntimeError("Llamá fit() primero.")
        X = np.asarray(X, dtype=np.float32)
        yr = X[:, self.year_idx].astype(float)
        return self.slope_ * yr + self.intercept_ + self._inner.predict(X)

    def get_config(self) -> Dict:
        inner = (self._inner.get_config() if self._inner is not None
                 else {"model_type": getattr(self.inner_cls, "model_type", "?"),
                       **self.inner_params})
        return {"model_type": self.model_type, "year_idx": self.year_idx,
                "slope": getattr(self, "slope_", None), "inner": inner}
