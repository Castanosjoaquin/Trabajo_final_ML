"""Modelos de predicción de rinde del Componente B.

Todos comparten la interfaz `Regressor` (`fit(X, y)` / `predict(X)` /
`get_config()`) y exponen sus hiperparámetros y opciones de regularización en el
constructor, listos para tunear desde los notebooks:

    from componente_b.modelos import (LinearRegressor, NeuralNetRegressor,
                                       XGBoostRegressor, RandomForestRegressorModel,
                                       HistGBMRegressor, StackingRegressorModel)

    modelo = XGBoostRegressor(max_depth=5, reg_lambda=2.0, learning_rate=0.03)
    modelo.fit(X_train, y_train)
    y_hat = modelo.predict(X_test)
"""
from .base import Regressor
from .linear import LinearRegressor
from .random_forest import RandomForestRegressorModel
from .hist_gbm import HistGBMRegressor
# IMPORTANTE: xgboost DEBE importarse antes que torch. Ambos traen su propio
# runtime de OpenMP (libomp.dylib) y en macOS conviven mal: si torch carga el
# suyo primero, entrenar xgboost segfaultea. Cargando xgboost primero, su libomp
# queda en memoria y torch la reusa sin conflicto. Por eso xgboost_model va antes
# que neural_net acá.
# Si xgboost no está instalado, el resto del paquete sigue siendo usable:
# XGBoostRegressor queda como stub que avisa recién al instanciarse.
try:
    from .xgboost_model import XGBoostRegressor
except ImportError as _e:
    _XGB_ERR = _e

    class XGBoostRegressor:  # type: ignore[no-redef]
        """Stub: xgboost no está instalado (`pip install xgboost`)."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "XGBoostRegressor requiere el paquete 'xgboost' "
                f"(pip install xgboost). Error original: {_XGB_ERR}")
from .neural_net import NeuralNetRegressor
# stacking no importa xgboost a nivel de módulo (import perezoso en fit), así
# que es seguro importarlo siempre, esté o no instalado xgboost.
from .stacking import StackingRegressorModel
from .detrend import DetrendedRegressor

__all__ = [
    "Regressor",
    "LinearRegressor",
    "NeuralNetRegressor",
    "XGBoostRegressor",
    "RandomForestRegressorModel",
    "HistGBMRegressor",
    "StackingRegressorModel",
    "DetrendedRegressor",
]
