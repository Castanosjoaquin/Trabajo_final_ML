"""Modelos de predicción de rinde del Componente B.

Todos comparten la interfaz `Regressor` (`fit(X, y)` / `predict(X)` /
`get_config()`) y exponen sus hiperparámetros y opciones de regularización en el
constructor, listos para tunear desde los notebooks:

    from componente_b.modelos import LinearRegressor, NeuralNetRegressor, XGBoostRegressor

    modelo = XGBoostRegressor(max_depth=5, reg_lambda=2.0, learning_rate=0.03)
    modelo.fit(X_train, y_train)
    y_hat = modelo.predict(X_test)
"""
from .base import Regressor
from .linear import LinearRegressor
# IMPORTANTE: xgboost DEBE importarse antes que torch. Ambos traen su propio
# runtime de OpenMP (libomp.dylib) y en macOS conviven mal: si torch carga el
# suyo primero, entrenar xgboost segfaultea. Cargando xgboost primero, su libomp
# queda en memoria y torch la reusa sin conflicto. Por eso xgboost_model va antes
# que neural_net acá.
from .xgboost_model import XGBoostRegressor
from .neural_net import NeuralNetRegressor

__all__ = [
    "Regressor",
    "LinearRegressor",
    "NeuralNetRegressor",
    "XGBoostRegressor",
]
