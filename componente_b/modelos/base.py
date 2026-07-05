"""Contrato mínimo para los regresores del Componente B (predicción de rinde).

A diferencia del Componente A (detección de anomalías, `fit(X)` no supervisado),
acá el problema es supervisado: predecir `rinde_kgha` a partir de las features
climáticas. La interfaz es la clásica de sklearn — `fit(X, y)` / `predict(X)` —
más un `get_config()` para registrar y comparar hiperparámetros entre corridas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Regressor(ABC):
    """Interfaz compartida de todos los modelos de rinde.

    Convención: `fit(X, y)` aprende, `predict(X)` devuelve el rinde estimado
    (misma unidad que `y`, kg/ha), `get_config()` expone los hiperparámetros
    para dejar traza de cada experimento.
    """

    model_type: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regressor": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Rinde estimado por muestra (kg/ha)."""

    @abstractmethod
    def get_config(self) -> Dict:
        """Hiperparámetros del modelo (para registrar/comparar)."""
