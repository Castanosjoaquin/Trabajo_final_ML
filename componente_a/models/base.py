"""Contrato mínimo para todos los detectores del Componente A."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class AnomalyDetector(ABC):
    """Interfaz compartida: fit(X) sobre normales, score_samples(X) mayor = más anómalo."""

    model_type: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "AnomalyDetector": ...

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score de anomalía: mayor = más anómalo."""

    @abstractmethod
    def get_config(self) -> Dict:
        """Hiperparámetros del modelo (para registrar/comparar)."""
