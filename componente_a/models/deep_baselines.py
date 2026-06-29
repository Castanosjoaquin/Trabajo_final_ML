"""Baselines de detección de anomalías profundos modernos (librería `deepod`).

Envuelve métodos SOTA de AD tabular en la interfaz `AnomalyDetector`, para
benchmarkear contra el VAE recon_prob. Todos no supervisados (entrenan sobre el
train normal, igual que el resto del Componente A):

- 'icl'     : Internal Contrastive Learning (Shenkar & Wolf 2022) — predice un
              subconjunto de features desde el resto; SOTA en tabular.
- 'deepsvdd': Deep SVDD (Ruff 2018) — deep one-class (esfera mínima en el latente).
- 'goad'    : Classification-based AD con transformaciones (Bergman 2020).
- 'neutral' : Neutralizing transformations contrastivas (Qiu 2021).

deepod expone `fit(X)` y `decision_function(X)` (mayor = más anómalo), que mapean
directo a nuestro `fit`/`score_samples`.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .base import AnomalyDetector

# Se importan adentro de fit() para no pagar el import de deepod si no se usa.
_ALGOS = {"icl": "ICL", "deepsvdd": "DeepSVDD", "goad": "GOAD", "neutral": "NeuTraL"}


class DeepODDetector(AnomalyDetector):
    """Wrapper de un modelo de `deepod`. `algo` ∈ {icl, deepsvdd, goad, neutral}."""

    def __init__(self, algo: str = "icl", epochs: int = 50, batch_size: int = 64,
                 lr: float = 1e-3, random_state: int = 42, **extra):
        if algo not in _ALGOS:
            raise ValueError(f"algo debe ser uno de {list(_ALGOS)}; got {algo!r}")
        self.algo = algo
        self.model_type = f"deepod_{algo}"
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.extra = extra
        self._m = None

    def fit(self, X: np.ndarray) -> "DeepODDetector":
        import deepod.models as dm
        cls = getattr(dm, _ALGOS[self.algo])
        self._m = cls(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                      random_state=self.random_state, device="cpu", verbose=0,
                      **self.extra)
        self._m.fit(np.asarray(X, dtype=np.float64))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._m is None:
            raise RuntimeError("Llamá fit() primero.")
        return np.asarray(self._m.decision_function(np.asarray(X, dtype=np.float64)))

    def get_config(self) -> Dict:
        return {"model_type": self.model_type, "algo": self.algo,
                "epochs": self.epochs, "batch_size": self.batch_size,
                "lr": self.lr, "random_state": self.random_state, **self.extra}
