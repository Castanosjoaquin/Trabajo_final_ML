"""Ensemble de detectores: combina los scores normalizados de varios miembros.

Una sola clase cubre los dos casos:
- **Seed-ensemble**: N copias del MISMO modelo con semillas distintas → promedia
  los scores para BAJAR LA VARIANZA entre semillas (deep ensemble; ataca el
  problema #1 diagnosticado en AE/VAE). Ref: RandNet (Chen et al. 2017),
  deep ensembles (Lakshminarayanan et al. 2017).
- **Hetero-ensemble**: modelos distintos (p. ej. VAE recon_prob + IsolationForest)
  → combina señales complementarias; suele superar a cada parte por separado.

La normalización per-miembro (z-score con stats de train) es esencial: los
scores viven en escalas distintas (recon_prob ~ cientos, IForest ~ 0.x), así que
promediar crudo dejaría que el de mayor escala domine.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np

from .base import AnomalyDetector


class EnsembleDetector(AnomalyDetector):
    """Combina varios `AnomalyDetector` ya construidos. mayor = más anómalo."""

    model_type = "ensemble"

    def __init__(self, members: List[AnomalyDetector],
                 normalize: str = "zscore", combine: str = "mean"):
        if not members:
            raise ValueError("EnsembleDetector requiere al menos un miembro.")
        if normalize not in ("zscore", "rank"):
            raise ValueError(f"normalize debe ser 'zscore' | 'rank'; got {normalize!r}")
        if combine not in ("mean", "max"):
            raise ValueError(f"combine debe ser 'mean' | 'max'; got {combine!r}")
        self.members = list(members)
        self.normalize = normalize
        self.combine = combine
        # stats de normalización por miembro, calculadas en fit() sobre train.
        self._stats: List[Union[Tuple[float, float], np.ndarray]] = []

    def fit(self, X: np.ndarray) -> "EnsembleDetector":
        self._stats = []
        for m in self.members:
            m.fit(X)
            s = m.score_samples(X)
            if self.normalize == "zscore":
                mu, sd = float(np.mean(s)), float(np.std(s))
                self._stats.append((mu, sd if sd > 1e-12 else 1.0))
            else:  # rank: scores de train ordenados como referencia
                self._stats.append(np.sort(s))
        # Expone la curva de loss del primer miembro que la tenga (para el runner).
        for m in self.members:
            h = getattr(m, "history_", None)
            if h:
                self.history_ = h
                break
        return self

    def _normalize_one(self, i: int, s: np.ndarray) -> np.ndarray:
        if self.normalize == "zscore":
            mu, sd = self._stats[i]  # type: ignore[misc]
            return (s - mu) / sd
        # rank: fracción de scores de train por debajo de cada valor ∈ [0, 1]
        ref = self._stats[i]
        return np.searchsorted(ref, s, side="right") / len(ref)

    def _score_matrix(self, X: np.ndarray) -> np.ndarray:
        """Matriz (n_members, n) de scores normalizados por miembro."""
        if not self._stats:
            raise RuntimeError("Llamá fit() primero.")
        cols = [self._normalize_one(i, m.score_samples(X))
                for i, m in enumerate(self.members)]
        return np.vstack(cols)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        M = self._score_matrix(X)
        return M.mean(axis=0) if self.combine == "mean" else M.max(axis=0)

    def score_std(self, X: np.ndarray) -> np.ndarray:
        """Desacuerdo del ensemble por muestra: desvío entre los scores
        normalizados de los miembros. Es la incertidumbre epistémica del
        ensemble (filosofía 'consistentemente difícil = anomalía'): std BAJO con
        score alto = anomalía robusta (todos los miembros coinciden); std ALTO =
        muestra ambigua/borderline donde los miembros no se ponen de acuerdo."""
        return self._score_matrix(X).std(axis=0)

    def get_config(self) -> Dict:
        return {
            "model_type": self.model_type,
            "normalize": self.normalize,
            "combine": self.combine,
            "n_members": len(self.members),
            "members": [m.get_config() for m in self.members],
        }
