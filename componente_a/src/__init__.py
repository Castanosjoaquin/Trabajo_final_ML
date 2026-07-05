"""
Componente A — Detección de anomalías de rinde (soja y maíz).

Paquete mínimo compartido por los notebooks de `experiments/`:
  - config : constantes (rutas, features, splits, etiqueta).
  - data   : pipeline de datos (panel → etiqueta → splits → normalización).
  - models : detectores con interfaz común fit(X) / score_samples(X).

Todo lo demás (entrenamiento, evaluación, comparaciones, gráficos) vive
VISIBLE en los notebooks. Principio source-only: el panel nunca se modifica.
"""

from . import config, data, models  # noqa: F401

__all__ = ["config", "data", "models"]
