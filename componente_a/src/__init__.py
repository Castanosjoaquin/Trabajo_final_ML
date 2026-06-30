"""
Componente A — Detección de anomalías (región núcleo, soja y maíz).

Paquete compartido para entrenar, evaluar y comparar modelos de detección
de anomalías de forma estandarizada. Cada modelo produce un RunResult con
el mismo esquema, persistido vía ResultsStore (run-dirs locales en `runs/`)
y consumido por la app Streamlit de comparación.

Principio source-only: el panel NUNCA se modifica; todo feature engineering,
splits y normalización se computan acá.
"""

from . import config, data, models, evaluate, embeddings, store, runner  # noqa: F401

__all__ = ["config", "data", "models", "evaluate", "embeddings", "store", "runner"]
