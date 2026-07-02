"""Re-exports del subpaquete models.

Import público preservado:
    from src.models import IsolationForestDetector   # sigue funcionando
    from src.models import AEDetector, VAEDetector   # nuevo
"""
from .base import AnomalyDetector
from .baselines import IsolationForestDetector
from .ae import AEDetector, DenoisingAEDetector
from .vae import VAEDetector
from .hybrid import AEIForestDetector
from .deep_baselines import DeepODDetector
from .ensemble import EnsembleDetector

__all__ = [
    "AnomalyDetector",
    "IsolationForestDetector",
    "AEDetector",
    "DenoisingAEDetector",
    "VAEDetector",
    "AEIForestDetector",
    "DeepODDetector",
    "EnsembleDetector",
]
