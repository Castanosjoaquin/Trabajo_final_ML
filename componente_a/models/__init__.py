"""Re-exports del subpaquete models.

Import público preservado:
    from componente_a.models import IsolationForestDetector   # sigue funcionando
    from componente_a.models import AEDetector, VAEDetector   # nuevo
"""
from .base import AnomalyDetector
from .baselines import IsolationForestDetector, PCAReconDetector
from .ae import AEDetector, DenoisingAEDetector
from .vae import VAEDetector

__all__ = [
    "AnomalyDetector",
    "IsolationForestDetector",
    "PCAReconDetector",
    "AEDetector",
    "DenoisingAEDetector",
    "VAEDetector",
]
