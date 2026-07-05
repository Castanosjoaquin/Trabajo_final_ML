"""Re-exports del subpaquete models.

Todos los detectores comparten la interfaz AnomalyDetector:
fit(X_train_normal) + score_samples(X) (mayor = más anómalo).
"""
from .base import AnomalyDetector
from .baselines import IsolationForestDetector, OneClassSVMDetector
from .ae import AEDetector, DenoisingAEDetector
from .vae import VAEDetector
from .deep_baselines import DeepODDetector
from .ensemble import EnsembleDetector

__all__ = [
    "AnomalyDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "AEDetector",
    "DenoisingAEDetector",
    "VAEDetector",
    "DeepODDetector",
    "EnsembleDetector",
]
