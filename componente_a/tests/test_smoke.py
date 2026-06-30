"""
Smoke tests: datos sintéticos 700×40, cubre todos los detectores.

Verifican que el contrato ABC se respeta y que los modelos producen scores
finitos con PR-AUC mejor que random en datos claramente separados.
No requieren datos reales.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.models import (
    AEDetector,
    AnomalyDetector,
    DenoisingAEDetector,
    IsolationForestDetector,
    PCAReconDetector,
    VAEDetector,
)

# ---------------------------------------------------------------------------
# Datos sintéticos: normal ~ N(0,1), anomalía ~ N(3,2)  →  separación clara
# ---------------------------------------------------------------------------
_N_TOTAL = 700
_N_FEATURES = 40
_ANOMALY_RATE = 0.10


def _make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_anom = int(_N_TOTAL * _ANOMALY_RATE)
    n_norm = _N_TOTAL - n_anom
    X_norm = rng.normal(0, 1, (n_norm, _N_FEATURES)).astype(np.float32)
    X_anom = rng.normal(3, 2, (n_anom, _N_FEATURES)).astype(np.float32)
    X = np.vstack([X_norm, X_anom])
    y = np.array([0] * n_norm + [1] * n_anom)
    idx = rng.permutation(_N_TOTAL)
    return X[idx], y[idx]


def _split(X, y):
    n = len(X)
    i1, i2 = int(n * 0.60), int(n * 0.80)
    # Train: solo filas normales del 60% inicial
    mask = y[:i1] == 0
    return X[:i1][mask], X[i1:i2], y[i1:i2], X[i2:], y[i2:]


_X, _y = _make_data(seed=0)
X_train, X_val, y_val, X_test, y_test = _split(_X, _y)

# ---------------------------------------------------------------------------
# Parámetros de entrenamiento reducidos para velocidad en tests
# ---------------------------------------------------------------------------
_FAST = dict(hidden_dims=(32, 16), latent_dim=4, max_epochs=30, patience=10, random_state=0)

DETECTORS: list[tuple[str, AnomalyDetector]] = [
    ("isolation_forest", IsolationForestDetector(n_estimators=50, random_state=0)),
    ("pca_recon",        PCAReconDetector(n_components=10, random_state=0)),
    ("ae",               AEDetector(**_FAST)),
    ("dae_salt_pepper",  DenoisingAEDetector(**_FAST, corruption=0.1, noise_type="salt_pepper")),
    ("dae_gaussian",     DenoisingAEDetector(**_FAST, corruption=0.3, noise_type="gaussian")),
    ("vae_recon_error",  VAEDetector(**_FAST, score_mode="recon_error")),
    ("vae_neg_elbo",     VAEDetector(**_FAST, score_mode="neg_elbo")),
    ("vae_recon_prob",   VAEDetector(**_FAST, score_mode="recon_prob", n_mc_samples=5)),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,detector", DETECTORS, ids=[d[0] for d in DETECTORS])
def test_fit_score_shapes(name, detector):
    """fit() + score_samples() devuelven arrays 1D con el tamaño correcto."""
    detector.fit(X_train)
    sc_val = detector.score_samples(X_val)
    sc_test = detector.score_samples(X_test)
    assert sc_val.shape == (len(X_val),),  f"{name}: val shape {sc_val.shape}"
    assert sc_test.shape == (len(X_test),), f"{name}: test shape {sc_test.shape}"


@pytest.mark.parametrize("name,detector", DETECTORS, ids=[d[0] for d in DETECTORS])
def test_scores_finite(name, detector):
    """No hay NaN ni Inf en los scores."""
    detector.fit(X_train)
    for split, X in [("val", X_val), ("test", X_test)]:
        sc = detector.score_samples(X)
        assert np.isfinite(sc).all(), f"{name} {split}: scores no finitos"


@pytest.mark.parametrize("name,detector", DETECTORS, ids=[d[0] for d in DETECTORS])
def test_pr_auc_above_random(name, detector):
    """PR-AUC en test > 0.5 con datos claramente separados (N(0,1) vs N(3,2))."""
    detector.fit(X_train)
    sc_test = detector.score_samples(X_test)
    pr_auc = average_precision_score(y_test, sc_test)
    assert pr_auc > 0.5, f"{name}: PR-AUC={pr_auc:.3f} (esperado > 0.5)"


@pytest.mark.parametrize("name,detector", DETECTORS, ids=[d[0] for d in DETECTORS])
def test_get_config_has_model_type(name, detector):
    """get_config() devuelve dict con 'model_type'."""
    cfg = detector.get_config()
    assert isinstance(cfg, dict), f"{name}: get_config() no devolvió dict"
    assert "model_type" in cfg, f"{name}: falta 'model_type' en get_config()"
    assert cfg["model_type"] == detector.model_type


def test_deep_model_trains_and_scores():
    """Un AE entrena y puntúa sobre datos chicos sin errores."""
    det = AEDetector(hidden_dims=(16,), latent_dim=4, max_epochs=5, random_state=0)
    det.fit(X_train)
    sc = det.score_samples(X_val)
    assert sc.shape == (len(X_val),)
    assert np.isfinite(sc).all()


def test_import_backward_compat():
    """from src.models import IsolationForestDetector sigue funcionando."""
    from src.models import IsolationForestDetector as IFD  # noqa: F401
    assert IFD.model_type == "isolation_forest"
