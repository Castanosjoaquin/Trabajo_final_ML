"""Guard de reproducibilidad del dataset.

Hashea el CONTENIDO que entra al modelo (X_train/X_test/y_test después de todo
el pipeline de data.py: dedup, etiqueta z_rinde, split temporal, normalización
por depto) y lo compara contra un baseline fijo. Es robusto a re-escrituras del
parquet (otra versión de pandas/pyarrow cambia los bytes pero no el contenido).

Como el resto del pipeline ya es determinístico (seeds fijas: ver test abajo),
si estos hashes coinciden, TODOS los experimentos dan resultados idénticos.
Pensado para correr antes/después de mover o reconstruir el dataset.

Si cambiás el dataset a propósito, actualizá BASELINE con los nuevos hashes.
Referencia adicional (solo válida si se copia el archivo, no si se regenera):
    panel_union.parquet  md5 = 0ed6b26aa1c432b4961023990bce5bab
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from src.config import ExperimentConfig, PANEL_PATH
from src import data as cdata

# Baseline registrado el 2026-06-30 (panel limpio: dedup + clave provincia).
BASELINE = {
    "soja": {"X_train": "b0d0e874711ccc40", "X_test": "06197c9db67e42ff",
             "y_test": "3950e6765e4197f3", "n_train": 5717, "n_test": 954},
    "maiz": {"X_train": "4c0f75db1c882bc2", "X_test": "783bdec7ff57db11",
             "y_test": "3b9534a774778c17", "n_train": 7548, "n_test": 1172},
}


def _h(a) -> str:
    return hashlib.md5(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


@pytest.mark.skipif(not os.path.exists(PANEL_PATH),
                    reason="panel real no disponible (data/processed/panel_union.parquet)")
@pytest.mark.parametrize("cultivo", ["soja", "maiz"])
def test_dataset_content_unchanged(cultivo):
    """El contenido del dataset (post data.py) coincide con el baseline → los
    experimentos siguen dando los mismos resultados."""
    cfg = ExperimentConfig()
    panel, _ = cdata.prepare(cfg)
    ds = cdata.build_crop_dataset(panel, cultivo, cfg)
    exp = BASELINE[cultivo]
    assert len(ds.X_train) == exp["n_train"], f"{cultivo}: n_train cambió"
    assert len(ds.X_test) == exp["n_test"],   f"{cultivo}: n_test cambió"
    assert _h(ds.X_train) == exp["X_train"], f"{cultivo}: X_train cambió"
    assert _h(ds.X_test) == exp["X_test"],   f"{cultivo}: X_test cambió"
    assert _h(ds.y_test) == exp["y_test"],   f"{cultivo}: y_test (etiqueta) cambió"
