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

from src.config import PANEL_PATH
from src import data as cdata

# Baseline re-registrado el 2026-07-08 tras regenerar el panel con CHIRPS (panel
# 27865 filas -> 20672 tras dedup; pipeline 72 features) y el clip de rinde_kgha
# 0.5%/99.5% por cultivo en compute_z_rinde.
BASELINE = {
    "soja": {"X_train": "c5d286c94ad362dc", "X_test": "a9736293782a332c",
             "y_test": "7d395383a26c032f", "n_train": 6142, "n_test": 942},
    "maiz": {"X_train": "61c19c94a7fe94c9", "X_test": "0c7d2bd8ce1e05e3",
             "y_test": "969f07acd8815182", "n_train": 8082, "n_test": 1160},
}


def _h(a) -> str:
    return hashlib.md5(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


@pytest.mark.skipif(not os.path.exists(PANEL_PATH),
                    reason="panel real no disponible (data/processed/panel_union.parquet)")
@pytest.mark.parametrize("cultivo", ["soja", "maiz"])
def test_dataset_content_unchanged(cultivo):
    """El contenido del dataset (post data.py) coincide con el baseline → los
    experimentos siguen dando los mismos resultados."""
    panel_z = cdata.prepare()
    ds = cdata.build_crop_dataset(panel_z, cultivo)
    exp = BASELINE[cultivo]
    assert len(ds.X_train) == exp["n_train"], f"{cultivo}: n_train cambió"
    assert len(ds.X_test) == exp["n_test"],   f"{cultivo}: n_test cambió"
    assert _h(ds.X_train) == exp["X_train"], f"{cultivo}: X_train cambió"
    assert _h(ds.X_test) == exp["X_test"],   f"{cultivo}: X_test cambió"
    assert _h(ds.y_test) == exp["y_test"],   f"{cultivo}: y_test (etiqueta) cambió"
