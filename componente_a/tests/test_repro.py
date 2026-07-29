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

# Baseline re-registrado el 2026-07-28 al pasar `_h` a hash redondeado (ver abajo).
# Los hashes cambiaron por el cambio de `_h`, NO por un cambio de datos: se verificó
# que el panel de git HEAD y el de trabajo son idénticos columna por columna, que
# n_train/n_test no se movieron, y que el parche de la columna `region` de ese mismo
# día es neutral (X_train idéntico antes y después). El baseline anterior
# (2026-07-08) había quedado inservible por deriva de floats tras actualizar a
# pandas 3.0 / numpy 2.4.
#
# Panel: 27865 filas -> 20672 tras dedup; pipeline 72 features; clip de rinde_kgha
# 0.5%/99.5% por cultivo en compute_z_rinde.
BASELINE = {
    "soja": {"X_train": "2546b1bddc254af7", "X_test": "b4b33cb812f08628",
             "y_test": "659cc215951a6495", "n_train": 6142, "n_test": 942},
    "maiz": {"X_train": "2f3a9ec86b2ecda8", "X_test": "b8550e158e9994c1",
             "y_test": "e551c44a99117f35", "n_train": 8082, "n_test": 1160},
}


_DECIMALES = 6


def _h(a) -> str:
    """Hash del contenido, TOLERANTE al ruido de punto flotante.

    Antes se hasheaban los bytes crudos y eso rompía el test con cada upgrade de
    librería: pandas/numpy reordenan internamente las sumas del groupby de
    `_normalize_per_depto` y los z-scores se mueven ~1e-15, sin que cambie ni una
    fila del panel. Redondear a 6 decimales borra esa deriva y sigue detectando
    cualquier cambio real del dataset, que mueve los z-scores muchos órdenes de
    magnitud más.

    Detalles que importan: se castea a float64 para que un cambio de dtype no
    altere los bytes, y se suma 0.0 para canonicalizar -0.0 -> 0.0 (tienen
    representación binaria distinta).

    Tolerancia REAL, medida sobre X_train de soja (6142x72): perturbaciones de
    1e-15 y 1e-12 no mueven el hash; 1e-9 ya lo mueve. El redondeo es un corte,
    no un margen: con ~442k elementos, basta que uno caiga sobre el borde para
    que el hash cambie, y esa probabilidad crece con el tamaño de la
    perturbación. O sea: cubre de sobra la deriva entre versiones (~1e-15, unos
    3 órdenes de margen) y NO sirve como test de "casi igual" para nada más
    grosero. Cualquier cambio real de datos, que mueve z-scores en 1e-3 o más,
    salta sin ambigüedad.
    """
    a = np.round(np.asarray(a, dtype=np.float64), _DECIMALES) + 0.0
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


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
