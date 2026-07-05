"""Arregla el segfault de xgboost+torch en macOS (doble runtime de OpenMP).

En macOS, xgboost trae su propio `libomp.dylib` y torch trae OTRO. Cuando ambos
conviven en el mismo proceso, entrenar xgboost segfaultea (exit 139) — pasa
siempre que torch se importe antes que xgboost, que es lo habitual (el pipeline
de datos del Componente A importa torch de entrada).

La solución robusta es que xgboost use EXACTAMENTE el mismo libomp que torch, así
hay una sola copia del runtime. Este script reescribe la dependencia de
`libxgboost.dylib` para que apunte al libomp de torch (via `install_name_tool`).

Es idempotente: corrélo una vez, y de nuevo si reinstalás xgboost o torch.

    python componente_b/fix_openmp_macos.py
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    if sys.platform != "darwin":
        print("No es macOS: no hace falta el fix.")
        return 0

    try:
        import torch
    except ImportError:
        print("torch no está instalado; nada que arreglar.")
        return 0
    try:
        import xgboost
    except ImportError:
        print("xgboost no está instalado; nada que arreglar.")
        return 0

    torch_omp = os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")
    xgb_lib = os.path.join(os.path.dirname(xgboost.__file__), "lib", "libxgboost.dylib")

    if not os.path.exists(torch_omp):
        print(f"No encontré el libomp de torch en {torch_omp}; abortando.")
        return 1
    if not os.path.exists(xgb_lib):
        print(f"No encontré {xgb_lib}; abortando.")
        return 1

    deps = subprocess.check_output(["otool", "-L", xgb_lib], text=True)
    if torch_omp in deps:
        print("Ya está apuntando al libomp de torch. Nada que hacer.")
        return 0

    # xgboost declara la dependencia como @rpath/libomp.dylib.
    subprocess.check_call([
        "install_name_tool", "-change",
        "@rpath/libomp.dylib", torch_omp, xgb_lib,
    ])
    print(f"OK: libxgboost ahora usa {torch_omp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
