"""
momentum.py — Memoria inter-campaña: promedio móvil exponencial con decaimiento.

Responde al eje *inter-campaña* del requisito de producto: además de saber qué pasó
en la campaña actual, el modelo tiene que recordar **qué viene pasando en esa zona**,
dándole más peso a lo reciente.

La intuición de partida era un esquema tipo RL con un factor `gamma` que penaliza las
campañas más viejas. Formalmente eso es un EWMA, que tiene la misma recursión que un
retorno descontado:

    M_t = gamma * M_(t-1) + (1 - gamma) * x_t

con `gamma` grande = más memoria (decae lento), `gamma` chico = solo importa lo último.
En términos de pandas es `ewm(alpha=1-gamma, adjust=False)`.

REGLA CRÍTICA (anti-leakage)
----------------------------
La feature de la campaña `t` es `M_(t-1)`, **no** `M_t`: se construye únicamente con
campañas ESTRICTAMENTE anteriores. Sin el `shift(1)`, el momentum de la campaña `t`
incluiría `x_t` — que para la señal `z_rinde` es una función directa del target, o
sea leakage puro. Ese es el error a evitar, y no es sutil: verificado sobre series
aleatorias, omitir el shift cambia los valores hasta en 1.5 desvíos.

Dónde va el shift da igual, y conviene saberlo para no "arreglar" código correcto:
`s.ewm(...).mean().shift(1)` y `s.shift(1).ewm(...).mean()` dan resultados
**idénticos** salvo el primer elemento. El EWMA con `adjust=False` es un filtro
causal lineal, así que conmuta con el desplazamiento (comprobado para gamma en
{0.5, 0.7, 0.85, 0.95}). Lo que NO es equivalente es no desplazar en absoluto.

Es compatible con el walk-forward de `evaluacion.py`: el EWMA solo mira el pasado de
la propia serie del departamento, así que no filtra información del fold de validación
aunque se calcule una sola vez sobre todo el frame.

Este módulo no importa `datos` ni `latente` a propósito (evita un ciclo de imports):
recibe el DataFrame ya armado y, para el score de anomalía, el array ya alineado.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "componente_a")
if _p not in sys.path:
    sys.path.insert(0, _p)

from src import data as _A_data  # noqa: E402

# Valores de gamma a barrer. NO es un hiperparámetro del estimador sino del dataset:
# no puede ir en el grid de `evaluacion.buscar` (que barre parámetros del modelo).
# Se elige construyendo un RegDataset por gamma y comparando con `evaluacion.cv_score`.
GAMMAS: Sequence[float] = (0.5, 0.7, 0.85)

# Tope para |z_rinde| antes de promediarlo.
#
# `compute_z_rinde` divide por un desvío móvil con `min_periods=3`. Cuando un
# departamento arranca con 3 campañas de rinde casi idéntico, ese desvío queda
# diminuto y el z_rinde explota: La Paz (Entre Ríos) tiene rindes de soja de 1408,
# 1400 y 1412 kg/ha y al año siguiente 756 → z = -106. Son artefactos del
# denominador, no anomalías agronómicas.
#
# Para la ETIQUETA no molesta (es un umbral: -106 y -2 son ambos "anómalo"), pero
# como señal continua el EWMA los promedia y, con memoria, un solo valor extremo
# contamina las campañas siguientes del departamento durante años. Sin clip, la
# feature escalada llega a -52 desvíos y comprime todo el resto contra el cero, lo
# que castiga sobre todo a los modelos lineales y a la red.
#
# ±10 es el punto justo: toca el 0.33 % de las filas y deja el rango escalado en
# [-10.6, 10.4]. Es una constante fija, no un percentil de train → sin leakage.
# `compute_z_rinde` NO se toca: el clip vive acá, solo para la feature de momentum.
Z_CLIP: float = 10.0


def ewma_pasado(s: pd.Series, gamma: float) -> pd.Series:
    """`M_(t-1)`: EWMA construido SOLO con campañas estrictamente anteriores.

        s.ewm(...).mean()          -> M_t      incluye x_t   ← LEAKAGE
        s.ewm(...).mean().shift(1) -> M_(t-1)  solo pasado   ← correcto

    El `shift(1)` es lo que hace la diferencia; que vaya antes o después del `ewm`
    no cambia nada (ver la nota del módulo). La primera campaña de cada serie queda
    en NaN (no hay pasado); el relleno lo decide `add_momentum`, que conoce el split.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma debe estar en [0, 1), recibido {gamma}")
    return s.ewm(alpha=1.0 - gamma, adjust=False).mean().shift(1)


def add_momentum(df: pd.DataFrame, geo: List[str], gamma: float,
                 score: Optional[np.ndarray] = None,
                 train_mask: Optional[np.ndarray] = None,
                 clip_z: Optional[float] = Z_CLIP) -> tuple:
    """Agrega las features de momentum al frame de un cultivo.

    Devuelve `(df_con_cols, nombres_nuevos)`.

    PRECONDICIÓN: `df` tiene que venir ordenado por `geo + ["campania_inicio"]` y con
    índice 0..n-1 — que es exactamente lo que garantiza `datos.crop_frame`. Si el
    orden no fuera ese, el `shift(1)` tomaría la campaña equivocada y el resultado
    sería silenciosamente incorrecto: se valida abajo en vez de confiar.

    Señales:
    - `momentum_z_rinde`: EWMA del `z_rinde` histórico del departamento. Mide si la
      zona viene rindiendo por encima o por debajo de su propia norma.
    - `momentum_anomaly_score`: EWMA del score de anomalía del Componente A en
      campañas pasadas. Solo se agrega si se pasa `score` (requiere entrenar el VAE;
      se obtiene de `latente.vae_features`, ver `momentum_score_alineado`).

    `train_mask` se usa para rellenar el arranque de cada serie con una estadística
    de TRAIN únicamente (mismo criterio que los lags en `datos.build_reg_dataset`).
    """
    if not len(df):
        return df, []
    esperado = list(geo) + ["campania_inicio"]
    orden = df[esperado]
    if not orden.equals(orden.sort_values(esperado)):
        raise ValueError(
            "add_momentum requiere df ordenado por geo + campania_inicio "
            "(usar datos.crop_frame). Con otro orden el shift(1) toma la campaña "
            "equivocada y el momentum queda mal sin dar error.")

    df = df.copy()
    new_cols: List[str] = []
    if train_mask is None:
        train_mask = np.ones(len(df), dtype=bool)
    train_mask = np.asarray(train_mask, dtype=bool)

    # --- Señal 1: z_rinde ---
    # El z_rinde ya viene en el panel (lo calcula compute_z_rinde). Si no está, se
    # mergea con "cultivo" en la clave: sin él, el drop_duplicates se queda con la
    # fila de maíz por orden alfabético y soja hereda la etiqueta ajena (es el bug
    # de leakage cruzado ya documentado en latente.py).
    if "z_rinde" in df.columns:
        col = "momentum_z_rinde"
        # Clip de los z_rinde degenerados ANTES de promediar (ver Z_CLIP).
        df["_z_tmp"] = (df["z_rinde"] if clip_z is None
                        else df["z_rinde"].clip(-clip_z, clip_z))
        df[col] = df.groupby(geo, sort=False)["_z_tmp"].transform(
            lambda s: ewma_pasado(s, gamma))
        df = df.drop(columns=["_z_tmp"])
        # Arranque de serie: 0 = "sin información previa". z_rinde ya está
        # estandarizado, así que 0 es su valor neutro y no introduce sesgo.
        df[col] = df[col].fillna(0.0)
        new_cols.append(col)

    # --- Señal 2: score de anomalía del Componente A ---
    if score is not None:
        score = np.asarray(score, dtype=float)
        if len(score) != len(df):
            raise ValueError(f"score tiene {len(score)} filas y df {len(df)}: "
                             "tiene que venir alineado al mismo frame")
        col = "momentum_anomaly_score"
        df["_score_tmp"] = score
        df[col] = df.groupby(geo, sort=False)["_score_tmp"].transform(
            lambda s: ewma_pasado(s, gamma))
        # El score no está centrado, así que el neutro es la media de TRAIN.
        relleno = float(np.nanmean(score[train_mask])) if train_mask.any() else 0.0
        df[col] = df[col].fillna(relleno)
        df = df.drop(columns=["_score_tmp"])
        new_cols.append(col)

    return df, new_cols


def momentum_score_alineado(vf: dict, tr_mask: np.ndarray,
                            te_mask: np.ndarray) -> np.ndarray:
    """Rearma el score del VAE como un array del largo del frame completo.

    `latente.vae_features` devuelve el score ya partido en train/test; para el EWMA
    hace falta la serie entera y en orden, porque la memoria de una campaña de test
    se construye con campañas de train.
    """
    tr_mask, te_mask = np.asarray(tr_mask, bool), np.asarray(te_mask, bool)
    n = len(tr_mask)
    if not (tr_mask | te_mask).all():
        raise ValueError("hay filas que no caen ni en train ni en test")
    out = np.empty(n, dtype=float)
    out[tr_mask] = vf["score_train"]
    out[te_mask] = vf["score_test"]
    return out
