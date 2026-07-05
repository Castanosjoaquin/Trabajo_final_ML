"""Utilidades de evaluación para los notebooks del Componente B.

Herramientas de bajo nivel: métricas de regresión, baselines, búsqueda de
hiperparámetros con validación cruzada TEMPORAL y gráficos. El entrenamiento del
modelo final va VISIBLE en cada notebook (mismo criterio que el Componente A).

Uso típico:

    import evaluacion as ev
    from componente_b.modelos import XGBoostRegressor
    grid = {"max_depth": [3, 4, 6], "reg_lambda": [1.0, 5.0]}
    tabla, best = ev.buscar(XGBoostRegressor, grid, ds, metric="rmse")
    modelo = XGBoostRegressor(**best).fit(ds.X_train, ds.y_train)
    print(ev.metricas(ds.y_test, modelo.predict(ds.X_test)))
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Hacer importable el paquete componente_b desde los notebooks.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Paleta consistente con el Componente A.
C_BASE, C_LINEAR, C_XGB, C_NN = "#DD8452", "#4C72B0", "#55A868", "#C44E52"


# ===========================================================================
# Métricas
# ===========================================================================
def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Métricas de regresión de rinde (kg/ha).

    - mae  : error absoluto medio (kg/ha), robusto e interpretable.
    - rmse : raíz del error cuadrático medio (kg/ha), penaliza errores grandes.
    - r2   : proporción de varianza explicada (1 = perfecto, 0 = predecir la media).
    - mape : error porcentual absoluto medio (%), relativo al rinde real.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0                      # MAPE indefinido en rinde 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100),
    }


# ===========================================================================
# Baselines
# ===========================================================================
def pred_media(ds) -> np.ndarray:
    """Baseline trivial: predecir la media de rinde de train para todo test."""
    return np.full(len(ds.y_test), ds.y_train_mean, dtype=float)


def pred_media_depto(ds) -> np.ndarray:
    """Baseline agronómico: predecir, para cada fila de test, el rinde medio
    histórico de su departamento en train (climatología). Deptos nuevos → media
    global de train."""
    geo = [c for c in ["provincia", "departamento"] if c in ds.meta_test.columns]
    keys = list(ds.meta_test[geo].itertuples(index=False, name=None))
    return np.array([ds.depto_mean.get(k, ds.y_train_mean) for k in keys], dtype=float)


# ===========================================================================
# Búsqueda de hiperparámetros (CV temporal)
# ===========================================================================
def _param_combos(grid: Dict[str, Sequence], n_iter: Optional[int],
                  random_state: int) -> List[Dict]:
    """Producto cartesiano del grid, o muestra aleatoria de n_iter combinaciones."""
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    if n_iter is not None and n_iter < len(combos):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(combos), size=n_iter, replace=False)
        combos = [combos[i] for i in idx]
    return combos


def _temporal_folds(years: np.ndarray, n_splits: int):
    """Folds de CV que respetan el tiempo (ventana expansiva): validación siempre
    posterior al train de cada fold. Ordena por año y usa TimeSeriesSplit sobre las
    posiciones, así ningún fold valida sobre años vistos en su entrenamiento."""
    order = np.argsort(years, kind="stable")
    for tr_pos, va_pos in TimeSeriesSplit(n_splits=n_splits).split(order):
        yield order[tr_pos], order[va_pos]


def buscar(model_cls: Callable, grid: Dict[str, Sequence], ds,
           metric: str = "rmse", n_splits: int = 4, n_iter: Optional[int] = None,
           random_state: int = 42, fixed: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """Búsqueda de hiperparámetros con CV temporal sobre el TRAIN (el test nunca se
    toca acá). Para cada combinación entrena en cada fold y promedia la métrica de
    validación. Devuelve (tabla ordenada, mejores_params).

    - `grid`   : dict {hiperparámetro: [valores]}.
    - `metric` : 'rmse' | 'mae' | 'r2' (se optimiza; menor es mejor salvo r2).
    - `n_iter` : si se pasa, búsqueda aleatoria de n_iter combos (si no, grid full).
    - `fixed`  : params fijos que se pasan a todos los modelos (no se buscan).
    """
    fixed = fixed or {}
    years = ds.meta_train["campania_inicio"].values
    folds = list(_temporal_folds(years, n_splits))
    combos = _param_combos(grid, n_iter, random_state)
    greater_better = (metric == "r2")

    filas = []
    for params in combos:
        fold_scores = []
        for tr_idx, va_idx in folds:
            model = model_cls(**{**fixed, **params})
            model.fit(ds.X_train[tr_idx], ds.y_train[tr_idx])
            pred = model.predict(ds.X_train[va_idx])
            fold_scores.append(metricas(ds.y_train[va_idx], pred)[metric])
        filas.append({**params,
                      f"cv_{metric}": float(np.mean(fold_scores)),
                      f"cv_{metric}_std": float(np.std(fold_scores))})

    tabla = pd.DataFrame(filas).sort_values(
        f"cv_{metric}", ascending=not greater_better).reset_index(drop=True)
    best = {k: tabla.loc[0, k] for k in grid}
    # Castear a tipos nativos (evita np.int64/np.float64 en los constructores).
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return tabla, best


# ===========================================================================
# Evaluación final + tabla comparativa
# ===========================================================================
def evaluar(nombre: str, y_pred: np.ndarray, ds) -> Dict:
    """Fila de métricas de test para un modelo/baseline ya predicho."""
    return {"modelo": nombre, **metricas(ds.y_test, y_pred)}


def tabla_comparativa(filas: List[Dict], ordenar_por: str = "rmse") -> pd.DataFrame:
    """Ensambla las filas de `evaluar` en una tabla ordenada."""
    df = pd.DataFrame(filas)
    return df.sort_values(ordenar_por, ascending=(ordenar_por != "r2")).reset_index(drop=True)


# ===========================================================================
# Datasets (base vs era5+ndvi) + estrategias con el latente del Componente A
# ===========================================================================
def comparar_datasets_y_latente(model_cls, params: Dict, cultivo: str,
                                fixed: Optional[Dict] = None,
                                vae_kwargs: Optional[Dict] = None,
                                ordenar_por: str = "rmse"):
    """Toma una configuración YA elegida y la evalúa, en test, sobre:

      1. dataset 'base' (solo clima),
      2. dataset 'era5_ndvi' (clima + NDVI + ERA5),
      3–5. sobre el dataset que mejor anduvo de (1)/(2): + latente del VAE,
           solo el espacio latente, y + la categórica `es_anomalo`.

    El latente y el `es_anomalo` salen del mejor detector del Componente A (VAE
    recon_prob) vía `latente.vae_features`. Devuelve (tabla, mejor_dataset).
    """
    import datos, latente                     # import perezoso (torch/VAE)
    fixed = fixed or {}
    vae_kwargs = vae_kwargs or {}

    def _fit_eval(nombre, dv):
        model = model_cls(**{**fixed, **params}).fit(dv.X_train, dv.y_train)
        return {"variante": nombre, "n_feats": dv.X_train.shape[1],
                **metricas(dv.y_test, model.predict(dv.X_test))}

    # (1)-(2) los dos datasets
    dss = {name: datos.prepare(cultivo, dataset=name) for name in datos.DATASETS}
    filas = [_fit_eval(name, dss[name]) for name in datos.DATASETS]

    # elegir el mejor dataset por la métrica pedida (menor mejor, salvo r2)
    key = (lambda f: -f[ordenar_por]) if ordenar_por == "r2" else (lambda f: f[ordenar_por])
    mejor = min(filas, key=key)["variante"]
    ds = dss[mejor]

    # (3)-(5) estrategias con el latente del VAE sobre el mejor dataset
    vf = latente.vae_features(cultivo, dataset=mejor, **vae_kwargs)
    filas.append(_fit_eval(f"{mejor} + latente", latente.concat_latente(ds, vf)))
    filas.append(_fit_eval("solo latente",       latente.solo_latente(ds, vf)))
    filas.append(_fit_eval(f"{mejor} + es_anomalo", latente.add_es_anomalo(ds, vf)))

    tabla = pd.DataFrame(filas).sort_values(
        ordenar_por, ascending=(ordenar_por != "r2")).reset_index(drop=True)
    return tabla, mejor


# ===========================================================================
# Gráficos
# ===========================================================================
def plot_pred_vs_real(y_true: np.ndarray, y_pred: np.ndarray, titulo: str = "",
                      color: str = C_XGB, ax=None):
    """Dispersión predicho vs real con la diagonal ideal y=x."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, color=color, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
    ax.set_xlabel("Rinde real (kg/ha)")
    ax.set_ylabel("Rinde predicho (kg/ha)")
    m = metricas(y_true, y_pred)
    ax.set_title(f"{titulo}\nMAE={m['mae']:.0f}  RMSE={m['rmse']:.0f}  R²={m['r2']:.3f}",
                 fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_residuos(y_true: np.ndarray, y_pred: np.ndarray, titulo: str = "",
                  color: str = C_XGB, ax=None):
    """Residuos (real − predicho) vs predicho: chequea sesgo/heterocedasticidad."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    resid = np.asarray(y_true, float) - np.asarray(y_pred, float)
    ax.scatter(y_pred, resid, s=8, alpha=0.35, color=color, edgecolors="none")
    ax.axhline(0, ls="--", color="0.4", lw=1)
    ax.set_xlabel("Rinde predicho (kg/ha)")
    ax.set_ylabel("Residuo (real − pred)")
    ax.set_title(titulo, fontsize=10)
    return ax


def plot_comparativa(tabla: pd.DataFrame, metrica: str = "rmse", ax=None):
    """Barras horizontales de una métrica por modelo (menor mejor salvo r2)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.5 * len(tabla) + 1))
    t = tabla.sort_values(metrica, ascending=(metrica == "r2"))
    ax.barh(t["modelo"], t[metrica], color=C_LINEAR)
    ax.set_xlabel(metrica.upper())
    ax.set_title(f"Comparación de modelos — {metrica.upper()}", fontsize=11)
    for i, v in enumerate(t[metrica]):
        ax.text(v, i, f" {v:.1f}" if metrica != "r2" else f" {v:.3f}",
                va="center", fontsize=9)
    return ax
