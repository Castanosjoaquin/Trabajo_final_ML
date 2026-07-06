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

    - mae   : error absoluto medio (kg/ha), robusto e interpretable.
    - rmse  : raíz del error cuadrático medio (kg/ha), penaliza errores grandes.
    - r2    : proporción de varianza explicada (1 = perfecto, 0 = predecir la media).
    - mape  : error porcentual absoluto medio (%), relativo al rinde real.
    - smape : MAPE simétrico (%), acotado en [0, 200]; la métrica porcentual que
              pide la propuesta (no explota con rindes chicos como el MAPE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0                      # MAPE indefinido en rinde 0
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smask = denom != 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100),
        "smape": float(np.mean(np.abs(y_true[smask] - y_pred[smask]) / denom[smask]) * 100),
    }


def skill_score(y_true: np.ndarray, y_pred: np.ndarray,
                y_ref: np.ndarray) -> float:
    """Skill score de RMSE contra una referencia (típicamente la climatología
    `pred_media_depto`): 1 − RMSE_modelo / RMSE_ref. Positivo = el modelo le
    gana a la referencia; 0 = empata; negativo = pierde. Es la métrica central
    que pide la propuesta: cuánta información REAL aporta el clima del año por
    encima de saber "cuánto suele rendir este departamento"."""
    rmse_m = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_r = float(np.sqrt(mean_squared_error(y_true, y_ref)))
    return 1.0 - rmse_m / rmse_r


def diebold_mariano(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    loss: str = "se", h: int = 1) -> Dict[str, float]:
    """Test de Diebold–Mariano: ¿la diferencia de precisión entre A y B es
    estadísticamente significativa?

    H0: ambos modelos tienen la misma pérdida esperada. d_t = L(e_A) − L(e_B);
    el estadístico es la media de d sobre su error estándar HAC (Newey–West con
    h−1 lags). DM < 0 → A pierde menos que B (A mejor); p < 0.05 → diferencia
    significativa. `loss`: 'se' (cuadrática) o 'ae' (absoluta).

    Nota de uso en este panel: las "observaciones" son filas depto×campaña de
    test, no una serie temporal pura; con h=1 el test asume d_t sin
    autocorrelación (aproximación razonable si se compara sobre las mismas
    filas). Para una versión por año, agregarse los d por campaña antes.
    """
    from scipy import stats
    e_a = np.asarray(y_true, float) - np.asarray(pred_a, float)
    e_b = np.asarray(y_true, float) - np.asarray(pred_b, float)
    if loss == "se":
        d = e_a ** 2 - e_b ** 2
    elif loss == "ae":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"loss desconocida: {loss!r} (opciones: 'se', 'ae')")
    n = len(d)
    d_mean = d.mean()
    # Varianza HAC (Newey–West) de la media de d con h−1 lags.
    gamma0 = np.mean((d - d_mean) ** 2)
    var = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        var += 2.0 * (1.0 - k / h) * cov
    dm = d_mean / np.sqrt(var / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return {"dm": float(dm), "p_value": float(p), "n": int(n),
            "mejor": "A" if dm < 0 else "B"}


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


def fold_X(ds, tr_idx: np.ndarray, va_idx: np.ndarray):
    """X de un fold de CV con el `depto_enc` recomputado SOLO con el train del fold.

    El target encoding usa el rinde (el target): si se deja el `depto_enc` del
    RegDataset — calculado con TODO el train — cada fold de CV "conoce" el rinde
    de sus propias filas de validación a través de esa feature, y la métrica de
    CV sale optimista (fuga detectada en la auditoría). Acá se recalcula la media
    por depto con las filas de train del fold únicamente, y se re-estandariza con
    su media/desvío. Las demás features (clima) no usan el target: su escalado
    global de train es inocuo para la selección de hiperparámetros."""
    Xtr = ds.X_train[tr_idx].copy()
    Xva = ds.X_train[va_idx].copy()
    if "depto_enc" not in ds.feature_cols:
        return Xtr, Xva
    j = ds.feature_cols.index("depto_enc")
    meta = ds.meta_train
    geo = [c for c in ("provincia", "departamento") if c in meta.columns]
    tr_meta = meta.iloc[tr_idx]
    m_global = float(tr_meta["rinde_kgha"].mean())
    dm = tr_meta.groupby(geo)["rinde_kgha"].mean()

    def _enc(idx):
        keys = list(meta.iloc[idx][geo].itertuples(index=False, name=None))
        return np.array([dm.get(k, m_global) for k in keys], dtype=float)

    e_tr, e_va = _enc(tr_idx), _enc(va_idx)
    mu, sd = float(e_tr.mean()), float(e_tr.std()) or 1.0
    Xtr[:, j] = (e_tr - mu) / sd
    Xva[:, j] = (e_va - mu) / sd
    return Xtr, Xva


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

    # X por fold con depto_enc honesto (ver fold_X); se computa una sola vez.
    fold_Xs = [fold_X(ds, tr_idx, va_idx) for tr_idx, va_idx in folds]

    filas = []
    for params in combos:
        fold_scores = []
        for (tr_idx, va_idx), (Xtr, Xva) in zip(folds, fold_Xs):
            model = model_cls(**{**fixed, **params})
            model.fit(Xtr, ds.y_train[tr_idx])
            pred = model.predict(Xva)
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


# Métricas que se muestran por defecto en el Componente B (regresión). El resto
# (mape, smape) se calcula igual pero no se muestra salvo pedido.
COLS_STD = ("rmse", "r2", "mae")


def tabla(filas, cols=COLS_STD) -> pd.DataFrame:
    """Presentación ESTÁNDAR de métricas de regresión (Componente B): un DataFrame
    con una fila por modelo y las columnas reducidas (RMSE, R², MAE), ordenado por
    la primera. Es la forma única de mostrar métricas — reemplaza los `print(f"MAE=
    ...")` sueltos. `filas` = lista de dicts de `evaluar` (o un solo dict)."""
    if isinstance(filas, dict):
        filas = [filas]
    df = pd.DataFrame(filas)
    id_cols = [c for c in ("modelo", "variante") if c in df.columns]
    df = df.sort_values(cols[0], ascending=(cols[0] != "r2")).reset_index(drop=True)
    return df[id_cols + list(cols)].round(3)


# ===========================================================================
# Datasets (base vs era5+ndvi) + estrategias con el latente del Componente A
# ===========================================================================
def cv_score(model_cls, params: Dict, ds, metric: str = "rmse",
             n_splits: int = 4, fixed: Optional[Dict] = None) -> float:
    """Métrica promedio de CV temporal (ventana expansiva) DENTRO de train, para
    una configuración fija. Es el evaluador honesto para cualquier decisión
    (dataset, features, zona vs. pooled): no toca el test."""
    fixed = fixed or {}
    years = ds.meta_train["campania_inicio"].values
    scores = []
    for tr_idx, va_idx in _temporal_folds(years, n_splits):
        Xtr, Xva = fold_X(ds, tr_idx, va_idx)
        m = model_cls(**{**fixed, **params}).fit(Xtr, ds.y_train[tr_idx])
        scores.append(metricas(ds.y_train[va_idx], m.predict(Xva))[metric])
    return float(np.mean(scores))


def comparar_latente(model_cls, params: Dict, cultivo: str,
                     fixed: Optional[Dict] = None,
                     vae_kwargs: Optional[Dict] = None,
                     ordenar_por: str = "rmse", ds=None):
    """Toma una configuración YA elegida y la evalúa, en test, sobre el dataset
    (único, unificado) y las tres estrategias que reusan el mejor detector del
    Componente A (VAE recon_prob, vía `latente.vae_features`): + espacio latente,
    solo el espacio latente, y + la categórica `es_anomalo`.

    Devuelve la tabla comparativa (ordenada por `ordenar_por`).
    """
    import datos, latente                     # import perezoso (torch/VAE)
    fixed = fixed or {}
    vae_kwargs = vae_kwargs or {}
    if ds is None:
        ds = datos.prepare(cultivo)

    def _fit_eval(nombre, dv):
        model = model_cls(**{**fixed, **params}).fit(dv.X_train, dv.y_train)
        return {"variante": nombre, "n_feats": dv.X_train.shape[1],
                **metricas(dv.y_test, model.predict(dv.X_test))}

    vf = latente.vae_features(cultivo, **vae_kwargs)
    filas = [
        _fit_eval("dataset unificado", ds),
        _fit_eval("+ latente", latente.concat_latente(ds, vf)),
        _fit_eval("solo latente", latente.solo_latente(ds, vf)),
        _fit_eval("+ es_anomalo", latente.add_es_anomalo(ds, vf)),
    ]
    return pd.DataFrame(filas).sort_values(
        ordenar_por, ascending=(ordenar_por != "r2")).reset_index(drop=True)


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
