"""Features derivadas del mejor detector del Componente A (VAE recon_prob).

El Componente A aprendió una representación no supervisada del clima "normal": un
VAE cuyo espacio latente resume cada campaña y cuyo score de reconstrucción mide
cuán anómala es. Acá reusamos ese modelo para *alimentar* la regresión de rinde
del Componente B, con tres estrategias que se prueban en los notebooks:

  1. concatenar el latente del VAE a las features del mejor dataset,
  2. hacer la regresión SOLO en el espacio latente,
  3. agregar una feature categórica `es_anomalo` (el score del VAE, umbralado).

Alineación: se usa el mismo `datos.crop_frame` (orden determinístico) que arma el
RegDataset, así el latente/score de cada fila corresponde exactamente a la fila de
`ds.X_train`/`ds.X_test`.

El VAE se entrena con la config ganadora del Componente A (recon_prob, latente 16,
β=1) sobre las filas NORMALES de train, con las features climáticas normalizadas
por departamento (igual que en el Componente A). El resultado se cachea en disco
(`.latente_cache/`) porque es lo caro de recomputar entre notebooks.

Caveat metodológico (auditoría): esos hiperparámetros "ganadores" del Componente A
(latent_dim=16, β=1, recon_prob) se seleccionaron ilustrando en el test de A
(data snooping declarado en `componente_a/src/config.py`). No es una fuga del
test de B, pero la elección de arquitectura hereda ese sesgo; si se quisiera
eliminar, habría que re-seleccionarlos con validación temporal dentro del train
de A.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd

import datos
from datos import _A_config, _A_data  # config (EXCLUDED_TRAIN_YEARS, ...) y helpers

# El VAE vive en el Componente A.
from src.models.vae import VAEDetector  # noqa: E402

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".latente_cache")


# ===========================================================================
# Normalización por departamento (estilo Componente A) para alimentar el VAE
# ===========================================================================
def _normalize_per_depto(df: pd.DataFrame, clim_cols, geo, normal_mask: np.ndarray) -> np.ndarray:
    """z-score por departamento con parámetros calculados SOLO en las filas normales
    de train. Cuando la std por-depto es 0/NaN (feature constante en ese depto),
    cae a la std global; lo que quede NaN se rellena con 0 (la media). Rellenar en
    vez de descartar mantiene la alineación fila a fila con el RegDataset."""
    normal = df[normal_mask]
    stats = normal.groupby(geo)[clim_cols].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    global_std = normal[clim_cols].std()

    out = np.empty((len(df), len(clim_cols)), dtype=np.float32)
    stats_r = stats.reset_index()
    for j, col in enumerate(clim_cols):
        m = df[geo].merge(stats_r[geo + [f"{col}_mean", f"{col}_std"]], on=geo, how="left")
        mean = m[f"{col}_mean"].values.astype(float)
        std = m[f"{col}_std"].values.astype(float)
        bad = ~np.isfinite(std) | (std == 0)
        std[bad] = global_std.get(col, np.nan)
        std = np.where((std == 0) | ~np.isfinite(std), np.nan, std)
        z = (df[col].values - mean) / std
        out[:, j] = np.nan_to_num(z, nan=0.0)
    return out


# ===========================================================================
# Extracción de features del VAE (con caché)
# ===========================================================================
def vae_features(cultivo: str, latent_dim: int = 16,
                 score_seeds: Sequence[int] = (42, 43, 44), latent_seed: int = 42,
                 max_epochs: int = 150, force: bool = False) -> dict:
    """Entrena el VAE del Componente A y devuelve, alineadas al RegDataset:

    - `Z_train`, `Z_test` : espacio latente (mu_z) de cada fila.
    - `score_train`, `score_test` : score de anomalía (recon_prob; mayor = más raro).

    El latente sale de un VAE (seed `latent_seed`); el score se promedia sobre
    `score_seeds` (el ensemble de semillas es lo que estabiliza el score en el
    Componente A). Resultado cacheado en `.latente_cache/`."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    # "v2": el fix de la etiqueta por cultivo (merge con "cultivo" en la clave)
    # invalida los caches anteriores; el sufijo evita reusar .npz viejos.
    # "v3": panel unificado (era5+ndvi por defecto) → invalida caches previos.
    tag = f"{cultivo}_lat{latent_dim}_ep{max_epochs}_ns{len(score_seeds)}_v3"
    cache = os.path.join(_CACHE_DIR, f"{tag}.npz")
    if os.path.exists(cache) and not force:
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    # Mismo frame determinístico que el RegDataset → alineación garantizada.
    panel = datos.load_panel()
    df, tr_mask, te_mask, clim_cols, geo = datos.crop_frame(panel, cultivo)

    # Filas normales de train (para entrenar el VAE): sin anomalías ni años excluidos.
    # OJO: "cultivo" DEBE estar en la clave del merge — compute_z_rinde da una
    # etiqueta por cultivo, y sin él el drop_duplicates se quedaba con la fila de
    # maíz (orden alfabético) y soja heredaba anomalías ajenas (bug de leakage
    # cruzado detectado en la auditoría).
    panel_z = _A_data.compute_z_rinde(panel)
    key = geo + ["campania_inicio", "cultivo"]
    z = df[key].merge(panel_z[key + ["anomalia"]].drop_duplicates(key), on=key, how="left")
    anom = z["anomalia"].fillna(0).values.astype(int)
    excl = df["campania_inicio"].isin(list(_A_config.EXCLUDED_TRAIN_YEARS)).values
    normal_train = tr_mask.values & (anom == 0) & ~excl

    X = _normalize_per_depto(df, clim_cols, geo, normal_train)
    X_fit = X[normal_train]

    # Latente: un VAE (recon_prob, β=1) con la config ganadora del Componente A.
    vae = VAEDetector(latent_dim=latent_dim, beta=1.0, score_mode="recon_prob",
                      max_epochs=max_epochs, random_state=latent_seed).fit(X_fit)
    Z = vae.encode(X).astype(np.float32)

    # Score: ensemble de semillas (promedio del recon_prob).
    scores = np.zeros(len(df), dtype=np.float64)
    for s in score_seeds:
        m = (vae if s == latent_seed else
             VAEDetector(latent_dim=latent_dim, beta=1.0, score_mode="recon_prob",
                         max_epochs=max_epochs, random_state=s).fit(X_fit))
        scores += m.score_samples(X)
    scores /= len(score_seeds)

    out = {
        "Z_train": Z[tr_mask.values], "Z_test": Z[te_mask.values],
        "score_train": scores[tr_mask.values].astype(np.float32),
        "score_test": scores[te_mask.values].astype(np.float32),
    }
    np.savez(cache, **out)
    return out


def anomaly_flag(vf: dict, contamination: float = 0.10):
    """Feature categórica `es_anomalo` ∈ {0,1}: 1 si el score del VAE supera el
    umbral del (1−contamination) percentil de train (top-`contamination` de las
    filas más raras). El umbral se fija SOLO con train (sin leakage)."""
    thr = np.quantile(vf["score_train"], 1.0 - contamination)
    return ((vf["score_train"] >= thr).astype(np.float32),
            (vf["score_test"] >= thr).astype(np.float32))


# ===========================================================================
# Estrategias de aumento: devuelven un RegDataset nuevo
# ===========================================================================
def _std_train(a_tr: np.ndarray, a_te: np.ndarray):
    """Estandariza con media/desvío de train (para el latente antes de concatenar)."""
    mu = a_tr.mean(axis=0)
    sd = a_tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (a_tr - mu) / sd, (a_te - mu) / sd


def concat_latente(ds, vf: dict):
    """Features del dataset + espacio latente del VAE (estandarizado)."""
    Ztr, Zte = _std_train(vf["Z_train"], vf["Z_test"])
    Xtr = np.hstack([ds.X_train, Ztr]).astype(np.float32)
    Xte = np.hstack([ds.X_test, Zte]).astype(np.float32)
    cols = list(ds.feature_cols) + [f"z{i}" for i in range(Ztr.shape[1])]
    return replace(ds, X_train=Xtr, X_test=Xte, feature_cols=cols)


def solo_latente(ds, vf: dict):
    """Regresión ÚNICAMENTE en el espacio latente del VAE."""
    Ztr, Zte = _std_train(vf["Z_train"], vf["Z_test"])
    cols = [f"z{i}" for i in range(Ztr.shape[1])]
    return replace(ds, X_train=Ztr.astype(np.float32), X_test=Zte.astype(np.float32),
                   feature_cols=cols)


def add_es_anomalo(ds, vf: dict, contamination: float = 0.10):
    """Features del dataset + la categórica `es_anomalo` del VAE."""
    f_tr, f_te = anomaly_flag(vf, contamination)
    Xtr = np.hstack([ds.X_train, f_tr[:, None]]).astype(np.float32)
    Xte = np.hstack([ds.X_test, f_te[:, None]]).astype(np.float32)
    return replace(ds, X_train=Xtr, X_test=Xte,
                   feature_cols=list(ds.feature_cols) + ["es_anomalo"])
