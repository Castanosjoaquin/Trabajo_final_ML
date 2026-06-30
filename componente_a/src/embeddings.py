"""
Proyecciones 2D del espacio de features para visualización.

t-SNE (sklearn) y UMAP (umap-learn). Se proyecta la matriz normalizada
concatenando los splits, conservando identificadores, score y etiqueta para
colorear en la app de comparación.
"""
from __future__ import annotations

import os
import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def compute_projection(X: np.ndarray, method: str = "umap",
                       random_state: int = 42) -> np.ndarray:
    """Devuelve un array (n, 2). method ∈ {'umap', 'tsne'}.

    Silencia RuntimeWarnings espurios de las descomposiciones internas
    (Apple Accelerate BLAS): X ya está validado sin nan/inf, no afectan el
    resultado.
    """
    method = method.lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "umap":
            import umap  # import perezoso: dependencia pesada (numba)
            # n_neighbors acotado a n-1 para datasets chicos
            n_neighbors = max(2, min(15, len(X) - 1))
            reducer = umap.UMAP(n_components=2, random_state=random_state,
                                n_neighbors=n_neighbors)
            return reducer.fit_transform(X)
        if method == "tsne":
            # perplexity acotada a n-1 para datasets chicos
            perplexity = max(5, min(30, len(X) - 1))
            return TSNE(n_components=2, random_state=random_state,
                        perplexity=perplexity, init="pca").fit_transform(X)
    raise ValueError(f"Método desconocido: {method!r} (usar 'umap' o 'tsne')")


def build_embedding_table(X: np.ndarray, meta: pd.DataFrame, scores: np.ndarray,
                          split: str, methods: List[str],
                          random_state: int = 42) -> pd.DataFrame:
    """Tabla larga con proyecciones por método para un split.

    Columnas: method, dim1, dim2, score, split + identificadores/etiqueta.
    """
    frames = []
    for m in methods:
        proj = compute_projection(X, m, random_state)
        df = meta.copy()
        df["method"] = m
        df["dim1"] = proj[:, 0]
        df["dim2"] = proj[:, 1]
        df["score"] = scores
        df["split"] = split
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(meta.columns) + ["method", "dim1", "dim2", "score", "split"])
    return pd.concat(frames, ignore_index=True)


def compute_embeddings(run_id: str, runs_dir: str = "runs",
                       methods: Sequence[str] = ("umap", "tsne"),
                       cache: bool = True) -> pd.DataFrame:
    """Proyecciones 2D **on-demand** para un run ya guardado.

    No se calculan al entrenar (son lentas). Acá se reconstruye X (val+test)
    desde el config del run (data.prepare → build_crop_dataset), se proyecta con
    UMAP/t-SNE y se colorea con los scores YA guardados en scores.parquet. **No
    hace falta el modelo entrenado**: las coordenadas dependen solo de X, que es
    reproducible desde el config. El resultado se cachea en `embeddings.parquet`
    del run-dir, así la primera llamada lo calcula y las siguientes lo leen.

    Devuelve la tabla larga (method, dim1, dim2, score, split + identificadores).
    """
    from dataclasses import fields
    from .config import ExperimentConfig
    from .store import ResultsStore
    from . import data as cdata

    store = ResultsStore(runs_dir)
    run = store.load_run(run_id)
    if cache and run.embeddings is not None and not run.embeddings.empty:
        return run.embeddings

    # Reconstruir el ExperimentConfig desde el config guardado (solo sus campos).
    valid = {f.name for f in fields(ExperimentConfig)}
    exp_cfg = ExperimentConfig(**{k: v for k, v in run.config.items() if k in valid})

    panel_z, _ = cdata.prepare(exp_cfg)
    ds = cdata.build_crop_dataset(panel_z, run.cultivo, exp_cfg)

    X_all = np.vstack([ds.X_val, ds.X_test])
    meta_all = pd.concat([ds.meta_val, ds.meta_test], ignore_index=True)
    split_all = (["val"] * len(ds.X_val)) + (["test"] * len(ds.X_test))

    # Scores guardados, en el MISMO orden val→test que el pipeline determinístico.
    sc = run.scores
    scores = np.concatenate([sc[sc["split"] == "val"]["score"].to_numpy(),
                             sc[sc["split"] == "test"]["score"].to_numpy()])
    if len(scores) != len(X_all):
        raise ValueError(
            f"Desalineación scores/X en {run_id}: {len(scores)} vs {len(X_all)}. "
            "¿Cambió el config o los datos desde que se guardó el run?")

    emb_df = build_embedding_table(X_all, meta_all, scores, "valtest",
                                   list(methods), exp_cfg.random_state)
    emb_df["split"] = split_all * len(methods)

    if cache:
        emb_df.to_parquet(os.path.join(store._dir(run_id), "embeddings.parquet"),
                          index=False)
    return emb_df
