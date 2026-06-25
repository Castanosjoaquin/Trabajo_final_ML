"""
Proyecciones 2D del espacio de features para visualización.

t-SNE (sklearn) y UMAP (umap-learn). Se proyecta la matriz normalizada
concatenando los splits, conservando identificadores, score y etiqueta para
colorear en la app y para exportar como tabla a W&B (render 'Combined 2D
Projection').
"""
from __future__ import annotations

import warnings
from typing import List

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
    return pd.concat(frames, ignore_index=True)
