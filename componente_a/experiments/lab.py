"""
lab.py — utilidades para los notebooks de experiments/.

Solo herramientas de bajo nivel: métricas, agregación multi-semilla y gráficos.
El **entrenamiento** (crear el detector, fit, score) va VISIBLE en cada notebook,
no acá — la idea es que se pueda leer todo el flujo sin saltar a un módulo opaco.

Uso típico en un notebook:

    import lab, numpy as np
    filas = []
    for seed in lab.SEEDS:
        det = VAEDetector(..., random_state=seed).fit(ds.X_train)
        scores = det.score_samples(ds.X_test)
        filas.append(lab.metrics(scores, ds.y_test))
    lab.mean_std(filas)          # media ± std entre semillas
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)

# Hacer importable el paquete `src` desde los notebooks (experiments/ es hijo de raíz).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Semillas del estudio (10). El profesor puede reducirlas arriba de cada notebook
# para que el "Run all" sea más rápido (los números cambian dentro del ±std).
# También se puede fijar por entorno: LAB_SEEDS="42,43,44" para una corrida veloz.
SEEDS = ([int(s) for s in os.environ["LAB_SEEDS"].split(",")]
         if os.environ.get("LAB_SEEDS") else list(range(42, 52)))

# Paleta consistente en todos los notebooks.
C_NORMAL, C_ANOM, C_BASE, C_VAE = "#4C72B0", "#C44E52", "#DD8452", "#4C72B0"


# ===========================================================================
# Métricas
# ===========================================================================
def metrics(scores: np.ndarray, y: np.ndarray, contamination: float = 0.10) -> dict:
    """Métricas de un vector de scores (mayor = más anómalo) vs etiquetas y.

    - pr_auc  : área PR (principal, libre de umbral).
    - roc_auc : área ROC (referencia).
    - prec_at_k       : precisión en los top-k, k = nº de anomalías reales.
    - recall_at_contam: recall usando como corte el top-(contamination) de scores.
    """
    y = np.asarray(y).astype(int)
    k = int(y.sum())
    order = np.argsort(scores)[::-1]
    k_cont = max(1, int(len(y) * contamination))
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "prec_at_k": float(y[order[:k]].sum() / k) if k else np.nan,
        "recall_at_contam": float(y[order[:k_cont]].sum() / k) if k else np.nan,
    }


def mean_std(rows: list[dict]) -> pd.Series:
    """Media ± std de una lista de dicts de métricas (una fila por semilla).
    Devuelve una Series con strings 'media±std' listos para mostrar."""
    df = pd.DataFrame(rows)
    return pd.Series({c: f"{df[c].mean():.3f}±{df[c].std():.3f}" for c in df.columns})


def evaluate(make_detector, ds, seeds, metric="pr_auc") -> tuple[float, float]:
    """Evaluación estándar EN IGUALDAD DE CONDICIONES: entrena el detector una vez
    por semilla y devuelve (media, desvío) de `metric` sobre test. Así toda
    comparación de modelos se hace multi-seed, nunca con una corrida suelta.

    make_detector(seed) -> detector sin entrenar (con ese random_state).
    Para detectores deterministas (p. ej. OneClassSVM, que no usa semilla) el
    desvío da 0 — se reportan igual, para que la tabla sea homogénea."""
    vals = [metrics(make_detector(s).fit(ds.X_train).score_samples(ds.X_test),
                    ds.y_test)[metric] for s in seeds]
    return float(np.mean(vals)), float(np.std(vals))


def fmt(mean_std_tuple) -> str:
    """Formatea (media, desvío) como 'media±desvío'."""
    m, s = mean_std_tuple
    return f"{m:.3f}±{s:.3f}"


def leaderboard(resultados: dict[str, list[dict]]) -> pd.DataFrame:
    """resultados = {nombre_modelo: [métricas por semilla]}. Tabla comparativa
    ordenada por PR-AUC medio (de mayor a menor)."""
    filas = []
    for nombre, rows in resultados.items():
        df = pd.DataFrame(rows)
        fila = {"modelo": nombre}
        for c in df.columns:
            fila[c] = df[c].mean()
            # 1 corrida (modelo determinista, p. ej. OCSVM) → std 0, no NaN.
            fila[c + "_std"] = df[c].std() if len(df) > 1 else 0.0
        filas.append(fila)
    out = pd.DataFrame(filas).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return out


# Métricas que se muestran por defecto en TODO el proyecto (Componente A). El resto
# (prec_at_k, recall_at_contam) se calcula igual pero no se muestra salvo pedido.
COLS_STD = ("pr_auc", "roc_auc")


def tabla(resultados: dict[str, list[dict]], cols=COLS_STD) -> pd.DataFrame:
    """Presentación ESTÁNDAR de resultados del Componente A: un DataFrame con una
    fila por modelo y las métricas como 'media±std'. Es la forma única de mostrar
    métricas en el proyecto — reemplaza imprimir dicts crudos o `print(f"...")`
    sueltos, para que toda tabla de resultados se vea igual.

    resultados = {nombre_modelo: [métricas por semilla]} (una sola corrida vale; los
    modelos deterministas dan ±0.000). Ordena por la primera métrica (desc)."""
    lb = leaderboard(resultados).sort_values(cols[0], ascending=False)
    filas = {r["modelo"]: {c: f"{r[c]:.3f}±{r[c + '_std']:.3f}" for c in cols}
             for _, r in lb.iterrows()}
    return pd.DataFrame(filas).T


# ===========================================================================
# Gráficos
# ===========================================================================
def plot_scores(scores: np.ndarray, y: np.ndarray, ax=None, titulo=""):
    """Histograma del score separando normales vs anómalas. Si se separan, el
    modelo rankea bien."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    y = np.asarray(y).astype(int)
    ax.hist(scores[y == 0], bins=40, alpha=0.6, density=True, label="normal", color=C_NORMAL)
    ax.hist(scores[y == 1], bins=40, alpha=0.6, density=True, label="anómala", color=C_ANOM)
    ax.set_yscale("log"); ax.set_xlabel("score de anomalía"); ax.set_ylabel("densidad")
    ax.set_title(titulo, fontsize=9); ax.legend(fontsize=8)
    return ax


def plot_pr(curvas: dict[str, np.ndarray], y: np.ndarray, ax=None, titulo="Curvas PR (test)"):
    """curvas = {nombre: scores}. Dibuja la curva Precision-Recall de cada uno."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    y = np.asarray(y).astype(int)
    for nombre, sc in curvas.items():
        prec, rec, _ = precision_recall_curve(y, sc)
        ax.plot(rec, prec, lw=1.8, label=f"{nombre} (PR-AUC {average_precision_score(y, sc):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(titulo)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return ax


def confusion_top(scores: np.ndarray, y: np.ndarray, q: float, ax=None, titulo=""):
    """Matriz de confusión marcando como anómalo el top-q de los scores del
    propio split (sin usar etiquetas para el corte — evita el problema del umbral
    de val que no transfiere a test). q=0.10 → top-10%; q='base' → top-(tasa real)."""
    y = np.asarray(y).astype(int)
    qf = float(y.mean()) if q in ("base", "base_rate") else float(q)
    thr = np.quantile(scores, 1 - qf)
    yp = (scores >= thr).astype(int)
    cm = confusion_matrix(y, yp, labels=[0, 1])
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "anómala"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "anómala"])
    ax.set_xlabel("predicho"); ax.set_ylabel("real")
    etiqueta = "top-tasa real" if q in ("base", "base_rate") else f"top-{int(qf*100)}%"
    ax.set_title(f"{titulo} · {etiqueta}", fontsize=9)
    thr_txt = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr_txt else "black")
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def plot_leaderboard(lb: pd.DataFrame, ax=None, titulo="PR-AUC (test, media±std)"):
    """Barras horizontales de un leaderboard (salida de lab.leaderboard)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(lb))))
    d = lb.iloc[::-1]
    colores = [C_ANOM if ("IForest" in m or "OCSVM" in m) else C_VAE for m in d["modelo"]]
    ax.barh(d["modelo"], d["pr_auc"], xerr=d.get("pr_auc_std"), capsize=3, color=colores, alpha=0.85)
    for i, v in enumerate(d["pr_auc"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("PR-AUC (test)"); ax.set_title(titulo); ax.grid(axis="x", alpha=0.3)
    return ax


# ===========================================================================
# Proyecciones 2D (para el t-SNE de VAE / ensemble)
# ===========================================================================
def embed_2d(X: np.ndarray, method: str = "tsne", seed: int = 42) -> np.ndarray:
    """Proyecta X a 2D con t-SNE o PCA (para visualizar el espacio)."""
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(X)


def plot_embedding(emb: np.ndarray, color_by: np.ndarray, kind: str = "label",
                   ax=None, titulo=""):
    """Scatter 2D. kind='label' colorea normal/anómala; kind='score' usa un
    gradiente continuo por score."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    if kind == "label":
        c = np.asarray(color_by).astype(int)
        ax.scatter(emb[c == 0, 0], emb[c == 0, 1], s=8, alpha=0.35, color=C_NORMAL, label="normal")
        ax.scatter(emb[c == 1, 0], emb[c == 1, 1], s=10, alpha=0.8, color=C_ANOM, label="anómala")
        ax.legend(fontsize=8)
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=10, alpha=0.8, c=color_by, cmap="YlOrRd")
        plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(titulo, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    return ax
