"""
App de comparación de modelos — Componente A
============================================
Compara dos corridas lado a lado (modelo A izquierda, modelo B derecha):
métricas, specs, curvas PR/ROC, distribución de score, proyección 2D
(t-SNE/UMAP), top-10 y heatmap depto × campaña.

Lee del ResultsStore (LocalBackend ahora; cambiar a 'wandb' en la barra
lateral cuando se active). Ejecutar:

    streamlit run app_streamlit.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from componente_a.config import RUNS_DIR, WANDB_PROJECT, WANDB_ENTITY
from componente_a.store import get_store

st.set_page_config(page_title="Componente A — Comparador", layout="wide")

# Métricas comparables (clave en summary -> etiqueta)
METRICS = {
    "test_pr_auc": "PR-AUC (test) ↑",
    "test_roc_auc": "ROC-AUC (test) ↑",
    "test_f1": "F1 anómala (test) ↑",
    "test_precision_at_k": "Precision@k (test) ↑",
    "val_pr_auc": "PR-AUC (val) ↑",
}


# -----------------------------------------------------------------------------
# Carga (cacheada) del store y de runs
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=30)
def _list_runs(backend, runs_dir):
    return get_store(backend, runs_dir=runs_dir, project=WANDB_PROJECT,
                     entity=WANDB_ENTITY).list_runs()


@st.cache_data(show_spinner=False, ttl=300)
def _load_run(backend, runs_dir, run_id):
    store = get_store(backend, runs_dir=runs_dir, project=WANDB_PROJECT,
                      entity=WANDB_ENTITY)
    r = store.load_run(run_id)
    return {"meta": {"run_id": r.run_id, "model_name": r.model_name,
                     "cultivo": r.cultivo, "created_at": r.created_at},
            "config": r.config, "summary": r.summary,
            "scores": r.scores, "curves": r.curves,
            "embeddings": r.embeddings, "sweep": r.sweep}


# -----------------------------------------------------------------------------
# Sidebar — backend, cultivo, selección de runs
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
backend = st.sidebar.selectbox("Backend de resultados", ["local", "wandb"], index=0,
                               help="local = run-dirs en disco; wandb = nube (cuando se active)")
runs_dir = st.sidebar.text_input("Runs dir", RUNS_DIR)
if st.sidebar.button("🔄 Recargar runs"):
    st.cache_data.clear()
    st.rerun()

try:
    all_runs = _list_runs(backend, runs_dir)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo listar runs ({backend}): {e}")
    st.stop()

if not all_runs:
    st.warning(f"No hay runs en '{runs_dir}'. Corré primero: `python run_isoforest.py`")
    st.stop()

cultivos = sorted({r["cultivo"] for r in all_runs})
cultivo = st.sidebar.selectbox("Cultivo", cultivos)
runs_c = [r for r in all_runs if r["cultivo"] == cultivo]

def _fmt(r):
    return f'{r["model_name"]} · {r["run_id"].split("__")[-1]}'

labels = {_fmt(r): r["run_id"] for r in runs_c}
opts = list(labels.keys())
st.sidebar.markdown("### Modelos a comparar")
sel_a = st.sidebar.selectbox("Modelo A (izquierda)", opts, index=0)
sel_b = st.sidebar.selectbox("Modelo B (derecha)", opts,
                             index=min(1, len(opts) - 1))
metric_key = st.sidebar.selectbox("Métrica destacada", list(METRICS.keys()),
                                  format_func=lambda k: METRICS[k])
proj_method = st.sidebar.radio("Proyección 2D", ["umap", "tsne"], horizontal=True)
color_by = st.sidebar.radio("Colorear por", ["label", "score"], horizontal=True)

run_a = _load_run(backend, runs_dir, labels[sel_a])
run_b = _load_run(backend, runs_dir, labels[sel_b])

KNOWN_EVENTS = ["2008/09", "2017/18", "2022/23"]  # para contraste cualitativo


# -----------------------------------------------------------------------------
# Helpers de ploteo
# -----------------------------------------------------------------------------
def _metric_val(run, key):
    v = run["summary"].get(key)
    return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None


def plot_curve(ax, curves_df, curve, split="test"):
    sub = curves_df[(curves_df["curve"] == curve) & (curves_df["split"] == split)]
    if sub.empty:
        ax.text(0.5, 0.5, f"sin datos ({split})", ha="center", va="center")
        return
    ax.plot(sub["x"], sub["y"])
    if curve == "roc":
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    else:
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")


def plot_score_dist(ax, scores_df, threshold):
    test = scores_df[scores_df["split"] == "test"]
    for lab, color in [(0, "steelblue"), (1, "tomato")]:
        s = test[test["anomalia"] == lab]["score"]
        if len(s):
            ax.hist(s, bins=25, alpha=0.6, color=color, density=True,
                    label="normal" if lab == 0 else "anómala")
    if threshold is not None:
        ax.axvline(threshold, color="black", ls=":", label="umbral")
    ax.set_xlabel("Score"); ax.set_ylabel("Densidad"); ax.legend(fontsize=8)


def plot_projection(ax, emb_df, method, color_by):
    sub = emb_df[emb_df["method"] == method]
    if sub.empty:
        ax.text(0.5, 0.5, "sin proyección", ha="center", va="center"); return
    if color_by == "label":
        for lab, color in [(0, "steelblue"), (1, "tomato")]:
            s = sub[sub["anomalia"] == lab]
            ax.scatter(s["dim1"], s["dim2"], s=18, alpha=0.7, color=color,
                       label="normal" if lab == 0 else "anómala")
        ax.legend(fontsize=8)
    else:
        sc = ax.scatter(sub["dim1"], sub["dim2"], s=18, alpha=0.8,
                        c=sub["score"], cmap="YlOrRd")
        plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_xlabel(f"{method.upper()} 1"); ax.set_ylabel(f"{method.upper()} 2")


def plot_heatmap(ax, scores_df):
    test = scores_df[scores_df["split"] == "test"]
    if test.empty:
        ax.text(0.5, 0.5, "sin test", ha="center", va="center"); return
    piv = test.pivot_table(index="departamento", columns="campania",
                           values="score", aggfunc="mean")
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=5)
    plt.colorbar(im, ax=ax, shrink=0.7)


def render_column(run, other_run, metric_key):
    st.subheader(run["meta"]["model_name"])
    st.caption(f'`{run["meta"]["run_id"]}` · {run["meta"]["created_at"]}')

    # Métrica destacada con delta vs el otro modelo
    v = _metric_val(run, metric_key)
    v_other = _metric_val(other_run, metric_key)
    delta = (v - v_other) if (v is not None and v_other is not None) else None
    st.metric(METRICS[metric_key], f"{v:.4f}" if v is not None else "—",
              delta=f"{delta:+.4f}" if delta is not None else None)

    # Tabla de todas las métricas
    rows = [{"métrica": lbl, "valor": _metric_val(run, k)} for k, lbl in METRICS.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    thr = run["summary"].get("threshold")

    # Curvas PR / ROC
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    plot_curve(axs[0], run["curves"], "pr"); axs[0].set_title("PR (test)")
    plot_curve(axs[1], run["curves"], "roc"); axs[1].set_title("ROC (test)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Distribución score + proyección 2D
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    plot_score_dist(axs[0], run["scores"], thr); axs[0].set_title("Score (test)")
    plot_projection(axs[1], run["embeddings"], proj_method, color_by)
    axs[1].set_title(f"{proj_method.upper()} ({color_by})")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Heatmap depto × campaña
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_heatmap(ax, run["scores"]); ax.set_title("Score depto × campaña (test)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Top-10 con marca de eventos conocidos
    test = run["scores"][run["scores"]["split"] == "test"].copy()
    top = test.sort_values("score", ascending=False).head(10)
    top = top[["departamento", "campania", "z_rinde", "anomalia", "score"]]
    top["evento_conocido"] = top["campania"].isin(KNOWN_EVENTS)
    st.markdown("**Top-10 por score (test)**")
    st.dataframe(top, hide_index=True, width='stretch')

    # Specs del modelo
    with st.expander("Especificaciones / config"):
        cfg = {k: val for k, val in run["config"].items() if k != "feature_cols"}
        st.json(cfg)
        st.caption(f'{run["config"].get("n_features", "?")} features')


# -----------------------------------------------------------------------------
# Layout principal — dos columnas
# -----------------------------------------------------------------------------
st.title("🛰️ Componente A — Comparador de detectores de anomalías")
st.caption(f"Cultivo: **{cultivo}** · backend: `{backend}`")

col_a, col_b = st.columns(2)
with col_a:
    render_column(run_a, run_b, metric_key)
with col_b:
    render_column(run_b, run_a, metric_key)

# Barrido de hiperparámetros (si existe)
if run_a["sweep"] is not None or run_b["sweep"] is not None:
    st.markdown("---")
    st.subheader("Barrido de hiperparámetros (PR-AUC val)")
    sc1, sc2 = st.columns(2)
    for c, run in [(sc1, run_a), (sc2, run_b)]:
        with c:
            st.caption(run["meta"]["model_name"])
            if run["sweep"] is not None:
                st.dataframe(run["sweep"], hide_index=True, width='stretch')
            else:
                st.info("Sin barrido para este run.")
