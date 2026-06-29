"""Utilidades para los notebooks de experiments/.

Carga los runs ya guardados en runs/ (no re-entrena nada), expone las tablas de
resultados validadas (panel limpio) y helpers de ploteo. Los notebooks importan
esto y quedan finos: narrativa + tabla + gráfico.
"""
from __future__ import annotations

import os
import sys
import json
import glob

import numpy as np
import pandas as pd

# --- Raíz del repo (este archivo vive en experiments/) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
RUNS = os.path.join(ROOT, "runs")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402
from componente_a.store import get_store  # noqa: E402


# ===========================================================================
# Carga de runs guardados
# ===========================================================================
def load_run_table() -> pd.DataFrame:
    """Todas las corridas guardadas en runs/ con sus métricas y flags de config."""
    rows = []
    if not os.path.isdir(RUNS):
        return pd.DataFrame()
    for mt in sorted(os.listdir(RUNS)):
        td = os.path.join(RUNS, mt)
        if not os.path.isdir(td):
            continue
        for rid in sorted(os.listdir(td)):
            d = os.path.join(td, rid)
            try:
                meta = json.load(open(os.path.join(d, "meta.json")))
                s = json.load(open(os.path.join(d, "summary.json")))
                c = json.load(open(os.path.join(d, "config.json")))
            except Exception:
                continue
            rows.append(dict(
                run_id=rid, model_type=mt, name=meta["model_name"],
                cultivo=meta["cultivo"], created=meta.get("created_at", ""),
                pr_auc=s.get("test_pr_auc_mean", s.get("test_pr_auc")),
                pr_std=s.get("test_pr_auc_std", 0.0),
                roc=s.get("test_roc_auc_mean", s.get("test_roc_auc")),
                rec_k=s.get("test_recall_at_contamination_mean",
                            s.get("test_recall_at_contamination")),
                n_seeds=s.get("n_seeds", 1), n_features=c.get("n_features"),
                panel=os.path.basename(str(c.get("panel_path", ""))),
                use_ndvi=c.get("use_ndvi"), use_era5=c.get("use_era5_features"),
                use_agro=c.get("use_agro_features")))
    return pd.DataFrame(rows)


def latest_run(name: str, cultivo: str | None = None) -> str | None:
    """run_id de la corrida más reciente que matchea `name` (y cultivo)."""
    df = load_run_table()
    m = df[df["name"] == name]
    if cultivo:
        m = m[m["cultivo"] == cultivo]
    return m.sort_values("created")["run_id"].iloc[-1] if len(m) else None


def load_run(run_id: str):
    return get_store("local", runs_dir=RUNS).load_run(run_id)


# ===========================================================================
# Plots
# ===========================================================================
def plot_pr_curves(runs, cultivo, title="Curvas PR (test)"):
    """runs = lista de (name, etiqueta). Grafica la curva PR de test de cada uno."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, label in runs:
        rid = latest_run(name, cultivo)
        if not rid:
            continue
        cur = load_run(rid).curves
        pr = cur[(cur["curve"] == "pr") & (cur["split"] == "test")]
        if not pr.empty:
            ax.plot(pr["x"], pr["y"], label=label, lw=1.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{title} — {cultivo}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); return fig


def plot_score_hist(name, cultivo, bins=40):
    """Histograma del score de test, separado por clase (normal vs anómala)."""
    rid = latest_run(name, cultivo)
    sc = load_run(rid).scores
    t = sc[sc["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(t[t["anomalia"] == 0]["score"], bins=bins, alpha=0.6, label="normal", density=True)
    ax.hist(t[t["anomalia"] == 1]["score"], bins=bins, alpha=0.6, label="anómala", density=True)
    ax.set_xlabel("score de anomalía"); ax.set_ylabel("densidad")
    ax.set_title(f"Distribución de scores — {name} ({cultivo})")
    ax.legend(); fig.tight_layout(); return fig


def barh_leaderboard(df, value="pr_auc", err="pr_std", label="model", title=""):
    """Barras horizontales de un leaderboard (DataFrame con columnas model/pr_auc/pr_std)."""
    d = df.sort_values(value)
    fig, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(d))))
    ax.barh(d[label], d[value], xerr=d.get(err), color="#4C72B0", alpha=0.85)
    for i, v in enumerate(d[value]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("PR-AUC (test)"); ax.set_title(title); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); return fig


def show(df):
    """Display redondeado para notebooks."""
    return df.round(4)


# --- Métricas completas de un run (media±std multi-seed) ---
_METRICS = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
            ("precision_at_k", "Precision@k"), ("recall_at_contamination", "Recall@contam")]


def run_metrics(name, cultivo, split="test") -> pd.DataFrame:
    """Todas las métricas de `split` de un run, con su desvío si es multi-seed."""
    s = load_run(latest_run(name, cultivo)).summary
    rows = []
    for key, lab in _METRICS:
        mean = s.get(f"{split}_{key}_mean", s.get(f"{split}_{key}"))
        std = s.get(f"{split}_{key}_std", 0.0)
        if mean is not None and mean == mean:
            rows.append((lab, float(mean), float(std or 0.0)))
    return pd.DataFrame(rows, columns=["metrica", "valor", "std"])


def plot_all_metrics(name, cultivo, split="test"):
    """Barras de TODAS las métricas del run, con barra de error (±std entre seeds)."""
    m = run_metrics(name, cultivo, split)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(m["metrica"], m["valor"], yerr=m["std"], capsize=4, color="#4C72B0", alpha=0.85)
    for i, (v, e) in enumerate(zip(m["valor"], m["std"])):
        ax.text(i, v + e + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel(f"valor ({split})")
    ax.set_title(f"Métricas de {split} — {name} ({cultivo})")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout(); return fig


def plot_loss(name, cultivo):
    """Curva de loss de entrenamiento (train/val por época) desde curves.parquet."""
    cur = load_run(latest_run(name, cultivo)).curves
    loss = cur[cur["curve"] == "loss"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for split, color in [("train", "#4C72B0"), ("val", "#DD8452")]:
        d = loss[loss["split"] == split].sort_values("x")
        if not d.empty:
            ax.plot(d["x"], d["y"], label=split, color=color, lw=1.8)
    ax.set_xlabel("época"); ax.set_ylabel("loss")
    ax.set_title(f"Curva de loss — {name} ({cultivo})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); return fig


def plot_metric_comparison(names_labels, cultivo, metric="pr_auc"):
    """Barras horizontales comparando modelos en `metric`, con ±std (desde runs)."""
    lab = dict(_METRICS).get(metric, metric)
    rows = []
    for name, label in names_labels:
        rid = latest_run(name, cultivo)
        if not rid:
            continue
        s = load_run(rid).summary
        v = s.get(f"test_{metric}_mean", s.get(f"test_{metric}"))
        e = s.get(f"test_{metric}_std", 0.0)
        if v is not None and v == v:
            rows.append((label, float(v), float(e or 0.0)))
    d = pd.DataFrame(rows, columns=["model", "value", "std"]).sort_values("value")
    fig, ax = plt.subplots(figsize=(7, max(2, 0.55 * len(d))))
    ax.barh(d["model"], d["value"], xerr=d["std"], capsize=4, color="#55A868", alpha=0.85)
    for i, (v, e) in enumerate(zip(d["value"], d["std"])):
        ax.text(v + e + 0.005, i, f"{v:.3f}±{e:.3f}", va="center", fontsize=8)
    ax.set_xlabel(f"{lab} (test)"); ax.set_title(f"Comparación de modelos — {lab} ({cultivo})")
    ax.grid(axis="x", alpha=0.3); fig.tight_layout(); return fig


def plot_leaderboard_compare(metric="pr"):
    """Comparación FINAL soja vs maíz con barras de error (de tbl_leaderboard)."""
    t = tbl_leaderboard().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y - 0.2, t["soja_pr"], height=0.38, xerr=t["soja_std"], capsize=3,
            color="#4C72B0", alpha=0.85, label="soja")
    ax.barh(y + 0.2, t["maiz_pr"], height=0.38, xerr=t["maiz_std"], capsize=3,
            color="#DD8452", alpha=0.85, label="maíz")
    ax.set_yticks(y); ax.set_yticklabels(t["model"])
    ax.set_xlabel("PR-AUC (test, media±std)")
    ax.set_title("Comparación final de modelos (¿cuál es mejor?)")
    ax.legend(); ax.grid(axis="x", alpha=0.3); fig.tight_layout(); return fig


# ===========================================================================
# Tablas de resultados VALIDADAS (panel limpio, multi-seed) — fuente única
# ===========================================================================
def tbl_leaderboard() -> pd.DataFrame:
    """Leaderboard final (panel limpio). PR-AUC media±std, test."""
    return pd.DataFrame([
        ("vae_seedens_v1 ⭐",   "VAE seed-ensemble (lat16)",     0.592, 0.010, 0.508, 0.005),
        ("vae_seedens_deep",    "VAE seed-ensemble (deep)",      0.585, 0.007, 0.500, 0.003),
        ("vae_v4 (single)",     "VAE recon_prob single",         0.559, 0.035, 0.479, 0.070),
        ("vae_v15 (single)",    "VAE recon_prob deep single",    0.553, 0.036, 0.489, 0.060),
        ("iforest_v2",          "Isolation Forest (mf=0.3)",     0.511, 0.031, 0.492, 0.030),
    ], columns=["model", "desc", "soja_pr", "soja_std", "maiz_pr", "maiz_std"])


def tbl_baselines() -> pd.DataFrame:
    """Modelos base + el hallazgo del score (recon_prob vs MSE/max/topk). Soja."""
    return pd.DataFrame([
        ("IForest (baseline tuneado)",       0.511, "no neuronal, ensamble estable"),
        ("VAE recon_prob (single)",          0.559, "score probabilístico (An & Cho)"),
        ("AE seed-ensemble (MSE)",           0.339, "MSE plano: dramáticamente peor"),
        ("DAE seed-ensemble (MSE)",          0.413, "denoising ayuda algo vs AE"),
        ("AE score=max",                     0.329, "max error por feature: no ayuda"),
    ], columns=["modelo", "soja_pr", "nota"])


def tbl_ensembles() -> pd.DataFrame:
    """El ensemble de semillas resuelve la varianza. Soja."""
    return pd.DataFrame([
        ("VAE single (vae_v4)",          0.559, 0.035, "1 modelo: alta varianza"),
        ("VAE seed-ensemble ×10",        0.592, 0.010, "promedia 10 seeds: var ÷3, media ↑"),
        ("Hetero VAE+IForest",           0.427, 0.025, "el IForest más débil arrastra"),
        ("AE seed-ensemble ×10",         0.339, 0.003, "ensemble no salva base débil"),
    ], columns=["modelo", "soja_pr", "soja_std", "nota"])


def tbl_hybrids() -> pd.DataFrame:
    """Híbridos y scoring alternativo (todos por debajo del base). Soja, panel union."""
    return pd.DataFrame([
        ("VAE recon_prob (referencia)",     0.435, "el mejor single en su momento"),
        ("AE-latente + IForest (híbrido)",  0.307, "el latente del AE destruye la señal"),
        ("AE score=max",                    0.329, "max error por feature"),
        ("AE score=topk5",                  0.303, "suma top-5 errores"),
    ], columns=["modelo", "soja_pr", "nota"])


def tbl_features() -> pd.DataFrame:
    """Features nuevas: efecto sobre el VAE (SOTA) y el IForest. Soja, test sets emparejados."""
    return pd.DataFrame([
        ("base (limpio, sin features extra)", 0.592, 0.511, "VAE / IForest"),
        ("+ NDVI-AVHRR (1981+)",              0.561, 0.503, "redundante con clima"),
        ("+ agronómicas (ventana crítica)",   0.422, 0.421, "ayuda IForest, perjudica VAE"),
        ("+ ERA5 (suelo + heladas)",          0.580, 0.542, "ayuda IForest, perjudica VAE"),
    ], columns=["features", "VAE_seedens", "IForest", "nota"])


def tbl_modern() -> pd.DataFrame:
    """Comparación con métodos modernos de AD tabular profundo (deepod)."""
    return pd.DataFrame([
        ("VAE seed-ensemble (nuestro)", 0.592, 0.508, "SOTA"),
        ("IForest (baseline)",          0.511, 0.492, "—"),
        ("DeepSVDD",                    0.432, 0.426, "deep one-class"),
        ("ICL",                         0.332, 0.342, "internal contrastive learning"),
        ("NeuTraL",                     0.240, 0.234, "transformaciones; bajo azar"),
        ("GOAD",                        np.nan, np.nan, "impracticablemente lento"),
    ], columns=["modelo", "soja_pr", "maiz_pr", "nota"])
