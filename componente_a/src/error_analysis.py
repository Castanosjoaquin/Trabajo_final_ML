"""Análisis de errores cross-modelo del Componente A.

Responde: ¿en qué departamentos y campañas se equivocan los modelos, y se
equivocan TODOS en lo mismo? Si todos fallan en las mismas muestras, el problema
no está en el modelo sino en los datos / la etiqueta / las features — esas
muestras no se diferencian del resto en el espacio de features (el "techo
estructural" clima→rinde, a nivel de muestra individual).

Read-only sobre `runs/`: alinea las muestras de test de varios runs por
(departamento, campania), computa el consenso de scores/errores y lo cruza con la
firma climática del CropDataset. No toca el pipeline de entrenamiento.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import CropDataset

# Clave de alineación entre runs.
_KEY = ["departamento", "campania"]
# Fracción marcada como anómala (= eval_contamination): top-10% por score.
_TOP_DECILE = 0.90


# ---------------------------------------------------------------------------
# Selección de runs: el mejor de cada model_type para un cultivo
# ---------------------------------------------------------------------------
def _iter_run_meta(runs_dir: str):
    """Yield (model_type, run_id, cultivo, summary) leyendo los json livianos
    (sin cargar los parquet). Recorre runs_dir/<model_type>/<run_id>/."""
    if not os.path.isdir(runs_dir):
        return
    for model_type in sorted(os.listdir(runs_dir)):
        type_dir = os.path.join(runs_dir, model_type)
        if not os.path.isdir(type_dir):
            continue
        for run_id in sorted(os.listdir(type_dir)):
            d = os.path.join(type_dir, run_id)
            meta_p, summ_p = os.path.join(d, "meta.json"), os.path.join(d, "summary.json")
            if not (os.path.exists(meta_p) and os.path.exists(summ_p)):
                continue
            with open(meta_p) as f:
                meta = json.load(f)
            with open(summ_p) as f:
                summary = json.load(f)
            yield model_type, run_id, meta.get("cultivo", ""), summary


def select_best_runs(runs_dir: str, cultivo: str,
                     metric: str = "test_pr_auc") -> List[Tuple[str, str]]:
    """El mejor run (por `metric`, usando *_mean si existe) de cada model_type
    para `cultivo`. Devuelve [(model_type, run_id), ...]."""
    best: Dict[str, Tuple[float, str]] = {}
    for model_type, run_id, cult, summary in _iter_run_meta(runs_dir):
        if cult != cultivo:
            continue
        score = summary.get(f"{metric}_mean", summary.get(metric))
        if score is None or (isinstance(score, float) and np.isnan(score)):
            continue
        score = float(score)
        if model_type not in best or score > best[model_type][0]:
            best[model_type] = (score, run_id)
    return [(mt, rid) for mt, (_, rid) in sorted(best.items())]


# ---------------------------------------------------------------------------
# Firma climática por muestra (reusa la lógica de stratified_recall)
# ---------------------------------------------------------------------------
def climate_signature(dataset: CropDataset) -> np.ndarray:
    """precip_z medio (features de precipitación normalizadas) por muestra de
    test, en el ORDEN de meta_test. precip_z bajo = firma climática adversa
    (sequía)."""
    precip_idx = [i for i, c in enumerate(dataset.feature_cols)
                  if "chirps_precip" in c or "prectotcorr" in c]
    if not precip_idx:
        return np.full(len(dataset.X_test), np.nan)
    return dataset.X_test[:, precip_idx].mean(axis=1)


# ---------------------------------------------------------------------------
# Consenso de errores (alineación POSICIONAL)
# ---------------------------------------------------------------------------
def build_consensus(scores_by_run: Dict[str, pd.DataFrame],
                    dataset: CropDataset, split: str = "test") -> pd.DataFrame:
    """Alinea las muestras de `split` de varios runs y arma la tabla de consenso.

    Alineación POSICIONAL (la fila i es la misma muestra en todos los runs,
    porque todos salen del mismo build_crop_dataset). NO se usa (departamento,
    campania) como clave porque los nombres de depto se repiten entre provincias.
    Valida que cada run tenga el mismo orden que `dataset.meta_test`; descarta con
    aviso los que no coincidan.

    Por muestra: mean_pct/std_pct del rank-percentil (desacuerdo entre modelos),
    n_flag (cuántos la ponen en el top-10%), n_wrong (cuántos y_pred≠anomalia),
    flags de consenso, y firma climática."""
    canon = dataset.meta_test[_KEY].reset_index(drop=True)
    y = dataset.y_test.astype(int)
    n = len(canon)

    pct_mat, flag_mat, wrong_mat, used = [], [], [], []
    for label, df in scores_by_run.items():
        d = df[df["split"] == split].reset_index(drop=True)
        if len(d) != n or not (d[_KEY].reset_index(drop=True).values == canon.values).all():
            print(f"  [aviso] '{label}' no alinea con el dataset (orden/filas distintos) "
                  f"→ descartado")
            continue
        pct = d["score"].rank(pct=True).to_numpy()                 # ∈ (0,1]
        pct_mat.append(pct)
        flag_mat.append((pct >= _TOP_DECILE).astype(int))          # marcada anómala
        wrong_mat.append((d["y_pred"].astype(int).to_numpy() != y).astype(int))
        used.append(label)

    if not used:
        return pd.DataFrame()

    P = np.vstack(pct_mat)                 # (n_models, n)
    n_models = len(used)

    out = dataset.meta_test[_KEY].reset_index(drop=True).copy()
    out["z_rinde"] = dataset.meta_test["z_rinde"].to_numpy()
    out["anomalia"] = y
    out["n_models"] = n_models
    out["mean_pct"] = P.mean(axis=0)
    out["std_pct"] = P.std(axis=0)                       # desacuerdo entre modelos
    out["n_flag"] = np.vstack(flag_mat).sum(axis=0)      # cuántos la marcan anómala
    out["n_wrong"] = np.vstack(wrong_mat).sum(axis=0)
    out["precip_z"] = climate_signature(dataset)

    out["consensus_miss"] = ((out["anomalia"] == 1) & (out["n_flag"] == 0)).astype(int)
    out["consensus_fp"] = ((out["anomalia"] == 0) & (out["n_flag"] == n_models)).astype(int)

    out.attrs["n_models"] = n_models
    out.attrs["models"] = used
    return out


# ---------------------------------------------------------------------------
# Vistas derivadas
# ---------------------------------------------------------------------------
def missed_anomalies(consensus: pd.DataFrame) -> pd.DataFrame:
    """Anomalías reales ordenadas por cuán universalmente se fallan (menor
    mean_pct = todos las scorean normal). Las de arriba con precip_z alto son las
    estructuralmente invisibles (sin firma climática)."""
    m = consensus[consensus["anomalia"] == 1].copy()
    cols = [c for c in ["departamento", "campania", "z_rinde", "precip_z",
                        "mean_pct", "std_pct", "n_flag", "n_models"] if c in m.columns]
    return m.sort_values("mean_pct")[cols].reset_index(drop=True)


def false_alarms(consensus: pd.DataFrame) -> pd.DataFrame:
    """Normales ordenadas por cuán universalmente se marcan anómalas
    (mayor mean_pct)."""
    f = consensus[consensus["anomalia"] == 0].copy()
    cols = [c for c in ["departamento", "campania", "z_rinde", "precip_z",
                        "mean_pct", "std_pct", "n_flag", "n_models"] if c in f.columns]
    return f.sort_values("mean_pct", ascending=False)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Auditoría de calidad de la etiqueta (z_rinde)
# ---------------------------------------------------------------------------
# Umbrales (tuneables) para marcar una anomalía como sospechosa de ARTEFACTO:
_CV_MIN = 0.07         # coef. de variación del baseline < esto → roll_std diminuto
_Z_IMPLAUSIBLE = -5.0  # z_rinde más negativo que esto → físicamente improbable
_N_MIN = 3             # baseline armado con <= esto campañas previas → endeble


def label_reliability(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Recalcula z_rinde reteniendo los componentes del rolling (roll_mean,
    roll_std, n, CV) y clasifica cada ANOMALÍA según cuán confiable es su
    etiqueta. Read-only; no toca el pipeline.

    La etiqueta z_rinde = (rinde − roll_mean) / roll_std puede ser un ARTEFACTO
    cuando el denominador es frágil: si un depto tuvo rindes casi idénticos
    (roll_std diminuto, CV bajo), cualquier desvío chico explota el z (de ahí los
    z=−13.7). Categorías para anomalia==1:
      - 'z_extremo'     : z_rinde < −5 (caída de rinde implausible).
      - 'cv_inestable'  : CV del baseline < _CV_MIN (denominador diminuto).
      - 'historia_corta': baseline con <= _N_MIN campañas.
      - 'genuina'       : ninguna de las anteriores → anomalía creíble.
    """
    geo = ["provincia", "departamento"] if "provincia" in panel.columns else ["departamento"]
    df = panel.sort_values(geo + ["cultivo", "campania_inicio"]).copy()
    grp = df.groupby(geo + ["cultivo"])["rinde_kgha"]
    roll_mean = grp.transform(lambda s: s.shift(1).rolling(cfg.rolling_window, min_periods=3).mean())
    roll_std = grp.transform(lambda s: s.shift(1).rolling(cfg.rolling_window, min_periods=3).std())
    roll_n = grp.transform(lambda s: s.shift(1).rolling(cfg.rolling_window, min_periods=3).count())

    df["z_rinde"] = (df["rinde_kgha"] - roll_mean) / roll_std.replace(0, np.nan)
    df["anomalia"] = (df["z_rinde"] < cfg.z_thresh).astype(int)
    df["roll_std"] = roll_std
    df["roll_cv"] = roll_std / roll_mean
    df["roll_n"] = roll_n

    def categorize(r) -> str:
        if r["anomalia"] != 1 or pd.isna(r["z_rinde"]):
            return ""
        if r["z_rinde"] < _Z_IMPLAUSIBLE:
            return "z_extremo"
        if pd.notna(r["roll_cv"]) and r["roll_cv"] < _CV_MIN:
            return "cv_inestable"
        if pd.notna(r["roll_n"]) and r["roll_n"] <= _N_MIN:
            return "historia_corta"
        return "genuina"

    df["label_cat"] = df.apply(categorize, axis=1)
    return df


def training_data_map(per_sample_err: np.ndarray,
                      meta_train: pd.DataFrame) -> pd.DataFrame:
    """Data map de dinámica de entrenamiento (Dataset Cartography, Swayamdipta
    et al. 2020) aplicado al VAE: a partir del error de reconstrucción por
    muestra a lo largo de las épocas (`per_sample_err`, shape n_epochs × n_train),
    deriva por muestra de TRAIN:

    - `mean_err`: error medio entre épocas (= 1/confidence; alto = el modelo
      nunca la reconstruye bien).
    - `var_err`: desvío del error entre épocas (= variability; alto = inestable).

    Tres regiones (análogas a la cartografía, con error en vez de probabilidad):
    - **hard-to-learn**: error alto + baja variabilidad → "el modelo NUNCA la
      aprende". Como el train son SOLO normales (z_rinde≥−1.5), estas son
      candidatas a **anomalías residuales / ruido de etiqueta** en el set normal.
    - **ambiguous**: alta variabilidad → el modelo oscila.
    - **easy-to-learn**: error bajo → bien reconstruidas.

    Es la traducción directa de la idea 'lo que siempre se predice mal es
    anomalía' al VAE, sobre el set de entrenamiento (auditoría de calidad)."""
    mean_err = per_sample_err.mean(axis=0)
    var_err = per_sample_err.std(axis=0)
    df = meta_train[["departamento", "campania", "z_rinde", "anomalia"]].reset_index(drop=True).copy()
    df["mean_err"] = mean_err
    df["var_err"] = var_err

    hi_var = var_err >= np.quantile(var_err, 0.66)
    hi_err = mean_err >= np.quantile(mean_err, 0.66)
    lo_err = mean_err <= np.quantile(mean_err, 0.33)
    df["region"] = np.where(hi_var, "ambiguous",
                    np.where(hi_err, "hard-to-learn",
                    np.where(lo_err, "easy-to-learn", "medium")))
    return df


def plot_data_map(data_map: pd.DataFrame, path: str, cultivo: str = "") -> None:
    """Scatter variability (x) vs mean_err (y), coloreado por región. Guarda PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"easy-to-learn": "#2ca02c", "medium": "#7f7f7f",
              "ambiguous": "#ff7f0e", "hard-to-learn": "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for region, c in colors.items():
        sub = data_map[data_map["region"] == region]
        ax.scatter(sub["var_err"], sub["mean_err"], s=14, alpha=0.6,
                   color=c, label=f"{region} (n={len(sub)})")
    ax.set_xlabel("variability (desvío del error entre épocas)")
    ax.set_ylabel("mean_err (error de reconstrucción medio)")
    ax.set_title(f"Data map — dinámica de entrenamiento VAE ({cultivo})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_stats(consensus: pd.DataFrame, precip_z_thresh: float = -0.5) -> Dict:
    """Estadísticas para el techo estructural a nivel muestra."""
    anom = consensus[consensus["anomalia"] == 1]
    missed = anom[anom["consensus_miss"] == 1]
    s = {
        "n_test": int(len(consensus)),
        "n_anomalias": int(len(anom)),
        "n_missed_por_todos": int(len(missed)),
        "frac_anomalias_missed": float(len(missed) / len(anom)) if len(anom) else np.nan,
        "n_false_alarms_por_todos": int(consensus["consensus_fp"].sum()),
    }
    if "precip_z" in consensus.columns and len(missed):
        sin_firma = (missed["precip_z"] >= precip_z_thresh).sum()
        s["frac_missed_sin_firma_climatica"] = float(sin_firma / len(missed))
        if len(anom):
            detect = anom[anom["consensus_miss"] == 0]
            s["precip_z_medio_missed"] = float(missed["precip_z"].mean())
            s["precip_z_medio_detectadas"] = float(detect["precip_z"].mean()) if len(detect) else np.nan
    return s
