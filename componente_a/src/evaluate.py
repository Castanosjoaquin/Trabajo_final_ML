"""
Evaluación estandarizada de detectores de anomalías.

Métricas: PR-AUC (principal), ROC-AUC, F1 de la clase anómala, precision@k,
recall estratificado (firma climática adversa vs no). Umbral calibrado en
validación (máximo F1) y aplicado a test. Todo en un esquema plano apto para
la app de comparación.
"""
from __future__ import annotations

from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve, f1_score)


def calibrate_threshold(scores_val: np.ndarray, y_val: np.ndarray,
                        mode: str = "contamination",
                        contamination: float = 0.10,
                        fallback_pct: float = 85.0) -> Dict:
    """Elige el umbral de decisión sobre los scores de validación.

    mode='contamination' (default): umbral = cuantil (1-contamination) de los
        scores de val → se marca aprox. la fracción `contamination` como anómala.
        Operating point ESTABLE e independiente de las pocas anomalías de val,
        evita la degeneración del max-F1 (que con val chico calibra un umbral
        tan bajo que en test marca TODO → recall=1, precision=tasa base).
    mode='f1': umbral que maximiza F1 en val (sensible al ruido de val chico;
        se mantiene solo por comparación / retrocompatibilidad).

    Devuelve threshold + diagnósticos. `calibrated` indica si pudo usar labels.
    """
    if mode == "contamination":
        thr = float(np.quantile(scores_val, 1.0 - contamination))
        return {"threshold": thr, "mode": "contamination",
                "contamination": float(contamination),
                "f1_val": np.nan, "calibrated": bool(y_val.sum() > 0)}

    # mode == "f1"
    if y_val.sum() > 0:
        prec, rec, thr = precision_recall_curve(y_val, scores_val)
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        i = int(np.argmax(f1s))
        return {"threshold": float(thr[i]), "mode": "f1",
                "f1_val": float(f1s[i]), "calibrated": True}
    return {"threshold": float(np.percentile(scores_val, fallback_pct)),
            "mode": "f1", "f1_val": np.nan, "calibrated": False}


def evaluate_split(scores: np.ndarray, y_true: np.ndarray,
                   threshold: float, contamination: float = 0.10) -> Dict:
    """Métricas de un split. NaN si no hay anómalos reales (no definidas).

    recall_at_contamination: de los top-(contamination*n) por score,
        ¿qué fracción de las anomalías reales está incluida?
        Responde directamente "¿cuántas anomalías capturo si uso el prior?"
    """
    out = {"n": int(len(y_true)), "n_anomalias": int(y_true.sum())}
    if y_true.sum() == 0:
        out.update({"pr_auc": np.nan, "roc_auc": np.nan, "f1": np.nan,
                    "precision_at_k": np.nan, "recall_at_contamination": np.nan})
        return out
    out["pr_auc"] = float(average_precision_score(y_true, scores))
    out["roc_auc"] = float(roc_auc_score(y_true, scores))
    y_pred = (scores >= threshold).astype(int)
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    k = int(y_true.sum())
    top_k = np.argsort(scores)[::-1][:k]
    out["precision_at_k"] = float(y_true[top_k].sum() / k)
    out["k"] = k
    # Recall al usar el prior de contaminación como presupuesto de alertas
    k_cont = max(1, int(len(y_true) * contamination))
    top_k_cont = np.argsort(scores)[::-1][:k_cont]
    out["recall_at_contamination"] = float(y_true[top_k_cont].sum() / y_true.sum())
    return out


def pr_curve(scores: np.ndarray, y_true: np.ndarray) -> Optional[pd.DataFrame]:
    """Puntos de la curva Precision-Recall (para graficar)."""
    if y_true.sum() == 0:
        return None
    prec, rec, _ = precision_recall_curve(y_true, scores)
    return pd.DataFrame({"recall": rec, "precision": prec})


def roc_curve_df(scores: np.ndarray, y_true: np.ndarray) -> Optional[pd.DataFrame]:
    """Puntos de la curva ROC."""
    if y_true.sum() == 0:
        return None
    fpr, tpr, _ = roc_curve(y_true, scores)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def stratified_recall(scores: np.ndarray, y_true: np.ndarray, threshold: float,
                      X_norm: np.ndarray, feature_cols: List[str],
                      precip_z_thresh: float = -0.5) -> Dict:
    """Recall separando anomalías reales según tengan o no firma climática
    adversa (precipitación normalizada media < umbral). Visibiliza el techo
    estructural: el detector solo 've' anomalías con causa climática."""
    precip_idx = [i for i, c in enumerate(feature_cols)
                  if "chirps_precip" in c or "prectotcorr" in c]
    if not precip_idx or y_true.sum() == 0:
        return {"recall_clima_adverso": np.nan, "n_clima_adverso": 0,
                "recall_otros": np.nan, "n_otros": 0}

    mean_precip_z = X_norm[:, precip_idx].mean(axis=1)
    y_pred = (scores >= threshold).astype(int)
    m_adv = (mean_precip_z < precip_z_thresh) & (y_true == 1)
    m_oth = (~(mean_precip_z < precip_z_thresh)) & (y_true == 1)

    def _recall(mask):
        return float(y_pred[mask].sum() / mask.sum()) if mask.sum() else np.nan

    return {"recall_clima_adverso": _recall(m_adv), "n_clima_adverso": int(m_adv.sum()),
            "recall_otros": _recall(m_oth), "n_otros": int(m_oth.sum())}


def hyperparam_sweep(detector_factory, X_train, scores_eval_fn,
                     y_val, n_estimators_grid, max_samples_grid) -> pd.DataFrame:
    """Barrido PR-AUC en val variando n_estimators y max_samples.

    detector_factory(n_estimators, max_samples) -> AnomalyDetector
    scores_eval_fn(detector) -> scores_val
    """
    rows = []
    for n_est in n_estimators_grid:
        for max_s in max_samples_grid:
            det = detector_factory(n_est, max_s).fit(X_train)
            scores_val = scores_eval_fn(det)
            prauc = (float(average_precision_score(y_val, scores_val))
                     if y_val.sum() > 0 else np.nan)
            rows.append({"n_estimators": n_est, "max_samples": max_s,
                         "pr_auc_val": prauc})
    return pd.DataFrame(rows)
