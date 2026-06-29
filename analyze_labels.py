"""Auditoría de calidad de la etiqueta z_rinde — Componente A.

¿Cuánto del "techo estructural" (anomalías que todos los modelos fallan) es
RUIDO DE ETIQUETA y cuánto anomalía genuina? Clasifica cada anomalía según la
fragilidad del denominador del z-score (CV del baseline, z implausible, historia
corta) y lo cruza con los aciertos/errores del mejor modelo. Si las anomalías que
el modelo falla están enriquecidas en ruido, parte del techo es FALSO → al
excluirlas, la PR-AUC sobre anomalías creíbles sube.

Read-only sobre el panel y los runs guardados. No toca el pipeline.

Uso:
    python analyze_labels.py --cultivo soja
    python analyze_labels.py --cultivo maiz --run <run_id>   # modelo específico
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from componente_a.config import CULTIVOS, RUNS_DIR, ExperimentConfig
from componente_a import data as cdata
from componente_a import error_analysis as ea
from componente_a.store import get_store

_NOISE = {"z_extremo", "cv_inestable", "historia_corta"}
_KEY = ["provincia", "departamento", "campania_inicio"]


def _top_run(runs_dir: str, cultivo: str) -> str:
    """run_id con mayor test_pr_auc (usa *_mean si está) para el cultivo."""
    best = (-np.inf, None)
    for _mt, run_id, cult, summary in ea._iter_run_meta(runs_dir):
        if cult != cultivo:
            continue
        s = summary.get("test_pr_auc_mean", summary.get("test_pr_auc"))
        if s is not None and not (isinstance(s, float) and np.isnan(s)) and float(s) > best[0]:
            best = (float(s), run_id)
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditoría de la etiqueta z_rinde")
    ap.add_argument("--cultivo", choices=CULTIVOS, required=True)
    ap.add_argument("--run", help="run_id del modelo (default: el de mayor test_pr_auc)")
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- Etiqueta + confiabilidad (recalcula z_rinde con sus componentes) ---
    cfg = ExperimentConfig()
    panel = cdata.load_panel(cfg)
    lab = ea.label_reliability(panel, cfg)
    lab = lab[lab["cultivo"] == args.cultivo]

    # --- Scores del mejor modelo (test) ---
    run_id = args.run or _top_run(args.runs_dir, args.cultivo)
    if not run_id:
        ap.error(f"No hay runs para cultivo={args.cultivo}")
    store = get_store("local", runs_dir=args.runs_dir)
    scores = store.load_run(run_id).scores
    test = scores[scores["split"] == "test"].copy()

    keys = [k for k in _KEY if k in test.columns and k in lab.columns]
    merged = test.merge(lab[keys + ["label_cat", "roll_cv", "roll_n", "z_rinde"]],
                        on=keys, how="left", suffixes=("", "_lab"))

    anom = merged[merged["anomalia"] == 1].copy()
    n_anom = len(anom)

    # --- Distribución de confiabilidad entre las anomalías de test ---
    dist = anom["label_cat"].value_counts()
    frac_ruido = anom["label_cat"].isin(_NOISE).mean() if n_anom else np.nan

    # --- ¿El modelo falla MÁS en las ruidosas? (missed = fuera del top-decil) ---
    k_cont = max(1, int(len(merged) * cfg.eval_contamination))
    thr = merged["score"].nlargest(k_cont).min()
    anom["detectada"] = anom["score"] >= thr
    by_cat = anom.groupby("label_cat")["detectada"].agg(["mean", "size"]).rename(
        columns={"mean": "recall", "size": "n"})

    # --- Re-evaluación: PR-AUC con TODAS vs excluyendo las ruidosas ---
    y = merged["anomalia"].to_numpy()
    sc = merged["score"].to_numpy()
    ap_all = average_precision_score(y, sc) if y.sum() else np.nan
    keep = ~((merged["anomalia"] == 1) & (merged["label_cat"].isin(_NOISE))).to_numpy()
    yk, sck = y[keep], sc[keep]
    ap_clean = average_precision_score(yk, sck) if yk.sum() else np.nan

    # --- Reporte ---
    c = args.cultivo
    print(f"cultivo={c}  modelo={run_id}")
    print(f"\nanomalías de test: {n_anom}")
    print("distribución por confiabilidad de etiqueta:")
    for cat, n in dist.items():
        print(f"  {cat:16s}: {n:4d} ({n/n_anom:.1%})")
    print(f"  → sospechosas de RUIDO: {frac_ruido:.1%}")
    print("\nrecall del modelo por categoría (¿falla más en las ruidosas?):")
    for cat, row in by_cat.iterrows():
        print(f"  {cat:16s}: recall={row['recall']:.2f}  (n={int(row['n'])})")
    print(f"\nPR-AUC test  (todas las anomalías): {ap_all:.4f}")
    print(f"PR-AUC test  (excluyendo ruido)   : {ap_clean:.4f}   "
          f"Δ={ap_clean-ap_all:+.4f}")
    print(f"  (se excluyeron {int((merged['anomalia']==1).sum() - yk.sum())} "
          f"anomalías sospechosas de ruido)")

    # --- Artefactos ---
    susp = anom[anom["label_cat"].isin(_NOISE)].sort_values("z_rinde")
    cols = [k for k in ["provincia", "departamento", "campania", "z_rinde",
                        "roll_cv", "roll_n", "label_cat", "score"] if k in susp.columns]
    susp[cols].to_csv(os.path.join(args.out, f"label_noise_{c}.csv"), index=False)
    print(f"\nsospechosas escritas en {args.out}/label_noise_{c}.csv")


if __name__ == "__main__":
    main()
