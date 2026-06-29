"""Análisis de errores cross-modelo — Componente A.

Alinea las muestras de test de varios runs y encuentra en qué (departamento,
campaña) se equivocan TODOS los modelos. Escribe artefactos en analysis/.

Uso:
    python analyze_errors.py --cultivo soja
    python analyze_errors.py --cultivo maiz --out analysis/
    python analyze_errors.py --cultivo soja --runs RUN_ID_1 RUN_ID_2 ...

Por defecto compara el mejor run (por test_pr_auc) de cada model_type.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from componente_a.config import CULTIVOS, RUNS_DIR, ExperimentConfig
from componente_a import data as cdata
from componente_a import error_analysis as ea
from componente_a.store import get_store


def _md_table(df: pd.DataFrame) -> str:
    """Tabla markdown sin depender de `tabulate`. Redondea floats a 3 dec."""
    d = df.copy()
    for col in d.select_dtypes("float").columns:
        d[col] = d[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    cols = list(d.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _heatmap(consensus: pd.DataFrame, cultivo: str, path: str) -> None:
    """depto × campaña coloreado por n_wrong (cuántos modelos se equivocan)."""
    piv = consensus.pivot_table(index="departamento", columns="campania",
                                values="n_wrong", aggfunc="max")
    # ordena deptos por total de errores (los más conflictivos arriba)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * piv.shape[1]),
                                    max(4, 0.22 * piv.shape[0])))
    im = ax.imshow(piv.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_title(f"Errores por consenso ({cultivo}, test) — n modelos que fallan")
    fig.colorbar(im, ax=ax, label="n_wrong")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Análisis de errores cross-modelo")
    ap.add_argument("--cultivo", choices=CULTIVOS, required=True)
    ap.add_argument("--runs", nargs="*", help="run_ids explícitos (default: mejor de cada model_type)")
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    store = get_store("local", runs_dir=args.runs_dir)

    # --- Selección de runs ---
    if args.runs:
        run_ids = args.runs
        labels = {r: r for r in run_ids}
    else:
        best = ea.select_best_runs(args.runs_dir, args.cultivo)
        run_ids = [rid for _, rid in best]
        labels = {rid: mt for mt, rid in best}
    if not run_ids:
        ap.error(f"No hay runs para cultivo={args.cultivo} en {args.runs_dir}")

    print(f"cultivo={args.cultivo}  modelos comparados ({len(run_ids)}):")
    for rid in run_ids:
        print(f"  [{labels[rid]:<16}] {rid}")

    # --- Cargar scores de cada run ---
    scores_by_run: Dict[str, pd.DataFrame] = {}
    for rid in run_ids:
        r = store.load_run(rid)
        scores_by_run[labels[rid]] = r.scores

    # --- Dataset para la firma climática ---
    exp_cfg = ExperimentConfig()
    panel_z, _ = cdata.prepare(exp_cfg)
    ds = cdata.build_crop_dataset(panel_z, args.cultivo, exp_cfg)

    # --- Consenso + vistas ---
    consensus = ea.build_consensus(scores_by_run, ds, split="test")
    missed = ea.missed_anomalies(consensus)
    fa = ea.false_alarms(consensus)
    stats = ea.summary_stats(consensus)

    # --- Escribir artefactos ---
    c = args.cultivo
    consensus.to_csv(os.path.join(args.out, f"consensus_{c}.csv"), index=False)
    missed.to_csv(os.path.join(args.out, f"missed_anomalies_{c}.csv"), index=False)
    fa.to_csv(os.path.join(args.out, f"false_alarms_{c}.csv"), index=False)
    _heatmap(consensus, c, os.path.join(args.out, f"consensus_heatmap_{c}.png"))

    md_path = os.path.join(args.out, f"summary_{c}.md")
    with open(md_path, "w") as f:
        f.write(f"# Análisis de errores cross-modelo — {c} (test)\n\n")
        f.write(f"Modelos comparados ({len(run_ids)}): "
                + ", ".join(f"`{labels[r]}`" for r in run_ids) + "\n\n")
        f.write("## Estadísticas de consenso\n\n")
        for k, v in stats.items():
            vs = f"{v:.3f}" if isinstance(v, float) else str(v)
            f.write(f"- **{k}**: {vs}\n")
        f.write("\n## Anomalías falladas por TODOS los modelos (top 15)\n\n")
        f.write(_md_table(missed.head(15)) + "\n")
        f.write("\n## Falsas alarmas universales (top 15)\n\n")
        f.write(_md_table(fa.head(15)) + "\n")

    # --- Resumen en consola ---
    print("\n--- consenso ---")
    for k, v in stats.items():
        vs = f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"  {k}: {vs}")
    print(f"\nartefactos en {args.out}/ (consensus_{c}.csv, missed_anomalies_{c}.csv, "
          f"false_alarms_{c}.csv, consensus_heatmap_{c}.png, summary_{c}.md)")


if __name__ == "__main__":
    main()
