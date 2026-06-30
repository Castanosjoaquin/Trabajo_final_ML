"""Corre varios configs de train en secuencia (o en paralelo) y muestra tabla resumen.

Uso:
    python batch_train.py configs/ae/*.yaml
    python batch_train.py configs/ae/ae_v1_1capa.yaml configs/ae/ae_v3_latent16.yaml
    python batch_train.py configs/ae/*.yaml configs/iforest/*.yaml
    python batch_train.py configs/ae/*.yaml --cultivo soja   # solo soja
    python batch_train.py configs/ae/*.yaml --workers 4      # 4 en paralelo
"""
from __future__ import annotations

# --- bootstrap de rutas: agrega esta carpeta y la raíz del repo al path
# (para importar `train` y `src` desde cualquier cwd / multiprocessing)
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)),
                 _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))]

import argparse
import functools
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.config import CULTIVOS, PANEL_PATH, RUNS_DIR, ExperimentConfig
from src import data as cdata
from src.runner import run_model_multiseed
from src.store import ResultsStore
from train import build_detector


# ---------------------------------------------------------------------------
# Worker — función de módulo (picklable para ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_single(task: Tuple[str, str]) -> Dict:
    """Entrena un (config_path, cultivo) y devuelve dict de métricas.

    Debe ser función de módulo (no lambda) para ser picklable.
    """
    cfg_path, cultivo = task
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    name      = cfg.get("name", Path(cfg_path).stem)
    base_seed = int(cfg.get("random_state", 42))
    n_seeds   = int(cfg.get("n_seeds", 5))

    exp_cfg = ExperimentConfig(
        panel_path=cfg.get("panel_path") or PANEL_PATH,
        use_ndvi=cfg.get("use_ndvi", False),
        use_agro_features=cfg.get("use_agro_features", False),
        use_era5_features=cfg.get("use_era5_features", False),
        train_start=cfg.get("train_start", None),
        rolling_window=cfg.get("rolling_window", 5),
        z_thresh=cfg.get("z_thresh", -1.5),
        threshold_mode=cfg.get("threshold_mode", "contamination"),
        eval_contamination=cfg.get("eval_contamination", 0.10),
        random_state=base_seed,
    )

    panel_z, _ = cdata.prepare(exp_cfg)
    store = ResultsStore(RUNS_DIR)

    ds = cdata.build_crop_dataset(panel_z, cultivo, exp_cfg)

    # functools.partial es picklable; lambda no lo es
    detector_fn = functools.partial(build_detector, cfg)

    result = run_model_multiseed(
        name, detector_fn, ds, exp_cfg,
        n_seeds=n_seeds, base_seed=base_seed,
    )
    store.save(result)

    t = result.summary
    return {
        "name":     name,
        "cultivo":  cultivo,
        "pr_auc":   t.get("test_pr_auc_mean",  t.get("test_pr_auc",  float("nan"))),
        "pr_std":   t.get("test_pr_auc_std",   0.0),
        "roc_auc":  t.get("test_roc_auc_mean", t.get("test_roc_auc", float("nan"))),
        "f1":       t.get("test_f1_mean",       t.get("test_f1",      float("nan"))),
        "recall_k": t.get("test_recall_at_contamination_mean",
                          t.get("test_recall_at_contamination", float("nan"))),
        "n_seeds":  t.get("n_seeds", 1),
    }


# ---------------------------------------------------------------------------
# Tabla resumen
# ---------------------------------------------------------------------------

def print_table(rows: List[Dict]) -> None:
    if not rows:
        return
    header = f"{'modelo':<30} {'cultivo':<8} {'PR-AUC':>8} {'±std':>6} {'ROC-AUC':>8} {'Rec@k':>7} {'F1':>7} {'seeds':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (x["name"], x["cultivo"])):
        print(f"{r['name']:<30} {r['cultivo']:<8} "
              f"{r['pr_auc']:>8.4f} {r['pr_std']:>6.4f} "
              f"{r['roc_auc']:>8.4f} {r['recall_k']:>7.4f} "
              f"{r['f1']:>7.4f} {r['n_seeds']:>6}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Batch training — corre varios configs y muestra tabla resumen")
    ap.add_argument("configs", nargs="+", help="Archivos YAML de configuración")
    ap.add_argument("--cultivo", choices=CULTIVOS + ["ambos"], default="ambos")
    ap.add_argument("--workers", type=int, default=1,
                    help="Procesos en paralelo. Cada worker entrena un (config, cultivo). "
                         "Default=1 (secuencial). Recomendado: cpu_count // 2.")
    args = ap.parse_args()

    cultivos = CULTIVOS if args.cultivo == "ambos" else [args.cultivo]

    # Armar lista de tareas: una por (config, cultivo)
    tasks: List[Tuple[str, str]] = [
        (cfg_path, c)
        for cfg_path in args.configs
        for c in cultivos
    ]

    n_tasks = len(tasks)
    print(f"\n{n_tasks} tareas ({len(args.configs)} configs × {len(cultivos)} cultivos), "
          f"workers={args.workers}")

    all_results: List[Dict] = []

    if args.workers <= 1:
        for cfg_path, cultivo in tasks:
            print(f"\n{'─'*55}")
            print(f"Config: {cfg_path}  cultivo: {cultivo}")
            print(f"{'─'*55}")
            try:
                row = _run_single((cfg_path, cultivo))
                all_results.append(row)
                print(f"  [{row['cultivo'].upper()}] PR-AUC={row['pr_auc']:.4f}±{row['pr_std']:.4f}  "
                      f"ROC-AUC={row['roc_auc']:.4f}  Rec@k={row['recall_k']:.4f}  F1={row['f1']:.4f}")
            except Exception:
                print(f"  ERROR en {cfg_path} ({cultivo}):")
                traceback.print_exc(file=sys.stdout)
    else:
        # Paralelo: los prints internos de cada worker pueden entrelazarse,
        # solo imprimimos completions desde el proceso principal.
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_to_task = {ex.submit(_run_single, t): t for t in tasks}
            done = 0
            for future in as_completed(future_to_task):
                cfg_path, cultivo = future_to_task[future]
                done += 1
                try:
                    row = future.result()
                    all_results.append(row)
                    print(f"[{done}/{n_tasks}] {row['name']} ({cultivo})  "
                          f"PR-AUC={row['pr_auc']:.4f}±{row['pr_std']:.4f}  "
                          f"ROC-AUC={row['roc_auc']:.4f}  Rec@k={row['recall_k']:.4f}")
                except Exception:
                    print(f"[{done}/{n_tasks}] ERROR en {cfg_path} ({cultivo}):")
                    traceback.print_exc(file=sys.stdout)

    print_table(all_results)


if __name__ == "__main__":
    main()
