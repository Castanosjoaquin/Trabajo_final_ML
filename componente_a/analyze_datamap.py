"""Data map de dinámica de entrenamiento del VAE — Componente A.

Aplica la idea "lo que el modelo SIEMPRE reconstruye mal es anomalía" (Dataset
Cartography, Swayamdipta et al. 2020) al set de entrenamiento: entrena un VAE
registrando el error de reconstrucción por muestra en cada época, y clasifica
cada muestra de TRAIN en easy / ambiguous / hard-to-learn.

Como el train son SOLO normales (z_rinde ≥ −1.5 por construcción), las muestras
hard-to-learn son candidatas a **anomalías residuales / ruido de etiqueta** que
se colaron en el set "normal" → auditoría de calidad del entrenamiento.

Uso:
    python analyze_datamap.py --cultivo soja
    python analyze_datamap.py --cultivo maiz --config configs/vae/vae_v4_reconprob_lat16.yaml
"""
from __future__ import annotations

import argparse
import os

import yaml

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "training"))

from src.config import CULTIVOS, PANEL_PATH, ExperimentConfig
from src import data as cdata
from src import error_analysis as ea
from train import build_detector


def main() -> None:
    ap = argparse.ArgumentParser(description="Data map de dinámica de entrenamiento del VAE")
    ap.add_argument("--cultivo", choices=CULTIVOS, required=True)
    ap.add_argument("--config", default="configs/vae/vae_v4_reconprob_lat16.yaml",
                    help="Config YAML de un VAE (model: vae). Default: el ganador vae_v4.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("model") != "vae":
        ap.error(f"--config debe ser un VAE (model: vae); es {cfg.get('model')!r}")

    # --- Dataset (mismo pipeline que el entrenamiento) ---
    exp_cfg = ExperimentConfig(
        panel_path=cfg.get("panel_path") or PANEL_PATH,
        use_ndvi=cfg.get("use_ndvi", False),
        use_agro_features=cfg.get("use_agro_features", False),
        use_era5_features=cfg.get("use_era5_features", False),
        train_start=cfg.get("train_start", None),
        random_state=args.seed,
    )
    panel_z, _ = cdata.prepare(exp_cfg)
    ds = cdata.build_crop_dataset(panel_z, args.cultivo, exp_cfg)

    # --- VAE con tracking de dinámica de entrenamiento ---
    det = build_detector(cfg, args.seed)
    det.track_datamap = True            # activa el registro por muestra/época
    print(f"cultivo={args.cultivo}  config={args.config}  "
          f"n_train_normal={len(ds.X_train)}  entrenando VAE con tracking…")
    det.fit(ds.X_train)

    if det.per_sample_error_ is None:
        ap.error("No se registró la dinámica por muestra (¿el modelo expone "
                 "per_sample_error?). Asegurate de usar un VAE.")

    # --- Data map ---
    dm = ea.training_data_map(det.per_sample_error_, ds.meta_train)
    hard = dm[dm["region"] == "hard-to-learn"].sort_values("mean_err", ascending=False)

    # --- Artefactos ---
    c = args.cultivo
    dm.to_csv(os.path.join(args.out, f"datamap_{c}.csv"), index=False)
    hard.to_csv(os.path.join(args.out, f"hard_to_learn_{c}.csv"), index=False)
    ea.plot_data_map(dm, os.path.join(args.out, f"datamap_{c}.png"), cultivo=c)

    # --- Resumen en consola ---
    counts = dm["region"].value_counts()
    print("\n--- regiones del data map (train normal) ---")
    for region in ["easy-to-learn", "medium", "ambiguous", "hard-to-learn"]:
        print(f"  {region:<15}: {int(counts.get(region, 0))}")
    print(f"\nhard-to-learn (siempre mal reconstruidas): {len(hard)} muestras normales")
    print("  z_rinde medio de hard-to-learn:", round(float(hard['z_rinde'].mean()), 3)
          if len(hard) else "n/a",
          "| vs train completo:", round(float(dm['z_rinde'].mean()), 3))
    print("\n  top 10 candidatas a anomalía/ruido en el set normal:")
    for _, r in hard.head(10).iterrows():
        print(f"    {r['departamento']:<22} {int(r['campania']) if str(r['campania']).isdigit() else r['campania']}"
              f"  z_rinde={r['z_rinde']:.2f}  mean_err={r['mean_err']:.3f}")
    print(f"\nartefactos en {args.out}/ (datamap_{c}.csv, hard_to_learn_{c}.csv, datamap_{c}.png)")


if __name__ == "__main__":
    main()
