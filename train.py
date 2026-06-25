"""Entrypoint único de entrenamiento — Componente A.

Uso básico:
    python train.py configs/ae.yaml
    python train.py configs/ae.yaml --cultivo soja --name run_prueba
    python train.py configs/vae.yaml --backend local   # override backend

W&B Sweeps (búsqueda de hiperparámetros):
    wandb sweep sweeps/ae.yaml          # crea sweep → imprime SWEEP_ID
    python train.py --sweep SWEEP_ID --cultivo soja --count 20
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
import yaml

from componente_a.config import (
    CULTIVOS, RUNS_DIR, WANDB_ENTITY, WANDB_PROJECT, ExperimentConfig,
)
from componente_a import data as cdata
from componente_a.models import (
    AEDetector, DenoisingAEDetector, IsolationForestDetector,
    PCAReconDetector, VAEDetector,
)
from componente_a.runner import run_model
from componente_a.store import get_store


_IFOREST_SWEEP_GRIDS = {
    "n_estimators": [100, 200],
    "max_samples": [64, 128, 256],
}


# ---------------------------------------------------------------------------
# Construcción del detector desde dict de config
# ---------------------------------------------------------------------------
def build_detector(cfg: Dict[str, Any]):
    model = cfg["model"]
    rs = cfg.get("random_state", 42)

    if model == "iforest":
        return IsolationForestDetector(
            n_estimators=cfg.get("n_estimators", 100),
            max_samples=cfg.get("max_samples", "auto"),
            max_features=cfg.get("max_features", 1.0),
            contamination=cfg.get("contamination", "auto"),
            random_state=rs,
        )
    if model == "pca_recon":
        return PCAReconDetector(
            n_components=cfg.get("n_components", 0.95),
            random_state=rs,
        )
    if model == "ae":
        return AEDetector(
            hidden_dims=tuple(cfg.get("hidden_dims", [64, 32])),
            latent_dim=cfg.get("latent_dim", 8),
            lr=cfg.get("lr", 1e-3),
            max_epochs=cfg.get("max_epochs", 200),
            patience=cfg.get("patience", 15),
            batch_size=cfg.get("batch_size", 64),
            random_state=rs,
        )
    if model == "dae":
        return DenoisingAEDetector(
            hidden_dims=tuple(cfg.get("hidden_dims", [64, 32])),
            latent_dim=cfg.get("latent_dim", 8),
            corruption=cfg.get("corruption", 0.1),
            noise_type=cfg.get("noise_type", "salt_pepper"),
            lr=cfg.get("lr", 1e-3),
            max_epochs=cfg.get("max_epochs", 200),
            patience=cfg.get("patience", 15),
            batch_size=cfg.get("batch_size", 64),
            random_state=rs,
        )
    if model == "vae":
        return VAEDetector(
            hidden_dims=tuple(cfg.get("hidden_dims", [64, 32])),
            latent_dim=cfg.get("latent_dim", 8),
            beta=cfg.get("beta", 1.0),
            score_mode=cfg.get("score_mode", "recon_error"),
            n_mc_samples=cfg.get("n_mc_samples", 20),
            lr=cfg.get("lr", 1e-3),
            max_epochs=cfg.get("max_epochs", 200),
            patience=cfg.get("patience", 15),
            batch_size=cfg.get("batch_size", 64),
            random_state=rs,
        )
    raise ValueError(
        f"Modelo desconocido: {model!r}. "
        "Opciones: iforest, pca_recon, ae, dae, vae"
    )


# ---------------------------------------------------------------------------
# Comando: entrenamiento normal desde YAML
# ---------------------------------------------------------------------------
def cmd_train(args) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Overrides de CLI (tienen prioridad sobre el YAML)
    if args.name:    cfg["name"] = args.name
    if args.cultivo: cfg["cultivo"] = args.cultivo
    if args.backend: cfg["backend"] = args.backend

    name    = cfg.get("name", f'{cfg["model"]}_run')
    cultivo = cfg.get("cultivo", "ambos")
    backend = cfg.get("backend", "wandb")

    exp_cfg = ExperimentConfig(
        use_ndvi=cfg.get("use_ndvi", False),
        rolling_window=cfg.get("rolling_window", 5),
        z_thresh=cfg.get("z_thresh", -1.5),
        random_state=cfg.get("random_state", 42),
    )

    cultivos = CULTIVOS if cultivo == "ambos" else [cultivo]
    print(f"modelo={cfg['model']}  cultivos={cultivos}  backend={backend}")
    print("=" * 55)

    panel_z, _ = cdata.prepare(exp_cfg)

    for c in cultivos:
        print(f"\n--- {c.upper()} ---")
        ds = cdata.build_crop_dataset(panel_z, c, exp_cfg)

        detector = build_detector(cfg)

        sweep_factory = sweep_grids = None
        if cfg["model"] == "iforest":
            sweep_factory = lambda n, m, _rs=exp_cfg.random_state: \
                IsolationForestDetector(n_estimators=n, max_samples=m, random_state=_rs)
            sweep_grids = _IFOREST_SWEEP_GRIDS

        result = run_model(name, detector, ds, exp_cfg,
                           sweep_factory=sweep_factory, sweep_grids=sweep_grids)

        store = get_store(backend, runs_dir=RUNS_DIR,
                          project=WANDB_PROJECT, entity=WANDB_ENTITY)
        path = store.save(result)

        t = result.summary
        print(f"  val  PR-AUC={t.get('val_pr_auc', 0):.3f}  "
              f"ROC-AUC={t.get('val_roc_auc', 0):.3f}")
        print(f"  test PR-AUC={t.get('test_pr_auc', 0):.3f}  "
              f"ROC-AUC={t.get('test_roc_auc', 0):.3f}  "
              f"F1={t.get('test_f1', 0):.3f}")
        print(f"  guardado: {path if isinstance(path, str) else result.run_id}")


# ---------------------------------------------------------------------------
# Comando: agente de W&B Sweep
# ---------------------------------------------------------------------------
def cmd_sweep(args) -> None:
    import wandb

    sweep_id = args.sweep
    cultivo_arg = args.cultivo or "soja"

    def sweep_fn():
        with wandb.init() as run:
            wc = wandb.config
            model_type = wc.get("model_type", "ae")
            n_seeds    = int(wc.get("n_seeds", 3))

            exp_cfg = ExperimentConfig()
            panel_z, _ = cdata.prepare(exp_cfg)
            ds = cdata.build_crop_dataset(panel_z, cultivo_arg, exp_cfg)

            hidden_dims = tuple(int(d) for d in wc.get("hidden_dims", [64, 32]))
            common = dict(
                hidden_dims=hidden_dims,
                latent_dim=int(wc.get("latent_dim", 8)),
                lr=float(wc.get("lr", 1e-3)),
                max_epochs=int(wc.get("max_epochs", 200)),
                patience=int(wc.get("patience", 15)),
                batch_size=int(wc.get("batch_size", 64)),
            )

            val_pr_aucs = []
            for i in range(n_seeds):
                if model_type == "ae":
                    det = AEDetector(**common, random_state=42 + i)
                elif model_type in ("dae", "denoising_ae"):
                    det = DenoisingAEDetector(
                        **common,
                        corruption=float(wc.get("corruption", 0.1)),
                        noise_type=wc.get("noise_type", "salt_pepper"),
                        random_state=42 + i,
                    )
                elif model_type == "vae":
                    det = VAEDetector(
                        **common,
                        beta=float(wc.get("beta", 1.0)),
                        score_mode=wc.get("score_mode", "recon_error"),
                        n_mc_samples=int(wc.get("n_mc_samples", 20)),
                        random_state=42 + i,
                    )
                else:
                    raise ValueError(f"model_type desconocido: {model_type!r}")

                result = run_model(
                    f"{model_type}_sweep_s{i}", det, ds, exp_cfg, emb_methods=[]
                )
                val_pr_aucs.append(float(result.summary.get("val_pr_auc", 0.0)))

            run.summary["val_pr_auc_mean"] = float(np.mean(val_pr_aucs))
            run.summary["val_pr_auc_std"]  = float(np.std(val_pr_aucs))
            run.summary["cultivo"] = cultivo_arg

    entity = WANDB_ENTITY
    project = WANDB_PROJECT
    path = f"{entity + '/' if entity else ''}{project}/{sweep_id}"
    wandb.agent(path, function=sweep_fn, count=args.count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Componente A — entrenamiento y W&B sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ejemplos:
  python train.py configs/ae.yaml
  python train.py configs/ae.yaml --cultivo soja --name ae_test
  python train.py configs/vae.yaml --backend local

  wandb sweep sweeps/ae.yaml              # crea sweep → imprime SWEEP_ID
  python train.py --sweep SWEEP_ID --cultivo soja --count 20
        """,
    )
    ap.add_argument("config", nargs="?", help="YAML de configuración del modelo")
    ap.add_argument("--sweep", metavar="SWEEP_ID",
                    help="Lanzar agente de W&B Sweep con este ID")
    ap.add_argument("--cultivo", choices=CULTIVOS + ["ambos"],
                    help="Override del cultivo definido en el YAML")
    ap.add_argument("--name",    help="Override del nombre del run")
    ap.add_argument("--backend", choices=["local", "wandb"],
                    help="Override del backend (local | wandb)")
    ap.add_argument("--count", type=int, default=20,
                    help="Número máximo de runs para el sweep agent (default: 20)")

    args = ap.parse_args()

    if args.sweep:
        cmd_sweep(args)
    elif args.config:
        cmd_train(args)
    else:
        ap.error("Se necesita un config YAML o --sweep SWEEP_ID.\n"
                 "Ejemplo: python train.py configs/ae.yaml")


if __name__ == "__main__":
    main()
