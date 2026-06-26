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
from componente_a.runner import run_model, run_model_multiseed
from componente_a.store import get_store


_IFOREST_SWEEP_GRIDS = {
    "n_estimators": [100, 200],
    "max_samples": [64, 128, 256],
}


# ---------------------------------------------------------------------------
# Construcción del detector desde dict de config
# ---------------------------------------------------------------------------
def build_detector(cfg: Dict[str, Any], seed: int | None = None):
    """Construye el detector desde el dict de config. `seed` (si se pasa)
    sobreescribe random_state — lo usa el barrido multi-semilla."""
    model = cfg["model"]
    rs = cfg.get("random_state", 42) if seed is None else seed

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
            weight_decay=cfg.get("weight_decay", 0.0),
            dropout=cfg.get("dropout", 0.0),
            use_batch_norm=bool(cfg.get("use_batch_norm", False)),
            grad_clip_norm=cfg.get("grad_clip_norm", 0.0),
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
            weight_decay=cfg.get("weight_decay", 0.0),
            dropout=cfg.get("dropout", 0.0),
            use_batch_norm=bool(cfg.get("use_batch_norm", False)),
            grad_clip_norm=cfg.get("grad_clip_norm", 0.0),
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
            weight_decay=cfg.get("weight_decay", 0.0),
            dropout=cfg.get("dropout", 0.0),
            use_batch_norm=bool(cfg.get("use_batch_norm", False)),
            grad_clip_norm=cfg.get("grad_clip_norm", 0.0),
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
    base_seed = cfg.get("random_state", 42)
    n_seeds = int(cfg.get("n_seeds", 5))

    from componente_a.config import PANEL_PATH
    exp_cfg = ExperimentConfig(
        panel_path=cfg.get("panel_path", PANEL_PATH),
        use_ndvi=cfg.get("use_ndvi", False),
        rolling_window=cfg.get("rolling_window", 5),
        z_thresh=cfg.get("z_thresh", -1.5),
        threshold_mode=cfg.get("threshold_mode", "contamination"),
        eval_contamination=cfg.get("eval_contamination", 0.10),
        random_state=base_seed,
    )

    cultivos = CULTIVOS if cultivo == "ambos" else [cultivo]
    print(f"modelo={cfg['model']}  cultivos={cultivos}  backend={backend}  "
          f"n_seeds={n_seeds}  umbral={exp_cfg.threshold_mode}")
    print("=" * 55)

    panel_z, _ = cdata.prepare(exp_cfg)

    for c in cultivos:
        print(f"\n--- {c.upper()} ---")
        ds = cdata.build_crop_dataset(panel_z, c, exp_cfg)

        sweep_factory = sweep_grids = None
        if cfg["model"] == "iforest":
            sweep_factory = lambda n, m, _rs=base_seed: \
                IsolationForestDetector(n_estimators=n, max_samples=m, random_state=_rs)
            sweep_grids = _IFOREST_SWEEP_GRIDS

        result = run_model_multiseed(
            name, lambda s: build_detector(cfg, s), ds, exp_cfg,
            n_seeds=n_seeds, base_seed=base_seed,
            sweep_factory=sweep_factory, sweep_grids=sweep_grids)

        store = get_store(backend, runs_dir=RUNS_DIR,
                          project=WANDB_PROJECT, entity=WANDB_ENTITY)
        path = store.save(result)

        t = result.summary
        ns = t.get("n_seeds", 1)
        if ns > 1:
            print(f"  [{ns} seeds] test PR-AUC={t.get('test_pr_auc_mean', 0):.3f}"
                  f"±{t.get('test_pr_auc_std', 0):.3f}  "
                  f"ROC-AUC={t.get('test_roc_auc_mean', 0):.3f}"
                  f"±{t.get('test_roc_auc_std', 0):.3f}  "
                  f"F1={t.get('test_f1_mean', 0):.3f}±{t.get('test_f1_std', 0):.3f}")
        else:
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

            hidden_dims = tuple(int(d) for d in wc.get("hidden_dims", [32]))
            common = dict(
                hidden_dims=hidden_dims,
                latent_dim=int(wc.get("latent_dim", 8)),
                lr=float(wc.get("lr", 1e-3)),
                weight_decay=float(wc.get("weight_decay", 0.0)),
                dropout=float(wc.get("dropout", 0.0)),
                use_batch_norm=bool(wc.get("use_batch_norm", False)),
                grad_clip_norm=float(wc.get("grad_clip_norm", 0.0)),
                max_epochs=int(wc.get("max_epochs", 300)),
                patience=int(wc.get("patience", 20)),
                batch_size=int(wc.get("batch_size", 64)),
            )

            val_losses = []
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

                run_model(f"{model_type}_sweep_s{i}", det, ds, exp_cfg, emb_methods=[])
                # val_loss mínima = mejor época alcanzada por early stopping
                best_val_loss = min(h["val_loss"] for h in det.history_)
                val_losses.append(best_val_loss)

            run.summary["val_loss_mean"] = float(np.mean(val_losses))
            run.summary["val_loss_std"]  = float(np.std(val_losses))
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
