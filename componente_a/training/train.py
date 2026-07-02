"""Entrypoint de entrenamiento — Componente A.

Uso:
    python training/train.py configs/ae/ae_v1_1capa.yaml
    python training/train.py configs/ae/ae_v1_1capa.yaml --cultivo soja --name run_prueba
"""
from __future__ import annotations

# --- bootstrap de rutas: agrega esta carpeta y la raíz del repo al path
# (para importar `train` y `src` desde cualquier cwd / multiprocessing)
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)),
                 _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))]

import argparse
from typing import Any, Dict

import yaml

from src.config import CULTIVOS, RUNS_DIR, ExperimentConfig
from src import data as cdata
from src.models import (
    AEDetector, AEIForestDetector, DenoisingAEDetector, EnsembleDetector,
    IsolationForestDetector, VAEDetector,
)
from src.runner import run_model, run_model_multiseed
from src.store import ResultsStore


# ---------------------------------------------------------------------------
# Construcción del detector desde dict de config
# ---------------------------------------------------------------------------
def build_detector(cfg: Dict[str, Any], seed: int | None = None):
    """Construye el detector desde el dict de config. `seed` (si se pasa)
    sobreescribe random_state — lo usa el barrido multi-semilla."""
    model = cfg["model"]
    rs = cfg.get("random_state", 42) if seed is None else seed

    if model == "ensemble":
        # Construye los miembros recursivamente con el mismo build_detector.
        # - Seed-ensemble: 'base' (un config de modelo) + 'n_members' → N copias
        #   con semillas rs, rs+1, … (baja la varianza promediando scores).
        # - Hetero-ensemble: 'members' = lista de configs de modelos distintos.
        normalize = cfg.get("normalize", "zscore")
        combine = cfg.get("combine", "mean")
        if cfg.get("members"):
            members = [build_detector(mc, rs) for mc in cfg["members"]]
        elif cfg.get("base"):
            n_members = int(cfg.get("n_members", 10))
            members = [build_detector(cfg["base"], rs + i) for i in range(n_members)]
        else:
            raise ValueError(
                "ensemble necesita 'base'+'n_members' (seed-ensemble) "
                "o 'members' (hetero-ensemble)")
        return EnsembleDetector(members, normalize=normalize, combine=combine)
    if model == "deepod":
        from src.models import DeepODDetector
        return DeepODDetector(
            algo=cfg.get("algo", "icl"),
            epochs=cfg.get("epochs", 50),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            random_state=rs,
            **cfg.get("extra", {}),
        )
    if model == "iforest":
        return IsolationForestDetector(
            n_estimators=cfg.get("n_estimators", 100),
            max_samples=cfg.get("max_samples", "auto"),
            max_features=cfg.get("max_features", 1.0),
            contamination=cfg.get("contamination", "auto"),
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
            activation=cfg.get("activation", "relu"),
            lr_schedule=cfg.get("lr_schedule", None),
            max_epochs=cfg.get("max_epochs", 200),
            patience=cfg.get("patience", 15),
            batch_size=cfg.get("batch_size", 64),
            score_mode=cfg.get("score_mode", "mse"),
            top_k=cfg.get("top_k", 5),
            random_state=rs,
        )
    if model == "ae_iforest":
        return AEIForestDetector(
            hidden_dims=tuple(cfg.get("hidden_dims", [64, 32])),
            latent_dim=cfg.get("latent_dim", 8),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 0.0),
            dropout=cfg.get("dropout", 0.0),
            use_batch_norm=bool(cfg.get("use_batch_norm", False)),
            grad_clip_norm=cfg.get("grad_clip_norm", 0.0),
            activation=cfg.get("activation", "relu"),
            lr_schedule=cfg.get("lr_schedule", None),
            max_epochs=cfg.get("max_epochs", 200),
            patience=cfg.get("patience", 15),
            batch_size=cfg.get("batch_size", 64),
            n_estimators=cfg.get("n_estimators", 100),
            max_samples=cfg.get("max_samples", "auto"),
            max_features=cfg.get("max_features", 1.0),
            contamination=cfg.get("contamination", "auto"),
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
            activation=cfg.get("activation", "relu"),
            lr_schedule=cfg.get("lr_schedule", None),
            max_epochs=cfg.get("max_epochs", 200),
            patience=cfg.get("patience", 15),
            batch_size=cfg.get("batch_size", 64),
            score_mode=cfg.get("score_mode", "mse"),
            top_k=cfg.get("top_k", 5),
            random_state=rs,
        )
    if model == "vae":
        return VAEDetector(
            hidden_dims=tuple(cfg.get("hidden_dims", [64, 32])),
            latent_dim=cfg.get("latent_dim", 8),
            beta=cfg.get("beta", 1.0),
            score_mode=cfg.get("score_mode", "recon_error"),
            n_mc_samples=cfg.get("n_mc_samples", 20),
            decoder_dist=cfg.get("decoder_dist", "gaussian"),
            student_t_df=cfg.get("student_t_df", 4.0),
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
        "Opciones: iforest, ae, ae_iforest, dae, vae, ensemble, deepod"
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

    name    = cfg.get("name", f'{cfg["model"]}_run')
    cultivo = cfg.get("cultivo", "ambos")
    base_seed = cfg.get("random_state", 42)
    n_seeds = int(cfg.get("n_seeds", 5))

    from src.config import PANEL_PATH
    exp_cfg = ExperimentConfig(
        panel_path=cfg.get("panel_path", PANEL_PATH),
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

    cultivos = CULTIVOS if cultivo == "ambos" else [cultivo]
    print(f"modelo={cfg['model']}  cultivos={cultivos}  "
          f"n_seeds={n_seeds}  umbral={exp_cfg.threshold_mode}")
    print("=" * 55)

    panel_z, _ = cdata.prepare(exp_cfg)

    for c in cultivos:
        print(f"\n--- {c.upper()} ---")
        ds = cdata.build_crop_dataset(panel_z, c, exp_cfg)

        result = run_model_multiseed(
            name, lambda s: build_detector(cfg, s), ds, exp_cfg,
            n_seeds=n_seeds, base_seed=base_seed)

        path = ResultsStore(RUNS_DIR).save(result)

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
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Componente A — entrenamiento de un config YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ejemplos:
  python training/train.py configs/ae/ae_v1_1capa.yaml
  python training/train.py configs/vae/vae_v1_1capa.yaml --cultivo soja --name vae_test
        """,
    )
    ap.add_argument("config", help="YAML de configuración del modelo")
    ap.add_argument("--cultivo", choices=CULTIVOS + ["ambos"],
                    help="Override del cultivo definido en el YAML")
    ap.add_argument("--name", help="Override del nombre del run")
    cmd_train(ap.parse_args())


if __name__ == "__main__":
    main()
