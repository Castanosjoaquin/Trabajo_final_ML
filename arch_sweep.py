"""Barrido de arquitecturas — Componente A.

Corre IForest (baseline) + varias arquitecturas de AE y VAE, cada una con
evaluación robusta (multi-seed + umbral por contaminación), y guarda cada
corrida en el ResultsStore para compararlas después en el visualizador.

Uso:
    python arch_sweep.py                        # local, 5 seeds, ambos cultivos
    python arch_sweep.py --backend wandb --n-seeds 10
    python arch_sweep.py --skip-existing        # solo corre lo que aún no está

Nomenclatura de runs:
    ae_61-32-8        → hidden=(32,), latent=8, sin dropout
    ae_61-32-8_drop   → ídem con dropout=0.1
    ae_61-32-16-8     → hidden=(32,16), latent=8
"""
from __future__ import annotations

import argparse

from componente_a.config import (CULTIVOS, RUNS_DIR, WANDB_ENTITY,
                                  WANDB_PROJECT, ExperimentConfig)
from componente_a import data as cdata
from componente_a.runner import run_model_multiseed
from componente_a.store import get_store
from componente_a.models import (AEDetector, IsolationForestDetector,
                                  VAEDetector)

# ---------------------------------------------------------------------------
# Grilla de arquitecturas: (hidden_dims, latent_dim, dropout)
# ---------------------------------------------------------------------------
AE_ARCHS = [
    ((),        8,  0.0),   # ae_61-8        1 capa, sin ocultas
    ((),       16,  0.0),   # ae_61-16       1 capa, latente grande
    ((32,),     8,  0.0),   # ae_61-32-8     ← mejor media del 1er barrido
    ((64, 32),  8,  0.0),   # ae_61-64-32-8  profunda con expansión
    ((16,),     4,  0.0),   # ae_61-16-4     bottleneck chico
    # Nueva arquitectura recomendada por el profe
    ((32, 16),  8,  0.0),   # ae_61-32-16-8  dos ocultas, monótona
    # Variantes con dropout (las más prometedoras)
    ((32,),     8,  0.1),   # ae_61-32-8_drop
    ((64, 32),  8,  0.1),   # ae_61-64-32-8_drop
    ((32, 16),  8,  0.1),   # ae_61-32-16-8_drop
]

VAE_ARCHS = [
    ((),        8,  0.0),   # vae_61-8       1 capa
    ((16,),     8,  0.0),   # vae_61-16-8
    ((32,),     8,  0.0),   # vae_61-32-8    ← mejor media del 1er barrido
    ((64, 32),  8,  0.0),   # vae_61-64-32-8 profunda
    # Nueva
    ((32, 16),  8,  0.0),   # vae_61-32-16-8
    # Variantes con dropout
    ((32,),     8,  0.1),   # vae_61-32-8_drop
    ((64, 32),  8,  0.1),   # vae_61-64-32-8_drop
    ((32, 16),  8,  0.1),   # vae_61-32-16-8_drop
]


def _label(prefix: str, hidden: tuple, latent: int, dropout: float) -> str:
    arch = "-".join(["61"] + [str(h) for h in hidden] + [str(latent)])
    return f"{prefix}_{arch}" + ("_drop" if dropout > 0 else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Barrido de arquitecturas AE/VAE")
    ap.add_argument("--backend", choices=["local", "wandb"], default="local")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--cultivo", choices=CULTIVOS + ["ambos"], default="ambos")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Omite runs cuyo model_name ya está guardado en el store")
    args = ap.parse_args()

    exp = ExperimentConfig()
    panel_z, _ = cdata.prepare(exp)
    store = get_store(args.backend, runs_dir=RUNS_DIR,
                      project=WANDB_PROJECT, entity=WANDB_ENTITY)
    cultivos = CULTIVOS if args.cultivo == "ambos" else [args.cultivo]

    # Nombres ya existentes (para --skip-existing)
    existing: set[str] = set()
    if args.skip_existing:
        existing = {r["model_name"] for r in store.list_runs()}
        if existing:
            print(f"Saltando {len(existing)} runs ya guardadas: {sorted(existing)}")

    def _run(name: str, factory, ds):
        if name in existing:
            print(f"  {name:22s} [ya existe, saltando]")
            return
        res = run_model_multiseed(name, factory, ds, exp, n_seeds=args.n_seeds)
        store.save(res)
        t = res.summary
        print(f"  {name:22s} testPR={t.get('test_pr_auc_mean', 0):.3f}"
              f"±{t.get('test_pr_auc_std', 0):.3f}  "
              f"testROC={t.get('test_roc_auc_mean', 0):.3f}"
              f"±{t.get('test_roc_auc_std', 0):.3f}")

    for c in cultivos:
        print(f"\n===== {c.upper()} =====")
        ds = cdata.build_crop_dataset(panel_z, c, exp)

        _run("iforest", lambda s: IsolationForestDetector(random_state=s), ds)

        for hidden, latent, drop in AE_ARCHS:
            _run(_label("ae", hidden, latent, drop),
                 lambda s, h=hidden, l=latent, d=drop: AEDetector(
                     hidden_dims=h, latent_dim=l, weight_decay=1e-4,
                     dropout=d, random_state=s),
                 ds)

        for hidden, latent, drop in VAE_ARCHS:
            _run(_label("vae", hidden, latent, drop),
                 lambda s, h=hidden, l=latent, d=drop: VAEDetector(
                     hidden_dims=h, latent_dim=l, weight_decay=1e-4,
                     dropout=d, random_state=s),
                 ds)

    print("\nLISTO. Abrí el comparador con: streamlit run app_streamlit.py")


if __name__ == "__main__":
    main()
