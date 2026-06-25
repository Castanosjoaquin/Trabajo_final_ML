"""
Capa de persistencia de resultados (ResultsStore).

Dos backends de IDÉNTICO esquema:
- LocalBackend  (activo): escribe/lee run-dirs en disco.
- WandbBackend  (después): mapea el mismo RunResult a un run de W&B.

La app Streamlit consume runs vía esta interfaz, sin saber qué backend hay
detrás. El esquema local es un mirror del de W&B, así el switch es transparente:

  run-dir/
    config.json        -> wandb.config (specs del modelo + experimento)
    summary.json       -> wandb.summary (métricas planas: val_*, test_*)
    scores.parquet     -> wandb.Table (depto, campania, split, score, label, ...)
    embeddings.parquet -> wandb.Table (proyecciones 2D, render projection)
    curves.parquet     -> wandb.plot.pr_curve / roc_curve
    sweep.parquet      -> wandb.Table (barrido de hiperparámetros)
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


# =============================================================================
# RunResult: todo lo que produce una corrida, en un esquema único
# =============================================================================
@dataclass
class RunResult:
    run_id: str
    model_name: str               # etiqueta legible elegida por el usuario
    cultivo: str
    config: Dict                  # specs (experimento + modelo) -> config.json
    summary: Dict                 # métricas planas -> summary.json
    scores: pd.DataFrame          # filas val+test con score/label
    curves: pd.DataFrame          # puntos de PR y ROC (col 'curve','split')
    embeddings: pd.DataFrame      # proyecciones 2D
    sweep: Optional[pd.DataFrame] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def make_run_id(model_name: str, cultivo: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in model_name.lower())
    return f"{safe}__{cultivo}__{ts}"


# =============================================================================
# Interfaz
# =============================================================================
class ResultsStore(ABC):
    @abstractmethod
    def save(self, result: RunResult) -> str: ...

    @abstractmethod
    def list_runs(self) -> List[Dict]:
        """Lista runs disponibles (metadatos livianos para el selector)."""

    @abstractmethod
    def load_run(self, run_id: str) -> RunResult: ...


# =============================================================================
# LocalBackend (activo ahora)
# =============================================================================
class LocalBackend(ResultsStore):
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        os.makedirs(runs_dir, exist_ok=True)

    def _dir(self, run_id: str) -> str:
        return os.path.join(self.runs_dir, run_id)

    def save(self, result: RunResult) -> str:
        d = self._dir(result.run_id)
        os.makedirs(d, exist_ok=True)
        meta = {"run_id": result.run_id, "model_name": result.model_name,
                "cultivo": result.cultivo, "created_at": result.created_at}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(result.config, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(result.summary, f, indent=2, ensure_ascii=False, default=str)
        result.scores.to_parquet(os.path.join(d, "scores.parquet"), index=False)
        result.curves.to_parquet(os.path.join(d, "curves.parquet"), index=False)
        result.embeddings.to_parquet(os.path.join(d, "embeddings.parquet"), index=False)
        if result.sweep is not None:
            result.sweep.to_parquet(os.path.join(d, "sweep.parquet"), index=False)
        return d

    def list_runs(self) -> List[Dict]:
        runs = []
        if not os.path.isdir(self.runs_dir):
            return runs
        for run_id in sorted(os.listdir(self.runs_dir)):
            meta_path = os.path.join(self._dir(run_id), "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    runs.append(json.load(f))
        return runs

    def load_run(self, run_id: str) -> RunResult:
        d = self._dir(run_id)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Run no encontrado: {run_id}")
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        with open(os.path.join(d, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(d, "summary.json")) as f:
            summary = json.load(f)
        sweep_path = os.path.join(d, "sweep.parquet")
        return RunResult(
            run_id=meta["run_id"], model_name=meta["model_name"],
            cultivo=meta["cultivo"], created_at=meta.get("created_at", ""),
            config=config, summary=summary,
            scores=pd.read_parquet(os.path.join(d, "scores.parquet")),
            curves=pd.read_parquet(os.path.join(d, "curves.parquet")),
            embeddings=pd.read_parquet(os.path.join(d, "embeddings.parquet")),
            sweep=pd.read_parquet(sweep_path) if os.path.exists(sweep_path) else None,
        )


# =============================================================================
# WandbBackend (stub — activar después)
# =============================================================================
class WandbBackend(ResultsStore):
    """Mapea el MISMO RunResult a W&B. No instala nada hasta que se use.

    save():
        wandb.init(project=..., name=model_name, config=result.config)
        wandb.summary.update(result.summary)
        wandb.log({"scores": wandb.Table(dataframe=result.scores)})
        wandb.log({"embeddings": wandb.Table(dataframe=result.embeddings)})  # render 2D
        wandb.log({"pr": wandb.plot.pr_curve(...), "roc": wandb.plot.roc_curve(...)})

    Para leer (list_runs/load_run) se usaría wandb.Api(). Streamlit consumiría
    estos métodos sin cambios, porque la interfaz es la misma.
    """

    def __init__(self, project: str, entity: Optional[str] = None):
        self.project = project
        self.entity = entity

    def _require_wandb(self):
        try:
            import wandb  # noqa: F401
            return wandb
        except ImportError as e:
            raise ImportError(
                "WandbBackend requiere 'wandb'. Instalá con: pip install wandb\n"
                "y autenticá con: wandb login"
            ) from e

    def save(self, result: RunResult) -> str:
        wandb = self._require_wandb()
        import numpy as np

        run = wandb.init(
            project=self.project, entity=self.entity,
            name=result.model_name, config=result.config,
            group=result.cultivo, reinit="finish_previous",
        )
        # --- Métricas planas (aparecen en la tabla de runs) ---
        run.summary.update(result.summary)

        # --- Curvas PR y ROC como gráficos nativos de W&B ---
        for split in ("val", "test"):
            pr = result.curves[
                (result.curves["curve"] == "pr") & (result.curves["split"] == split)
            ]
            roc = result.curves[
                (result.curves["curve"] == "roc") & (result.curves["split"] == split)
            ]
            if not pr.empty:
                run.log({
                    f"pr_curve_{split}": wandb.plot.line(
                        wandb.Table(data=list(zip(pr["x"], pr["y"])),
                                    columns=["recall", "precision"]),
                        x="recall", y="precision",
                        title=f"PR Curve ({split})",
                    )
                })
            if not roc.empty:
                run.log({
                    f"roc_curve_{split}": wandb.plot.line(
                        wandb.Table(data=list(zip(roc["x"], roc["y"])),
                                    columns=["fpr", "tpr"]),
                        x="fpr", y="tpr",
                        title=f"ROC Curve ({split})",
                    )
                })

        # --- Distribución de scores (histograma por clase) ---
        test_scores = result.scores[result.scores["split"] == "test"]
        if not test_scores.empty:
            run.log({
                "score_distribution": wandb.plot.histogram(
                    wandb.Table(
                        data=[[row["score"], "anómala" if row["anomalia"] else "normal"]
                              for _, row in test_scores.iterrows()],
                        columns=["score", "clase"],
                    ),
                    value="score",
                    title="Distribución de scores (test)",
                )
            })

        # --- Tablas completas (scores, embeddings) ---
        run.log({
            "scores":     wandb.Table(dataframe=result.scores),
            "embeddings": wandb.Table(dataframe=result.embeddings),
        })
        if result.sweep is not None:
            run.log({"sweep": wandb.Table(dataframe=result.sweep)})

        run.finish()
        return run.id

    def list_runs(self) -> List[Dict]:
        wandb = self._require_wandb()
        api = wandb.Api()
        path = f"{self.entity + '/' if self.entity else ''}{self.project}"
        return [{"run_id": r.id, "model_name": r.name,
                 "cultivo": r.config.get("cultivo", ""),
                 "created_at": r.created_at} for r in api.runs(path)]

    def load_run(self, run_id: str) -> RunResult:
        raise NotImplementedError(
            "Lectura desde W&B aún no implementada; usar LocalBackend por ahora."
        )


def get_store(backend: str = "local", runs_dir: str = "runs",
              project: str = "componente-a-anomalias",
              entity: Optional[str] = None) -> ResultsStore:
    """Factory. backend ∈ {'local', 'wandb'}. Flag único para el switch."""
    if backend == "local":
        return LocalBackend(runs_dir)
    if backend == "wandb":
        return WandbBackend(project, entity)
    raise ValueError(f"Backend desconocido: {backend!r}")
