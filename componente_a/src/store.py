"""
Persistencia de resultados (local, en disco).

Cada corrida se guarda como un run-dir:

  runs/<model_type>/<run_id>/
    meta.json          -> run_id, model_name, cultivo, created_at
    config.json        -> specs del modelo + experimento
    summary.json       -> métricas planas (val_*, test_*)
    scores.parquet     -> filas val+test (depto, campania, split, score, label, ...)
    embeddings.parquet -> proyecciones 2D (UMAP / t-SNE)
    curves.parquet     -> puntos de PR / ROC / loss
    sweep.parquet      -> barrido de hiperparámetros (opcional)

La app Streamlit y los notebooks consumen los runs vía `ResultsStore(runs_dir)`.
"""
from __future__ import annotations

import json
import os
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
    curves: pd.DataFrame          # puntos de PR / ROC / loss
    # proyecciones 2D: vacío al entrenar; se computan on-demand (ver embeddings.py)
    embeddings: pd.DataFrame = field(default_factory=pd.DataFrame)
    sweep: Optional[pd.DataFrame] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def make_run_id(model_name: str, cultivo: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in model_name.lower())
    return f"{safe}__{cultivo}__{ts}"


# =============================================================================
# ResultsStore: lee/escribe run-dirs en disco
# =============================================================================
class ResultsStore:
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        os.makedirs(runs_dir, exist_ok=True)

    def _model_type(self, config: Dict) -> str:
        return config.get("model_type") or config.get("model", "other")

    def _dir(self, run_id: str) -> str:
        """Busca run_id en cualquier subcarpeta de modelo; fallback a raíz."""
        for entry in os.listdir(self.runs_dir):
            candidate = os.path.join(self.runs_dir, entry, run_id)
            if os.path.isdir(candidate):
                return candidate
        return os.path.join(self.runs_dir, run_id)

    def save(self, result: RunResult) -> str:
        model_type = self._model_type(result.config)
        d = os.path.join(self.runs_dir, model_type, result.run_id)
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
        if result.embeddings is not None and not result.embeddings.empty:
            result.embeddings.to_parquet(os.path.join(d, "embeddings.parquet"), index=False)
        if result.sweep is not None:
            result.sweep.to_parquet(os.path.join(d, "sweep.parquet"), index=False)
        return d

    def list_runs(self) -> List[Dict]:
        runs = []
        if not os.path.isdir(self.runs_dir):
            return runs
        for model_type in sorted(os.listdir(self.runs_dir)):
            type_dir = os.path.join(self.runs_dir, model_type)
            if not os.path.isdir(type_dir):
                continue
            for run_id in sorted(os.listdir(type_dir)):
                meta_path = os.path.join(type_dir, run_id, "meta.json")
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
        emb_path = os.path.join(d, "embeddings.parquet")
        return RunResult(
            run_id=meta["run_id"], model_name=meta["model_name"],
            cultivo=meta["cultivo"], created_at=meta.get("created_at", ""),
            config=config, summary=summary,
            scores=pd.read_parquet(os.path.join(d, "scores.parquet")),
            curves=pd.read_parquet(os.path.join(d, "curves.parquet")),
            embeddings=pd.read_parquet(emb_path) if os.path.exists(emb_path) else pd.DataFrame(),
            sweep=pd.read_parquet(sweep_path) if os.path.exists(sweep_path) else None,
        )
