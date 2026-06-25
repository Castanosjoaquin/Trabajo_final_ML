"""
Orquestador de una corrida de modelo.

Toma un CropDataset + un detector, entrena, evalúa, proyecta y arma un
RunResult estandarizado que se persiste vía ResultsStore. Cualquier modelo
que cumpla AnomalyDetector pasa por acá sin cambios → comparación justa.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import CropDataset
from .models import AnomalyDetector
from . import evaluate as ev
from . import embeddings as emb
from .store import RunResult, make_run_id


def _flatten_summary(cfg_model: Dict, val_m: Dict, test_m: Dict,
                     thr: Dict, strat: Dict, dataset: CropDataset) -> Dict:
    """Métricas planas (prefijo por split) aptas para wandb.summary y tablas."""
    s = {"model_type": cfg_model.get("model_type"),
         "threshold": thr["threshold"], "threshold_calibrated": thr["calibrated"],
         "n_train_normal": int(len(dataset.X_train)),
         "n_excluded_year": dataset.n_excluded_year,
         "n_excluded_anom": dataset.n_excluded_anom}
    for split, m in [("val", val_m), ("test", test_m)]:
        for k, v in m.items():
            s[f"{split}_{k}"] = v
    s.update({f"test_{k}": v for k, v in strat.items()})
    return s


def run_model(model_name: str, detector: AnomalyDetector, dataset: CropDataset,
              cfg: ExperimentConfig, emb_methods: List[str] = ("umap", "tsne"),
              sweep_factory: Optional[Callable] = None,
              sweep_grids: Optional[Dict] = None) -> RunResult:
    """Entrena y evalúa un detector sobre un cultivo, devuelve un RunResult.

    sweep_factory(n_estimators, max_samples) -> AnomalyDetector (opcional).
    """
    # --- Entrenar (solo train normal) y scorear ---
    detector.fit(dataset.X_train)
    sc_train = detector.score_samples(dataset.X_train)
    sc_val = detector.score_samples(dataset.X_val)
    sc_test = detector.score_samples(dataset.X_test)

    # --- Calibrar umbral en val, evaluar ---
    thr = ev.calibrate_threshold(sc_val, dataset.y_val)
    val_m = ev.evaluate_split(sc_val, dataset.y_val, thr["threshold"])
    test_m = ev.evaluate_split(sc_test, dataset.y_test, thr["threshold"])
    strat = ev.stratified_recall(sc_test, dataset.y_test, thr["threshold"],
                                 dataset.X_test, dataset.feature_cols)

    # --- Tabla de scores (val + test) ---
    def _scored(meta, scores, split, y):
        df = meta.copy()
        df["split"] = split
        df["score"] = scores
        df["y_pred"] = (scores >= thr["threshold"]).astype(int)
        return df
    scores_df = pd.concat([
        _scored(dataset.meta_val, sc_val, "val", dataset.y_val),
        _scored(dataset.meta_test, sc_test, "test", dataset.y_test),
    ], ignore_index=True)

    # --- Curvas (PR + ROC) en formato largo ---
    curve_frames = []
    for split, sc, y in [("val", sc_val, dataset.y_val), ("test", sc_test, dataset.y_test)]:
        pr = ev.pr_curve(sc, y)
        if pr is not None:
            pr = pr.assign(curve="pr", split=split,
                           x=pr["recall"], y=pr["precision"])
            curve_frames.append(pr[["curve", "split", "x", "y"]])
        roc = ev.roc_curve_df(sc, y)
        if roc is not None:
            roc = roc.assign(curve="roc", split=split, x=roc["fpr"], y=roc["tpr"])
            curve_frames.append(roc[["curve", "split", "x", "y"]])
    curves_df = (pd.concat(curve_frames, ignore_index=True) if curve_frames
                 else pd.DataFrame(columns=["curve", "split", "x", "y"]))

    # --- Embeddings (val + test juntos) ---
    X_all = np.vstack([dataset.X_val, dataset.X_test])
    meta_all = pd.concat([dataset.meta_val, dataset.meta_test], ignore_index=True)
    split_all = (["val"] * len(dataset.X_val)) + (["test"] * len(dataset.X_test))
    sc_all = np.concatenate([sc_val, sc_test])
    emb_df = emb.build_embedding_table(X_all, meta_all, sc_all, "valtest",
                                       list(emb_methods), cfg.random_state)
    emb_df["split"] = split_all * len(emb_methods)

    # --- Barrido opcional ---
    sweep_df = None
    if sweep_factory is not None and sweep_grids is not None:
        sweep_df = ev.hyperparam_sweep(
            sweep_factory, dataset.X_train,
            lambda det: det.score_samples(dataset.X_val), dataset.y_val,
            sweep_grids["n_estimators"], sweep_grids["max_samples"])

    # --- Config completo (experimento + modelo) y summary plano ---
    full_config = {**cfg.to_dict(), **detector.get_config(),
                   "cultivo": dataset.cultivo, "model_name": model_name,
                   "feature_cols": dataset.feature_cols,
                   "n_features": len(dataset.feature_cols)}
    summary = _flatten_summary(detector.get_config(), val_m, test_m, thr, strat, dataset)

    return RunResult(
        run_id=make_run_id(model_name, dataset.cultivo),
        model_name=model_name, cultivo=dataset.cultivo,
        config=full_config, summary=summary,
        scores=scores_df, curves=curves_df, embeddings=emb_df, sweep=sweep_df,
    )
