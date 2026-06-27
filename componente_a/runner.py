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
    thr = ev.calibrate_threshold(sc_val, dataset.y_val,
                                 mode=cfg.threshold_mode,
                                 contamination=cfg.eval_contamination)
    val_m = ev.evaluate_split(sc_val, dataset.y_val, thr["threshold"],
                              contamination=cfg.eval_contamination)
    test_m = ev.evaluate_split(sc_test, dataset.y_test, thr["threshold"],
                               contamination=cfg.eval_contamination)
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
    # --- Curva de loss de entrenamiento (solo modelos con historial: AE/DAE/VAE) ---
    history = getattr(detector, "history_", None)
    if history:
        hist = pd.DataFrame(history)
        for split, col in [("train", "train_loss"), ("val", "val_loss")]:
            if col in hist.columns:
                curve_frames.append(pd.DataFrame(
                    {"curve": "loss", "split": split,
                     "x": hist["epoch"], "y": hist[col]}))

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


# Métricas sobre las que se agrega media/desvío entre semillas.
_AGG_KEYS = ["val_pr_auc", "val_roc_auc", "val_f1",
            "test_pr_auc", "test_roc_auc", "test_f1",
            "test_precision_at_k", "test_recall_at_contamination"]


def run_model_multiseed(model_name: str, detector_factory: Callable[[int], AnomalyDetector],
                        dataset: CropDataset, cfg: ExperimentConfig,
                        n_seeds: int = 5, base_seed: int = 42,
                        emb_methods: List[str] = ("umap", "tsne"),
                        sweep_factory: Optional[Callable] = None,
                        sweep_grids: Optional[Dict] = None) -> RunResult:
    """Corre el detector con `n_seeds` semillas y agrega media/desvío de las
    métricas. Devuelve UN RunResult representativo (semilla mediana por
    val_pr_auc — la métrica de selección, nunca toca labels de test) con los
    campos `*_mean` / `*_std` añadidos al summary. Resuelve la alta varianza de
    AE/VAE en n≈750: una sola semilla no es representativa.

    detector_factory(seed) -> AnomalyDetector  (con ese random_state).
    """
    if n_seeds <= 1:
        res = run_model(model_name, detector_factory(base_seed), dataset, cfg,
                        emb_methods, sweep_factory, sweep_grids)
        res.summary["n_seeds"] = 1
        return res

    # --- Pasada liviana por semilla: solo métricas (sin embeddings/curvas) ---
    per_seed: List[Dict] = []
    for i in range(n_seeds):
        seed = base_seed + i
        det = detector_factory(seed).fit(dataset.X_train)
        sc_val = det.score_samples(dataset.X_val)
        sc_test = det.score_samples(dataset.X_test)
        thr = ev.calibrate_threshold(sc_val, dataset.y_val,
                                     mode=cfg.threshold_mode,
                                     contamination=cfg.eval_contamination)
        val_m = ev.evaluate_split(sc_val, dataset.y_val, thr["threshold"])
        test_m = ev.evaluate_split(sc_test, dataset.y_test, thr["threshold"])
        flat = {f"val_{k}": v for k, v in val_m.items()}
        flat.update({f"test_{k}": v for k, v in test_m.items()})
        per_seed.append({"seed": seed, "flat": flat})

    # --- Agregar media/desvío ---
    agg: Dict[str, float] = {"n_seeds": n_seeds,
                             "seeds": [s["seed"] for s in per_seed]}
    for key in _AGG_KEYS:
        vals = np.array([s["flat"].get(key, np.nan) for s in per_seed], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

    # --- Semilla representativa: mediana por val_pr_auc (no usa test) ---
    val_prs = [s["flat"].get("val_pr_auc", np.nan) for s in per_seed]
    order = np.argsort(np.nan_to_num(val_prs, nan=-np.inf))
    rep = per_seed[int(order[len(order) // 2])]["seed"]

    # --- Run completo (scores/curvas/embeddings) en la semilla representativa ---
    result = run_model(model_name, detector_factory(rep), dataset, cfg,
                       emb_methods, sweep_factory, sweep_grids)
    result.summary.update(agg)
    result.summary["representative_seed"] = int(rep)
    return result
