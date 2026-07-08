"""Re-tuning completo de TODOS los modelos con CV temporal honesta, sobre el
panel actual (con CHIRPS). Guarda best-params + métricas de test en
retuning_cv_honesta.json, para ambos cultivos. Uso:

    python experimentos/_retune_all.py            # soja + maiz
    python experimentos/_retune_all.py soja       # solo un cultivo

La comparación final (Diebold-Mariano, skill) la hace el notebook 05 leyendo
este json. Acá solo se BUSCA (CV en train) y se evalúa el ganador en test.
"""
import sys, os, json, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import datos, evaluacion as ev
from modelos import (LinearRegressor, XGBoostRegressor, RandomForestRegressorModel,
                     HistGBMRegressor, NeuralNetRegressor, StackingRegressorModel)

# Config del dataset: la mejor establecida (agro + suavizado del target encoding).
DS_KW = dict(use_agro=True, enc_smooth=10.0)

GRIDS = {
    "linear": (LinearRegressor, {
        "penalty": ["none", "ridge", "lasso", "elasticnet"],
        "alpha":   [0.1, 1.0, 10.0, 50.0, 100.0],
        "l1_ratio": [0.2, 0.5, 0.8],
    }, {}, None),
    "xgb": (XGBoostRegressor, {
        "max_depth": [3, 4, 6], "learning_rate": [0.02, 0.05, 0.1],
        "n_estimators": [200, 400, 800], "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 5],
        "reg_lambda": [1.0, 5.0], "reg_alpha": [0.0, 1.0], "gamma": [0.0, 1.0],
    }, {"random_state": 42, "n_jobs": 4}, 40),
    "rf": (RandomForestRegressorModel, {
        "n_estimators": [400], "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5],
    }, {"random_state": 42, "n_jobs": 4}, 12),
    "hist_gbm": (HistGBMRegressor, {
        "learning_rate": [0.05, 0.1], "max_depth": [None, 6],
        "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 1.0, 10.0],
        "max_iter": [200, 400],
    }, {"random_state": 42}, 16),
    "nn": (NeuralNetRegressor, {
        "hidden_dims": [(16,), (32,), (64, 32)], "dropout": [0.1, 0.3, 0.5],
        "weight_decay": [1e-3, 1e-2, 3e-2], "lr": [1e-3, 3e-3],
    }, {"max_epochs": 200, "patience": 25, "l1_lambda": 0.0, "random_state": 42}, 14),
}


def tune_one(cultivo):
    ds = datos.prepare(cultivo, **DS_KW)
    out = {"_dataset": {"cultivo": cultivo, "n_train": int(len(ds.y_train)),
                        "n_test": int(len(ds.y_test)), "n_feats": len(ds.feature_cols),
                        **DS_KW}}
    base = ev.metricas(ds.y_test, ev.pred_media_depto(ds))
    out["baseline"] = {"best_params": {}, "test": base}
    for name, (cls, grid, fixed, n_iter) in GRIDS.items():
        t0 = time.time()
        tabla, best = ev.buscar(cls, grid, ds, metric="rmse", n_iter=n_iter,
                                random_state=42, n_splits=4, fixed=fixed)
        model = cls(**{**fixed, **best}).fit(ds.X_train, ds.y_train)
        test = ev.metricas(ds.y_test, model.predict(ds.X_test))
        out[name] = {"best_params": best, "cv_rmse": float(tabla["cv_rmse"].iloc[0]),
                     "test": test, "secs": round(time.time() - t0, 1)}
        print(f"[{cultivo}] {name:9s} cv_rmse={out[name]['cv_rmse']:.0f} "
              f"test_rmse={test['rmse']:.0f} r2={test['r2']:.3f} ({out[name]['secs']}s)",
              flush=True)

    # Stacking: usa los best de xgb y nn ya encontrados.
    t0 = time.time()
    xgb_p = {k: v for k, v in out["xgb"]["best_params"].items()}
    nn_p = {**{"max_epochs": 200, "patience": 25}, **out["nn"]["best_params"]}
    stk = StackingRegressorModel(xgb_params=xgb_p, nn_params=nn_p, ridge_alpha=1.0)
    stk.fit(ds.X_train, ds.y_train, years=ds.meta_train["campania_inicio"].values)
    test = ev.metricas(ds.y_test, stk.predict(ds.X_test))
    out["stacking"] = {"best_params": {"xgb_params": xgb_p, "nn_params": nn_p,
                                       "ridge_alpha": 1.0},
                       "test": test, "secs": round(time.time() - t0, 1)}
    print(f"[{cultivo}] stacking  test_rmse={test['rmse']:.0f} r2={test['r2']:.3f} "
          f"({out['stacking']['secs']}s)", flush=True)
    return out


if __name__ == "__main__":
    cultivos = sys.argv[1:] or ["soja", "maiz"]
    path = os.path.join(os.path.dirname(__file__), "retuning_cv_honesta.json")
    result = {}
    for c in cultivos:
        print(f"\n===== {c} =====", flush=True)
        result[c] = tune_one(c)
        # guardado incremental por si se corta
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nGuardado en", path, flush=True)
