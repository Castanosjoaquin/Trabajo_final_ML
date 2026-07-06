"""Integración Componente A ↔ Componente B — "el aporte distintivo del proyecto".

La propuesta pide tres análisis de integración que acá se implementan:

1. **Consistencia cruzada** (`consistencia_cruzada`): correlación de Spearman
   entre el score de anomalía del VAE (Componente A) y el |residuo| del
   predictor de rinde (Componente B) sobre el test. Si ambos componentes
   detectan la misma estructura, una campaña anómala para el VAE también
   debería ser difícil de predecir para el regresor.

2. **Rinde contrafactual bajo clima normal** (`contrafactual_normal`): qué
   hubiera rendido cada depto×campaña si el clima hubiese sido el típico de ese
   departamento. Como las features climáticas están estandarizadas con
   media/desvío de train, "clima normal" = poner esas features en 0 (la media)
   dejando intactas las estructurales (depto_enc, year, lags). No requiere
   re-entrenar nada.

3. **Cuantificación económica** (`cuantificar_perdida`): pérdida de producción
   de cada anomalía = (rinde contrafactual − rinde real) × superficie sembrada.
   `resumen_2223` agrega la campaña 2022/23 (la sequía que la BCR valuó en
   USD 14.140 M) para comparar contra el benchmark externo.

Uso típico (ver también el __main__):

    import datos, integracion, latente
    from modelos import XGBoostRegressor

    ds = datos.prepare("soja", use_agro=True, enc_smooth=10.0)
    model = XGBoostRegressor(...).fit(ds.X_train, ds.y_train)
    vf = latente.vae_features("soja")                  # score del VAE (cacheado)
    print(integracion.consistencia_cruzada(ds, model.predict(ds.X_test), vf))
    y_cf = integracion.contrafactual_normal(model, ds)
    print(integracion.resumen_2223(ds, y_cf))
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Features estructurales: NO se anulan en el contrafactual (definen al depto y
# la tendencia, no al clima de la campaña).
_ESTRUCTURALES = ("depto_enc", "year")


def _idx_climaticas(feature_cols: List[str]) -> np.ndarray:
    """Índices de las features climáticas/satelitales (todo lo que no es
    estructural ni lag del rinde). Incluye las agro: derivan del clima."""
    return np.array([i for i, c in enumerate(feature_cols)
                     if c not in _ESTRUCTURALES
                     and not c.startswith("rinde_lag") and c != "rinde_ma5"])


# ===========================================================================
# 1. Consistencia cruzada (Spearman score del VAE vs |residuo| del predictor)
# ===========================================================================
def consistencia_cruzada(ds, y_pred_test: np.ndarray, vf: Dict) -> Dict[str, float]:
    """Spearman(score de anomalía del VAE, |residuo| del predictor) en test.

    `vf` es el dict de `latente.vae_features` (usa `score_test`, alineado fila a
    fila con el RegDataset). Correlación alta = ambos componentes ven la misma
    estructura (validación cruzada metodológica de la propuesta)."""
    from scipy import stats
    resid = np.abs(ds.y_test - np.asarray(y_pred_test, float))
    rho, p = stats.spearmanr(vf["score_test"], resid)
    return {"spearman_rho": float(rho), "p_value": float(p), "n": len(resid)}


# ===========================================================================
# 2. Contrafactual: rinde predicho bajo clima "normal" del departamento
# ===========================================================================
def contrafactual_normal(model, ds, split: str = "test") -> np.ndarray:
    """Predicción con las features climáticas puestas en su media de train.

    En el espacio escalado (StandardScaler de train) la media es 0, así que el
    contrafactual es simplemente X con las columnas climáticas anuladas. Las
    estructurales (depto_enc, year, lags) se conservan: el "qué hubiera pasado"
    es del clima, no del departamento ni de la tendencia tecnológica."""
    X = ds.X_test if split == "test" else ds.X_train
    X_cf = X.copy()
    X_cf[:, _idx_climaticas(ds.feature_cols)] = 0.0
    return np.asarray(model.predict(X_cf), dtype=float)


# ===========================================================================
# 3. Cuantificación económica de anomalías
# ===========================================================================
def cuantificar_perdida(ds, y_cf: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Pérdida por fila de test: (rinde contrafactual − real) × sup. sembrada.

    Devuelve un DataFrame con la metadata de test + `rinde_cf`, `perdida_kgha`
    y `perdida_tn` (toneladas; positiva = la campaña rindió MENOS que bajo
    clima normal). `mask` opcional restringe a las filas anómalas detectadas
    (p. ej. `latente.anomaly_flag(vf)[1] == 1`)."""
    out = ds.meta_test.copy()
    out["rinde_cf"] = y_cf
    out["perdida_kgha"] = out["rinde_cf"] - out["rinde_kgha"]
    sup = out.get("sup_sembrada_ha")
    if sup is None:
        raise ValueError("meta_test no tiene sup_sembrada_ha: regenerá el "
                         "RegDataset (datos.py ya la incluye en _META_COLS).")
    out["perdida_tn"] = out["perdida_kgha"] * sup / 1000.0
    if mask is not None:
        out = out[np.asarray(mask, dtype=bool)]
    return out


def resumen_2223(ds, y_cf: np.ndarray, mask: Optional[np.ndarray] = None,
                 campania_inicio: int = 2022) -> Dict:
    """Agrega la pérdida de la campaña 2022/23 (o la que se pida) para comparar
    contra benchmarks externos (BCR estimó USD 14.140 M en 2022/23, todos los
    cultivos y todo el país — acá solo los deptos/cultivo del panel)."""
    t = cuantificar_perdida(ds, y_cf, mask)
    t = t[t["campania_inicio"] == campania_inicio]
    top = (t.sort_values("perdida_tn", ascending=False)
             .head(10)[["provincia", "departamento", "rinde_kgha",
                        "rinde_cf", "perdida_tn"]])
    return {
        "campania": f"{campania_inicio}/{str(campania_inicio + 1)[-2:]}",
        "n_filas": len(t),
        "perdida_total_tn": float(t["perdida_tn"].sum()),
        "perdida_media_kgha": float(t["perdida_kgha"].mean()) if len(t) else np.nan,
        "top_deptos": top,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    import warnings; warnings.filterwarnings("ignore")
    import json, os

    import datos, latente
    from modelos import XGBoostRegressor
    import evaluacion as ev

    # Config: la retuneada con CV honesta si existe, si no la vieja del nb 03.
    cfg_path = os.path.join("experimentos", "retuning_cv_honesta.json")
    params = dict(max_depth=3, learning_rate=0.05, n_estimators=200,
                  subsample=0.7, colsample_bytree=1.0, min_child_weight=1,
                  reg_lambda=5.0, reg_alpha=1.0, gamma=1.0)
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            saved = json.load(f)
        if "xgboost" in saved:
            params = {k: v for k, v in saved["xgboost"]["best_params"].items()
                      if k != "n_jobs"}
            print("[integracion] usando XGBoost retuneado (CV honesta)")

    ds = datos.prepare("soja", dataset="base", use_agro=True, enc_smooth=10.0)
    model = XGBoostRegressor(**params).fit(ds.X_train, ds.y_train)
    y_pred = model.predict(ds.X_test)
    print("test:", {k: round(v, 3) for k, v in ev.metricas(ds.y_test, y_pred).items()})

    # 1. consistencia cruzada (el VAE se cachea en .latente_cache)
    vf = latente.vae_features("soja", dataset="base")
    print("consistencia cruzada:", consistencia_cruzada(ds, y_pred, vf))

    # 2-3. contrafactual + pérdida 2022/23 (en las anomalías del VAE y en total)
    y_cf = contrafactual_normal(model, ds)
    _, flag_te = latente.anomaly_flag(vf)
    res_all = resumen_2223(ds, y_cf)
    res_anom = resumen_2223(ds, y_cf, mask=flag_te == 1)
    print(f"\n2022/23 — todas las filas: {res_all['n_filas']} filas, "
          f"pérdida total {res_all['perdida_total_tn']:,.0f} tn "
          f"({res_all['perdida_media_kgha']:.0f} kg/ha promedio)")
    print(f"2022/23 — solo anomalías del VAE: {res_anom['n_filas']} filas, "
          f"pérdida total {res_anom['perdida_total_tn']:,.0f} tn")
    print("\nTop deptos por pérdida (todas las filas 2022/23):")
    print(res_all["top_deptos"].to_string(index=False))
