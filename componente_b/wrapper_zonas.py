"""Wrapper por zonas: router que activa, para cada zona, el modelo que mejor
predice ahí — versión inicial.

Motivación (notebook 06): especializar un modelo por zona ayuda donde la señal
clima→rinde es fuerte y hay datos (zonas pampeanas), pero colapsa en zonas
ruidosas con pocas filas (norte subtropical), y esa zona sola hunde el neto
global. La idea del wrapper: quedarse con lo mejor de los dos mundos — modelo
especializado SOLO en las zonas donde mejora, y el pooled (todas las filas, con
`depto_enc` para lo espacial) en el resto.

La regla anti-trampa, y el corazón de esta clase: la decisión "especializado vs.
pooled" de cada zona se toma con **CV temporal DENTRO de train** (ventana
expansiva, ver `evaluacion._temporal_folds`), NUNCA mirando métricas de test.
Elegir por test sería ajustar el router al test set (leakage de selección): el
número final saldría inflado y no generalizaría. El test se toca una única vez,
al final, con las asignaciones ya congeladas.

Uso típico:

    from wrapper_zonas import WrapperPorZona
    from modelos import XGBoostRegressor

    w = WrapperPorZona(XGBoostRegressor, dict(max_depth=3, n_estimators=200),
                       cultivo="soja", use_agro=True)
    w.fit()
    print(w.resumen_zonas())     # decisión y CV por zona
    print(w.evaluar())           # pooled vs router vs por-zona-puro, en test

Extensiones naturales (todavía no implementadas): comparar varios algoritmos por
zona (no solo pooled vs. especializado del mismo modelo), partial pooling
(encoger el modelo de zona hacia el pooled según n), y pasar `zona` como feature
categórica del pooled.
"""
from __future__ import annotations

from typing import Dict, Optional, Type

import numpy as np
import pandas as pd

import datos
import evaluacion as ev
from modelos.base import Regressor


class WrapperPorZona:
    """Router de modelos por zona geográfica con selección por CV en train.

    Parámetros
    ----------
    model_cls / params : clase del regresor (interfaz `Regressor`) y sus
        hiperparámetros. Se usa la MISMA config para el pooled y para cada zona
        (comparación limpia: lo único que cambia es el subconjunto de datos).
    cultivo, dataset, use_agro, enc_smooth : passthrough al pipeline de datos.
        `enc_smooth` > 0 es recomendable acá: dentro de una zona los deptos
        tienen pocas filas y el target encoding crudo se vuelve ruidoso.
    n_zonas / method / seed : cómo se asignan las zonas (ver `datos.assign_zonas`).
    metric : métrica de la decisión por zona ('rmse' | 'mae' | 'r2').
    n_splits : folds de la CV temporal.
    min_train / min_test : zonas con menos filas que esto no tienen modelo
        propio (van directo al pooled).
    """

    def __init__(self, model_cls: Type[Regressor], params: Optional[Dict] = None,
                 cultivo: str = "soja",
                 use_agro: bool = True, enc_smooth: float = 10.0,
                 n_zonas: int = 6, method: str = "geo", seed: int = 42,
                 metric: str = "rmse", n_splits: int = 4,
                 min_train: int = 100, min_test: int = 20):
        self.model_cls = model_cls
        self.params = dict(params or {})
        self.cultivo = cultivo
        self.use_agro = use_agro
        self.enc_smooth = enc_smooth
        self.n_zonas = n_zonas
        self.method = method
        self.seed = seed
        self.metric = metric
        self.n_splits = n_splits
        self.min_train = min_train
        self.min_test = min_test

        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self) -> "WrapperPorZona":
        """Arma los datasets, corre la CV de decisión y entrena los modelos.

        Pasos:
        1. Pooled: un RegDataset con todas las filas + modelo global.
        2. CV del pooled DESAGREGADA por zona: predicciones out-of-fold de la
           CV temporal del pooled, y métrica de esas predicciones restringida a
           las filas de cada zona → cuánto rinde el pooled EN esa zona, sin test.
        3. CV de cada modelo de zona dentro de su propio train.
        4. Decisión por zona: especializado si su CV es mejor que la del pooled
           en esa zona; si no (o si la zona es muy chica), pooled.
        5. Re-entrena los especializados elegidos con todo su train.
        """
        kw = dict(use_agro=self.use_agro, enc_smooth=self.enc_smooth)

        # --- 1. pooled ---
        panel = datos.assign_zonas(datos.load_panel(self.dataset),
                                   n_zonas=self.n_zonas, method=self.method,
                                   seed=self.seed)
        self.ds_pool = datos.build_reg_dataset(panel, self.cultivo, **kw)
        self.model_pool = self.model_cls(**self.params).fit(
            self.ds_pool.X_train, self.ds_pool.y_train)

        # --- 2. CV del pooled, desagregada por zona (todo dentro de train) ---
        years = self.ds_pool.meta_train["campania_inicio"].values
        zonas_train = self.ds_pool.meta_train["zona"].values
        oof_pred = np.full(len(years), np.nan)
        for tr_idx, va_idx in ev._temporal_folds(years, self.n_splits):
            # fold_X recomputa el depto_enc con el train del fold (sin fuga del
            # target de validación); ver evaluacion.fold_X.
            Xtr, Xva = ev.fold_X(self.ds_pool, tr_idx, va_idx)
            m = self.model_cls(**self.params).fit(Xtr, self.ds_pool.y_train[tr_idx])
            oof_pred[va_idx] = m.predict(Xva)
        seen = ~np.isnan(oof_pred)   # filas que alguna vez fueron validación

        def _pooled_cv_en_zona(z) -> float:
            mask = seen & (zonas_train == z)
            if mask.sum() == 0:
                return np.nan
            return ev.metricas(self.ds_pool.y_train[mask], oof_pred[mask])[self.metric]

        # --- 3. datasets y CV de los modelos por zona ---
        self.zds = datos.build_zona_datasets(
            self.cultivo, n_zonas=self.n_zonas,
            method=self.method, min_train=self.min_train, min_test=self.min_test,
            seed=self.seed, **kw)

        # --- 4. decisión por zona ---
        better = (lambda a, b: a > b) if self.metric == "r2" else (lambda a, b: a < b)
        self.decision_: Dict[str, str] = {}
        self.cv_tabla_: pd.DataFrame
        filas = []
        todas = sorted(pd.unique(zonas_train[pd.notna(zonas_train)]))
        for z in todas:
            cv_pool = _pooled_cv_en_zona(z)
            if z in self.zds:
                cv_zona = ev.cv_score(self.model_cls, self.params, self.zds[z],
                                      metric=self.metric, n_splits=self.n_splits)
                elegido = "zona" if better(cv_zona, cv_pool) else "pooled"
            else:
                cv_zona, elegido = np.nan, "pooled"   # zona muy chica
            self.decision_[z] = elegido
            filas.append({"zona": z, f"cv_{self.metric}_pooled": cv_pool,
                          f"cv_{self.metric}_zona": cv_zona, "elegido": elegido,
                          "n_train": len(self.zds[z].y_train) if z in self.zds else 0})
        self.cv_tabla_ = pd.DataFrame(filas)

        # --- 5. re-entrenar los especializados elegidos con todo su train ---
        self.models_zona_: Dict[str, Regressor] = {
            z: self.model_cls(**self.params).fit(self.zds[z].X_train,
                                                 self.zds[z].y_train)
            for z, e in self.decision_.items() if e == "zona"}

        self._fitted = True
        return self

    # -------------------------------------------------------------- predict
    def predict_test(self) -> np.ndarray:
        """Predicción ruteada sobre el test completo, alineada a `ds_pool`.

        Cada fila de test se mapea a su zona: si la zona tiene modelo propio
        elegido, predice ese (con SU escalado/encoding); si no, el pooled.
        """
        if not self._fitted:
            raise RuntimeError("Llamá fit() primero.")
        y_pred = self.model_pool.predict(self.ds_pool.X_test).astype(float)
        zt = self.ds_pool.meta_test["zona"].values
        for z, model in self.models_zona_.items():
            pos = np.where(zt == z)[0]
            dz = self.zds[z]
            # Sanity check de alineación: mismo orden determinístico (geo + año)
            # en el subset pooled de la zona y en el dataset de la zona.
            if len(pos) != len(dz.y_test) or not np.allclose(
                    self.ds_pool.y_test[pos], dz.y_test):
                raise RuntimeError(
                    f"Desalineación de filas de test en zona {z}: revisá que "
                    "ambos datasets usen el mismo panel/orden.")
            y_pred[pos] = model.predict(dz.X_test)
        return y_pred

    # ------------------------------------------------------------- reportes
    def resumen_zonas(self) -> pd.DataFrame:
        """Tabla de decisión por zona (CV pooled vs. CV zona, elegido, n)."""
        if not self._fitted:
            raise RuntimeError("Llamá fit() primero.")
        return self.cv_tabla_.copy()

    def evaluar(self) -> pd.DataFrame:
        """Métricas de TEST: pooled vs. router vs. por-zona-puro.

        'por zona (puro)' fuerza el especializado en TODAS las zonas con modelo
        (la variante ingenua del nb 06) — se incluye para mostrar qué aporta la
        selección por CV del router. Las tres variantes se evalúan sobre las
        mismas filas (el test completo del pooled)."""
        if not self._fitted:
            raise RuntimeError("Llamá fit() primero.")
        y_te = self.ds_pool.y_test
        pred_pool = self.model_pool.predict(self.ds_pool.X_test)

        # por-zona puro: especializado donde exista dataset de zona, pooled resto
        pred_puro = pred_pool.astype(float).copy()
        zt = self.ds_pool.meta_test["zona"].values
        for z, dz in self.zds.items():
            pos = np.where(zt == z)[0]
            if len(pos) != len(dz.y_test):
                continue
            m = self.models_zona_.get(z) or self.model_cls(**self.params).fit(
                dz.X_train, dz.y_train)
            pred_puro[pos] = m.predict(dz.X_test)

        return ev.tabla_comparativa([
            ev.evaluar("pooled", pred_pool, self.ds_pool),
            ev.evaluar("router por zona (CV)", self.predict_test(), self.ds_pool),
            ev.evaluar("por zona (puro)", pred_puro, self.ds_pool),
        ])


if __name__ == "__main__":
    # Demo: soja con la config ganadora de XGBoost del nb 03/06.
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from modelos import XGBoostRegressor

    BEST = dict(max_depth=3, learning_rate=0.05, n_estimators=200, subsample=0.7,
                colsample_bytree=1.0, min_child_weight=1, reg_lambda=5.0,
                reg_alpha=1.0, gamma=1.0)
    w = WrapperPorZona(XGBoostRegressor, BEST, cultivo="soja", use_agro=True)
    w.fit()
    print("\n--- Decisión por zona (CV en train) ---")
    print(w.resumen_zonas().to_string(index=False))
    print("\n--- Test final ---")
    print(w.evaluar().to_string(index=False))
