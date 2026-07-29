"""Pipeline de datos del Componente B — predicción de rinde (regresión).

Construye, por cultivo, un dataset listo para entrenar los regresores de
`modelos/`: X (features escaladas) e y (`rinde_kgha`, kg/ha), con split temporal
train/test consistente con el Componente A (train ≤2020, test ≥2021).

Reutiliza del Componente A la carga del panel (dedup de las filas espurias por
lat/lon) y la lista de features climáticas mensuales; ver `componente_a/src/data.py`.

Las features que ve el modelo son de tres tipos, todas escaladas con parámetros
calculados SOLO en train (sin leakage):

  1. Clima mensual (Sep–Mar): las 54 columnas de NASA POWER + ONI.
  2. Codificación del departamento por su rinde medio en train ("mean/target
     encoding"): captura la estructura espacial (un depto rinde sistemáticamente
     más que otro) sin explotar en cientos de dummies. Deptos nuevos en test →
     media global de train.
  3. Año (`campania_inicio`): captura la tendencia tecnológica (el rinde sube con
     las décadas). OJO: en test (2021–2024) el año queda fuera del rango de train,
     así que los árboles (XGBoost) no extrapolan la tendencia y los lineales sí.

El target `rinde_kgha` se deja crudo (kg/ha); cada modelo lo escala internamente
si lo necesita (la red neuronal lo hace).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Reutilizar el pipeline del Componente A (carga + dedup + lista de features).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_A_ROOT = os.path.join(_REPO_ROOT, "componente_a")
_B_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _A_ROOT, _B_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import data as _A_data      # noqa: E402  (componente_a/src)
from src import config as _A_config  # noqa: E402

# Import PLANO, no relativo: latente.py hace `import datos`, así que este módulo
# también se carga como top-level y un `from . import ...` reventaría ahí.
# momentum no importa datos → sin ciclo.
import momentum as _momentum  # noqa: E402

# Split temporal (mismo criterio que el Componente A).
TRAIN_END = 2020
TEST_START = 2021

# Meses de la campaña, en orden Sep..Mar. Se toma del Componente A en vez de
# redefinirlo: es el mismo orden que usan los sufijos de las columnas del panel,
# y de él depende la regla de corte de los checkpoints.
MESES = _A_config.MESES

@dataclass
class RegDataset:
    """Datos de un cultivo listos para regresión de rinde.

    X ya están escaladas (StandardScaler ajustado en train). y es `rinde_kgha`
    crudo. Los meta_* conservan identificadores y el rinde para baselines/plots.
    """

    cultivo: str
    feature_cols: List[str]

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray          # rinde_kgha (kg/ha)
    y_test: np.ndarray

    meta_train: pd.DataFrame
    meta_test: pd.DataFrame

    # para baselines
    y_train_mean: float                    # media global de train
    depto_mean: pd.Series                  # rinde medio por (prov, depto) en train


_META_COLS = ["provincia", "departamento", "campania", "campania_inicio",
              "rinde_kgha", "sup_sembrada_ha", "zona"]

# Checkpoints del calendario agrícola: en qué momento de la campaña se consulta el
# modelo. Cada uno se define por su MES DE CORTE — el último mes cuyo clima ya se
# observó. `None` = la campaña todavía no empezó.
#
#   'pre_siembra' : antes de sembrar. Solo estructurales + ONI + sm_winter.
#   'nov'         : campaña arrancada, clima Sep–Nov observado.
#   'ene'         : mitad de campaña, Sep–Ene.
#   'pre_cosecha' : Sep–Feb, para pronosticar antes de cosechar.
#   'full'        : campaña completa Sep–Mar (comportamiento histórico).
#
# Los meses posteriores al corte se EXCLUYEN, no se imputan.
CHECKPOINTS = {
    "pre_siembra": None,
    "nov":         "nov",
    "ene":         "ene",
    "pre_cosecha": "feb",
    "full":        "mar",
}

# Alias histórico: los notebooks 08-10 importan MOMENTOS. Los tres nombres viejos
# siguen devolviendo exactamente el mismo conjunto de columnas que antes de sumar
# los checkpoints intermedios (verificado columna por columna, ambos cultivos).
MOMENTOS = tuple(CHECKPOINTS)

# Features estacionales de ERA5: no llevan sufijo de mes, así que la regla de corte
# no las alcanza y hay que declarar a mano desde qué checkpoint se observan.
#   sm_winter        (Jun–Ago) : previa a la campaña -> disponible siempre.
#   sm_planting      (Sep–Nov) : completa en noviembre.
#   frost_days_early (Sep–Nov) : idem.
#   frost_days       (Sep–Mar) : recién con la campaña terminada.
_ERA5_DESDE = {
    "sm_winter": None,          # siempre
    "sm_planting": "nov",
    "frost_days_early": "nov",
    "frost_days": "mar",
}


def _filter_momento(clim_cols: List[str], momento: str) -> List[str]:
    """Restringe las columnas climáticas/satelitales a lo observable en un checkpoint.

    Regla general: una columna `<prefijo>_<mes>` se observa si `mes` no es posterior
    al mes de corte. Aprovecha que todas las columnas mensuales del panel ya llevan
    el sufijo Sep..Mar.

    Excepción deliberada — ONI: se mantiene en TODOS los checkpoints, incluido
    pre-siembra, donde `oni_ene` y `oni_feb` todavía no ocurrieron. Es la convención
    que ya traía el repo, y se sostiene porque en producción ese valor sería el
    PRONÓSTICO de ENSO, disponible por adelantado. Es la única concesión de
    información futura del esquema y conviene declararla al reportar resultados.
    """
    if momento not in CHECKPOINTS:
        raise ValueError(f"checkpoint desconocido: {momento!r}. "
                         f"Opciones: {tuple(CHECKPOINTS)}")
    corte = CHECKPOINTS[momento]
    hasta = -1 if corte is None else MESES.index(corte)

    def observable(c: str) -> bool:
        if c.startswith("oni_"):
            return True                       # pronóstico ENSO, ver docstring
        if c in _ERA5_DESDE:
            desde = _ERA5_DESDE[c]
            return desde is None or hasta >= MESES.index(desde)
        mes = c.rsplit("_", 1)[-1]
        if mes in MESES:
            return hasta >= MESES.index(mes)
        # Sin sufijo de mes ni regla propia: se asume estructural, disponible
        # siempre. Si se suma una fuente estacional nueva, va en _ERA5_DESDE.
        return True

    return [c for c in clim_cols if observable(c)]


def load_panel() -> pd.DataFrame:
    """Panel ÚNICO del proyecto (base + NDVI + ERA5), deduplicado. Delega en el
    Componente A, que ya incluye NDVI/ERA5 por defecto en el panel unificado."""
    return _A_data.load_panel()


def crop_frame(panel: pd.DataFrame, cultivo: str,
               train_end: int = TRAIN_END, test_start: int = TEST_START):
    """Filas de un cultivo en orden DETERMINÍSTICO, con sus features climáticas
    (clima + NDVI + ERA5, del panel unificado).

    Devuelve (df, tr_mask, te_mask, clim_cols, geo). El orden fijo (ordenar por
    geo + campania_inicio y resetear el índice) es lo que garantiza que el
    RegDataset (`datos`) y las features del VAE (`latente`) queden ALINEADOS fila
    a fila aunque partan de paneles ordenados distinto."""
    clim_cols = _A_data.build_feature_list(panel)
    geo = ["provincia", "departamento"] if "provincia" in panel.columns else ["departamento"]
    df = (panel[panel["cultivo"] == cultivo]
          .dropna(subset=clim_cols + ["rinde_kgha"])
          .sort_values(geo + ["campania_inicio"])
          .reset_index(drop=True))
    tr_mask = df["campania_inicio"] <= train_end
    te_mask = df["campania_inicio"] >= test_start
    return df, tr_mask, te_mask, clim_cols, geo


def build_reg_dataset(panel: pd.DataFrame, cultivo: str,
                      use_depto_encoding: bool = True, use_year: bool = True,
                      use_agro: bool = False, use_suelo: bool = False,
                      use_momentum: Optional[float] = None,
                      enc_smooth: float = 0.0,
                      use_lags: int = 0, momento: str = "full",
                      train_end: int = TRAIN_END,
                      test_start: int = TEST_START) -> RegDataset:
    """Arma el RegDataset de un cultivo: split temporal, features y escalado.

    Las features climáticas salen del panel unificado (clima + NDVI + ERA5).
    `use_momentum` (gamma en [0,1)) agrega `momentum_z_rinde`: el EWMA del z_rinde
    histórico del departamento, calculado SOLO con campañas anteriores. gamma alto
    = más memoria. Es un parámetro del DATASET, no del estimador, así que no puede
    barrerse con `evaluacion.buscar`: hay que armar un RegDataset por gamma y
    comparar con `evaluacion.cv_score`. El momentum del score de anomalía
    (`momentum_anomaly_score`) se agrega aparte, porque requiere entrenar el VAE
    (ver `momentum.momentum_score_alineado` + `latente.vae_features`).

    `use_suelo` agrega las 13 features derivadas de suelo y geografía
    (`add_suelo_features`): agua útil integrada, textura y químicas 0-30 cm,
    elevación, pendiente y distancia a cursos de agua. Son estáticas por
    departamento, así que valen para TODOS los momentos, incluido pre-siembra.

    `use_agro` agrega las features agronómicas de ventana crítica del Componente A
    (balance hídrico, estrés térmico; ver `add_agro_features`).

    `enc_smooth` (m del m-estimate) suaviza el target encoding del departamento
    hacia la media global de train: enc = (n·media_depto + m·media_global)/(n+m).
    Con m=0 se recupera la media cruda (comportamiento histórico de los
    notebooks); m≈10 protege a los deptos con pocas campañas en train, cuyas
    medias crudas son ruidosas — recomendado al modelar por zona, donde los n
    por depto se achican.

    `use_lags` (k>0) agrega los k lags del rinde (`rinde_lag1..k`, con shift —
    NUNCA el año actual) y la media móvil de 5 campañas previas (`rinde_ma5`),
    las features autorregresivas que pide la propuesta. Los NaN del arranque de
    cada serie (primeras k campañas del depto) se rellenan con la media de
    train. Nota: si a un depto le falta una campaña intermedia, el lag es la
    última campaña DISPONIBLE (no el año calendario exacto).

    `momento` restringe las features al calendario agrícola (ver MOMENTOS):
    'pre_siembra' deja solo ONI + sm_winter + estructurales; 'pre_cosecha'
    excluye marzo. Con momento != 'full' no se admite `use_agro` (las ventanas
    críticas agronómicas llegan a marzo y filtrarían clima futuro)."""
    if momento != "full" and use_agro:
        raise ValueError("use_agro=True requiere momento='full': las ventanas "
                         "críticas agronómicas usan clima hasta marzo.")
    df, tr_mask, te_mask, clim_cols, geo = crop_frame(
        panel, cultivo, train_end, test_start)

    feature_cols = _filter_momento(clim_cols, momento)

    # --- Features agronómicas de dominio (ventana crítica del cultivo) ---
    if use_agro:
        df, agro_cols = _A_data.add_agro_features(df, cultivo)
        feature_cols += agro_cols

    # --- Suelo y geografía (estáticas por departamento) ---
    # NO se filtran por `momento`, a diferencia de las agronómicas: no dependen de
    # la campaña, así que están disponibles en todos los checkpoints —incluso
    # pre-siembra— sin ningún riesgo de leakage.
    # Nota: sirven acá porque el escalado de B es GLOBAL. En el pipeline del
    # Componente A, que normaliza por departamento, una feature estática tiene
    # varianza 0 dentro del grupo y se anula (queda exactamente en 0).
    if use_suelo:
        df, suelo_cols = _A_data.add_suelo_features(df)
        feature_cols += suelo_cols

    # --- Momentum inter-campaña (EWMA con decaimiento `use_momentum` = gamma) ---
    # Resume la historia del departamento dándole más peso a lo reciente. Como solo
    # mira campañas ESTRICTAMENTE anteriores (ver momentum.ewma_pasado), está
    # disponible en todos los momentos, igual que las estáticas.
    if use_momentum is not None:
        # `z_rinde` no viene en el panel crudo. El merge lleva "cultivo" en la clave:
        # sin él, drop_duplicates se queda con la fila de maíz por orden alfabético
        # y soja hereda un z_rinde ajeno (mismo bug documentado en latente.py).
        key = geo + ["campania_inicio", "cultivo"]
        z = (_A_data.compute_z_rinde(panel)[key + ["z_rinde"]]
             .drop_duplicates(key))
        n0 = len(df)
        df = df.merge(z, on=key, how="left")
        assert len(df) == n0, f"el merge de z_rinde duplicó filas: {n0} -> {len(df)}"
        df, mom_cols = _momentum.add_momentum(df, geo, use_momentum,
                                              train_mask=tr_mask.values)
        df = df.drop(columns=["z_rinde"])   # la señal cruda no entra en X
        feature_cols += mom_cols

    # --- Lags del rinde (autorregresivas, sin filtrar el año actual) ---
    lag_cols: List[str] = []
    if use_lags > 0:
        g = df.groupby(geo)["rinde_kgha"]
        for k in range(1, use_lags + 1):
            col = f"rinde_lag{k}"
            df[col] = g.shift(k)
            lag_cols.append(col)
        df["rinde_ma5"] = df.groupby(geo)["rinde_kgha"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        lag_cols.append("rinde_ma5")
        feature_cols += lag_cols

    tr = df[tr_mask].copy()
    te = df[te_mask].copy()

    if lag_cols:
        # Relleno de los NaN del arranque de serie con la media de TRAIN (stat
        # de train → sin fuga hacia test; solo toca las primeras campañas).
        fill = float(tr["rinde_kgha"].mean())
        tr[lag_cols] = tr[lag_cols].fillna(fill)
        te[lag_cols] = te[lag_cols].fillna(fill)

    # --- Codificación del departamento por su rinde medio en train (sin leakage) ---
    y_train_mean = float(tr["rinde_kgha"].mean())
    grp = tr.groupby(geo)["rinde_kgha"]
    if enc_smooth > 0:
        # m-estimate: encoge la media del depto hacia la global según cuántas
        # campañas de train lo respaldan (n chico → más cerca de la global).
        agg = grp.agg(["mean", "size"])
        depto_mean = ((agg["size"] * agg["mean"] + enc_smooth * y_train_mean)
                      / (agg["size"] + enc_smooth))
    else:
        depto_mean = grp.mean()
    if use_depto_encoding:
        def _enc(d: pd.DataFrame) -> np.ndarray:
            keys = list(d[geo].itertuples(index=False, name=None))
            return np.array([depto_mean.get(k, y_train_mean) for k in keys], dtype=float)
        tr["depto_enc"] = _enc(tr)
        te["depto_enc"] = _enc(te)
        feature_cols.append("depto_enc")

    # --- Año (tendencia) ---
    if use_year:
        tr["year"] = tr["campania_inicio"].astype(float)
        te["year"] = te["campania_inicio"].astype(float)
        feature_cols.append("year")

    # --- Escalado: StandardScaler ajustado SOLO en train ---
    mu = tr[feature_cols].mean()
    sd = tr[feature_cols].std().replace(0, 1.0)
    Xtr = ((tr[feature_cols] - mu) / sd).values.astype(np.float32)
    Xte = ((te[feature_cols] - mu) / sd).values.astype(np.float32)

    return RegDataset(
        cultivo=cultivo,
        feature_cols=feature_cols,
        X_train=Xtr,
        X_test=Xte,
        y_train=tr["rinde_kgha"].values.astype(np.float32),
        y_test=te["rinde_kgha"].values.astype(np.float32),
        meta_train=tr[[c for c in _META_COLS if c in tr.columns]].reset_index(drop=True),
        meta_test=te[[c for c in _META_COLS if c in te.columns]].reset_index(drop=True),
        y_train_mean=y_train_mean,
        depto_mean=depto_mean,
    )


def prepare(cultivo: str, **kwargs) -> RegDataset:
    """Atajo: carga el panel unificado y arma el RegDataset del cultivo."""
    return build_reg_dataset(load_panel(), cultivo, **kwargs)


def build_zona_datasets(cultivo: str, n_zonas: int = 6,
                        method: str = "geo", min_train: int = 100, min_test: int = 20,
                        seed: int = 42, **kwargs) -> dict:
    """Un RegDataset por zona (para entrenar un modelo por zona).

    Asigna zonas con `assign_zonas` y arma, para cada una, un RegDataset con SU
    propio split temporal, codificación de depto y escalado (todo calculado dentro
    de la zona). Descarta zonas con pocas filas de train/test. `kwargs` se pasan a
    `build_reg_dataset` (p. ej. `use_agro=True`). Devuelve {zona: RegDataset}."""
    panel = assign_zonas(load_panel(), n_zonas=n_zonas, method=method, seed=seed)
    out = {}
    for z in sorted(panel["zona"].dropna().unique()):
        ds = build_reg_dataset(panel[panel["zona"] == z], cultivo, **kwargs)
        if len(ds.y_train) >= min_train and len(ds.y_test) >= min_test:
            out[z] = ds
    return out


def assign_zonas(panel: pd.DataFrame, n_zonas: int = 6, method: str = "geo",
                 seed: int = 42) -> pd.DataFrame:
    """Agrega una columna `zona` para poder modelar/analizar por separado.

    - 'provincia' : zona = provincia (15 zonas, tamaños muy dispares).
    - 'geo'       : agrupa DEPARTAMENTOS cercanos con KMeans sobre su centroide
                    (lat, lon) en `n_zonas` zonas geográficas contiguas. Más
                    balanceadas que las provincias y respetan la cercanía (la
                    idea: dentro de una zona el clima es parecido, así se reduce
                    el ruido entre zonas de climas distintos).

    Devuelve una copia del panel con la columna `zona`."""
    out = panel.copy()
    if method == "provincia":
        out["zona"] = out["provincia"].astype(str)
        return out
    if method == "geo":
        from sklearn.cluster import KMeans
        geo = ["provincia", "departamento"] if "provincia" in out.columns else ["departamento"]
        cent = out.dropna(subset=["lat", "lon"]).groupby(geo)[["lat", "lon"]].mean()
        km = KMeans(n_clusters=n_zonas, random_state=seed, n_init=10)
        cent["zona"] = [f"z{c}" for c in km.fit_predict(cent[["lat", "lon"]].values)]
        out = out.merge(cent["zona"].reset_index(), on=geo, how="left")
        return out
    raise ValueError(f"method desconocido: {method!r}. Opciones: 'geo', 'provincia'.")
