# Plan de ejecución — Expansión del dataset (suelo/geografía) + dimensión temporal

**Branch:** `expansion-suelo-y-serie-temporal` (creada desde `rebuild-parte1` @ `baf42d4`)
**Origen:** `spec_expansion_dataset_serie_temporal.md`

---

## Contexto

El feature set actual (72 features, no 48) es enteramente climático: NASA POWER + ONI +
CHIRPS + NDVI-AVHRR + ERA5-Land. No hay ninguna variable que explique *por qué dos
departamentos con el mismo shock climático responden distinto* — es decir, la
variabilidad estructural del suelo. **Parte 1** agrega esa capa.

Por otro lado, el modelo hoy solo predice con la campaña completa (Sep–Mar) o con dos
recortes gruesos. Un productor necesita consultar el modelo **en cualquier momento** de
la campaña y que use tanto lo observado hasta ahí como la historia de su zona.
**Parte 2** resuelve esos dos ejes.

### Correcciones al spec (importante — el spec fue escrito contra un repo anterior)

| El spec asume | Realidad del repo |
|---|---|
| `panel_nucleo.parquet`, 2.287×76, 26 deptos | `data/processed/panel_union.parquet`, **27.865×84, 307 deptos / 15 provincias** (20.672 tras dedup) |
| `build_panel.py` + `validate_panel.py` | `componente_a/eda/build_panel_union.py` — único ETL. **No existe validador**; el guard es `componente_a/tests/test_repro.py` |
| Split 3 vías (train/val/test) | 2 vías: `TRAIN_END=2020`, `TEST_START=2021` (`componente_a/src/config.py:70`). El bloque de validación se plegó a train |
| 48 features climáticas | **72** (`data.build_feature_list`, `componente_a/src/data.py:36`) |
| "generalizar pre-siembra/pre-cosecha a N checkpoints" | Ya existen: `MOMENTOS` + `_filter_momento` (`componente_b/datos.py:87`). Se **extiende**, no se crea |
| momentum inter-campaña "no contemplado" | Existe parcialmente: `use_lags` → `rinde_lag1..k` + `rinde_ma5` (`componente_b/datos.py:170`) |
| `evaluate.py` | `componente_b/evaluacion.py` |

**Decisiones tomadas:** extraer suelo para los **307** deptos del panel; **IP INTA/IDECOR
diferido a fase condicional** (solo si el EDA muestra señal); **5 checkpoints
acumulativos**; **EWMA se suma a los lags existentes** y se decide por ablation.

---

## Limitaciones conocidas y decisiones tomadas durante la ejecución

- **Villa Constitución (Santa Fe) no está en el panel.** Es uno de los 26 departamentos
  del núcleo, pero MAGyP lo escribe `'Villa Constitución'` (con acento) y
  `CENTROIDES_NUCLEO` lo tiene sin acento, así que el merge de centroides no lo matchea,
  se queda sin lat/lon y se descarta. Solo tiene datos de maíz (56 filas). El panel
  contiene **25 de los 26** deptos núcleo. Se documenta como limitación, junto a `ws2m` y
  la napa freática. Recuperarlo exige normalizar el merge de centroides, lo que agrega
  filas y obliga a re-registrar `test_repro.py` — decisión diferida a propósito.
- **`region` corregida** (2026-07-28): se asignaba por nombre de departamento sin
  provincia, marcando 34 deptos núcleo en vez de 25. Ahora compara el par
  (departamento, provincia) normalizado con `_norm_one`. Neutral en filas.
- **Agua útil negativa en 7 deptos de Misiones**: artefacto de SoilGrids (predice cada
  profundidad de forma independiente y en suelos rojos lateríticos capacidad de campo y
  punto de marchitez se cruzan). El panel guarda `wv0033`/`wv1500` crudos (source-only) y
  `add_suelo_features` aplica `.clip(lower=0)` antes de integrar la lámina.
- **Las features estáticas NO sirven en Componente A tal como está.** Verificado
  empíricamente: `_normalize_per_depto` z-scorea por departamento, y una feature estática
  tiene varianza 0 dentro del grupo, así que queda en **exactamente 0** para todas las
  filas. En Componente B sí funcionan, porque su escalado es global (`suelo_awc_mm`
  escalada da rango [-3,17, 2,28]). Consecuencia para el ablation: suelo/geo se evalúa en
  B; para usarlas en A habría que cambiar el esquema de normalización, que no está en el
  alcance de esta iteración.

---

## Dependencias previas (bloqueante)

En `.venv` faltan: `xgboost` (está en `requirements.txt` pero no instalado), y
`earthengine-api` no figura en `requirements.txt` aunque `ee 1.7.32` sí está instalado.

```bash
.venv/bin/pip install xgboost earthengine-api
```

Actualizar `requirements.txt` agregando `earthengine-api`. `geopandas`/`shapely` **no**
se instalan en esta fase (solo harían falta para el IP diferido).

---

## PARTE 1 — Capas estáticas de suelo y geografía

### 1.1 Script de extracción — `componente_a/eda/build_capas_estaticas.py` (nuevo)

Calcado del patrón de `load_era5()` (`build_panel_union.py:703-752`), que ya resuelve
exactamente este problema: polígonos GAUL + `reduceRegions` + reintentos.

Reutilizar tal cual de `build_panel_union.py`:
- `_ee_init()` (`:445`) — auth con `EE_PROJECT = "tp-final-ml-499915"`
- `_gee_getinfo_retry(obj, descripcion)` (`:456`) — backoff exponencial
- `GAUL = "FAO/GAUL/2015/level2"` filtrado por `ADM0_NAME == "Argentina"`
- `_merge_satelital()` (`:761`) — join ASCII-insensible por `(provincia, departamento)`;
  hay que **derivar una variante sin `campania_inicio`** (`_merge_estatico`) porque estas
  capas no tienen eje temporal
- Convención de caché: `if out.exists(): return pd.read_parquet(out)`

**Diferencia clave con CHIRPS/ERA5:** no hay bucle temporal. El chunking necesario es
por *grupo de bandas* (no por fecha), porque 307 features × ~45 bandas en un solo
`getInfo` se acerca al límite de 5000 elementos. Un `getInfo` por propiedad
(≈307 elementos cada uno) es holgado y simple.

**Capas a extraer:**

| Fuente | Asset GEE | Salida |
|---|---|---|
| SoilGrids v2.0 | `projects/soilgrids-isric/{prop}_mean` para `bdod, cec, clay, sand, silt, nitrogen, phh2o, soc` | `suelo_{prop}_{0_5,5_15,15_30,30_60}` |
| SoilGrids agua | `ISRIC/SoilGrids250m/v2_0` (ImageCollection) → `wv0010, wv0033, wv1500` | `suelo_{wv}_{prof}` |
| SRTM | `USGS/SRTMGL1_003` banda `elevation` + `ee.Terrain.slope()` | `geo_elev_mean`, `geo_elev_std`, `geo_slope_mean` |
| HydroSHEDS | `WWF/HydroSHEDS/v1/FreeFlowingRivers`, filtrado por `RIV_ORD <= 5` | `geo_dist_rio_km` (distancia del centroide a la red) |

Profundidades: **0-5, 5-15, 15-30, 30-60 cm** (zona radicular soja/maíz).
`reduceRegion`/`reduceRegions` con `ee.Reducer.mean()` sobre el polígono departamental.

**Source-only:** el script guarda las bandas **crudas**, sin derivar nada. El
`agua_util_soilgrids = wv0033 - wv1500` y cualquier agregación por profundidad se
calculan en el notebook / en `data.py`, no en el ETL — igual que hoy las features
agronómicas viven en `add_agro_features` y no en el panel.

Caché intermedia: `data/processed/capas_estaticas_wide.parquet`.

**Descartado explícitamente** (documentar como limitación, junto a `ws2m` y napa
freática): `subregion_agro` (Pampa Ondulada / Interior) — el spec la plantea como un
mapeo manual de 26 valores, pero el panel tiene 307 deptos en 15 provincias; el mapeo
manual deja de ser de bajo esfuerzo y la clasificación de Soriano et al. no cubre las
provincias extra-pampeanas. Si se quiere, se aproxima post-hoc con `lat`/`lon`, que ya
están en el panel.

### 1.2 Integración al panel — `build_panel_union.py`

Agregar un paso 7 en `build_panel_union()` (`:776`), simétrico a los pasos NDVI/ERA5, con
flag `--skip-estaticas`. Merge por `(provincia, departamento)` únicamente → las columnas
se repiten en todas las campañas y cultivos del mismo departamento.

### 1.3 Consumo desde el modelo — `componente_a/src/{config,data}.py`

- `config.py`: nuevas constantes `SUELO_PROPS`, `SUELO_PROFS`, `GEO_COLS`.
- `data.py:36` `build_feature_list(panel, use_ndvi=True, use_era5=True, **use_suelo=False**)`.

**El default `use_suelo=False` es deliberado y no negociable en el primer paso:** si las
columnas nuevas entran en `build_feature_list`, `load_panel` y `crop_frame` las incluyen
en su `dropna(subset=clim_cols)` y **cambian `X_train`/`X_test`**, rompiendo los baselines
de `componente_a/tests/test_repro.py:30` y toda la comparabilidad con los notebooks ya
corridos. Con el flag apagado por defecto, agregar las columnas al parquet es
estrictamente aditivo y los tests siguen pasando sin tocar los hashes.

Cuando el ablation decida que suelo entra en la versión final, ahí sí se re-registra
`BASELINE` en `test_repro.py` con el comentario de rigor (ya hay precedente en el
docstring: *"Si cambiás el dataset a propósito, actualizá BASELINE"*).

Derivadas en `data.py` (una función nueva `add_suelo_features`, en la línea de
`add_agro_features` `:134`): `suelo_awc_{prof} = wv0033 - wv1500`, y agregados ponderados
por espesor a 0-30 cm para las variables químicas (`soc`, `nitrogen`, `phh2o`, `cec`,
`bdod`), para no inflar el feature set de 72 a ~120.

### 1.4 Validación

No hay `validate_panel.py`. Crear `componente_a/eda/validate_capas_estaticas.py` (script
corto, mismo estilo de logging que el builder):
- cobertura por departamento (esperado ≥95%; los deptos insulares/costeros pueden fallar)
- rangos plausibles: `clay+sand+silt ≈ 1000` (SoilGrids usa g/kg), `phh2o` en 3–10 (×10),
  `elev` en 0–5000, `slope` en 0–45, `wv0033 > wv1500` fila a fila
- constancia por departamento: `groupby([provincia,departamento])[col].nunique() == 1`

### 1.5 EDA — `componente_a/eda/eda_suelo_geografia.ipynb` (nuevo)

Estilo del repo: markdown en español, títulos `# N · Título`, secciones `## N.x ~ ...`,
cierre en `## Conclusión`. Contenido mínimo:
1. Cobertura y sanidad de las columnas nuevas (tabla, no prosa).
2. Correlación de suelo/geo con `z_rinde` (de `data.compute_z_rinde`), por cultivo.
3. Comparación contra el ranking de importancia climático ya obtenido — usar
   `agro_waterbal_crit` como referencia (**no `SPI_6_feb`: el SPI no existe en este repo**,
   grep vacío en todo el árbol).
4. Caveat obligatorio: SoilGrids sobreestima SOC ~2,4× en Argentina (Guevara et al. 2018)
   → tratar como **gradiente relativo entre departamentos**, no como valor absoluto.

### 1.6 Fase condicional — Índice de Productividad (solo si 1.5 muestra señal)

Instalar `geopandas shapely fiona pyproj`, bajar el shapefile del Atlas de Suelos
INTA-SAGyP-PNUD (GeoINTA, 1:500.000) e IDECOR para Córdoba (WFS, 1:50.000), intersectar
con GAUL nivel 2 y agregar **por área ponderada**, no por conteo de polígonos.
Cobertura esperada: ~3 de 15 provincias → nulls en el resto, a documentar.

---

## PARTE 2 — Dimensión temporal

### 2.1 Eje inter-campaña: momentum EWMA — `componente_b/momentum.py` (nuevo)

El esquema con factor γ del spec es exactamente un EWMA con `alpha = 1 - γ`:

```
M_t = γ · M_(t-1) + (1 - γ) · x_t
```

**Implementación anti-leakage (el detalle crítico):** la feature de la campaña `t` debe
ser `M_(t-1)`, no `M_t`. Por lo tanto el `shift(1)` va **después** del `ewm`, no antes:

```python
def ewma_pasado(s: pd.Series, gamma: float) -> pd.Series:
    """M_(t-1): EWMA construido SOLO con campañas estrictamente anteriores."""
    return s.ewm(alpha=1.0 - gamma, adjust=False).mean().shift(1)
```

Aplicado con `df.groupby(geo)[col].transform(...)`, sobre el `df` ya ordenado por
`geo + campania_inicio` que devuelve `crop_frame` (`componente_b/datos.py:108`) — ese
orden determinístico ya es una garantía del repo y acá se vuelve un requisito de
corrección, no solo de alineación.

API:
```python
def add_momentum(df, geo, gamma, señales=("z_rinde", "anomaly_score")) -> tuple[pd.DataFrame, list[str]]
```
Señales:
1. `momentum_z_rinde` — `x_t = z_rinde` de `_A_data.compute_z_rinde` (`componente_a/src/data.py:94`).
   Ya está resuelto el merge correcto en `componente_b/latente.py:106`, incluida la
   corrección del leak cruzado entre cultivos — **reutilizar ese merge, no reescribirlo**.
2. `momentum_anomaly_score` — `x_t` = score del VAE, vía `latente.vae_features(cultivo)`
   (`componente_b/latente.py:76`), que ya cachea en `componente_b/.latente_cache/`.
3. `momentum_embedding` — **no se implementa** salvo que el ablation de (1) y (2) lo pida.

NaN del arranque de cada serie: rellenar con 0 para `momentum_z_rinde` (ya está
estandarizado) y con la media de **train** para `momentum_anomaly_score` — mismo criterio
que ya usa `use_lags` (`componente_b/datos.py:186`).

**γ no es un hiperparámetro del modelo, es del dataset:** no puede entrar en el grid de
`evaluacion.buscar` (`componente_b/evaluacion.py:188`), que barre parámetros del
estimador. Se barre por afuera: construir un `RegDataset` por γ ∈ {0.5, 0.7, 0.85} y
elegir por `evaluacion.cv_score` (`:267`) sobre train. Documentar esto en el notebook.

Enganche en `datos.py`: nuevo kwarg `use_momentum: float | None = None` en
`build_reg_dataset` (`:128`) — `None` desactiva, un float es el γ.

### 2.2 Eje intra-campaña: 5 checkpoints — extender `_filter_momento`

Reemplazar la cadena de `if` de `componente_b/datos.py:87` por una regla de **mes de
corte** sobre `MESES = ["sep","oct","nov","dic","ene","feb","mar"]` (`config.py:27`),
aprovechando que todas las columnas climáticas ya tienen sufijo `_<mes>`:

```python
CHECKPOINTS = {           # nombre → mes de corte (None = nada observado aún)
    "pre_siembra": None,
    "nov":         "nov",
    "ene":         "ene",
    "pre_cosecha": "feb",
    "full":        "mar",
}
```

Regla: `<pfx>_<mes>` es observable si `MESES.index(mes) <= MESES.index(corte)`.
Los meses posteriores se **excluyen**, no se imputan.

Casos especiales (los estacionales de ERA5 no llevan sufijo de mes):
- `sm_winter` (Jun–Ago): observable en **todos** los checkpoints.
- `sm_planting` y `frost_days_early` (Sep–Nov): desde `"nov"` en adelante.
- `frost_days` (Sep–Mar): solo en `"full"`.
- `oni_*`: se mantienen en todos los checkpoints, respetando la convención ya
  documentada en `datos.py:78` (*en la práctica es el pronóstico ENSO*). **Es la única
  concesión de "información futura" del esquema y hay que decirlo explícitamente en el
  notebook**; queda como variante de ablation (`oni_solo_observado=True`).
- Suelo/geografía y `depto_enc`/`year`/lags/momentum: estáticas o pasadas → **todos** los
  checkpoints.

**Compatibilidad hacia atrás (verificada):** con esta regla, `pre_siembra` sigue dando
`oni_* + sm_winter`, `pre_cosecha` sigue excluyendo `*_mar` y `frost_days`, y `full` sigue
siendo todo. Los tres coinciden exactamente con el comportamiento actual → los resultados
del notebook `08_momentos_y_lags.ipynb` **no se mueven**, y `nov`/`ene` entran limpios.
Mantener `MOMENTOS` como alias de `tuple(CHECKPOINTS)`.

Restricción existente a preservar: `momento != "full" and use_agro` sigue siendo error
(`datos.py:147`) — las ventanas críticas agronómicas llegan a marzo.

### 2.3 Entrenamiento — opción (a) del spec: un XGBoost por checkpoint

10 modelos = 2 cultivos × 5 checkpoints. Reutilizar sin cambios:
- `componente_b/modelos/xgboost_model.py:35` `XGBoostRegressor`
- `componente_b/evaluacion.py:188` `buscar(...)` con `n_splits=4` (walk-forward
  `TimeSeriesSplit` expandido, `_temporal_folds` `:147`)
- `fold_X` (`:156`), que ya re-ajusta el target encoding por fold

Punto de atención: `fold_X` recalcula `depto_enc` por fold, **pero el momentum se calcula
una vez sobre todo el `df`**. Esto es correcto — el EWMA con `shift(1)` solo mira el
pasado de la propia serie, así que no filtra información del fold de validación. Dejarlo
argumentado en el notebook.

La opción (b) (modelo único con feature de checkpoint + máscara) **no se implementa**.

### 2.4 Componente A: sin cambios

Se mantiene un único score por campaña completa. No se rediseña el AE/VAE a arquitectura
secuencial (LSTM-AE / TCN-AE). En los checkpoints tempranos de B, lo que se usa es el
`momentum_anomaly_score` histórico, no un score de la campaña en curso.

### 2.5 Notebooks nuevos — `componente_b/experimentos/`

- `11_momentum_y_checkpoints.ipynb` — barrido de γ, tabla por checkpoint, chequeo de
  sanidad de monotonicidad.
- `12_ablations_finales.ipynb` — la grilla completa de ablations.

Estilo B: celda 0 `# NN — Título` (em dash), celda 1 setup con
`sys.path.insert(0, os.path.abspath('..'))` + `CULTIVOS = ['soja','maiz']` + `DSS`,
resultados siempre vía `evaluacion.tabla` (`:251`), cierre en `## Conclusión`.

---

## Orden de ejecución

| # | Paso | Depende de |
|---|---|---|
| 1 | Instalar `xgboost` + `earthengine-api`, actualizar `requirements.txt` | — |
| 2 | `build_capas_estaticas.py` (SoilGrids + SRTM + HydroSHEDS) | 1 |
| ~~3~~ | ~~`validate_capas_estaticas.py` + integración al ETL~~ **HECHO** — paso 6c + `_merge_estatico`, panel 84 → 132 columnas, `test_repro` verde sin re-baselinear | 2 |
| ~~4~~ | ~~`build_feature_list(use_suelo=...)` + `add_suelo_features`~~ **HECHO** — 13 features derivadas, `use_suelo` también en `componente_b/datos.py`, `test_repro` verde sin re-baselinear | 3 |
| ~~5~~ | ~~`eda_suelo_geografia.ipynb` — gate de decisión~~ **HECHO — GATE PASADO**: suelo entra en B con `use_suelo=True`. Mejora chica pero sistemática (soja +0,34 %, maíz +0,97 % de CV-RMSE, 5/5 semillas) y **creciente hacia los checkpoints tempranos** (pre-siembra: +1,4 % / +3,3 %) | 4 |
| ~~6~~ | ~~`componente_b/momentum.py` + kwarg `use_momentum`~~ **HECHO — RESULTADO NEGATIVO**: el EWMA es redundante con `use_lags`, que ya existía. Ver abajo | 1 |
| ~~7~~ | ~~`CHECKPOINTS` + `_filter_momento`~~ **HECHO** — 5 checkpoints por regla de mes de corte; los 3 momentos viejos devuelven columnas idénticas | — |
| ~~8~~ | ~~XGBoost por checkpoint~~ **HECHO** — `11_checkpoints_y_momentum.ipynb`, ejecutado. Test RMSE baja 15,9 % (soja) y 14,3 % (maíz) de pre-siembra a full | 6, 7 |
| ~~9~~ | ~~`12_ablations_finales.ipynb`~~ **HECHO** — grilla de 80 configuraciones. Orden de importancia: **checkpoint ≫ memoria ≫ suelo**. Ver abajo | 5, 8 |

### Resultado del paso 9 — el ablation final

Grilla completa: 2 cultivos × 5 checkpoints × 4 niveles de memoria × suelo on/off.
Moviendo cada eje de su peor a su mejor valor (CV-RMSE):

| eje | soja | maíz |
|---|---|---|
| **checkpoint** | 12,1 % | 14,1 % |
| memoria (lags) | 0,9 % | 5,6 % |
| suelo/geo | 0,1 % | 1,1 % |

**Límite encontrado: el modelo no sirve en pre-siembra.** Contra la climatología
departamental (predecir la media histórica del depto), en pre-siembra maíz **pierde**
(−1,9 %) y soja apenas empata (+3,3 %). La ventaja recién se vuelve clara desde noviembre
(+5 % a +13 %) y llega a +12,7 % / +18,6 % con la campaña completa. Contra la media global
gana siempre, pero ése es el rival fácil.

Es un límite del enfoque, no un defecto de la implementación: antes de sembrar las únicas
señales son ONI, humedad invernal, el histórico del departamento y el suelo. Para el caso
de uso de producto, lo honesto en pre-siembra es responder con la climatología del
departamento y reservar el modelo para cuando ya hay clima observado.

### Resultado del paso 6 — el momentum EWMA no aporta

Se implementaron y midieron las **tres** señales del spec, con XGBoost tuneado, walk-forward
CV de 4 folds y 3 semillas, sobre el dataset completo (clima + agro + suelo):

| variante | soja | maíz |
|---|---|---|
| `momentum_z_rinde` solo, vs. sin memoria | +0,22 % | +0,79 % |
| **`use_lags=3` solo** (ya existía en el repo) | **+0,98 %** | **+6,44 %** |
| lags + `momentum_z_rinde` | +0,92 % | +6,46 % |
| lags + EWMA sobre rinde crudo | −0,36 … −0,08 % | −0,05 … +0,50 % |
| lags + `momentum_anomaly_score` | −0,32 … −0,16 % | −0,01 … +0,11 % |

**La memoria inter-campaña existe y es grande** (en maíz vale 6,4 % de RMSE), **pero ya
está capturada por `rinde_lag1..3` + `rinde_ma5`.** Reformularla como EWMA no agrega nada
medible, en ninguna de las tres señales ni en ningún gamma.

La explicación es conceptual: `momentum_z_rinde` promedia un z-score normalizado por
departamento, o sea que descarta el *nivel* de rinde — y el nivel es lo que predice el
nivel. Los lags son rinde crudo en kg/ha. Por eso también se probó el EWMA sobre rinde
crudo, que sí conserva el nivel: tampoco aporta sobre `rinde_ma5`.

Decisiones que se derivan:
- `use_momentum` queda implementado y testeado, con **default `None`**. Sirve para
  documentar el resultado negativo en el ablation del paso 9, no para el modelo final.
- **No se implementa `momentum_embedding`.** El spec lo condicionaba a que las señales
  escalares no capturaran suficiente señal; el caso real es distinto y peor para la idea:
  la señal está, pero ya la toman features más simples que están hace rato en el repo.
- El eje inter-campaña del requisito de producto **se considera cubierto por `use_lags`**.

**Resultado del gate (paso 5), que condiciona los pasos siguientes:** suelo aporta señal
**de nivel entre departamentos**, no de anomalía intra-departamento, y su contribución
**crece cuanto menos clima observado hay**. Consecuencias concretas:
- El ablation de suelo va en **Componente B**, no en A (en A las estáticas se anulan).
- En el paso 8, los checkpoints tempranos son los que más ganan con `use_suelo=True`:
  conviene reportar la contribución de suelo **por checkpoint**, no solo agregada.
| 10 | (Condicional) IP INTA/IDECOR con geopandas | 5 en positivo |

Los pasos 2–5 y 6–8 son **independientes entre sí** y se pueden hacer en paralelo; solo
convergen en el paso 9.

---

## Verificación

**Parte 1**
```bash
.venv/bin/python componente_a/eda/build_capas_estaticas.py
.venv/bin/python componente_a/eda/validate_capas_estaticas.py
.venv/bin/python -m pytest componente_a/tests/ -q     # los hashes NO deben moverse
```
Criterios de aceptación:
- cobertura ≥95% de los 307 deptos, sin nulls inesperados
- `wv0033 > wv1500` en el 100% de las filas
- cada columna nueva es constante dentro de cada `(provincia, departamento)`
- `test_repro.py` pasa **sin tocar `BASELINE`** (prueba de que la expansión fue aditiva)

**Parte 2**
```bash
.venv/bin/python -c "
from componente_b import datos
for m in ['pre_siembra','nov','ene','pre_cosecha','full']:
    ds = datos.prepare('soja', momento=m)
    print(f'{m:12} {len(ds.feature_cols):3d} features')
"
```
Criterios de aceptación:
- el nº de features crece monótonamente de `pre_siembra` a `full`
- `pre_siembra`, `pre_cosecha` y `full` dan **exactamente** el mismo `feature_cols` que
  antes del cambio (diff contra la rama base)
- ninguna columna de un mes posterior al corte aparece en el checkpoint
- test de no-leakage del momentum: para un depto sintético con `z_rinde = [1,2,3,4]`, la
  feature de la fila `t` no puede depender de `x_t` (perturbar `x_t` y verificar que
  `momentum_z_rinde[t]` no cambia)
- **chequeo de sanidad final:** el RMSE de test mejora (o al menos no empeora) de
  `pre_siembra` → `full`. Si no es monótono, hay algo mal en el enmascarado.

**Ablations (paso 9)** — sobre RMSE (B) y PR-AUC (A→B), por cultivo:
suelo/geo on–off × momentum on–off × (sin lags / solo lags / lags+EWMA) × 5 checkpoints.

---

## Riesgos conocidos

1. **Cuota de GEE.** 307 deptos × ~45 bandas. Mitigado con un `getInfo` por propiedad y el
   `_gee_getinfo_retry` existente. Si aun así falla, chunkear por provincia.
2. **SoilGrids en Argentina** tiene habilidad local limitada (SOC sobreestimado ~2,4×).
   Documentado; se usa como gradiente relativo.
3. **Explosión del feature set.** 72 → ~120 si entran las 4 profundidades crudas. Mitigado
   con `add_suelo_features` (agregados 0-30 para las químicas) y con el gate del EDA.
4. **`test_repro.py` congelado.** Cualquier cambio en `build_feature_list` que aplique por
   defecto rompe los baselines. De ahí el `use_suelo=False`.
5. **`TRAIN_END`/`TEST_START` duplicados** en `componente_b/datos.py:46-47` en vez de
   importados de `_A_config` (que ya está en scope). Deuda preexistente; no la tocamos en
   esta iteración, pero conviene no agregar más constantes duplicadas.
