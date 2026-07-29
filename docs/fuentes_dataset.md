# Fuentes del dataset `panel_union.parquet`

El panel es un merge de **8 fuentes públicas**, construido por
`componente_a/eda/build_panel_union.py` (el ETL canónico, compartido por ambos
componentes). Granularidad: **departamento × campaña × cultivo** (soja y maíz).
Tamaño actual: **~27.865 filas × 132 columnas**, campañas 1981–2024; al cargarlo,
`src/data.py` deduplica las filas espurias del join de centroides por nombre
(→ ~20.672 filas efectivas). No se imputa ningún rinde.

Cobertura geográfica: región núcleo (centro-sur de Santa Fe, sudeste de Córdoba,
norte de Buenos Aires) más extensión a todos los departamentos con al menos
`MIN_CAMPANAS=20` campañas de rinde válido (sin restricción núcleo hardcodeada;
la columna `region` marca núcleo vs. resto para ablations). Son **307
departamentos de 15 provincias**, de los cuales **25 son de la región núcleo**.

> **Limitación conocida — Villa Constitución (Santa Fe).** Es uno de los 26 deptos
> del núcleo, pero MAGyP lo escribe `'Villa Constitución'` y `CENTROIDES_NUCLEO`
> lo tiene sin acento: el merge de centroides no lo matchea, se queda sin lat/lon
> y se descarta. Por eso el panel tiene 25 y no 26. Recuperarlo exige normalizar
> ese merge, lo que agrega filas y obliga a re-registrar el baseline de
> `tests/test_repro.py`.

---

## 1. MAGyP — Rendimientos agrícolas (target)

- **Fuente**: `datos.magyp.gob.ar`, estimaciones agrícolas departamentales.
- **Columnas**: `rinde_kgha` (**target** del Componente B), `sup_sembrada_ha`,
  `sup_cosechada_ha`, `produccion_tn`.
- **Filtro**: series con ≥20 campañas por (depto, provincia, cultivo); sin imputación.
- **Granularidad**: define la del panel — departamento × campaña × **cultivo**.

## 2. ONI — Índice Oceánico Niño (ENSO)

- **Fuente**: NOAA, vía GitHub (`ahuang11/oni`), serie mensual.
- **Columnas**: `oni_oct`, `oni_nov`, `oni_dic`, `oni_ene`, `oni_feb` (5 meses).
- **Granularidad**: serie global — el mismo valor se repite para todos los
  departamentos × cultivos de cada campaña (el ENSO no varía a esta escala).

## 3. NASA POWER — Clima diario agregado

- **Fuente**: API REST `power.larc.nasa.gov`, un request por centroide departamental.
- **Variables (7)**: `T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `RH2M`,
  `ALLSKY_SFC_SW_DWN`, `WS2M`.
- **Agregación**: promedio mensual, meses Sep–Mar → **7 × 7 = 49 columnas**.
- **Cobertura**: ~98%; los NaN se concentran en la **radiación solar** de las
  campañas más tempranas (1981–1983) — por eso el train del Componente B arranca
  en 1984 tras el `dropna` de features.
- **Granularidad**: departamento × campaña (no depende del cultivo).
- ⚠️ **Caveat**: la caché de NASA POWER se indexa por **nombre** de departamento,
  así que departamentos homónimos de provincias distintas comparten serie climática.

## 4. CHIRPS — Precipitación satelital

- **Fuente**: Google Earth Engine (`UCSB-CHG/CHIRPS/DAILY`), agregada al polígono
  de cada departamento.
- **Columnas**: `chirps_precip_sep` … `chirps_precip_mar` (**7 meses**).
- **Rol**: fuente de precipitación independiente de NASA POWER; incorporada al
  set de features (`CLIM_PREFIXES`), lleva la X del pipeline a 72 features base.

## 5. NDVI — Índice de vegetación (AVHRR/VIIRS)

- **Fuente**: Google Earth Engine, uniendo `NOAA/CDR/AVHRR/NDVI/V5` (1981–2013)
  con `NOAA/CDR/VIIRS/NDVI/V1` (2014+); compuesto de máximo mensual promediado
  sobre el polígono departamental (FAO GAUL nivel 2).
- **Columnas**: `ndvi_avhrr_sep` … `ndvi_avhrr_mar` (**7 meses**).
- **Por qué AVHRR y no MODIS**: el NDVI MODIS del pipeline viejo se **retiró**
  (2026-07) — solo cubría 2002+ (recortaba a la mitad el historial 1981–2001, que
  incluye campañas clave como 1988/89) y 3 de sus 4 columnas eran estáticas por
  depto. AVHRR/VIIRS da cobertura mensual completa **1981–2024** con señal temporal.

## 6. ERA5-Land — Estado del suelo y heladas

- **Fuente**: Google Earth Engine (`ECMWF/ERA5_LAND/DAILY_AGGR`).
- **Columnas (4)**: `sm_planting` (humedad de suelo en siembra, Sep–Nov),
  `sm_winter` (humedad de suelo de invierno, Jun–Ago), `frost_days` (días de
  helada Sep–Mar) y `frost_days_early` (heladas tempranas Sep–Nov).
- **Rol**: `sm_winter` es de las pocas señales *anticipables* pre-siembra (ver el
  notebook de momentos del Componente B).

## 7. SoilGrids v2.0 (ISRIC) — Propiedades de suelo

- **Fuente**: Google Earth Engine. Ocho propiedades desde las Images
  `projects/soilgrids-isric/{prop}_mean` (`bdod`, `cec`, `clay`, `sand`, `silt`,
  `nitrogen`, `phh2o`, `soc`) y tres contenidos volumétricos de agua desde la
  ImageCollection `ISRIC/SoilGrids250m/v2_0` (`wv0010`, `wv0033`, `wv1500`).
  Media sobre el polígono departamental (FAO GAUL nivel 2), 250 m.
- **Columnas (44)**: `suelo_<prop>_<prof>` para las profundidades **0-5, 5-15,
  15-30 y 30-60 cm** (zona radicular de soja/maíz; no se baja a 60-200 cm).
- **Estáticas**: no varían por campaña ni por cultivo. Se repiten en todas las
  filas del departamento.
- **Tres trampas, todas verificadas contra GEE** (están comentadas en el script):
  1. Los `wv*` **no** existen como Image suelta y sus bandas se llaman
     `val_<prof>cm_mean`, no `wv0033_<prof>_mean`.
  2. Los assets `projects/soilgrids-isric/*` están en proyección **World_Mollweide**:
     hay que pasar `crs="EPSG:4326"` al `reduceRegions` o devuelve `None` en casi
     toda Argentina **sin lanzar error**.
  3. Las dos familias escalan distinto: los assets de ISRIC dan enteros escalados
     (arcilla 275 g/kg → ×0.1 = 27,5 %), la colección curada por GEE ya da floats
     en unidades nativas (wv0033 = 0,328 cm³/cm³ → ×100 = 32,8 % vol).
- **Caveat de calidad**: SoilGrids tiene habilidad local limitada en Argentina —
  sobreestima el SOC ~2,4× frente a estimaciones país-específicas (Guevara et al.
  2018). **Tratar como gradiente relativo entre departamentos, no como valor
  absoluto de campo.** Lo mismo aplica al nitrógeno, que en suelos hidromórficos
  (bahía de Samborombón) llega a ~80 g/kg.
- **Agua útil**: `wv0033 - wv1500` NO se calcula en el ETL (convención source-only);
  va en `add_suelo_features`, con `clip(lower=0)` porque SoilGrids predice cada
  profundidad de forma independiente y en 7 deptos de Misiones se cruzan.

## 8. SRTM + HydroSHEDS — Geografía

- **Fuente**: Google Earth Engine, `USGS/SRTMGL1_003` (banda `elevation` +
  `ee.Terrain.slope()`) y `WWF/HydroSHEDS/v1/FreeFlowingRivers` filtrado por
  `RIV_ORD <= 5` (deja Paraná, Carcarañá y cursos relevantes).
- **Columnas (4)**: `geo_elev_mean`, `geo_elev_std` (proxy de rugosidad),
  `geo_slope_mean`, `geo_dist_rio_km`.
- La pendiente se reduce a 250 m, no a los 30 m nativos: en la Pampa la diferencia
  es despreciable para una media departamental y el reduce es mucho más barato.
- La distancia al río se promedia **sobre el polígono**, no desde el centroide,
  para quedar consistente con el resto de las capas.

**Extracción y validación**: las capas 7 y 8 NO se extraen dentro de
`build_panel_union.py`. Las produce `componente_a/eda/build_capas_estaticas.py`
(se corre una sola vez, son estáticas) y el paso 6c del ETL solo mergea ese
parquet. `componente_a/eda/validate_capas_estaticas.py` chequea cobertura, rangos
físicos y constancia por departamento — importante porque los fallos de GEE son
silenciosos. A diferencia del satelital, este merge **no descarta filas**: la
expansión es estrictamente aditiva para no mover `X_train`/`X_test`.

**Explícitamente descartado**: napa freática departamental (no existe dataset
público continuo; la base NAPA de CONICET/INTA son puntos dispersos y sesgados a
Pampa Interior) y el Índice de Productividad del Atlas de Suelos INTA (requiere
geopandas y solo cubre 3 de las 15 provincias del panel).

---

## Del panel a las features

- Los centroides departamentales salen de 26 deptos núcleo con coordenadas fijas
  más geocodificación vía Nominatim/OSM (con caché) para el resto.
- Las fuentes satelitales (CHIRPS, NDVI-AVHRR, ERA5) se extraen de Earth Engine
  dentro del mismo ETL (requiere auth de GEE; hay flags `--skip-chirps` /
  `--skip-satelital`). Las filas sin cobertura satelital se descartan.
- El **feature engineering** (features agronómicas de ventana crítica, lags del
  rinde, normalización por depto) vive en `componente_a/src/data.py` y
  `componente_b/datos.py`, no en el ETL — el panel guarda las columnas crudas.
