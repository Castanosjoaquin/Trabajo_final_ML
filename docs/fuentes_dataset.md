# Fuentes del dataset `panel_union.parquet`

El panel es un merge de **6 fuentes públicas**, construido por
`componente_a/eda/build_panel_union.py` (el ETL canónico, compartido por ambos
componentes). Granularidad: **departamento × campaña × cultivo** (soja y maíz).
Tamaño actual: **~27.865 filas × 84 columnas**, campañas 1981–2024; al cargarlo,
`src/data.py` deduplica las filas espurias del join de centroides por nombre
(→ ~20.672 filas efectivas). No se imputa ningún rinde.

Cobertura geográfica: región núcleo (centro-sur de Santa Fe, sudeste de Córdoba,
norte de Buenos Aires) más extensión a todos los departamentos con al menos
`MIN_CAMPANAS=20` campañas de rinde válido (sin restricción núcleo hardcodeada;
la columna `region` marca núcleo vs. resto para ablations).

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
