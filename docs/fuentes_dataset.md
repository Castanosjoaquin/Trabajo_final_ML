# Fuentes del dataset `panel_nucleo.parquet`

El panel es un merge puro de 4 fuentes públicas, sin normalización ni feature
engineering. Granularidad: **departamento × campaña × cultivo** (~2287 filas × 70 columnas).

---

## 1. MAGyP — Rendimientos agrícolas

- **Fuente**: `datos.magyp.gob.ar`, serie histórica departamental (1941–2024)
- **Filtro aplicado**: 26 departamentos de la región núcleo (BCR), `campania_inicio >= 1981`
- **Columnas**: `rinde_kgha` (target del Componente B), `sup_sembrada_ha`, `sup_cosechada_ha`, `produccion_tn`
- **Cobertura**: 100%, sin NaN
- **Granularidad real**: departamento × campaña × cultivo → es la fuente que define la granularidad del panel completo

---

## 2. ONI — Índice Oceánico Niño (ENSO)

- **Fuente**: NOAA, vía GitHub (`ahuang11/oni`), serie mensual 1950–2025
- **Procesamiento**: para cada campaña se extrae el valor mensual crudo del índice
- **Columnas**: `oni_oct`, `oni_nov`, `oni_dic`, `oni_ene`, `oni_feb` (anomalía SST mensual)
- **Cobertura**: 100%
- **Granularidad real**: una sola serie temporal global. El mismo valor se repite
  para los 26 departamentos × 2 cultivos de cada campaña — el ENSO es un fenómeno
  climático regional/global, no varía geográficamente a esta escala
- **Feature engineering (en EDA, no en ETL)**: `oni_oct_feb_mean`, `oni_oct_feb_min`,
  categoría Niña/Niño/Neutro — se calculan a partir de estas 5 columnas

---

## 3. NASA POWER — Clima diario agregado

- **Fuente**: API REST `power.larc.nasa.gov`, un request por centroide departamental
- **Variables crudas**: `T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `WS2M` (diarias, 1981–2024)
- **Agregación**: promedio mensual de cada variable para los meses Sep–Mar de la campaña
- **Columnas**: 49 en total (7 variables × 7 meses: `t2m_sep` … `ws2m_mar`)
- **Cobertura**: ~98% (1.8% NaN, principalmente en columnas de radiación solar para campañas muy tempranas)
- **Granularidad real**: departamento × campaña (no depende del cultivo — el clima es el mismo para soja y maíz del mismo depto/campaña)
- **Feature engineering (en EDA, no en ETL)**: GDD, días T>35°C, precipitación por fase
  fenológica (siembra/vegetativo/R1-R5/llenado), tmean, tmax_p95, rad_solar_mean, etc. —
  se calculan a partir de las columnas mensuales crudas

---

## 4. NDVI — Índice de vegetación (MODIS)

- **Fuente**: HDX/WFP, satélite MODIS C6.1, archivo `arg-ndvi-adm2-full.csv`
- **Procesamiento**: filtro admin2, mapeo PCODE→departamento, agregación oct–mar por campaña
- **Columnas**: `ndvi_mean`, `ndvi_min`, `ndvi_max`, `ndvi_anomalia_pct`
- **Granularidad real**: departamento × campaña

### ⚠️ El problema de cobertura

**MODIS empezó a operar en 2002.** Esto significa que para las campañas
1981/82 a 2001/02 (21 de 44 campañas, ~52% del panel) **el NDVI no existe
físicamente** — no es un dato faltante por error de descarga o de merge, es
una limitación de la fuente satelital.

| Período | Cobertura NDVI |
|---|---|
| 1981–2001 (21 campañas) | 0% — NaN total |
| 2002–2024 (23 campañas) | ~92% |
| **Total panel** | **~48%** |

### Implicancias para el modelado

No hay forma de "rellenar" estos NaN sin inventar datos — cualquier imputación
para 1981-2001 sería ficticia. Las dos alternativas reales son:

1. **Excluir NDVI** del set de features para el dataset completo (1981–2024)
2. **Usar NDVI solo en el subconjunto 2002–2024**, como ablation explícito —
   esto es justamente lo que pide el enunciado en la sección de ablations
   ("Componente B: aporte marginal de NDVI")

La decisión recomendada es la (2): en el dataset completo, el AE y el
predictor "base" no usan NDVI (evita perder la mitad del historial 1981–2001,
que incluye campañas clave como 1988/89). Para el ablation de NDVI, se filtra
`df[df.campania_inicio >= 2002]` y se comparan métricas con/sin las 4 columnas
NDVI sobre ese subconjunto reducido (~1100 filas).

Esta separación se hace en el notebook de modelado, no en el ETL — el panel
mantiene las 4 columnas NDVI con sus NaN tal cual vienen de la fuente.
