# Resumen del proyecto: predicción y detección de anomalías de rinde agrícola

Proyecto final de Machine Learning sobre rinde de **soja y maíz** en Argentina (con foco original en la región núcleo). El trabajo tiene tres partes: la construcción de un dataset panel a partir de fuentes públicas, un **Componente A** de detección no supervisada de campañas anómalas, y un **Componente B** de regresión supervisada del rinde.

---

## 1. Generación del dataset

### Fuentes de datos

El dataset se construye combinando cinco fuentes públicas, cada una con su script en `data_sources/` o en `componente_a/eda/build_panel_union.py` (el builder canónico):

- **MAGyP (rindes)** — Estimaciones agrícolas de `datos.magyp.gob.ar` (CSVs `magyp_soja.csv` / `magyp_maiz.csv`). Aportan el target `rinde_kgha` por departamento y campaña. Se filtran las series con al menos 20 campañas por (departamento, provincia, cultivo) y **no se imputa** ningún valor.
- **NASA POWER (clima diario)** — API REST de `power.larc.nasa.gov`. Se descargan 7 variables (T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, ALLSKY_SFC_SW_DWN, WS2M) en el centroide de cada departamento y se agregan a valores mensuales para los 7 meses de la campaña (septiembre a marzo): **7 × 7 = 49 columnas**.
- **ONI (ENSO)** — Índice Oceánico de El Niño de NOAA. Se usan 5 meses (oct, nov, dic, ene, feb): **5 columnas**.
- **NDVI AVHRR/VIIRS** (`data_sources/extract_avhrr_ndvi.py`) — vía **Google Earth Engine**, uniendo `NOAA/CDR/AVHRR/NDVI/V5` (1981–2013) con `NOAA/CDR/VIIRS/NDVI/V1` (2014+). Compuesto de máximo mensual (MVC) promediado sobre el polígono de cada departamento (FAO GAUL nivel 2), meses sep–mar: **7 columnas**. Se eligió AVHRR en lugar de MODIS justamente para no recortar el train a 2002+ (el NDVI MODIS original fue retirado del pipeline: solo existía desde 2002 y 3 de sus 4 columnas eran estáticas por departamento).
- **ERA5-Land** (`data_sources/extract_era5.py`) — vía Earth Engine (`ECMWF/ERA5_LAND/DAILY_AGGR`): **4 features** por depto × campaña: humedad de suelo en siembra (`sm_planting`, sep–nov) y en invierno (`sm_winter`, jun–ago), días de helada (`frost_days`, sep–mar) y heladas tempranas (`frost_days_early`, sep–nov).

Los centroides departamentales salen de 26 departamentos núcleo con coordenadas fijas más geocodificación vía Nominatim/OSM para el resto.

### Merge y variantes del panel

`componente_a/eda/build_panel_union.py` produce el panel base `data/processed/panel_union.parquet` (**28.683 filas × 66 columnas**, campañas 1981–2024). Los scripts `merge_avhrr_ndvi.py` y `merge_era5.py` agregan las columnas satelitales de forma no destructiva, generando dos variantes:

- `panel_union_ndvi.parquet` — 28.683 × 73 (+7 columnas `ndvi_avhrr_<mes>`).
- `panel_union_era5.parquet` — 28.683 × 70 (+4 columnas ERA5).

Al cargar los datos (`componente_a/src/data.py`) se hace un **dedup** por (provincia, departamento, campaña, cultivo) que elimina duplicados espurios del join geográfico: 28.683 → **21.418 filas** efectivas.

### EDA y feature engineering

El análisis exploratorio (`componente_a/eda/*.ipynb` y `componente_b/experimentos/00_eda_rinde_y_features.ipynb`) motivó los cambios principales del pipeline:

- Retiro del NDVI MODIS (sin señal temporal, cobertura corta) y su reemplazo por AVHRR/VIIRS con cobertura completa 1981–2025.
- Creación de **4 features agronómicas de "ventana crítica"** (`add_agro_features`): precipitación crítica, temperatura máxima crítica, amplitud térmica y balance hídrico (con PET proxy Hargreaves). Son opcionales (`use_agro`).
- Definición de **zonas geográficas** balanceadas vía KMeans sobre lat/lon, usadas luego en el Componente B.

### Features finales y granularidad

- Base climática: **54 features** (49 de NASA POWER + 5 de ONI).
- Aumentos: +7 NDVI AVHRR, +4 ERA5-Land, +4 agro (opcionales).
- El Componente B agrega además **`depto_enc`** (target encoding del rinde medio por departamento en train) y **`year`** (año de inicio de campaña, para capturar la tendencia tecnológica).

**Cada fila del dataset representa un departamento × una campaña agrícola × un cultivo (soja o maíz)**, con su rinde observado y el resumen climático/satelital de esa campaña.

---

## 2. Componente A: detección no supervisada de campañas anómalas

### Objetivo

Detectar, sin usar etiquetas en el entrenamiento, las campañas en las que el rinde de un departamento cae de forma anómala (típicamente sequías u otros eventos extremos).

### Pipeline

1. **Etiqueta proxy (solo para evaluar)**: `z_rinde` = z-score del rinde contra su media móvil de **5 años con `shift(1)`** (la ventana no incluye el año actual), por (provincia, departamento, cultivo). Una campaña es anómala si `z_rinde < −1.5`.
2. **Split temporal**: train ≤ campaña 2020/21, test ≥ 2021/22. Se excluyen del train los años de sequía generalizada **1988, 1996, 2008 y 2017** para que el modelo aprenda solo lo "normal".
3. **Normalización por departamento**: z-score por depto con media y desvío calculados **solo sobre las filas normales del train** (sin leakage), con fallback al desvío global cuando el desvío local es 0 (p. ej. `frost_days = 0` en zonas sin heladas).

### Modelos implementados (`componente_a/src/models/`)

- **Baselines clásicos**: `IsolationForest` y `OneClassSVM` (kernel RBF).
- **Autoencoders**: AE estándar y **Denoising AE** (ruido gaussiano o salt-and-pepper), score por error de reconstrucción.
- **VAE** con decoder **gaussiano o Student-t** y tres modos de score: `recon_error` (MSE), `recon_prob` (probabilidad de reconstrucción, An & Cho 2015, por Monte Carlo) y `neg_elbo`. Soporta β-VAE.
- **Ensembles**: seed-ensemble (mismo modelo, N semillas) y hetero-ensemble, con normalización z-score o por rango.
- **Baselines profundos modernos** (librería `deepod`): ICL, DeepSVDD, GOAD, NeuTraL.

### Qué se descubre en cada notebook (`experiments/00–07`)

- **00 — Datos y pipeline**: construcción del pipeline reproducible (carga, dedup, proxy label, split, normalización). Tamaños: soja train (6.377 × 54) con 257 anómalas en test; maíz train (8.347 × 54) con 349 anómalas.
- **01 — Evaluación y baselines**: IsolationForest tuneado ≈ 0.51–0.52 de PR-AUC; OCSVM-RBF ≈ 0.477.
- **02 — AE y DAE**: AE = 0.353, DAE = 0.399. El error MSE "plano" no alcanza; el denoising no cambia el cuello de botella.
- **03 — VAE**: VAE `recon_prob` single ≈ 0.587. El salto viene del **score probabilístico**, no de la arquitectura.
- **04 — Ensemble**: seed-ensemble ×10 del VAE (solo clima) → **0.616**, con mucha menos varianza entre corridas.
- **05 — Techo estructural**: ningún modelo supera ~0.6 con las 54 features climáticas; un PCA del test muestra que muchas anomalías no se separan del resto — **la señal no está en las features**, no es un problema de modelo.
- **06 — Features nuevas**: agro, NDVI-AVHRR y ERA5. Ayudan al IForest, pero en el VAE la mejora **solo se ve con el seed-ensemble** (single-seed es demasiado ruidoso para detectarla).
- **07 — Leaderboard y conclusiones**: comparación final con presupuesto parejo; ICL y NeuTraL se descartan por rendir al nivel del azar; DeepSVDD queda como referencia moderna.

### Resultado final (soja, test ≥ 2021, semillas 42–51)

| Modelo | PR-AUC | ROC-AUC |
|---|---:|---:|
| **VAE recon_prob seed-ensemble ×10 + ERA5+NDVI** ★ | **0.640 ± 0.003** | 0.783 |
| VAE seed-ensemble ×10 (solo clima) | 0.616 ± 0.005 | 0.756 |
| VAE recon_prob (single) | 0.587 ± 0.039 | 0.737 |
| IsolationForest | 0.517 | 0.696 |
| DeepSVDD | 0.512 | 0.759 |
| OneClassSVM-RBF | 0.477 | 0.706 |
| Denoising AE | 0.399 | 0.659 |
| AE | 0.353 | 0.579 |

**Conclusión**: el modelo aplicado es el **VAE con score `recon_prob`, seed-ensemble ×10, entrenado con clima + ERA5 + NDVI**, con **PR-AUC ≈ 0.640** en soja test. El techo estructural (~0.6) existe porque buena parte de las anomalías de rinde no tienen firma climática observable (manejo, plagas, granizo, ruido de MAGyP), pero las features de suelo y verdor lo corren un poco hacia arriba.

---

## 3. Componente B: regresión supervisada del rinde

### Objetivo

Predecir el **rinde en kg/ha** (`rinde_kgha`) de cada departamento × campaña a partir del clima y las features satelitales.

### Preparación del dataset (`componente_b/datos.py::build_reg_dataset`)

Reutiliza la carga, dedup y lista de features del Componente A, y agrega:

- **Split temporal** idéntico: train ≤ 2020, test ≥ 2021.
- Elección de dataset: solo clima (54 features) o clima + NDVI + ERA5 (`dataset='era5_ndvi'`).
- **`depto_enc`**: target encoding — rinde medio del departamento en train (deptos nuevos en test reciben la media global).
- **`year`**: año de campaña, para la tendencia tecnológica del rinde.
- **Escalado**: `StandardScaler` ajustado solo en train.
- Opcional `use_agro=True`: suma las 4 features agronómicas de ventana crítica.

### Modelos (`componente_b/modelos/`)

- `LinearRegressor` unificado: OLS / **Ridge / Lasso / ElasticNet** según el parámetro `penalty`.
- `XGBoostRegressor` (con early stopping opcional).
- `NeuralNetRegressor`: MLP en PyTorch con target estandarizado, dropout, weight decay, scheduler de learning rate y early stopping con validación interna.

### Metodología de evaluación (`componente_b/evaluacion.py`)

- Métricas: MAE, RMSE, R², MAPE.
- **Baselines**: media global del train (`pred_media`) y **media por departamento** (`pred_media_depto`, una "climatología" de referencia — todo modelo debe superarla para demostrar que el clima del año aporta información real).
- **`buscar()`**: búsqueda de hiperparámetros con **validación cruzada temporal expansiva** (`TimeSeriesSplit` sobre años ordenados, 4 folds) exclusivamente dentro del train; el test nunca se toca hasta la evaluación final.

### Resultados (soja, test ≥ 2021)

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| **XGBoost** | **510** | **655** | **0.272** |
| Red neuronal (MLP) | 513 | 667 | 0.243 |
| Baseline media por depto | 591 | 714 | 0.133 |
| Lineal (Lasso) | 555 | 720 | 0.119 |
| Baseline media global | 654 | 778 | −0.029 |

El mejor modelo es **XGBoost con R² ≈ 0.27 y RMSE ≈ 655 kg/ha** (una corrida con features agro llega a R² = 0.294, RMSE = 644). El R² parece modesto pero hay razones estructurales: gran parte de la varianza es espacial y de tendencia (ya capturada por los baselines), la agregación departamento-campaña diluye la relación clima→rinde de lote, hay variables no observadas (manejo, plagas, granizo), y el test 2021–2024 exige extrapolación temporal.

### Hallazgo: el latente del VAE no ayuda (`componente_b/latente.py`)

Se probó reutilizar el VAE ganador del Componente A (latente de dimensión 16) de tres formas: concatenar el latente a las features, usar solo el latente, y agregar una feature binaria `es_anomalo` (top-10% del score). **Conclusión (notebook 05): el latente empeora la regresión** — y "solo latente" es lo peor — porque ese latente resume el clima **normalizado por departamento**: descarta justamente la señal espacial y de tendencia que `depto_enc` y `year` sí capturan. ERA5/NDVI, por su parte, mueven poco la aguja en regresión.

El notebook 06 (modelo por zona geográfica) muestra que especializar por zona ayuda en zonas pampeanas (z4: R² 0.524 → 0.577) pero colapsa en el norte subtropical (z2: R² = −2.67); en neto, el modelo pooled le gana al esquema un-modelo-por-zona.

### Qué más se puede hacer

- **NDVI mensual intra-campaña** como predictor directo del estado del cultivo en floración/llenado, en vez de una columna más entre 60.
- Explotar más las **features agronómicas** (`use_agro`): balance hídrico, días de estrés térmico, grados-día.
- **Dos momentos de predicción**: un modelo pre-siembra (solo clima previo, humedad de suelo, ENSO) y otro pre-cosecha (con el clima y NDVI de la campaña ya observados), útiles para decisiones distintas.
- Medir el **skill contra la climatología** (media por depto + tendencia) como métrica central, y modelar directamente la **anomalía de rinde** (residuo) en lugar del rinde absoluto.
- Tratar la extrapolación temporal (detrend; no pasar `year` crudo a modelos de árboles) y explorar pooling parcial/jerárquico por zona.
