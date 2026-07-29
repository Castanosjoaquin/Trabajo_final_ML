# Detección de anomalías y predicción de rinde de soja y maíz (Argentina)

Trabajo final de I302 — Aprendizaje Automático y Aprendizaje Profundo (UdeSA, 2026).
Dos componentes acoplados sobre un mismo panel **departamento × campaña × cultivo**
(soja y maíz, 1981/82–2024/25, 307 departamentos de 15 provincias):

- **Componente A — detección de anomalías** (*one-class*): ¿qué campañas tuvieron un
  rinde anómalamente bajo, mirando solo el clima? Modelo final: **VAE con score
  `recon_prob` + seed-ensemble ×10** (PR-AUC test: **0.605 soja / 0.515 maíz**).
- **Componente B — predicción de rinde** (regresión supervisada): ¿cuántos kg/ha va a
  rendir cada departamento? Modelo final: **Random Forest en ambos cultivos**
  (RMSE test 593, R² 0.40 en soja; RMSE 1536, R² 0.41 en maíz).
- **Integración A↔B**: el latente del VAE como features del regresor (−3 a −7% de
  RMSE), consistencia cruzada score↔residuo (Spearman 0.26 soja), y cuantificación
  económica de la sequía 2022/23 (~14.7 Mtn soja / ~20.0 Mtn maíz) vía contrafactual.

El informe está en `latex.txt` (formato IEEE; figuras en `informe_figs/`) y las
predicciones de entrega en `entrega/`.

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> En Windows, correr los scripts con `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` (hay
> prints con caracteres Unicode).

---

## Estructura

```
Trabajo_final_ML/
│
├── data/processed/panel_union.parquet    # Dataset ÚNICO (8 fuentes; ver docs/fuentes_dataset.md)
│                                         #   lo construye componente_a/eda/build_panel_union.py
│
├── componente_a/                         # Detección de anomalías
│   ├── src/                              #   config.py · data.py (pipeline) · models/
│   │   └── models/                       #   IForest/OCSVM · AE/DAE · VAE · seed-ensemble · deepod
│   ├── experiments/                      #   EL TRABAJO: notebooks 00–07 (ver su README)
│   ├── eda/                              #   ETL del panel + capas estáticas + 3 notebooks EDA
│   └── tests/                            #   test_repro (hash del dataset) + test_smoke (modelos)
│
├── componente_b/                         # Predicción de rinde
│   ├── datos.py                          #   panel → features → checkpoint → split → RegDataset
│   ├── evaluacion.py                     #   métricas, CV temporal honesta, Diebold–Mariano
│   ├── momentum.py                       #   memoria inter-campaña (EWMA); off por default
│   ├── modelos/                          #   lineal · RF · HistGBM · XGBoost · MLP · stacking
│   ├── integracion.py / latente.py       #   acople con el Componente A
│   ├── wrapper_zonas.py                  #   router por zonas (especializa donde la CV lo avala)
│   └── experimentos/                     #   notebooks 00–12 (ver su README)
│
├── entrega/                              # Castanos_Oostdijk_Predictions_PF.zip (+ CSVs por modelo)
├── informe_figs/                         # figuras del informe (extraídas de los notebooks)
├── latex.txt                             # informe IEEE (fuente LaTeX)
├── docs/                                 # consigna + fuentes del dataset + docs de planificación
└── requirements.txt
```

No hay capa de configs ni orquestador: los hiperparámetros y el entrenamiento viven
**a la vista en los notebooks**; `src/` y `componente_b/*.py` son solo pipeline de
datos + modelos + evaluación.

---

## Los datos (resumen)

- **Panel**: `data/processed/panel_union.parquet` — 27.865 × 132 crudo; el loader
  deduplica el join de centroides por nombre → **20.672 filas efectivas**
  (8.964 soja / 11.708 maíz). Detalle de las 8 fuentes: `docs/fuentes_dataset.md`.
- **Features base (72)**: NASA POWER (49) + ONI (5) + NDVI-AVHRR (7) + ERA5-Land (4)
  + CHIRPS (7). El Componente B agrega `depto_enc` (target encoding con m-estimate),
  `year` y 4 features agronómicas de ventana crítica (78 en total; hasta 84 con lags).
- **Capas estáticas (48)**: suelo (SoilGrids v2.0, 44 columnas) y geografía (SRTM +
  HydroSHEDS, 4), un valor por departamento que se repite en todas sus campañas.
  **No entran en X por defecto** (`use_suelo=False`): `add_suelo_features` las
  condensa en 13 features derivadas y el ablation decide si activarlas. Las extrae
  `componente_a/eda/build_capas_estaticas.py`, aparte del ETL.
- **Checkpoints de campaña**: el Componente B puede restringir las features a lo
  observable en 5 momentos (`pre_siembra` → `nov` → `ene` → `pre_cosecha` → `full`),
  para poder consultar el modelo en cualquier punto del calendario agrícola.
- **Etiqueta proxy (solo evaluación de A)**: `z_rinde < −1.5` vs. la media móvil de
  las 5 campañas previas (con `shift(1)`, sin mirar el año en curso).
- **Split temporal (idéntico en A y B)**: train ≤2020/21 · test 2021/22–2024/25.
  Los HP se eligen por **CV temporal expansiva (4 folds) dentro del train**; el test
  se usa una sola vez. En A, el train se filtra a solo-normales.

---

## Cómo reproducir

```bash
# 1. (opcional) reconstruir el panel — requiere auth de Google Earth Engine
python componente_a/eda/build_capas_estaticas.py     # suelo + geografía (estáticas)
python componente_a/eda/validate_capas_estaticas.py  # chequeos de sanidad
python componente_a/eda/build_panel_union.py         # el panel (mergea lo anterior)

# 2. Componente A: abrir los notebooks de componente_a/experiments/ y "Run all" (00→07)
# 3. Componente B: ídem componente_b/experimentos/ (00→12)
#    (el re-tuning está persistido en experimentos/retuning_cv_honesta.json)

# 4. Tests
cd componente_a && python -m pytest tests/ -q     # 36 tests
```

**Modelo final ejecutable**: `componente_b/experimentos/10_prediccion_final.ipynb`
reconstruye los campeones, escribe `predicciones_test.csv` y define
`predecir_entrega(panel_nuevo)` para predecir sobre un test externo con el mismo
esquema del panel.

---

## Resultados (test, ambos cultivos)

**Componente A** (PR-AUC media±std entre semillas; prevalencia: 0.266 soja / 0.297 maíz):

| Modelo | soja | maíz |
|---|---|---|
| **VAE seed-ensemble ×10 ⭐** | **0.605 ± 0.008** | **0.515 ± 0.013** |
| VAE `recon_prob` (1 semilla) | 0.588 | 0.488 |
| z-score (media) | 0.553 | 0.515 |
| Deep SVDD (tuneado) | 0.534 | 0.425 |
| Isolation Forest | 0.517 | 0.506 |
| One-Class SVM (RBF) | 0.485 | 0.436 |
| AE / DAE (score MSE) | 0.31 / 0.35 | 0.32 / 0.33 |

El salto no es la arquitectura sino el **score probabilístico** (An & Cho) y el
**ensemble de semillas** (baja el desvío ~5×). En maíz la ventaja sobre el z-score es
marginal: buena parte de sus anomalías no tiene firma climática (techo estructural).

**Componente B** (modelo final: Random Forest en ambos cultivos):

| | RMSE (kg/ha) | R² | sMAPE | skill vs. climatología |
|---|---|---|---|---|
| soja — Random Forest | 593 | 0.401 | 21.9% | +0.16 |
| maíz — Random Forest | 1536 | 0.414 | 24.4% | +0.14 |

En soja RF gana también la CV en train y le gana a todos los modelos con
significancia (Diebold–Mariano, p<0.01). En maíz la CV daba un empate técnico entre
los métodos de árboles, pero en test RF se despega con claridad (la elección queda
justificada y transparente en el nb 10). Especializar por zonas geográficas también
mostró mejoras (nb 06/09).

---

## Limitaciones conocidas

1. **Techo estructural**: parte de las anomalías de rinde tiene causas no climáticas
   (plagas, granizo, manejo) invisibles a las features → acota la PR-AUC de cualquier
   detector y el R² de cualquier regresor a este nivel de agregación.
2. **Distribution shift temporal**: el test (2021–24) incluye la sequía 2022/23; la
   tasa de anomalías sube de ~11–13% (histórica) a 27–30% en test. Por eso la métrica
   principal de A es PR-AUC (ranking) y los umbrales se recalibran por split.
3. **Rindes reportados ≈ 0** (abandono de superficie): el principal modo de falla del
   regresor; ningún descriptor climático los anticipa por completo.
