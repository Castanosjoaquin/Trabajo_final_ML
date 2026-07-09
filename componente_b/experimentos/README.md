# experimentos/ — Predicción de rinde, notebook por notebook

Notebooks **narrativos y reproducibles** del Componente B: predecir el rinde
(`rinde_kgha`, kg/ha) de un departamento-campaña a partir del clima. Cada uno viene
ejecutado (tablas y gráficos embebidos) y se puede re-correr de cero.

| nb | notebook | qué hace |
|---|---|---|
| 0 | `00_eda_rinde_y_features` | **EDA para predicción de rinde**: target (distribución/tendencia/heterogeneidad), la relación clima–rinde **cambia por zona**, features **agronómicas** de ventana crítica, y zonas geográficas balanceadas |
| 1 | `01_baselines` | El problema, los datos y el split temporal. Baselines: media global, **media por departamento** (climatología), OLS |
| 2 | `02_hp_regresion_lineal` | Búsqueda de HP del lineal (Ridge/Lasso/ElasticNet); métricas + coeficientes; + latente del Componente A |
| 3 | `03_hp_xgboost` | Búsqueda aleatoria de XGBoost; métricas + importancias; + latente |
| 4 | `04_hp_red_neuronal` | Búsqueda de la red neuronal (MLP); métricas + curva de entrenamiento; + latente |
| 5 | `05_comparacion_modelos` | **Comparación integral**: los 6 modelos + baselines, con RMSE/R²/sMAPE, **skill score** y **matriz de Diebold–Mariano** (¿las diferencias son significativas?) |
| 6 | `06_modelo_por_zona` | **Un modelo por zona** (geo-clustering): pooled vs. por-zona, zona por zona. Con los HP re-tuneados, especializar mejora el global en ambos cultivos (aunque no en todas las zonas) |
| 7 | `07_integracion_A_B` | **Integración A↔B (núcleo del proyecto)**: consistencia cruzada Spearman (score VAE vs. residuo), lift del latente, y **cuantificación económica** de la sequía 2022/23 (contrafactual × superficie, vs. benchmark BCR) |
| 9 | `09_router_por_zona` | **Router por zonas** (`WrapperPorZona`): especializa por zona solo donde la **CV en train** lo respalda; supera al pooled en test (punto medio entre pooled y por-zona puro) |
| 10 | `10_prediccion_final` | **Modelo final ejecutable**: reconstruye el modelo final (**Random Forest en ambos cultivos**), lo justifica contra todos los demás (tabla + Diebold–Mariano), agrega el score de anomalía del Componente A, escribe `predicciones_test.csv` y define `predecir_entrega()` para un test externo |

## Panel unificado y el latente del Componente A

El Componente B usa **un único panel** (clima NASA POWER + ONI + CHIRPS + NDVI-AVHRR +
ERA5-Land) con features agronómicas de ventana crítica (`use_agro`) y suavizado del
target-encoding de depto (`enc_smooth`). Los notebooks de HP (02–04), con su config
final, evalúan además tres estrategias que reusan el **mejor detector del Componente A**
(VAE `recon_prob`, `../latente.py`): concatenar su **latente**, regresión **solo en el
latente**, y la categórica **`es_anomalo`** (score umbralado). El VAE se cachea en
`../.latente_cache/`.

## Metodología

- **Split temporal**: train ≤2020, test ≥2021 (consistente con el Componente A). El
  test se usa **una sola vez**, para las métricas del modelo final.
- **Selección de HP**: validación cruzada **temporal** (ventana expansiva), con el
  `depto_enc` recomputado por fold (sin fuga del target), en `evaluacion.buscar`. El
  re-tuning de todos los modelos está en `_retune_all.py` → `retuning_cv_honesta.json`.
- **Métricas**: MAE, RMSE (kg/ha), R², sMAPE; **skill score** vs. climatología y test de
  **Diebold–Mariano** para comparar modelos (`evaluacion.py`).

## Reproducir

```bash
pip install -r ../../requirements.txt
# En Windows: prefijar con  PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python _retune_all.py        # (opcional) re-tunea todos los modelos con CV honesta
                             #   → regenera retuning_cv_honesta.json (ya versionado)
jupyter nbconvert --to notebook --execute --inplace *.ipynb   # o "Run all" por notebook
```

Todos los notebooks corren sobre **los dos cultivos** (soja y maíz) en la misma
pasada; no hay que cambiar ninguna variable.

## Estructura interna
- **`../datos.py`** — pipeline: panel → features (clima + agro + depto-encoding + año +
  lags/momentos) → split temporal → escalado sin leakage → `RegDataset`.
- **`../evaluacion.py`** — métricas (incl. sMAPE, skill, Diebold–Mariano), baselines,
  `buscar` (CV temporal con `depto_enc` honesto) y gráficos.
- **`../modelos/`** — los regresores con interfaz común (lineal, RF, HistGBM, XGBoost,
  MLP, stacking, detrended).
- **`../integracion.py`** — acople A↔B (consistencia cruzada, contrafactual, económico).
- **`../wrapper_zonas.py`** — router por zonas con selección por CV.
- **`_retune_all.py`** — re-tuning de todos los modelos (CV temporal honesta) →
  `retuning_cv_honesta.json`, que los notebooks 05–10 leen para reconstruir los
  modelos finales.
