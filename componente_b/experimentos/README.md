# experimentos/ — Predicción de rinde, notebook por notebook

Notebooks **narrativos y reproducibles** del Componente B: predecir el rinde
(`rinde_kgha`, kg/ha) de un departamento-campaña a partir del clima. Cada uno viene
ejecutado (tablas y gráficos embebidos) y se puede re-correr de cero.

| nb | notebook | qué hace |
|---|---|---|
| 0 | `00_eda_rinde_y_features` | **EDA para predicción de rinde**: target (distribución/tendencia/heterogeneidad), la relación clima–rinde **cambia por zona** (motiva separar por zonas), features **agronómicas** de ventana crítica, y zonas geográficas balanceadas |
| 1 | `01_baselines` | El problema, los datos y el split temporal. Baselines (media global, **media por departamento**, OLS) sobre **ambos datasets**: `base` y `era5_ndvi` |
| 2 | `02_hp_regresion_lineal` | Búsqueda de HP del lineal (Ridge/Lasso/ElasticNet); métricas finales + coeficientes; + comparación de datasets y latente |
| 3 | `03_hp_xgboost` | Búsqueda aleatoria de XGBoost; métricas finales + importancias; + comparación de datasets y latente |
| 4 | `04_hp_red_neuronal` | Búsqueda de la red neuronal; métricas finales + curva de entrenamiento; + comparación de datasets y latente |
| 5 | `05_comparacion_modelos` | Comparación final + **análisis de por qué el R² es bajo y qué modificar** |
| 6 | `06_modelo_por_zona` | **Un modelo por zona** (geo-clustering): pooled vs. por-zona, zona por zona. Ayuda en la Pampa, colapsa en el norte ruidoso; neto global no supera al pooled |

## Dos datasets y el latente del Componente A

Cada notebook de HP (02–04), con su configuración final, evalúa:

- **`base`** (solo clima) vs **`era5_ndvi`** (clima + NDVI-AVHRR + ERA5-Land), y
- sobre el mejor, tres estrategias que reusan el **mejor detector del Componente A**
  (VAE `recon_prob`, `../latente.py`): concatenar su **espacio latente**, regresión
  **solo en el latente**, y una categórica **`es_anomalo`** (su score umbralado).

> Los notebooks se entregan **sin ejecutar**. La primera corrida entrena el VAE por
> cultivo/dataset (se cachea en `../.latente_cache/`), así que tarda unos minutos.

## Resultado (soja, test ≥2021)

| modelo | MAE | RMSE | R² |
|--------|----:|----:|----:|
| **XGBoost** | 510 | 655 | **0.272** |
| Red neuronal | 513 | 667 | 0.243 |
| media × depto | 591 | 714 | 0.133 |
| Lineal (Lasso) | 555 | 720 | 0.119 |
| media global | 654 | 778 | −0.029 |

Los modelos no lineales (XGBoost, red neuronal) superan claramente al baseline
agronómico (media por departamento); el margen mide cuánto aporta el **clima del año**
por encima de "cada depto rinde lo de siempre".

## Metodología

- **Split temporal**: train ≤2020, test ≥2021 (consistente con el Componente A). El
  test se usa **una sola vez**, para las métricas del modelo final.
- **Selección de HP**: validación cruzada **temporal** (ventana expansiva; la
  validación es siempre posterior al train de cada fold), en `evaluacion.buscar`.
- **Métricas**: MAE y RMSE (kg/ha), R² y MAPE (`evaluacion.metricas`).

## Reproducir

```bash
pip install -r ../requirements.txt nbformat nbconvert ipykernel
python fix_openmp_macos.py                      # macOS: ver ../modelos/README.md
python experimentos/_build_notebooks.py         # (re)genera los .ipynb
jupyter nbconvert --to notebook --execute --inplace experimentos/*.ipynb
```

Para reproducir con **maíz**: cambiar `CULTIVO = 'maiz'` en la celda de setup de cada
notebook.

## Estructura interna
- **`../datos.py`** — pipeline: panel → features (clima + depto-encoding + año) →
  split temporal → escalado sin leakage → `RegDataset`.
- **`../evaluacion.py`** — métricas, baselines, `buscar` (CV temporal) y gráficos.
- **`../modelos/`** — los tres regresores con interfaz común.
- **`_build_notebooks.py`** — generador de los `.ipynb`.
