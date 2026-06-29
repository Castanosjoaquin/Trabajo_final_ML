# experiments/ — El recorrido de los experimentos (notebooks)

Notebooks que cuentan, **en orden cronológico**, el viaje de modelos del Componente A.
Leen de los runs guardados en `../runs/` (no re-entrenan) y vienen ya ejecutados, con
tablas y gráficos embebidos.

| notebook | contenido |
|---|---|
| `00_intro_problema_y_baseline` | El problema, la etiqueta `z_rinde`, la metodología de evaluación y el **baseline IForest** (el listón). Sin spoiler del final. |
| `01_modelos_de_reconstruccion` | El recorrido: **AE** (top-5 de 24 variantes) → **DAE** → **híbrido AE+IForest** → **VAE** (el salto del `recon_prob`). Siempre vs baseline. |
| `02_ensembles_y_modelo_final` | Cómo atacamos la varianza con el **seed-ensemble** hasta el modelo final; + la idea de **Dataset Cartography** (qué es y cómo funciona). |
| `03_limpieza_de_datos_y_features` | El **punto de quiebre** (limpieza de datos, +0.09–0.14) y las features nuevas (NDVI, ERA5 suelo/heladas, agro). |
| `04_modernos_y_techo_estructural` | Benchmark contra métodos **modernos** (deepod) + el **techo estructural** + comparación final de todos los modelos. |

Cada modelo se muestra con **todas sus métricas** (PR-AUC, ROC-AUC, F1, Precision@k,
Recall@contam) **con desvío estándar**, curvas de loss y gráficos comparativos.

## Las dos eras (para entender los números)
El recorrido tiene dos etapas, y eso es parte de la historia:
1. **Exploración** (NB 1): sobre el panel **original**. Importa la comparación *relativa*.
2. **Modelos finales** (NB 2–4): sobre el panel **limpio** (NB 3 explica la limpieza, que
   subió a todos +0.09–0.14). El leaderboard final es sobre datos limpios.

`explib.py` selecciona la era correcta con el parámetro `era='dirty'|'clean'`.

## Uso

```bash
# leer:
.venv/bin/jupyter notebook experiments/

# regenerar (si cambian los runs o las tablas):
.venv/bin/python experiments/_build_notebooks.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace experiments/*.ipynb
```

## Estructura interna
- **`explib.py`** — motor: carga de runs (`load_run_table`, `top_runs`, `latest_run` con
  `era`), tablas comparativas (`compare_table`, `run_metrics`), plots (`plot_all_metrics`,
  `plot_loss`, `plot_compare`, `plot_pr_curves`, `plot_leaderboard_compare`), y las tablas
  curadas (`tbl_journey`, `tbl_features`, `tbl_modern`, `tbl_leaderboard`).
- **`_build_notebooks.py`** — generador de los `.ipynb` (narrativa + llamadas a `explib`).

Algunos paneles (data map, heatmap de errores) esperan los PNG de `analyze_datamap.py` /
`analyze_errors.py` en `../analysis/`; si no están, el notebook lo avisa.
