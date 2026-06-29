# experiments/ — Notebooks de los experimentos

Notebooks que reconstruyen y documentan todos los experimentos del Componente A,
**leyendo de los runs ya guardados en `../runs/`** (no re-entrenan nada). Vienen
ya ejecutados (con tablas y gráficos embebidos), así que se pueden leer directo.

| notebook | contenido |
|---|---|
| `00_overview.ipynb` | El problema, la etiqueta, el leaderboard final y el índice |
| `01_modelos_base_y_tuning.ipynb` | AE / DAE / VAE base + búsqueda de hiperparámetros (recon_prob vs MSE) |
| `02_ensembles.ipynb` | Seed-ensemble del VAE, AE/DAE ensembles, heterogéneo (resuelve la varianza) |
| `03_hibridos_y_scoring.ipynb` | AE-latente + IForest, scoring max/top-k, tuning IForest |
| `04_features_nuevas.ipynb` | Limpieza de datos, NDVI, ERA5 (suelo/heladas), agronómicas |
| `05_modernos_y_techo.ipynb` | Comparación con deepod (DeepSVDD/ICL/NeuTraL/GOAD) + techo estructural |

## Cómo usarlos

```bash
# leerlos: abrir en Jupyter / VS Code
.venv/bin/jupyter notebook experiments/

# regenerarlos (si cambian los runs o las tablas):
.venv/bin/python experiments/_build_notebooks.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace experiments/*.ipynb
```

## Estructura

- **`explib.py`** — motor: carga los runs guardados (`load_run_table`, `latest_run`,
  `load_run`), helpers de ploteo (`plot_pr_curves`, `plot_score_hist`,
  `barh_leaderboard`) y las **tablas de resultados validadas** (panel limpio) como
  fuente única (`tbl_leaderboard`, `tbl_baselines`, `tbl_ensembles`, `tbl_hybrids`,
  `tbl_features`, `tbl_modern`).
- **`_build_notebooks.py`** — generador de los `.ipynb` (narrativa + llamadas a `explib`).

Las tablas de números viven en `explib.py` (validadas, panel limpio); los gráficos
se cargan en vivo desde `runs/`. Algunos paneles del notebook 05 esperan los
artefactos de `analyze_errors.py` en `../analysis/` (si no están, lo avisan).
