# experiments/ — El proyecto contado de punta a punta (notebooks)

Notebooks **didácticos y reproducibles** que cuentan todo el Componente A: qué son los
datos, cómo es el pipeline, cómo se entrena cada modelo, y el recorrido de experimentos
hasta el modelo final. Pensados para que **alguien ajeno al proyecto** entienda y pueda
**correr todo**.

Vienen ya ejecutados (tablas + gráficos embebidos), así que se pueden leer directo; y
cualquier modelo se puede **re-entrenar en el notebook**.

| notebook | contenido |
|---|---|
| `00_datos_y_pipeline` | Qué son los datos, la etiqueta `z_rinde`, las features, los splits temporales y la normalización — **con plots del panel real** |
| `01_evaluacion_baseline_y_como_correrlo` | Cómo se entrena/evalúa, **cómo reproducirlo** (entrenar en el notebook), PR-AUC vs F1, **selección en val vs test**, y el baseline IForest |
| `02_modelos_de_reconstruccion` | AE (top-5 de 24) → DAE → híbrido AE+IForest → VAE — con **config y arquitectura** de cada uno |
| `03_ensembles_y_modelo_final` | Seed-ensemble (de la varianza al modelo final) + **Dataset Cartography** explicada |
| `04_limpieza_de_datos_y_features` | El punto de quiebre (limpieza, +0.09–0.14) + NDVI / suelo / heladas |
| `05_modernos_y_techo_estructural` | Benchmark vs métodos modernos (deepod) + el techo estructural + comparación final |

## Reproducibilidad: entrenar en el notebook

Cada modelo se carga de `runs/` (rápido) o se **entrena de verdad** con un flag:

```python
import explib
# carga el run guardado:
res = explib.run_experiment("configs/vae/vae_v4_reconprob_lat16.yaml", "soja")
# o entrena de cero con el pipeline real (y lo guarda en runs/):
res = explib.run_experiment("configs/vae/vae_v4_reconprob_lat16.yaml", "soja", force_train=True)
```

Ver la configuración y la arquitectura de cualquier modelo:
```python
explib.show_yaml("configs/vae/vae_v4_reconprob_lat16.yaml")
explib.describe_architecture("configs/vae/vae_v4_reconprob_lat16.yaml")
```

## Setup (cualquiera, desde cero)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt nbformat nbconvert ipykernel deepod
jupyter notebook experiments/        # leer / correr
```

## Nota metodológica (importante)
La **selección de modelo** debería hacerse en **validación** y el **test** usarse una sola
vez. Acá el val es chico/ruidoso (5–9 anomalías), así que mostramos test para *ilustrar* las
comparaciones, dejando la limitación explícita (NB 1). El ranking en val y test coincide.

## Dos eras de datos
El recorrido tiene dos etapas (es parte de la historia): **exploración** (panel original,
NB 2) y **modelos finales** (panel limpio, NB 3–5). La limpieza (NB 4) es el punto de quiebre
que subió a todos. `explib` selecciona la era con `era='dirty'|'clean'`.

## Estructura interna
- **`explib.py`** — motor: carga de runs, `run_experiment` (entrenar-o-cargar),
  `show_yaml`/`describe_architecture`, tablas (`compare_table`, `run_metrics`,
  `tbl_journey`, `tbl_leaderboard`…) y plots (`plot_all_metrics`, `plot_loss`,
  `plot_compare`, `plot_pr_curves`, `plot_leaderboard_compare`).
- **`_build_notebooks.py`** — generador de los `.ipynb`.
