# experiments/ — El recorrido experimental, notebook por notebook

Notebooks **narrativos y reproducibles** del Componente A: cada uno toma una hipótesis
de modelado, muestra el experimento que la puso a prueba y la conclusión que justifica
el diseño final. Pensados para que **alguien ajeno al proyecto** pueda leerlos de
corrido (vienen ejecutados, con tablas y gráficos embebidos) o **re-correr todo**.
Todos los números salen de los runs actuales (mismo panel, misma evaluación).

| nb | notebook | pregunta que responde |
|---|---|---|
| 0 | `00_el_problema_y_los_datos` | El problema, el panel, la etiqueta proxy `z_rinde` (y sus decisiones de diseño), splits y normalización |
| 1 | `01_evaluacion_y_baselines` | Cómo se evalúa (PR-AUC multi-seed, *distribution shift* del umbral, por qué no hay val) y **todos los baselines**: estadísticos (z-score, Mahalanobis) y de modelos (IForest, One-Class SVM), con la presentación estándar `lab.tabla` |
| 2 | `02_reconstruccion_ae_a_vae` | AE (los HP no eran el cuello) → score max/top-k (refutado) → DAE → híbrido AE+IForest (refutado) → **VAE `recon_prob`**: el salto es el *score*, no la arquitectura |
| 3 | `03_varianza_y_seed_ensemble` | De dónde viene la varianza del VAE (MC no, reg no, Student-t no) y cómo la elimina el **seed-ensemble ×10** → modelo final (soja 0.592, maíz 0.508) |
| 4 | `04_el_techo_estructural` | ¿Por qué nadie pasa de ~0.6? Cross-modelo (todos fallan en las mismas anomalías, sin firma climática), Dataset Cartography, auditoría de etiqueta, t-SNE |
| 5 | `05_features_nuevas` | Features agro, NDVI-AVHRR y ERA5: ayudan al IForest, perjudican al VAE, ninguna supera al base → el techo se confirma |
| 6 | `06_modernos_leaderboard_y_conclusiones` | Benchmark justo vs deepod (DeepSVDD/ICL/NeuTraL), leaderboard final, el modelo final por dentro, limitaciones y conclusiones |

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
pip install -r ../../requirements.txt
jupyter notebook experiments/        # leer / correr
```

## Nota metodológica (importante)
La **selección de modelo** debería hacerse en **validación** y el **test** usarse una
sola vez. Acá el val es tan chico (5–9 anomalías) que sus métricas son ruido (PR-AUC ≈
tasa base) y no permiten seleccionar; mostramos test para *ilustrar* las comparaciones,
con el riesgo de *data snooping* dejado explícito (nb. 1).

## Estructura interna
- **`explib.py`** — motor: carga de runs (siempre la corrida más reciente de cada
  config), `run_experiment` (entrenar-o-cargar), `show_yaml`/`describe_architecture`,
  tablas (`compare_table`, `run_metrics`, `summary_fields`, `tbl_leaderboard`,
  `tbl_modern`) y plots (`plot_all_metrics`, `plot_loss`, `plot_compare`,
  `plot_pr_curves`, `plot_confusion_grid`, `plot_score_hist_grid`, `plot_embeddings`,
  `plot_leaderboard_compare`).
- **`_build_notebooks.py`** — generador de los `.ipynb` (después se ejecutan con
  `jupyter nbconvert --execute --inplace` para embeber salidas).
