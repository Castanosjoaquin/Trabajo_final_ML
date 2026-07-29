# experiments/ — El recorrido experimental, notebook por notebook

Notebooks **narrativos y reproducibles** del Componente A: cada uno toma una hipótesis
de modelado, muestra el experimento que la puso a prueba y la conclusión que justifica
el diseño final. Vienen ejecutados (tablas y gráficos embebidos) y se pueden re-correr
de cero con "Run all": **entrenan de verdad**, no leen resultados guardados. Todos los
experimentos corren sobre **los dos cultivos** (soja y maíz).

| nb | notebook | pregunta que responde |
|---|---|---|
| 0 | `00_datos_y_pipeline` | El panel, la etiqueta proxy `z_rinde` (y sus decisiones de diseño), splits temporales y normalización z-score por departamento |
| 1 | `01_evaluacion_y_baselines` | Cómo se evalúa (PR-AUC multi-seed, *distribution shift* del umbral) + **todos los baselines**: estadísticos (z-score, Mahalanobis) y de modelos (IForest, One-Class SVM), con matrices de confusión |
| 2 | `02_ae_y_dae` | AE y Denoising AE con score MSE: el error de reconstrucción plano **no alcanza** (queda debajo de los baselines) — el cuello no son los HP. + IForest/OCSVM sobre el latente del AE |
| 3 | `03_vae` | **VAE**: búsqueda de arquitectura multi-seed (config final: 128–64, latente 16, β=1) y los tres scores — `recon_error` falla, **`recon_prob` ≈ `neg_elbo` es el salto** (soja 0.598 vs 0.31 del AE) |
| 4 | `04_ensemble` | **Seed-ensemble ×10 = el modelo final**: promedia el score de 10 inicializaciones → el desvío se desploma (soja ±0.041→±0.017) y la media sube (0.605 soja / 0.515 maíz en el leaderboard final) |
| 6 | `06_robustez` | Robustez a ruido gaussiano en las features de test: degradación **gradual y monótona**, sin colapso (≈−30% relativo recién a σ=2) |
| 7 | `07_leaderboard_y_conclusiones` | Leaderboard final soja+maíz (incluye **DeepSVDD tuneado** como referencia moderna, vía `deepod`), recapitulación del recorrido y conclusiones |

El **slot 5 queda reservado** para el notebook de features nuevas (suelo y geografía);
ver `docs/plan_expansion_serie_temporal.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ../../requirements.txt        # incluye deepod (para el nb 07)
jupyter notebook .                           # leer / correr
```

Con las **10 semillas** del estudio la corrida completa tarda un rato. Para una
corrida rápida, reducir `SEEDS` en la celda de setup de cada notebook (o exportar
`LAB_SEEDS="42,43,44"` antes de lanzar Jupyter).

## Estructura interna

- **`lab.py`** — utilidades compartidas de los notebooks: métricas y evaluación
  multi-seed (`lab.metrics`, `lab.evaluate`), tablas (`lab.tabla`,
  `lab.leaderboard`) y gráficos (`plot_leaderboard`, `plot_pr`, `confusion_top`,
  `plot_scores`, embeddings 2D).
- **`../src/data.py`** — pipeline: panel → etiqueta → split → normalización
  (`data.prepare()` + `data.build_crop_dataset(panel_z, cultivo)`).
- **`../src/models/`** — detectores con interfaz común `fit(X)` /
  `score_samples(X)`: baselines, AE/DAE, VAE, seed-ensemble y wrapper de `deepod`.

## Nota metodológica

No hay bloque de validación separado: los HP se comparan con la **media−desvío de
PR-AUC multi-seed** y el test se reserva para la comparación final (nb 07). Donde una
decisión se ilustra sobre test, el riesgo de *data snooping* queda declarado en el
propio notebook. La etiqueta proxy se usa **solo para evaluar**: el entrenamiento es
sin etiquetas, sobre campañas normales curadas por la proxy histórica (one-class con
supervisión débil en la curación, no "no supervisado puro").
