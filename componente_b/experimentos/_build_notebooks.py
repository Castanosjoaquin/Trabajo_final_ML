"""Genera los 5 notebooks de experimentos del Componente B.

Cada notebook es narrativo y reproducible: importa el pipeline (`datos.py`),
la evaluación (`evaluacion.py`) y los modelos (`modelos/`), y entrena VISIBLE.
Después se ejecutan con `jupyter nbconvert --execute --inplace` para embeber
salidas.

    python componente_b/experimentos/_build_notebooks.py
"""
from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Setup común: paths, imports, dataset. Va de primera en cada notebook.
SETUP = """\
import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))          # componente_b/ (datos, evaluacion)
warnings.filterwarnings('ignore')                  # silenciar ConvergenceWarning de sklearn

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')

import datos, evaluacion as ev
from modelos import LinearRegressor, XGBoostRegressor, NeuralNetRegressor

# Cultivo del estudio (cambiar a 'maiz' para reproducir con maíz).
CULTIVO = 'soja'
ds = datos.prepare(CULTIVO)
print(f'{CULTIVO}: {len(ds.feature_cols)} features | '
      f'train {ds.X_train.shape[0]} filas (≤{datos.TRAIN_END}) | '
      f'test {ds.X_test.shape[0]} filas (≥{datos.TEST_START})')
"""


# Sección transversal (nbs 02–04): probar la config final sobre ambos datasets y,
# sobre el mejor, las tres estrategias con el latente del mejor detector del
# Componente A (VAE recon_prob).
def _ds_latente_md():
    return new_markdown_cell("""\
## Ambos datasets + el mejor modelo del Componente A

Con la **configuración final ya elegida**, evaluamos en test:

1. dataset `base` (solo clima) vs `era5_ndvi` (clima + NDVI + ERA5), y
2. sobre el que mejor anduvo, tres estrategias que reusan el **mejor detector del
   Componente A** (el VAE `recon_prob`, que aprendió a representar el clima
   "normal"): concatenar su **espacio latente**, hacer la regresión **solo en el
   latente**, y agregar la categórica **`es_anomalo`** (su score umbralado).

El latente/score se computan en `latente.py` (y se cachean). *La primera corrida
entrena el VAE, así que tarda unos minutos.*""")


def ds_latente_code(model_name, params_var, fixed="None"):
    return new_code_cell(
        f"tabla_ds, mejor = ev.comparar_datasets_y_latente(\n"
        f"    {model_name}, {params_var}, CULTIVO, fixed={fixed},\n"
        f"    vae_kwargs=dict(score_seeds=(42, 43, 44)))\n"
        f"print('Mejor dataset base:', mejor)\n"
        f"tabla_ds")


DS_LATENTE_MD = _ds_latente_md()


def build(path, cells):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("escrito", os.path.relpath(path, HERE))


def md(t): return new_markdown_cell(t)
def code(t): return new_code_cell(t)


# ===========================================================================
# NB 01 — Baselines: la media y la regresión lineal
# ===========================================================================
def nb01():
    return [
        md("""\
# 01 — Baselines: la media y la regresión lineal

**Componente B — predicción de rinde.** El problema es de *regresión supervisada*:
estimar el rinde (`rinde_kgha`, kg/ha) de un departamento-campaña a partir de las
variables climáticas de la campaña.

Antes de tunear modelos hay que fijar el **piso**: ¿qué tan lejos llega lo trivial?
Este notebook establece tres referencias en orden creciente de sofisticación:

1. **Media global** — predecir siempre el rinde medio de train. El baseline más
   tonto posible; su R² es 0 por construcción sobre train.
2. **Media por departamento** (climatología) — predecir, para cada fila, el rinde
   histórico medio de su departamento. Captura la estructura *espacial* (un depto
   rinde sistemáticamente más que otro) y es el baseline agronómico honesto.
3. **Regresión lineal (OLS)** — el primer modelo que usa el clima.

Los datos, el split temporal (train ≤2020, test ≥2021) y las features están en
`datos.py`; las métricas y baselines en `evaluacion.py`."""),
        code(SETUP),
        md("""\
## Los datos

Las features que ve el modelo son de tres tipos (todas escaladas con parámetros de
train, sin leakage): **clima mensual** (Sep–Mar), **codificación del departamento**
por su rinde medio en train (estructura espacial) y el **año** (tendencia
tecnológica). El target es el rinde crudo en kg/ha."""),
        code("""\
print('features:', ds.feature_cols[:6], '...', ds.feature_cols[-2:])
print(f'\\nrinde train: media={ds.y_train.mean():.0f}  std={ds.y_train.std():.0f} kg/ha')
print(f'rinde test : media={ds.y_test.mean():.0f}  std={ds.y_test.std():.0f} kg/ha')
ds.meta_test.head(3)"""),
        md("""\
## Baselines

Para regresión reportamos cuatro métricas (definidas en `evaluacion.metricas`):
**MAE** y **RMSE** en kg/ha (RMSE castiga más los errores grandes), **R²** (varianza
explicada; 0 = tan bueno como predecir la media, negativo = peor) y **MAPE** (error
porcentual).

Evaluamos los baselines sobre **los dos datasets**: `base` (solo clima) y
`era5_ndvi` (clima + NDVI-AVHRR + ERA5-Land: humedad de suelo y heladas). Los
baselines no miran las features climáticas —solo la media y la media por depto—,
así que sirven de sanity check de que ambos datasets comparten la misma estructura
de rinde; las features extra recién pesan cuando entra un modelo (nbs 02–04)."""),
        code("""\
filas = []
for dsname in datos.DATASETS:
    d = datos.prepare(CULTIVO, dataset=dsname)
    filas.append(ev.evaluar(f'media global [{dsname}]',  ev.pred_media(d),      d))
    filas.append(ev.evaluar(f'media x depto [{dsname}]', ev.pred_media_depto(d), d))
tabla_base = ev.tabla_comparativa(filas, ordenar_por='rmse')
tabla_base"""),
        md("""\
La **media por departamento** ya explica bastante más varianza que la media global:
casi todo el poder predictivo "fácil" del rinde es *dónde* está el campo, no el
clima del año. Ese es el número que cualquier modelo con clima tiene que superar.
(Los dos datasets dan casi lo mismo en los baselines, como era de esperar.)"""),
        md("""\
## Regresión lineal (OLS, sin regularizar)

Primer modelo con clima. Sin tuning todavía — eso es el notebook 02."""),
        code("""\
ols = LinearRegressor(penalty='none').fit(ds.X_train, ds.y_train)
pred_ols = ols.predict(ds.X_test)

filas.append(ev.evaluar('OLS (lineal)', pred_ols, ds))
tabla = ev.tabla_comparativa(filas, ordenar_por='rmse')
tabla"""),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ev.plot_pred_vs_real(ds.y_test, ev.pred_media_depto(ds), 'Baseline: media x depto',
                     color=ev.C_BASE, ax=axes[0])
ev.plot_pred_vs_real(ds.y_test, pred_ols, 'OLS (lineal)', color=ev.C_LINEAR, ax=axes[1])
plt.tight_layout(); plt.show()"""),
        md("""\
## Conclusión

- La **media por departamento** es un baseline fuerte: el grueso de la señal del
  rinde es espacial.
- La **OLS** aprovecha el clima y la tendencia y mejora sobre la media por depto,
  pero sin regularizar es propensa a la colinealidad (las 54 columnas mensuales
  están muy correlacionadas entre sí).

Los notebooks siguientes tunean cada familia de modelos: regularización para el
lineal (02), XGBoost (03) y red neuronal (04); la comparación final es el 05."""),
    ]


# ===========================================================================
# NB 02 — Búsqueda de hiperparámetros: regresión lineal
# ===========================================================================
def nb02():
    return [
        md("""\
# 02 — Búsqueda de hiperparámetros: regresión lineal regularizada

La OLS del notebook 01 no regulariza y sufre la colinealidad de las features
mensuales. Acá buscamos la **regularización** óptima sobre una grilla:

- `penalty`: `none` (OLS) · `l2` (Ridge, encoge coeficientes) · `l1` (Lasso, hace
  selección de features) · `elasticnet` (mezcla L1+L2).
- `alpha`: fuerza de la penalización.
- `l1_ratio`: mezcla L1/L2 (solo aplica a elasticnet).

La selección se hace con **validación cruzada temporal** (`evaluacion.buscar`):
folds de ventana expansiva sobre el train, donde la validación es siempre
*posterior* al entrenamiento de cada fold. **El test (≥2021) no se toca en la
búsqueda** — se usa una sola vez, al final, para reportar las métricas del modelo
elegido."""),
        code(SETUP),
        md("## La búsqueda\n\nGrid completo, optimizando RMSE de validación."),
        code("""\
grid = {
    'penalty':  ['none', 'ridge', 'lasso', 'elasticnet'],
    'alpha':    [0.1, 1.0, 10.0, 50.0, 100.0],
    'l1_ratio': [0.2, 0.5, 0.8],
}
tabla, best = ev.buscar(LinearRegressor, grid, ds, metric='rmse', n_splits=4)
print('Mejores hiperparámetros:', best)
tabla.head(10)"""),
        md("""\
## Modelo final

Reentrenamos con los mejores hiperparámetros sobre **todo** el train y evaluamos en
el test. Estas son las **métricas del modelo final**."""),
        code("""\
modelo = LinearRegressor(**best).fit(ds.X_train, ds.y_train)
pred = modelo.predict(ds.X_test)

m = ev.metricas(ds.y_test, pred)
print('Config final:', modelo.get_config())
print(f\"\\nMAE  = {m['mae']:.1f} kg/ha\")
print(f\"RMSE = {m['rmse']:.1f} kg/ha\")
print(f\"R²   = {m['r2']:.3f}\")
print(f\"MAPE = {m['mape']:.1f} %\")
print(f\"\\nvs baseline media x depto: RMSE {ev.metricas(ds.y_test, ev.pred_media_depto(ds))['rmse']:.1f}\")"""),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ev.plot_pred_vs_real(ds.y_test, pred, f'Lineal {best[\"penalty\"]} (final)',
                     color=ev.C_LINEAR, ax=axes[0])
ev.plot_residuos(ds.y_test, pred, 'Residuos', color=ev.C_LINEAR, ax=axes[1])
plt.tight_layout(); plt.show()"""),
        md("## Coeficientes: qué features sobreviven\n\nCon Lasso/ElasticNet muchos coeficientes van a cero — el modelo se queda con las features informativas."),
        code("""\
coef = getattr(modelo._model, 'coef_', None)
if coef is not None:
    s = pd.Series(coef, index=ds.feature_cols)
    n_cero = int((s.abs() < 1e-8).sum())
    print(f'coeficientes en cero: {n_cero}/{len(s)}')
    print('\\nTop 10 |coef|:')
    print(s.reindex(s.abs().sort_values(ascending=False).index).head(10))"""),
        DS_LATENTE_MD, ds_latente_code("LinearRegressor", "best"),
        md("""\
## Conclusión

La regularización mejora sobre la OLS cruda y da un modelo lineal estable. El techo
del lineal está en su capacidad de capturar solo relaciones **lineales** entre clima
y rinde; los notebooks 03 (XGBoost) y 04 (red neuronal) prueban si lo no-lineal
paga. En cuanto a features: las de ERA5/NDVI y el latente del VAE se evalúan arriba
(ver la conclusión transversal del nb 05)."""),
    ]


# ===========================================================================
# NB 03 — Búsqueda de hiperparámetros: XGBoost
# ===========================================================================
def nb03():
    return [
        md("""\
# 03 — Búsqueda de hiperparámetros: XGBoost

Gradient boosting de árboles: modelo no lineal fuerte para datos tabulares. Tiene
muchos ejes de regularización, y los buscamos todos:

- Complejidad de cada árbol: `max_depth`, `min_child_weight`, `gamma` (poda).
- Regularización de las hojas: `reg_lambda` (L2), `reg_alpha` (L1).
- Randomización tipo bagging: `subsample`, `colsample_bytree`.
- Shrinkage vs. nº de árboles: `learning_rate`, `n_estimators`.

Como el espacio es enorme, usamos **búsqueda aleatoria** (`n_iter`) con la misma
CV temporal del notebook 02. El test no se toca hasta el final."""),
        code(SETUP),
        md("## La búsqueda\n\nBúsqueda aleatoria de 25 combinaciones sobre el grid, optimizando RMSE de validación temporal."),
        code("""\
grid = {
    'max_depth':        [3, 4, 6, 8],
    'learning_rate':    [0.02, 0.05, 0.1],
    'n_estimators':     [200, 400, 800],
    'subsample':        [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'min_child_weight': [1, 5],
    'reg_lambda':       [1.0, 5.0],
    'reg_alpha':        [0.0, 1.0],
    'gamma':            [0.0, 1.0],
}
tabla, best = ev.buscar(XGBoostRegressor, grid, ds, metric='rmse',
                        n_iter=25, random_state=42, n_splits=4)
print('Mejores hiperparámetros:', best)
tabla.head(10)"""),
        md("""\
## Modelo final

Reentrenamos con los mejores hiperparámetros sobre todo el train y evaluamos en el
test. **Métricas del modelo final:**"""),
        code("""\
modelo = XGBoostRegressor(**best, random_state=42).fit(ds.X_train, ds.y_train)
pred = modelo.predict(ds.X_test)

m = ev.metricas(ds.y_test, pred)
print(f\"MAE  = {m['mae']:.1f} kg/ha\")
print(f\"RMSE = {m['rmse']:.1f} kg/ha\")
print(f\"R²   = {m['r2']:.3f}\")
print(f\"MAPE = {m['mape']:.1f} %\")"""),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ev.plot_pred_vs_real(ds.y_test, pred, 'XGBoost (final)', color=ev.C_XGB, ax=axes[0])
ev.plot_residuos(ds.y_test, pred, 'Residuos', color=ev.C_XGB, ax=axes[1])
plt.tight_layout(); plt.show()"""),
        md("## Importancia de features\n\nQué variables usa más el modelo para partir los árboles."),
        code("""\
imp = pd.Series(modelo._model.feature_importances_, index=ds.feature_cols)
imp = imp.sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(imp.index[::-1], imp.values[::-1], color=ev.C_XGB)
ax.set_title('XGBoost — top 12 feature importances'); plt.tight_layout(); plt.show()"""),
        DS_LATENTE_MD, ds_latente_code("XGBoostRegressor", "best", "{'random_state': 42}"),
        md("""\
## Conclusión

XGBoost captura interacciones no lineales entre clima, espacio y tendencia que el
lineal no puede. Ojo con una limitación de los árboles: **no extrapolan** la
tendencia temporal fuera del rango de train (el `year` de test, 2021–2024, está por
encima de todo lo visto), algo que el lineal sí hace. La comparación final (05) lo
pone en contexto contra el lineal y la red neuronal."""),
    ]


# ===========================================================================
# NB 04 — Búsqueda de hiperparámetros: red neuronal
# ===========================================================================
def nb04():
    return [
        md("""\
# 04 — Búsqueda de hiperparámetros: red neuronal (MLP)

Un MLP feed-forward (PyTorch). Buscamos arquitectura y **regularización**:

- Arquitectura: `hidden_dims` (capas y anchos).
- Regularización: `dropout`, `weight_decay` (L2 vía el optimizador). Además, cada
  modelo hace *early stopping* sobre un split interno de validación.
- Optimización: `lr`.

La red estandariza el target internamente. Búsqueda aleatoria con la misma CV
temporal; el test se reserva para el final. (Menos combinaciones que XGBoost porque
cada entrenamiento es más caro.)

**Nota:** con solo unos miles de filas, una red poco regularizada sobreajusta con
ganas (memoriza el rinde medio de cada depto en train y no generaliza al test, que
además cae en años *fuera* del rango de train). Por eso la grilla apunta a redes
chicas y regularización fuerte — el `dropout` alto es el que más mueve la aguja."""),
        code(SETUP),
        md("## La búsqueda\n\nBúsqueda aleatoria de 14 combinaciones, optimizando RMSE de validación temporal."),
        code("""\
grid = {
    'hidden_dims':  [(16,), (32,), (64, 32)],
    'dropout':      [0.1, 0.3, 0.5],
    'weight_decay': [1e-3, 1e-2, 3e-2],
    'lr':           [1e-3, 3e-3],
}
tabla, best = ev.buscar(NeuralNetRegressor, grid, ds, metric='rmse',
                        n_iter=14, random_state=42, n_splits=4,
                        fixed={'max_epochs': 200, 'patience': 25, 'l1_lambda': 0.0})
print('Mejores hiperparámetros:', best)
tabla.head(10)"""),
        md("""\
## Modelo final

Reentrenamos con los mejores hiperparámetros sobre todo el train y evaluamos en el
test. **Métricas del modelo final:**"""),
        code("""\
modelo = NeuralNetRegressor(**best, l1_lambda=0.0, max_epochs=250, patience=30,
                            random_state=42).fit(ds.X_train, ds.y_train)
pred = modelo.predict(ds.X_test)

m = ev.metricas(ds.y_test, pred)
print(f\"MAE  = {m['mae']:.1f} kg/ha\")
print(f\"RMSE = {m['rmse']:.1f} kg/ha\")
print(f\"R²   = {m['r2']:.3f}\")
print(f\"MAPE = {m['mape']:.1f} %\")"""),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ev.plot_pred_vs_real(ds.y_test, pred, 'Red neuronal (final)', color=ev.C_NN, ax=axes[0])
ev.plot_residuos(ds.y_test, pred, 'Residuos', color=ev.C_NN, ax=axes[1])
h = modelo.history_
axes[2].plot(h['train_loss'], label='train', color=ev.C_LINEAR)
if h['val_loss']:
    axes[2].plot(h['val_loss'], label='val', color=ev.C_NN)
axes[2].set_title('Curva de entrenamiento (MSE, target escalado)')
axes[2].set_xlabel('época'); axes[2].legend()
plt.tight_layout(); plt.show()"""),
        DS_LATENTE_MD,
        ds_latente_code("NeuralNetRegressor", "best",
                        "dict(l1_lambda=0.0, max_epochs=250, patience=30, random_state=42)"),
        md("""\
## Conclusión

Con tan pocos miles de filas y features tabulares, la red neuronal compite pero no
suele destronar a un XGBoost bien tuneado. El notebook 05 compara las tres familias
más los baselines en igualdad de condiciones."""),
    ]


# ===========================================================================
# NB 05 — Comparación de modelos
# ===========================================================================
def nb05():
    return [
        md("""\
# 05 — Comparación de modelos

Juntamos todo: los baselines (01) y el mejor modelo de cada familia según su
búsqueda de hiperparámetros (02–04), entrenados sobre el mismo train y evaluados
sobre el mismo test (≥2021). Los hiperparámetros de abajo son los ganadores
reportados en los notebooks 02, 03 y 04.

**Métrica principal: RMSE (kg/ha).** Reportamos también MAE, R² y MAPE."""),
        code(SETUP),
        md("## Mejores configuraciones (de los notebooks 02–04)"),
        code("""\
best_linear = {'penalty': 'lasso', 'alpha': 10.0}
best_xgb = {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 200,
            'subsample': 0.7, 'colsample_bytree': 1.0, 'min_child_weight': 1,
            'reg_lambda': 5.0, 'reg_alpha': 1.0, 'gamma': 1.0}
best_nn = {'hidden_dims': (64, 32), 'dropout': 0.5, 'weight_decay': 1e-3, 'lr': 3e-3}"""),
        md("## Entrenamiento y evaluación en test"),
        code("""\
preds = {}
preds['media global']  = ev.pred_media(ds)
preds['media x depto'] = ev.pred_media_depto(ds)
preds['Lineal']  = LinearRegressor(**best_linear).fit(ds.X_train, ds.y_train).predict(ds.X_test)
preds['XGBoost'] = XGBoostRegressor(**best_xgb, random_state=42).fit(ds.X_train, ds.y_train).predict(ds.X_test)
preds['Red neuronal'] = NeuralNetRegressor(**best_nn, max_epochs=250, patience=30,
                                           random_state=42).fit(ds.X_train, ds.y_train).predict(ds.X_test)

tabla = ev.tabla_comparativa([ev.evaluar(k, v, ds) for k, v in preds.items()],
                             ordenar_por='rmse')
tabla"""),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ev.plot_comparativa(tabla, metrica='rmse', ax=axes[0])
ev.plot_comparativa(tabla, metrica='r2', ax=axes[1])
plt.tight_layout(); plt.show()"""),
        md("## Predicho vs. real de los tres modelos"),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (nombre, color) in zip(axes, [('Lineal', ev.C_LINEAR),
                                      ('XGBoost', ev.C_XGB),
                                      ('Red neuronal', ev.C_NN)]):
    ev.plot_pred_vs_real(ds.y_test, preds[nombre], nombre, color=color, ax=ax)
plt.tight_layout(); plt.show()"""),
        md("""\
## Conclusión

- Todos los modelos con clima **superan a la media global**; el listón real es la
  **media por departamento** (estructura espacial pura).
- **XGBoost** es el mejor: las interacciones no lineales entre clima, espacio y
  tendencia le dan la ventaja sobre el lineal, con la red neuronal cerca.
- El margen sobre el baseline por depto cuantifica cuánto aporta el **clima del año**
  por encima de "cada depto rinde lo de siempre" — que es, en el fondo, lo difícil
  de predecir del rinde."""),
        md("""\
## ¿Por qué el R² es "bajo" (~0.25–0.30)?

No es (solo) que falte tunear: hay un **techo estructural** en predecir rinde de
clima a este nivel de agregación.

1. **La varianza reducible es chica.** El grueso del rinde es *espacial* (qué depto)
   y *tendencia* (qué década) — cosas que los baselines ya capturan. Lo que queda por
   explicar con el **clima del año** es una porción menor y ruidosa, así que aunque
   el modelo la capture bien, el R² total sube poco.
2. **Agregación depto-campaña.** Cada fila promedia miles de lotes con siembras,
   cultivares, suelos y manejo distintos: la relación clima→rinde de lote se diluye.
3. **Variables no observadas.** Manejo (fertilización, fecha de siembra, genética),
   plagas/enfermedades, granizo, y ruido de reporte de MAGYP no están en las features.
   El Componente A ya mostró que ~2/3 de las anomalías de rinde **no tienen firma
   climática**.
4. **Extrapolación temporal.** El test (2021–2024) cae *fuera* del rango de train: la
   tendencia y el régimen climático se corren. Se ve en que la feature `year`
   extrapola (ayuda al lineal, la cortan los árboles) y en que el `es_anomalo` del
   VAE marca casi todo el test (el score sube por *distribution shift*, no por
   anomalía real).

**Sobre las features extra y el latente** (nbs 02–04): ERA5/NDVI mueven poco la
aguja, y el **latente del VAE empeora** la regresión (y solo-latente es lo peor):
ese latente resume el clima *normalizado por depto*, así que tira la señal espacial
y de tendencia que es justo la que más predice. `es_anomalo` tampoco ayuda, dominado
por el shift. Coherente con el techo del Componente A."""),
        md("""\
## ¿Qué podemos modificar para mejorar?

En orden aproximado de impacto esperado:

1. **Cambiar el target a algo aprendible.** En vez del rinde absoluto, predecir la
   **anomalía de rinde** (residuo sobre la media/tendencia por depto, p. ej. el
   `z_rinde` del Componente A). Saca la parte "fácil" (espacio+tendencia) y deja que
   el modelo se concentre en la señal climática — R² más honesto de lo que sí se
   puede predecir.
2. **Features agronómicas, no promedios mensuales crudos.** Índices en la **ventana
   crítica** por cultivo: balance hídrico acumulado, días de estrés térmico (Tmax>32
   en floración), rachas secas, grados-día. El Componente A ya tiene `add_agro_features`.
3. **NDVI como predictor directo, no como una feature más.** El NDVI de
   floración/llenado es un proxy casi directo del rinde; conviene usarlo con más peso
   (o un modelo aparte) en vez de mezclarlo entre 60+ columnas.
4. **Tratar la extrapolación temporal.** Detrendear el rinde antes de modelar, o usar
   validación que imite el gap train→test; para los árboles, no darles `year` crudo.
5. **Modelo residual / híbrido.** Baseline fuerte = media por depto (+ tendencia), y
   un modelo que aprenda **solo el residuo** con el clima. Suele ganarle a predecir
   el absoluto de una.
6. **Robustez al ruido de etiqueta.** Pérdida robusta (Huber/cuantil) por el ruido de
   reporte de MAGYP, y quizás modelar por región/cultivo por separado.

La limitación de fondo (agregación depto-campaña + variables no observadas) solo se
levanta con **datos más finos** (lote/píxel, manejo), que exceden este panel.

Para reproducir con maíz: cambiar `CULTIVO = 'maiz'` en la celda de setup y
re-ejecutar los notebooks."""),
    ]


EDA_SETUP = """\
import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))          # componente_b/
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda v: f'{v:,.1f}')

import datos
from src import data as A                          # componente_a (path ya en sys.path)

CULTIVO = 'soja'
panel = datos.load_panel()
df = panel[panel.cultivo == CULTIVO].copy()
df, AGRO = A.add_agro_features(df, CULTIVO)         # features agronómicas de ventana crítica
# atajos de clima en la ventana crítica del cultivo (floración + llenado)
MESES_CRIT = A.CRITICAL_MONTHS[CULTIVO]
df['precip_crit'] = df[[f'prectotcorr_{m}' for m in MESES_CRIT]].sum(axis=1)
df['tmax_crit']   = df[[f't2m_max_{m}'   for m in MESES_CRIT]].mean(axis=1)
print(f'{CULTIVO}: {len(df)} filas | {df.provincia.nunique()} provincias | '
      f'años {int(df.campania_inicio.min())}–{int(df.campania_inicio.max())} | '
      f'ventana crítica: {MESES_CRIT}')
"""


def nb00():
    return [
        md("""\
# 00 — EDA para predicción de rinde + feature engineering

EDA **enfocado en predecir el rinde** (no en describir por describir). Tres
objetivos, encadenados:

1. Entender el **target** (rinde): distribución, tendencia temporal, heterogeneidad
   espacial.
2. Mostrar que la **relación clima–rinde cambia según la zona** — el motivo por el
   que un modelo único "promedia" relaciones distintas y pierde señal. Esto motiva
   **separar el dataset por zonas** (climas homogéneos adentro).
3. **Feature engineering**: construir features **agronómicas** de la ventana crítica
   del cultivo (balance hídrico, estrés térmico) y ver que correlacionan mejor con
   el rinde que los promedios mensuales crudos.

Todo sobre soja (cambiar `CULTIVO` para maíz)."""),
        code(EDA_SETUP),
        md("""\
## 1. El target: rinde

Dos cosas saltan a la vista y condicionan todo el modelado: el rinde tiene una
**tendencia creciente** (mejora tecnológica a lo largo de las décadas) y una enorme
**dispersión espacial** (cada provincia rinde distinto)."""),
        code("""\
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df.rinde_kgha, bins=40, color='#4C72B0')
ax[0].set_title(f'Distribución de rinde ({CULTIVO})'); ax[0].set_xlabel('kg/ha')
g = df.groupby('campania_inicio').rinde_kgha.mean()
ax[1].plot(g.index, g.values, marker='o', ms=3, color='#55A868')
ax[1].set_title('Rinde medio por campaña (tendencia)'); ax[1].set_xlabel('año')
ax[1].set_ylabel('kg/ha'); plt.tight_layout(); plt.show()"""),
        code("""\
prov = (df.groupby('provincia')
          .agg(n=('rinde_kgha', 'size'), rinde=('rinde_kgha', 'mean'),
               precip=('precip_crit', 'mean'), tmax=('tmax_crit', 'mean'),
               wbal=('agro_waterbal_crit', 'mean')))
prov = prov[prov.n >= 50].sort_values('rinde')
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(prov.index, prov.rinde, color='#4C72B0')
ax.set_title('Rinde medio por provincia'); ax.set_xlabel('kg/ha')
plt.tight_layout(); plt.show()
prov.round(1)"""),
        md("""\
Las provincias no solo rinden distinto: tienen **climas distintos** (mirá `precip`,
`tmax`, `wbal` arriba). El norte (Chaco, Santiago) es más cálido y con otro régimen
de lluvias que la Pampa húmeda. Esto es la raíz del problema del punto 2."""),
        md("""\
## 2. La relación clima–rinde depende de la zona

Si el clima explicara el rinde de la misma forma en todos lados, la correlación
clima–rinde sería parecida en cada provincia. **No lo es**: la correlación entre la
lluvia de la ventana crítica y el rinde va de **fuerte y positiva** en la Pampa (el
agua es limitante) a **nula o negativa** en el subtrópico (donde no falta agua y
mandan otros factores). Un modelo pooled tiene que promediar señales opuestas."""),
        code("""\
sub = df[df.groupby('provincia').rinde_kgha.transform('size') >= 80].copy()
glob = np.corrcoef(sub.precip_crit, sub.rinde_kgha)[0, 1]
within = (sub.groupby('provincia')
             .apply(lambda x: np.corrcoef(x.precip_crit, x.rinde_kgha)[0, 1])
             .sort_values())
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(within.index, within.values,
        color=['#C44E52' if v < 0 else '#55A868' for v in within.values])
ax.axvline(glob, ls='--', color='k', label=f'pooled = {glob:.2f}')
ax.axvline(0, color='0.6', lw=0.8)
ax.set_title('corr(precip ventana crítica, rinde) por provincia')
ax.set_xlabel('correlación de Pearson'); ax.legend(); plt.tight_layout(); plt.show()"""),
        code("""\
# Mismas nubes de puntos, pendientes distintas por provincia
fig, ax = plt.subplots(figsize=(7, 5))
for prov_name, c in [('ENTRE RIOS', '#55A868'), ('BUENOS AIRES', '#4C72B0'),
                     ('CHACO', '#DD8452')]:
    s = df[df.provincia == prov_name]
    ax.scatter(s.precip_crit, s.rinde_kgha, s=8, alpha=0.3, color=c, label=prov_name)
    b = np.polyfit(s.precip_crit, s.rinde_kgha, 1)
    xs = np.linspace(s.precip_crit.min(), s.precip_crit.max(), 20)
    ax.plot(xs, np.polyval(b, xs), color=c, lw=2.5)
ax.set_xlabel('precip ventana crítica (mm/día promedio)')
ax.set_ylabel('rinde (kg/ha)'); ax.legend()
ax.set_title('Relación precip–rinde: pendiente distinta por zona')
plt.tight_layout(); plt.show()"""),
        md("""\
**Conclusión del punto 2:** conviene **separar por zonas de clima homogéneo**. Así
cada modelo aprende la relación clima–rinde propia de su zona, sin promediarla con
zonas donde es distinta (o de signo opuesto)."""),
        md("""\
## 3. Feature engineering: features agronómicas

En vez de darle al modelo 7 meses × 7 variables de promedios crudos, construimos
features con **sentido agronómico**, centradas en la ventana crítica del cultivo
(`add_agro_features` del Componente A):

- `agro_precip_crit`  — lluvia total en la ventana crítica.
- `agro_tmax_crit`    — temperatura máxima media (estrés térmico).
- `agro_thermamp_crit`— amplitud térmica media.
- `agro_waterbal_crit`— balance hídrico Σ(precip − PET) (Hargreaves).

Comparamos cuánto correlaciona cada una con el rinde vs. la lluvia mensual cruda."""),
        code("""\
month_cols = [f'prectotcorr_{m}' for m in ['nov', 'dic', 'ene', 'feb']]
cors = (df[list(AGRO) + month_cols + ['rinde_kgha']].corr()['rinde_kgha']
        .drop('rinde_kgha').sort_values())
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(cors.index, cors.values,
        color=['#55A868' if c in AGRO else '#4C72B0' for c in cors.index])
ax.axvline(0, color='0.6', lw=0.8)
ax.set_title('Correlación con el rinde: agro (verde) vs precip mensual (azul)')
ax.set_xlabel('correlación de Pearson'); plt.tight_layout(); plt.show()"""),
        md("""\
El **balance hídrico** de la ventana crítica es de las features más correlacionadas
con el rinde — más que cualquier mes de lluvia suelto —, y encima es interpretable.
Ya están cableadas en el pipeline: `datos.prepare(CULTIVO, use_agro=True)`."""),
        md("""\
## 4. Definición de zonas balanceadas

Las **provincias** separan por clima pero están muy desbalanceadas (Buenos Aires
tiene ~3600 filas, Formosa ~100). Alternativa: agrupar **departamentos cercanos**
con KMeans sobre su centroide (lat, lon) → zonas geográficas contiguas y más
parejas en tamaño (`datos.assign_zonas(method='geo')`)."""),
        code("""\
pz = datos.assign_zonas(panel, n_zonas=6, method='geo')
dz = pz[pz.cultivo == CULTIVO].dropna(subset=['zona', 'lat', 'lon']).copy()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for z, s in dz.groupby('zona'):
    ax[0].scatter(s.lon, s.lat, s=6, alpha=0.4, label=z)
ax[0].set_title('Zonas geográficas (KMeans lat/lon, 6 zonas)')
ax[0].set_xlabel('lon'); ax[0].set_ylabel('lat'); ax[0].legend(markerscale=2)
vc = dz.zona.value_counts().sort_values()
ax[1].barh(vc.index, vc.values, color='#4C72B0')
ax[1].set_title('Tamaño de cada zona (balanceadas)'); ax[1].set_xlabel('n filas')
plt.tight_layout(); plt.show()"""),
        code("""\
dz['precip_crit'] = dz[[f'prectotcorr_{m}' for m in MESES_CRIT]].sum(axis=1)
(dz.groupby('zona')
   .agg(n=('rinde_kgha', 'size'), rinde=('rinde_kgha', 'mean'),
        precip=('precip_crit', 'mean'), lat=('lat', 'mean'))
   .round(1).sort_values('lat'))"""),
        md("""\
## Conclusión y próximos pasos

- El rinde está dominado por **espacio** (zona) y **tendencia**; el clima del año
  aporta sobre eso, pero **su efecto depende de la zona**.
- **Separar por zonas** de clima homogéneo (geo-clustering balanceado) debería
  reducir el ruido entre-zonas y dejar que cada modelo capture su relación
  clima–rinde. Es el próximo experimento.
- Las **features agronómicas** (balance hídrico, estrés térmico de la ventana
  crítica) correlacionan mejor con el rinde que los promedios mensuales crudos y ya
  están disponibles con `use_agro=True`.

**Siguiente:** modelar por zona (un modelo por zona, o zona como feature) con las
features agro, y comparar contra el modelo pooled de los notebooks 02–05."""),
    ]


if __name__ == "__main__":
    build(os.path.join(HERE, "00_eda_rinde_y_features.ipynb"), nb00())
    build(os.path.join(HERE, "01_baselines.ipynb"), nb01())
    build(os.path.join(HERE, "02_hp_regresion_lineal.ipynb"), nb02())
    build(os.path.join(HERE, "03_hp_xgboost.ipynb"), nb03())
    build(os.path.join(HERE, "04_hp_red_neuronal.ipynb"), nb04())
    build(os.path.join(HERE, "05_comparacion_modelos.ipynb"), nb05())
