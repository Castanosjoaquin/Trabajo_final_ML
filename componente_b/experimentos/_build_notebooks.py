"""Genera los 5 notebooks de experimentos del Componente B.

Cada notebook es narrativo y reproducible: importa el pipeline (`datos.py`),
la evaluación (`evaluacion.py`) y los modelos (`modelos/`), y entrena VISIBLE.
Después se ejecutan con `jupyter nbconvert --execute --inplace` para embeber
salidas.

    python componente_b/experimentos/_build_notebooks.py
"""
from __future__ import annotations

import os
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Setup común: paths, imports, dataset. Va de primera en cada notebook.
# El dataset "principal" del Componente B usa las features agronómicas de ventana
# crítica (use_agro) y el suavizado del target encoding de depto (enc_smooth): es
# la configuración que mejor generaliza y la que se tunea/compara en todos lados.
SETUP = """\
import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))          # componente_b/ (datos, evaluacion)
warnings.filterwarnings('ignore')                  # silenciar ConvergenceWarning de sklearn

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')

import datos, evaluacion as ev
from modelos import (LinearRegressor, XGBoostRegressor, NeuralNetRegressor,
                     RandomForestRegressorModel, HistGBMRegressor, StackingRegressorModel)

# Cultivo del estudio (cambiar a 'maiz' para reproducir con maíz).
CULTIVO = 'soja'
ds = datos.prepare(CULTIVO, use_agro=True, enc_smooth=10.0)
print(f'{CULTIVO}: {len(ds.feature_cols)} features | '
      f'train {ds.X_train.shape[0]} filas (≤{datos.TRAIN_END}) | '
      f'test {ds.X_test.shape[0]} filas (≥{datos.TEST_START})')
"""


# Helper (para nb05+): carga los best-params re-tuneados con CV honesta si existen.
LOAD_BEST = """\
import json
_bpath = 'retuning_cv_honesta.json'
BEST_ALL = json.load(open(_bpath, encoding='utf-8'))[CULTIVO] if os.path.exists(_bpath) else {}
def best_of(name, fallback):
    \"\"\"best-params re-tuneados del modelo `name` (o `fallback` si no hay json).\"\"\"
    return BEST_ALL.get(name, {}).get('best_params', fallback)
print('re-tuning disponible:', sorted(k for k in BEST_ALL if not k.startswith('_')) or 'NO (usando fallbacks)')
"""


# Sección transversal (nbs 02–04): probar la config final sobre ambos datasets y,
# sobre el mejor, las tres estrategias con el latente del mejor detector del
# Componente A (VAE recon_prob).
def _ds_latente_md():
    return new_markdown_cell("""\
## El mejor modelo del Componente A como features

Con la **configuración final ya elegida**, evaluamos en test el dataset (único,
unificado: clima + NDVI + ERA5) contra tres estrategias que reusan el **mejor
detector del Componente A** (el VAE `recon_prob`, que aprendió a representar el
clima "normal"): concatenar su **espacio latente**, hacer la regresión **solo en
el latente**, y agregar la categórica **`es_anomalo`** (su score umbralado).

El latente/score se computan en `latente.py` (y se cachean). *La primera corrida
entrena el VAE, así que tarda unos minutos.*""")


def ds_latente_code(model_name, params_var, fixed="None"):
    return new_code_cell(
        f"tabla_lat = ev.comparar_latente(\n"
        f"    {model_name}, {params_var}, CULTIVO, fixed={fixed},\n"
        f"    vae_kwargs=dict(score_seeds=(42, 43, 44)))\n"
        f"tabla_lat")


DS_LATENTE_MD = _ds_latente_md()


# Si se pasan nombres por CLI, solo se (re)generan esos notebooks (así se puede
# regenerar UNO sin pisar los outputs ya ejecutados del resto).
#   python _build_notebooks.py 10_prediccion_final.ipynb
ONLY = set(sys.argv[1:])


def build(path, cells):
    if ONLY and os.path.basename(path) not in ONLY:
        return
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(path, "w", encoding="utf-8") as f:   # utf-8 explícito: Windows es cp1252
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
porcentual)."""),
        code("""\
filas = [
    ev.evaluar('media global',  ev.pred_media(ds),      ds),
    ev.evaluar('media x depto',  ev.pred_media_depto(ds), ds),
]
tabla_base = ev.tabla_comparativa(filas, ordenar_por='rmse')
tabla_base"""),
        md("""\
La **media por departamento** ya explica bastante más varianza que la media global:
casi todo el poder predictivo "fácil" del rinde es *dónde* está el campo, no el
clima del año. Ese es el número que cualquier modelo con clima tiene que superar."""),
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
print('Config final:', modelo.get_config())
# Métricas del modelo final vs el baseline, en la tabla estándar (RMSE/R²/MAE)
ev.tabla([ev.evaluar('Lineal (final)', pred, ds),
          ev.evaluar('baseline media x depto', ev.pred_media_depto(ds), ds)])"""),
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
ev.tabla([ev.evaluar('XGBoost (final)', pred, ds),
          ev.evaluar('baseline media x depto', ev.pred_media_depto(ds), ds)])"""),
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
ev.tabla([ev.evaluar('Red neuronal (final)', pred, ds),
          ev.evaluar('baseline media x depto', ev.pred_media_depto(ds), ds)])"""),
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
# 05 — Comparación integral de modelos

Juntamos **todos** los modelos: los baselines (01), las tres familias tuneadas
(02–04: lineal, XGBoost, red neuronal) y los **ensembles adicionales** —
Random Forest, HistGradientBoosting y un **stacking** (XGBoost + MLP con
meta-modelo Ridge). Todos entrenados sobre el mismo train y evaluados sobre el
mismo test (≥2021), con los hiperparámetros **re-tuneados por CV temporal
honesta** (`_retune_all.py` → `retuning_cv_honesta.json`).

Comparamos de las tres formas que importan para decidir el mejor:
1. **Métricas de error** en test: RMSE (principal), MAE, R², sMAPE.
2. **Skill score** vs. la climatología (media por depto): cuánto aporta el clima
   del año por encima de "cada depto rinde lo de siempre".
3. **Test de Diebold–Mariano**: ¿la diferencia entre modelos es *significativa* o
   ruido? (H0: misma precisión; p<0.05 ⇒ diferencia real)."""),
        code(SETUP),
        code(LOAD_BEST),
        md("""\
## Entrenamiento de todos los modelos

Cada modelo con sus hiperparámetros re-tuneados (o un fallback razonable si no
está el json). El stacking recibe los años para armar sus meta-features
out-of-fold de forma temporal (sin leakage)."""),
        code("""\
yrs = ds.meta_train['campania_inicio'].values
def fit_pred(model):
    return model.fit(ds.X_train, ds.y_train).predict(ds.X_test)

preds = {}
preds['media global']  = ev.pred_media(ds)
preds['media x depto'] = ev.pred_media_depto(ds)
preds['Lineal']       = fit_pred(LinearRegressor(**best_of('linear', {'penalty':'ridge','alpha':10.0})))
preds['Random Forest'] = fit_pred(RandomForestRegressorModel(**best_of('rf', {}), random_state=42, n_jobs=-1))
preds['HistGBM']       = fit_pred(HistGBMRegressor(**best_of('hist_gbm', {}), random_state=42))
preds['XGBoost']       = fit_pred(XGBoostRegressor(**best_of('xgb', {}), random_state=42))
preds['Red neuronal']  = fit_pred(NeuralNetRegressor(**best_of('nn', {'hidden_dims':(64,32),'dropout':0.3}),
                                                     max_epochs=250, patience=30, random_state=42))
_stk = StackingRegressorModel(**best_of('stacking', {}))
_stk.fit(ds.X_train, ds.y_train, years=yrs)
preds['Stacking']      = _stk.predict(ds.X_test)

filas = [ev.evaluar(k, v, ds) for k, v in preds.items()]
tabla = ev.tabla_comparativa(filas, ordenar_por='rmse')
tabla"""),
        md("## Skill score vs. climatología (media por depto)\n\nPositivo = el modelo le gana a la climatología; 0 = empata."),
        code("""\
ref = ev.pred_media_depto(ds)
skill = {k: ev.skill_score(ds.y_test, v, ref) for k, v in preds.items() if k != 'media x depto'}
skill = pd.Series(skill).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(skill.index[::-1], skill.values[::-1],
        color=['#55A868' if v > 0 else '#C44E52' for v in skill.values[::-1]])
ax.axvline(0, color='0.5', lw=1); ax.set_xlabel('skill score (1 - RMSE/RMSE_clima)')
ax.set_title('Skill vs. climatología por modelo'); plt.tight_layout(); plt.show()
skill.round(3)"""),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ev.plot_comparativa(tabla, metrica='rmse', ax=axes[0])
ev.plot_comparativa(tabla, metrica='r2', ax=axes[1])
plt.tight_layout(); plt.show()"""),
        md("""\
## Test de Diebold–Mariano: ¿las diferencias son significativas?

Comparamos el **mejor modelo** (menor RMSE) contra cada uno de los demás. `dm<0`
= el mejor pierde *menos*; `p<0.05` = la diferencia es estadísticamente
significativa (no es ruido de muestreo del test)."""),
        code("""\
mejor = tabla.iloc[0]['modelo']
print('Mejor modelo por RMSE:', mejor)
filas_dm = []
for k in preds:
    if k == mejor: continue
    dm = ev.diebold_mariano(ds.y_test, preds[mejor], preds[k], loss='se')
    filas_dm.append({'vs': k, 'DM': dm['dm'], 'p_value': dm['p_value'],
                     'significativo (p<0.05)': dm['p_value'] < 0.05})
pd.DataFrame(filas_dm).sort_values('p_value')"""),
        md("## Predicho vs. real de los mejores modelos"),
        code("""\
top3 = [m for m in tabla['modelo'] if m not in ('media global', 'media x depto')][:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, nombre in zip(axes, top3):
    ev.plot_pred_vs_real(ds.y_test, preds[nombre], nombre, color=ev.C_XGB, ax=ax)
plt.tight_layout(); plt.show()"""),
        md("""\
## Conclusión

- Todos los modelos con clima **superan a la media global**; el listón real es la
  **media por departamento** (estructura espacial pura), y el **skill score** mide
  cuánto agrega el clima del año sobre eso.
- Los **ensembles de árboles** (Random Forest / HistGBM / XGBoost) dominan: capturan
  interacciones no lineales entre clima, espacio y tendencia. El **Diebold–Mariano**
  dice si la diferencia entre el puntero y el resto es real o ruido.
- El margen sobre el baseline por depto cuantifica cuánto aporta el **clima del año**
  por encima de "cada depto rinde lo de siempre" — lo difícil de predecir del rinde."""),
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

**Sobre el latente del Componente A** (nbs 02–04): concatenar el **latente del VAE**
suele empeorar la regresión (y solo-latente es lo peor): ese latente resume el clima
*normalizado por depto*, así que tira la señal espacial y de tendencia que es justo la
que más predice. `es_anomalo` tampoco ayuda, dominado por el *distribution shift*.
Coherente con el techo del Componente A."""),
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


def nb06():
    return [
        md("""\
# 06 — Un modelo por zona (geo-clustering)

**Hipótesis (del EDA, nb 00):** la relación clima–rinde *cambia según la zona*
(fuerte en la Pampa húmeda, casi nula en el norte subtropical). Un único modelo
"pooled" tiene que promediar esas relaciones distintas. Entonces: ¿anda mejor
**un modelo por zona**, cada uno especializado en el clima de su región?

Lo probamos en serio y lo interpretamos zona por zona. Las zonas son las
**geográficas balanceadas** del nb 00 (KMeans sobre lat/lon de los departamentos),
y el modelo es el **XGBoost ganador (nb 03) + features agronómicas (nb 00)**, igual
para el pooled y para cada zona (comparación justa)."""),
        code("""\
import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
import datos, evaluacion as ev
from modelos import XGBoostRegressor

CULTIVO = 'soja'
N_ZONAS = 6
# Config ganadora de XGBoost (nb 03). Misma para pooled y por zona.
BEST = dict(max_depth=3, learning_rate=0.05, n_estimators=200, subsample=0.7,
            colsample_bytree=1.0, min_child_weight=1, reg_lambda=5.0,
            reg_alpha=1.0, gamma=1.0)
def fit_xgb(ds):
    return XGBoostRegressor(**BEST, random_state=42).fit(ds.X_train, ds.y_train)

# Panel con zonas + dataset pooled de referencia (base + agro).
panel = datos.assign_zonas(datos.load_panel(), n_zonas=N_ZONAS, method='geo')
ds_pool = datos.build_reg_dataset(panel, CULTIVO, use_agro=True)
print(f'{CULTIVO}: pooled train={len(ds_pool.y_train)} test={len(ds_pool.y_test)} '
      f'| {N_ZONAS} zonas geográficas')"""),
        md("## Composición de las zonas\n\nCada zona agrupa departamentos cercanos; las ordenamos de norte a sur (por latitud)."),
        code("""\
d = panel[panel.cultivo == CULTIVO].dropna(subset=['zona'])
comp = (d.groupby('zona')
          .agg(n=('rinde_kgha', 'size'),
               prov_principal=('provincia', lambda s: s.mode().iat[0]),
               rinde=('rinde_kgha', 'mean'), lat=('lat', 'mean'))
          .sort_values('lat', ascending=False))
comp"""),
        md("""\
## Modelo pooled y su desempeño DENTRO de cada zona

Primero el pooled (un solo modelo con todas las filas). Además de la métrica global,
medimos su R² **dentro del test de cada zona** — así después comparamos, zona por
zona, contra el modelo especializado."""),
        code("""\
mp = fit_xgb(ds_pool)
pred_pool = mp.predict(ds_pool.X_test)
met_pool = ev.metricas(ds_pool.y_test, pred_pool)
print('POOLED (base+agro) global:', {k: round(v, 3) for k, v in met_pool.items()})

zt = ds_pool.meta_test['zona'].values
pool_por_zona = {z: ev.metricas(ds_pool.y_test[zt == z], pred_pool[zt == z])['r2']
                 for z in np.unique(zt)}"""),
        md("""\
## Un modelo por zona

Para cada zona: su propio split temporal, codificación de depto, escalado y XGBoost
(todo calculado dentro de la zona). Reportamos la métrica por zona y la global
uniendo los tests de todas las zonas (que juntos son el test completo)."""),
        code("""\
zds = datos.build_zona_datasets(CULTIVO, n_zonas=N_ZONAS,
                                method='geo', use_agro=True)
filas, yt_all, yp_all = [], [], []
for z, dz in zds.items():
    m = fit_xgb(dz); pr = m.predict(dz.X_test)
    mm = ev.metricas(dz.y_test, pr)
    filas.append({'zona': z, 'n_train': len(dz.y_train), 'n_test': len(dz.y_test),
                  'R2_pooled': pool_por_zona.get(z, np.nan),
                  'R2_por_zona': mm['r2'], 'RMSE_por_zona': mm['rmse']})
    yt_all.append(dz.y_test); yp_all.append(pr)
tabla_zona = pd.DataFrame(filas).sort_values('R2_por_zona', ascending=False)
tabla_zona"""),
        md("""\
### Zona por zona: ¿dónde ayuda especializar?

La barra verde (modelo por zona) contra la azul (pooled) en el test de cada zona."""),
        code("""\
t = tabla_zona.sort_values('zona')
x = np.arange(len(t)); w = 0.4
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, t.R2_pooled, w, label='pooled', color='#4C72B0')
ax.bar(x + w/2, t.R2_por_zona, w, label='modelo por zona', color='#55A868')
ax.axhline(0, color='0.6', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(t.zona)
ax.set_ylabel('R² en el test de la zona')
ax.set_title('Pooled vs. modelo por zona, zona por zona'); ax.legend()
plt.tight_layout(); plt.show()"""),
        md("""\
## El neto global y la zona que rompe

Uniendo los tests, comparamos pooled vs un-modelo-por-zona. Y miramos la zona que
colapsa: es el **norte subtropical**, justo donde el EDA (nb 00) mostró correlación
clima–rinde casi nula — ahí el modelo especializado sobreajusta ruido y extrapola
pésimo, arrastrando el promedio."""),
        code("""\
yt_all = np.concatenate(yt_all); yp_all = np.concatenate(yp_all)
met_zona_glob = ev.metricas(yt_all, yp_all)
comp_tabla = ev.tabla_comparativa([
    {'modelo': 'pooled (base+agro)', **met_pool},
    {'modelo': 'un modelo por zona', **met_zona_glob},
], ordenar_por='rmse')
peor = tabla_zona.sort_values('R2_por_zona').iloc[0]['zona']
print('Zona que colapsa:', peor, '->',
      dict(panel[(panel.cultivo == CULTIVO) & (panel.zona == peor)]
           .provincia.value_counts().head(3)))
comp_tabla"""),
        md("""\
## Interpretación

- **Especializar por zona ayuda donde la señal climática es fuerte y hay datos**
  (las zonas pampeanas: R² sube claramente sobre el pooled) — la hipótesis del EDA
  se cumple *ahí*.
- **Pero rompe en el norte subtropical**: poca señal clima–rinde + menos datos por
  modelo ⇒ sobreajuste y extrapolación catastrófica en test. Esa zona sola hunde la
  métrica global.
- **Neto global: un modelo por zona NO le gana al pooled.** El pooled comparte
  "fuerza estadística" entre zonas (regulariza las zonas ruidosas con las buenas) y
  ya captura lo espacial vía `depto_enc`. Fragmentar los datos pierde eso.

### Cómo hacer que las zonas sí paguen (próximos pasos)
1. **Split selectivo**: modelo por zona solo en las zonas donde mejora (Pampa) y
   pooled para las ruidosas — lo mejor de los dos.
2. **Partial pooling / jerárquico**: un modelo que comparte parámetros entre zonas
   pero deja variar la relación clima–rinde (efectos por zona), sin fragmentar.
3. **Más regularización en zonas chicas/ruidosas** (menos profundidad, más shrinkage)
   o un mínimo de filas más alto por zona.
4. **Cambiar el target a residuo/anomalía** por zona (sacar el nivel espacial), que
   es lo que de verdad depende del clima.

**Conclusión:** el geo-clustering es muy útil como *diagnóstico* (muestra dónde el
clima predice y dónde no), pero "un modelo por zona" a secas no mejora el global;
el camino es el pooling parcial o el split selectivo."""),
    ]


# ===========================================================================
# NB 07 — Integración Componente A -> Componente B (el aporte distintivo)
# ===========================================================================
def nb07():
    return [
        md("""\
# 07 — Integración Componente A ↔ Componente B

El **núcleo del proyecto**: acoplar el detector no supervisado (Componente A, un
VAE) con el predictor de rinde (Componente B). Tres análisis que pide la consigna:

1. **Consistencia cruzada** — ¿el score de anomalía del VAE correlaciona con el
   |residuo| del predictor? Si ambos ven la misma estructura, una campaña "rara"
   para el VAE también le cuesta al regresor (Spearman sobre el test).
2. **Aporte del detector al predictor** (lift) — RMSE del predictor **con vs. sin**
   las features del VAE (latente / score), vía `ev.comparar_latente`.
3. **Cuantificación económica** — rinde **contrafactual** bajo clima normal −
   rinde real, × superficie sembrada → pérdida de la sequía 2022/23, comparable al
   benchmark de la BCR (USD 14.140 M)."""),
        code(SETUP),
        code(LOAD_BEST),
        code("""\
import integracion, latente
# Modelo final del predictor: XGBoost re-tuneado.
model = XGBoostRegressor(**best_of('xgb', {}), random_state=42).fit(ds.X_train, ds.y_train)
y_pred = model.predict(ds.X_test)
print('predictor (test):', {k: round(v, 3) for k, v in ev.metricas(ds.y_test, y_pred).items()})
# Features del VAE del Componente A (score + latente), cacheadas.
vf = latente.vae_features(CULTIVO)"""),
        md("""\
## 1. Consistencia cruzada (Spearman)

Correlación de Spearman entre el score de anomalía del VAE y el valor absoluto del
residuo del predictor, sobre el test. Positiva y significativa ⇒ **ambos
componentes detectan la misma estructura** (test de robustez metodológica)."""),
        code("""\
cc = integracion.consistencia_cruzada(ds, y_pred, vf)
print(f"Spearman(score VAE, |residuo predictor|) = {cc['spearman_rho']:.3f}  "
      f"(p={cc['p_value']:.1e}, n={cc['n']})")
resid = np.abs(ds.y_test - y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(vf['score_test'], resid, s=10, alpha=0.35, color=ev.C_XGB, edgecolors='none')
ax.set_xlabel('score de anomalía del VAE'); ax.set_ylabel('|residuo| del predictor')
ax.set_title(f"Consistencia cruzada  (Spearman ρ={cc['spearman_rho']:.3f})")
plt.tight_layout(); plt.show()"""),
        md("""\
## 2. Aporte del detector al predictor (lift)

RMSE del predictor sobre el dataset base vs. tres estrategias que reusan el VAE:
concatenar su **latente**, usar **solo el latente**, y agregar la categórica
**`es_anomalo`**. Si el detector aporta contexto climático comprimido, debería
bajar el RMSE (sobre todo en campañas extremas)."""),
        code("""\
tabla_lat = ev.comparar_latente(XGBoostRegressor, best_of('xgb', {}), CULTIVO,
                                fixed={'random_state': 42},
                                vae_kwargs=dict(score_seeds=(42, 43, 44)))
tabla_lat"""),
        md("""\
## 3. Cuantificación económica de la sequía 2022/23

Rinde **contrafactual** (qué habría rendido cada depto con clima normal, poniendo
las features climáticas en su media de train) − rinde real, ponderado por
superficie sembrada. Agregamos la campaña 2022/23 y miramos los deptos más
golpeados."""),
        code("""\
y_cf = integracion.contrafactual_normal(model, ds)
res = integracion.resumen_2223(ds, y_cf)
print(f"2022/23 ({res['n_filas']} deptos-cultivo): pérdida total "
      f"{res['perdida_total_tn']:,.0f} tn  ({res['perdida_media_kgha']:.0f} kg/ha promedio)")
res['top_deptos']"""),
        code("""\
top = res['top_deptos'].head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top['departamento'] + ' (' + top['provincia'].str[:3] + ')',
        top['perdida_tn'] / 1e3, color='#C44E52')
ax.set_xlabel('pérdida estimada (miles de tn)')
ax.set_title('2022/23 — deptos más golpeados (contrafactual clima normal − real)')
plt.tight_layout(); plt.show()"""),
        md("""\
## Conclusión

- **Consistencia cruzada:** la correlación positiva y significativa entre el score
  del VAE y el residuo del predictor indica que ambos componentes captan la misma
  señal de anomalía — el sistema es coherente de punta a punta.
- **Lift del latente:** en línea con el techo estructural del Componente A, el
  latente del VAE aporta poco a la regresión (resume el clima *normalizado por
  depto*, tira la señal espacial/tendencia que domina el rinde). El aporte real del
  acople es **conceptual y económico**, no un salto de RMSE.
- **Cuantificación:** el contrafactual aísla la campaña 2022/23 como pérdida masiva
  concentrada en el sur de Córdoba y Santa Fe, consistente con la sequía histórica y
  el orden de magnitud del benchmark BCR."""),
    ]


# ===========================================================================
# NB 08 — Dos momentos de predicción + lags del rinde (ablations)
# ===========================================================================
def nb08():
    return [
        md("""\
# 08 — Momentos de predicción y lags del rinde

Dos extensiones que pide la consigna, como **ablations** sobre el mejor modelo:

1. **Dos momentos del calendario agrícola** (`momento` en `datos.build_reg_dataset`):
   - `pre_siembra` — solo lo conocible antes de sembrar (ONI + humedad de suelo de
     invierno + estructurales). Sirve para decisiones tempranas.
   - `pre_cosecha` — clima y NDVI hasta febrero (sin marzo).
   - `full` — toda la campaña Sep–Mar (referencia).
2. **Lags del rinde** (`use_lags`): rinde de 1–k campañas previas + media móvil,
   todo con `shift` (nunca el año actual → sin leakage)."""),
        code("""\
import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
import datos, evaluacion as ev, json
from modelos import XGBoostRegressor
CULTIVO = 'soja'
BEST = json.load(open('retuning_cv_honesta.json', encoding='utf-8')).get(CULTIVO, {}).get('xgb', {}).get('best_params', {}) if os.path.exists('retuning_cv_honesta.json') else {}
def evalua(momento='full', use_lags=0, use_agro=None):
    if use_agro is None: use_agro = (momento == 'full')   # agro llega a marzo
    d = datos.prepare(CULTIVO, momento=momento, use_lags=use_lags, use_agro=use_agro, enc_smooth=10.0)
    m = XGBoostRegressor(**BEST, random_state=42).fit(d.X_train, d.y_train)
    p = m.predict(d.X_test)
    r = ev.metricas(d.y_test, p)
    r['skill'] = ev.skill_score(d.y_test, p, ev.pred_media_depto(d))
    r['n_feats'] = len(d.feature_cols)
    return r, d, p"""),
        md("## 1. Los tres momentos de predicción"),
        code("""\
filas = []
res = {}
for mom in ['pre_siembra', 'pre_cosecha', 'full']:
    r, d, p = evalua(momento=mom)
    res[mom] = (d, p)
    filas.append({'momento': mom, 'n_feats': r['n_feats'], 'RMSE': r['rmse'],
                  'R2': r['r2'], 'sMAPE': r['smape'], 'skill': r['skill']})
tabla_mom = pd.DataFrame(filas)
tabla_mom"""),
        md("""\
Cuanto más tarde en la campaña se predice, más información climática observada hay
→ mejor debería ser. `pre_siembra` es el caso más difícil (casi sin clima del año);
si aun así le gana a la climatología, hay señal anticipable (ENSO, humedad de suelo)."""),
        code("""\
# ¿La diferencia full vs pre_cosecha es significativa? (mismas filas de test)
d_full, p_full = res['full']; d_pc, p_pc = res['pre_cosecha']
dm = ev.diebold_mariano(d_full.y_test, p_full, p_pc, loss='se')
print(f"Diebold-Mariano full vs pre_cosecha: DM={dm['dm']:.2f}  p={dm['p_value']:.3f}  "
      f"-> {'diferencia significativa' if dm['p_value'] < 0.05 else 'sin diferencia significativa'}")"""),
        md("## 2. Lags del rinde (features autorregresivas)"),
        code("""\
filas = []
for k in [0, 1, 3, 5]:
    r, _, _ = evalua(momento='full', use_lags=k, use_agro=True)
    filas.append({'lags': k, 'n_feats': r['n_feats'], 'RMSE': r['rmse'],
                  'R2': r['r2'], 'skill': r['skill']})
tabla_lags = pd.DataFrame(filas)
tabla_lags"""),
        code("""\
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(tabla_mom['momento'], tabla_mom['R2'], color='#4C72B0')
ax[0].set_title('R² por momento de predicción'); ax[0].axhline(0, color='0.6', lw=0.8)
ax[1].plot(tabla_lags['lags'], tabla_lags['R2'], marker='o', color='#55A868')
ax[1].set_title('R² vs. nº de lags del rinde'); ax[1].set_xlabel('lags')
plt.tight_layout(); plt.show()"""),
        md("""\
## Conclusión

- **Momentos:** `full` y `pre_cosecha` rinden parecido (el clima de marzo aporta
  poco una vez que están sep–feb y el NDVI); el Diebold–Mariano dice si la diferencia
  es real. `pre_siembra` es mucho más difícil — su skill vs. climatología mide cuánto
  se puede **anticipar** antes de sembrar (ENSO + humedad de suelo).
- **Lags:** agregar el rinde de campañas previas aporta señal autorregresiva
  (persistencia depto), complementaria al `depto_enc`. El punto óptimo de k se ve en
  la curva; más lags achican el train (se pierden las primeras campañas de cada serie)."""),
    ]


# ===========================================================================
# NB 09 — Router por zonas (selección por CV): lo mejor de pooled y especializado
# ===========================================================================
def nb09():
    return [
        md("""\
# 09 — Router por zonas: especializar solo donde paga

El nb 06 mostró que "un modelo por zona" a secas **no** le gana al pooled: ayuda en
la Pampa pero colapsa en el norte subtropical (poca señal + pocos datos), y esa
zona hunde el neto. La solución: un **router** que, para cada zona, elige por
**validación cruzada temporal en train** si conviene el modelo especializado o el
pooled — sin mirar el test (eso sería leakage de selección).

Implementado en `wrapper_zonas.WrapperPorZona`. Comparamos router vs. pooled vs.
"un modelo por zona (puro)" en el test."""),
        code("""\
import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
import json
from wrapper_zonas import WrapperPorZona
from modelos import XGBoostRegressor
CULTIVO = 'soja'
BEST = json.load(open('retuning_cv_honesta.json', encoding='utf-8')).get(CULTIVO, {}).get('xgb', {}).get('best_params', {}) if os.path.exists('retuning_cv_honesta.json') else {}
BEST = {k: v for k, v in BEST.items() if k not in ('n_jobs',)}"""),
        md("## Entrenamiento del router (decisión por CV en train)"),
        code("""\
w = WrapperPorZona(XGBoostRegressor, BEST, cultivo=CULTIVO, use_agro=True,
                   enc_smooth=10.0, n_zonas=6, metric='rmse')
w.fit()
w.resumen_zonas()"""),
        md("La columna `elegido` dice, por zona, si el router se quedó con el modelo **especializado** (mejor CV) o el **pooled**."),
        md("## Resultado en test: router vs. pooled vs. por-zona puro"),
        code("""\
tabla = w.evaluar()
tabla"""),
        code("""\
fig, ax = plt.subplots(figsize=(7, 4))
t = tabla.sort_values('rmse')
ax.barh(t['modelo'][::-1], t['rmse'][::-1], color=['#55A868', '#4C72B0', '#C44E52'][:len(t)])
ax.set_xlabel('RMSE (kg/ha)'); ax.set_title('Router por zonas vs. alternativas (test)')
for i, v in enumerate(t['rmse'][::-1]):
    ax.text(v, i, f' {v:.0f}', va='center')
plt.tight_layout(); plt.show()"""),
        md("""\
## Conclusión

- El **router** se queda con el especializado solo en las zonas donde la CV en train
  lo respalda (típicamente las pampeanas) y usa el pooled en las ruidosas → acota el
  colapso del norte que hundía al "un modelo por zona" del nb 06.
- Así logra **igualar o superar al pooled** sin el riesgo del esquema ingenuo. La
  clave metodológica es que la decisión por zona sale de **CV temporal en train**,
  nunca del test.
- Es un ejemplo de *split selectivo* / mixture-of-experts liviano; el siguiente paso
  natural sería el **pooling parcial** (encoger cada zona hacia el pooled según su n)."""),
    ]


# ===========================================================================
# NB 10 — Modelos finales + predicciones sobre el test (archivo de entrega)
# ===========================================================================
def nb10():
    return [
        md("""\
# 10 — Modelos finales y predicciones sobre el test

Notebook de **cierre y entrega**. Reúne los modelos finales de los dos
componentes, con sus **hiperparámetros definitivos** (los re-tuneados por CV
temporal honesta en `retuning_cv_honesta.json`), los re-entrena sobre **todo el
train** y produce las **predicciones finales sobre el test** (≥2021):

- **Componente B (supervisado)** — rinde predicho (`rinde_pred_kgha`) del modelo
  campeón (elegido por **CV**, no por test), con la comparación de todos los
  modelos para justificar la elección.
- **Componente A (no supervisado)** — score de anomalía y flag `es_anomalo` del
  detector final (VAE `recon_prob` seed-ensemble).

Al final se arma **`predicciones_test.csv`** con el formato de entrega, y se deja
una función `predecir_entrega(...)` lista para re-apuntar al **test set oficial**
cuando se publique (24 h antes de la entrega).

> **Adaptación al test oficial:** la consigna dice que 24 h antes se publica un
> archivo de test con su estructura. Cuando llegue, sólo hay que pasarlo por
> `predecir_entrega(panel_nuevo)` (misma tubería de features y modelos ya
> entrenados) y entregar el CSV resultante. Acá lo demostramos sobre nuestro
> propio test temporal (≥2021), para el que sí tenemos el rinde real y podemos
> reportar métricas."""),
        code("""\
import sys, os, warnings, json
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')

import datos, evaluacion as ev, latente
from modelos import (LinearRegressor, XGBoostRegressor, NeuralNetRegressor,
                     RandomForestRegressorModel, HistGBMRegressor, StackingRegressorModel)

CULTIVOS = ['soja', 'maiz']
DS_KW = dict(use_agro=True, enc_smooth=10.0)          # config final del pipeline
BEST = json.load(open('retuning_cv_honesta.json', encoding='utf-8'))
def best_of(cultivo, name, fallback=None):
    return BEST.get(cultivo, {}).get(name, {}).get('best_params', fallback or {})
print('re-tuning cargado para:', list(BEST))"""),
        md("""\
## 1. Reconstrucción de los modelos finales

Cada modelo se instancia con sus hiperparámetros definitivos. El **campeón se
elige por el `cv_rmse`** (validación en train), de modo que la elección **no
depende del test** — el test sólo confirma."""),
        code("""\
def construir_modelos(cultivo):
    return {
        'Lineal':        LinearRegressor(**best_of(cultivo, 'linear', {'penalty': 'ridge', 'alpha': 10.0})),
        'Random Forest': RandomForestRegressorModel(**best_of(cultivo, 'rf'), random_state=42, n_jobs=-1),
        'HistGBM':       HistGBMRegressor(**best_of(cultivo, 'hist_gbm'), random_state=42),
        'XGBoost':       XGBoostRegressor(**best_of(cultivo, 'xgb'), random_state=42),
        'Red neuronal':  NeuralNetRegressor(**best_of(cultivo, 'nn', {'hidden_dims': (64, 32), 'dropout': 0.3}),
                                            max_epochs=250, patience=30, random_state=42),
    }

def campeon(cultivo):
    cvs = {n: BEST[cultivo][k].get('cv_rmse', np.inf)
           for n, k in [('Lineal','linear'), ('Random Forest','rf'), ('HistGBM','hist_gbm'),
                        ('XGBoost','xgb'), ('Red neuronal','nn')] if k in BEST.get(cultivo, {})}
    return min(cvs, key=cvs.get) if cvs else 'Random Forest'

for c in CULTIVOS:
    print(f'{c}: campeón por CV = {campeon(c)}')"""),
        md("""\
## 2. Comparación de todos los modelos en test (justifica el campeón)

Entrenamos cada modelo en el train y evaluamos en test. Reportamos RMSE/R²/sMAPE,
**skill vs. climatología** y marcamos el campeón elegido por CV."""),
        code("""\
def evaluar_cultivo(cultivo):
    ds = datos.prepare(cultivo, **DS_KW)
    yrs = ds.meta_train['campania_inicio'].values
    modelos = construir_modelos(cultivo)
    filas, preds = [], {}
    preds['media x depto'] = ev.pred_media_depto(ds)
    filas.append(ev.evaluar('media x depto', preds['media x depto'], ds))
    for nombre, m in modelos.items():
        p = m.fit(ds.X_train, ds.y_train).predict(ds.X_test)
        preds[nombre] = p
        filas.append(ev.evaluar(nombre, p, ds))
    # stacking (usa años para OOF temporal)
    stk = StackingRegressorModel(**best_of(cultivo, 'stacking', {})).fit(
        ds.X_train, ds.y_train, years=yrs)
    preds['Stacking'] = stk.predict(ds.X_test)
    filas.append(ev.evaluar('Stacking', preds['Stacking'], ds))
    tabla = ev.tabla_comparativa(filas, ordenar_por='rmse')
    ref = ev.pred_media_depto(ds)
    tabla['skill'] = [ev.skill_score(ds.y_test, preds[m], ref) if m in preds else np.nan
                      for m in tabla['modelo']]
    return ds, preds, tabla

resultados = {}
for c in CULTIVOS:
    ds, preds, tabla = evaluar_cultivo(c)
    resultados[c] = (ds, preds, tabla)
    print(f'\\n===== {c} (campeón CV: {campeon(c)}) =====')
    print(tabla.to_string(index=False))"""),
        md("Diebold–Mariano del campeón vs. el resto (¿la ventaja es significativa?)."),
        code("""\
for c in CULTIVOS:
    ds, preds, tabla = resultados[c]
    camp = campeon(c)
    print(f'--- {c}: {camp} vs. resto ---')
    for m in preds:
        if m == camp: continue
        dm = ev.diebold_mariano(ds.y_test, preds[camp], preds[m], loss='se')
        sig = 'sig.' if dm['p_value'] < 0.05 else 'n.s.'
        print(f'  vs {m:16s} DM={dm[\"dm\"]:+.2f}  p={dm[\"p_value\"]:.3f}  [{sig}]')"""),
        md("""\
## 3. Componente A: score de anomalía final sobre el test

El detector final es el **VAE `recon_prob` en seed-ensemble** (Componente A). Su
`anomaly_score` (continuo) va alineado fila a fila con el test del predictor y es
la señal principal.

**Caveat del flag binario.** El umbral calibrado en train marca casi TODO el test
como anómalo: entre 2021 y 2024 el score sube por *distribution shift* (años fuera
del rango de train), no porque cada campaña sea extrema. Para la entrega, definimos
`es_anomalo` como el **top-10% más anómalo por score dentro del set evaluado**
(ranking relativo, sin usar etiquetas) — así el flag señala las campañas *más*
anómalas de forma útil, y el score continuo queda para el ranking fino."""),
        code("""\
CONTAM = 0.10                                          # fracción marcada como anómala
scores_A = {}
for c in CULTIVOS:
    vf = latente.vae_features(c)                       # cacheado en ../.latente_cache
    score = vf['score_test']
    thr = np.quantile(score, 1.0 - CONTAM)             # umbral relativo al set evaluado
    flag_te = (score >= thr).astype(int)
    scores_A[c] = (score, flag_te)
    print(f'{c}: {int(flag_te.sum())} campañas marcadas anómalas de {len(flag_te)} '
          f'(top-{int(CONTAM*100)}% por score)')"""),
        md("""\
## 4. Archivo de entrega: `predicciones_test.csv`

Formato propuesto (adaptable a la estructura que publique la cátedra): una fila
por **departamento × campaña × cultivo** del test, con identificadores + la
predicción de rinde del campeón + el score/flag de anomalía. Se incluye
`rinde_real_kgha` **solo como referencia** para nuestra evaluación (en el test
oficial ciego no estaría)."""),
        code("""\
bloques = []
for c in CULTIVOS:
    ds, preds, tabla = resultados[c]
    camp = campeon(c)
    score, flag = scores_A[c]
    out = ds.meta_test[['provincia', 'departamento', 'campania']].copy()
    out.insert(0, 'cultivo', c)
    out['rinde_real_kgha'] = ds.y_test                 # referencia (no va en test ciego)
    out['rinde_pred_kgha'] = np.round(preds[camp], 1)
    out['modelo'] = camp
    out['anomaly_score'] = np.round(score, 4)
    out['es_anomalo'] = flag.astype(int)
    bloques.append(out)

entrega = pd.concat(bloques, ignore_index=True)
entrega.to_csv('predicciones_test.csv', index=False, encoding='utf-8')
print('guardado: predicciones_test.csv', entrega.shape)
entrega.head(8)"""),
        code("""\
# Métricas finales del archivo entregado (por cultivo), para el informe
for c in CULTIVOS:
    sub = entrega[entrega.cultivo == c]
    m = ev.metricas(sub.rinde_real_kgha.values, sub.rinde_pred_kgha.values)
    print(f'{c:5s} ({sub.modelo.iloc[0]:13s}): '
          f'RMSE={m[\"rmse\"]:.0f}  R2={m[\"r2\"]:.3f}  sMAPE={m[\"smape\"]:.1f}%')"""),
        md("""\
## 5. Predicción sobre el test set OFICIAL (cuando se publique)

Cuando la cátedra publique el archivo de test (24 h antes), guardarlo como panel
con el **mismo esquema** que `panel_union.parquet` (columnas de clima/NDVI/ERA5 +
identificadores) y correr la función de abajo. Entrena los modelos sobre **todo**
nuestro panel conocido y predice sobre las filas nuevas, devolviendo el CSV en el
mismo formato."""),
        code("""\
def predecir_entrega(panel_nuevo, cultivos=CULTIVOS, salida='predicciones_oficial.csv'):
    \"\"\"Predice rinde + anomalía sobre un panel de test externo.

    panel_nuevo: DataFrame con el MISMO esquema que panel_union (features de clima/
    NDVI/ERA5 + provincia/departamento/campania/campania_inicio/cultivo). Entrena
    sobre todo el panel conocido (train = histórico) y predice las filas nuevas.
    \"\"\"
    panel_base = datos.load_panel()
    te_start = int(panel_nuevo['campania_inicio'].min())
    panel = pd.concat([panel_base, panel_nuevo], ignore_index=True)
    bloques = []
    for c in cultivos:
        ds = datos.build_reg_dataset(panel, c, test_start=te_start,
                                     train_end=te_start - 1, **DS_KW)
        camp = campeon(c)
        modelo = construir_modelos(c)[camp]
        pred = modelo.fit(ds.X_train, ds.y_train).predict(ds.X_test)
        out = ds.meta_test[['provincia', 'departamento', 'campania']].copy()
        out.insert(0, 'cultivo', c)
        out['rinde_pred_kgha'] = np.round(pred, 1)
        out['modelo'] = camp
        bloques.append(out)
    entrega = pd.concat(bloques, ignore_index=True)
    entrega.to_csv(salida, index=False, encoding='utf-8')
    return entrega

# Ejemplo de uso (descomentar cuando exista el archivo oficial):
# panel_oficial = pd.read_parquet('test_oficial.parquet')
# predecir_entrega(panel_oficial)
print('predecir_entrega() lista para el test oficial.')"""),
        md("""\
## Conclusión

- El **campeón se elige por `cv_rmse`** (validación en train), no por test:
  **Random Forest en soja** y **HistGBM en maíz**. Nota honesta: en maíz, RF le
  gana a HistGBM en el test (ver la tabla), pero HistGBM ganó la CV — respetamos
  la elección por CV para **no decidir sobre el test**. Es el protocolo correcto y
  la comparación + Diebold–Mariano quedan transparentes para el lector.
- El **detector final** (VAE seed-ensemble) aporta el `anomaly_score` continuo por
  campaña; el flag `es_anomalo` marca el top-10% por score (ver caveat de
  distribution shift), integrando ambos componentes en una sola salida.
- `predicciones_test.csv` es el entregable; `predecir_entrega()` re-genera el CSV
  sobre el test oficial en cuanto se publique, sin tocar el resto del pipeline."""),
    ]


if __name__ == "__main__":
    build(os.path.join(HERE, "00_eda_rinde_y_features.ipynb"), nb00())
    build(os.path.join(HERE, "06_modelo_por_zona.ipynb"), nb06())
    build(os.path.join(HERE, "01_baselines.ipynb"), nb01())
    build(os.path.join(HERE, "02_hp_regresion_lineal.ipynb"), nb02())
    build(os.path.join(HERE, "03_hp_xgboost.ipynb"), nb03())
    build(os.path.join(HERE, "04_hp_red_neuronal.ipynb"), nb04())
    build(os.path.join(HERE, "05_comparacion_modelos.ipynb"), nb05())
    build(os.path.join(HERE, "07_integracion_A_B.ipynb"), nb07())
    build(os.path.join(HERE, "08_momentos_y_lags.ipynb"), nb08())
    build(os.path.join(HERE, "09_router_por_zona.ipynb"), nb09())
    build(os.path.join(HERE, "10_prediccion_final.ipynb"), nb10())
