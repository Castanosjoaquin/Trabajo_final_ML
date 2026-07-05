# Modelos — Componente B (predicción de rinde)

Tres regresores con una **interfaz común** para predecir `rinde_kgha` a partir de
las features climáticas del panel. Todos comparten el contrato de
[`base.py`](base.py):

```python
modelo.fit(X, y)      # entrena (X features, y = rinde en kg/ha)
modelo.predict(X)     # devuelve rinde estimado (kg/ha)
modelo.get_config()   # dict de hiperparámetros, para registrar/comparar
```

Uso:

```python
from componente_b.modelos import LinearRegressor, NeuralNetRegressor, XGBoostRegressor

modelo = XGBoostRegressor(max_depth=5, reg_lambda=2.0, learning_rate=0.03)
modelo.fit(X_train, y_train)
y_hat = modelo.predict(X_test)
```

## Modelos y su regularización

| Modelo | Clase | Regularización tuneable |
|--------|-------|-------------------------|
| Regresión lineal | `LinearRegressor` | `penalty` = `none` (OLS) / `l2`(=`ridge`) / `l1`(=`lasso`) / `elasticnet`; `alpha` (fuerza), `l1_ratio` (mezcla L1/L2 en elasticnet) |
| Red neuronal (MLP) | `NeuralNetRegressor` | `weight_decay` (L2), `l1_lambda` (L1), `dropout`, `use_batch_norm`, early stopping (`patience`, `val_frac`) |
| XGBoost | `XGBoostRegressor` | `reg_lambda` (L2), `reg_alpha` (L1), `gamma` (poda), `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `learning_rate`+`n_estimators` |

Detalles de arquitectura/optimización (capas, `lr`, `lr_schedule`, etc.) están en
el constructor de cada clase y documentados en su docstring.

### Notas de uso

- **Neural net**: estandariza el target internamente (guarda media/desvío de
  train) y usa un split interno de validación para early stopping. Las features
  `X` se asumen ya escaladas por el pipeline de datos.
- **XGBoost**: para early stopping, pasale `eval_set=(X_val, y_val)` a `fit()` y
  seteá `early_stopping_rounds`. `best_iteration` queda en `get_config()`.

## ⚠️ macOS: conflicto de OpenMP (xgboost + torch)

En macOS, xgboost y torch traen cada uno su propio `libomp.dylib`. Con ambos en
el mismo proceso, entrenar xgboost **segfaultea (exit 139)**. El paquete mitiga
esto importando xgboost antes que torch, pero eso no alcanza cuando otro módulo
(p. ej. el pipeline del Componente A) ya importó torch.

Fix definitivo (una sola vez, y de nuevo si reinstalás xgboost/torch):

```bash
python componente_b/fix_openmp_macos.py
```

Hace que xgboost use el mismo `libomp` que torch → una sola copia del runtime, sin
crash y sin importar el orden de import. Requiere `brew install libomp` para que
xgboost cargue (dependencia de la wheel).
