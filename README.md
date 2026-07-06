# Componente A — Detección de Anomalías Agroclimáticas

Detecta **campañas agrícolas con rinde anómalo** (soja y maíz, Argentina, 1981–2024)
usando solo variables climáticas mensuales. Es un problema **no supervisado**: los modelos
se entrenan únicamente con campañas normales y puntúan cada campaña por cuán "rara" es.

El trabajo está contado y **reproducido** en los notebooks de `componente_a/experiments/`:
cada uno entrena los modelos de verdad (no lee resultados guardados), así que se puede
abrir cualquiera, hacer **Run all**, y recrear todo desde cero.

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Todo el Componente A vive bajo `componente_a/` y se corre desde ahí (`cd componente_a`).
> El venv queda en la raíz del repo (`.venv/`).

---

## Estructura

```
Trabajo_final_ML/
│
├── data/processed/panel_union.parquet    # Dataset ÚNICO: clima + NDVI + ERA5 (source-only)
│                                         #   lo construye componente_a/eda/build_panel_union.py
│
├── componente_a/
│   ├── src/                              # Paquete mínimo (solo 3 módulos)
│   │   ├── config.py                     #   Constantes: rutas, features, splits, etiqueta
│   │   ├── data.py                       #   Pipeline: panel → etiqueta z_rinde → splits → normalización
│   │   └── models/                       #   Detectores (interfaz común fit / score_samples)
│   │       ├── base.py                   #     AnomalyDetector (ABC)
│   │       ├── baselines.py              #     IsolationForest, OneClassSVM
│   │       ├── ae.py                     #     AE + Denoising AE
│   │       ├── vae.py                    #     VAE (score recon_prob)
│   │       ├── ensemble.py               #     EnsembleDetector (seed-ensemble)
│   │       ├── deep_baselines.py         #     DeepODDetector (métodos modernos, deepod)
│   │       └── trainer.py                #     Loop de entrenamiento (Adam + early stopping)
│   │
│   ├── experiments/                      # EL TRABAJO: notebooks reproducibles (entrenan inline)
│   │   ├── lab.py                        #   Utilidades: métricas y gráficos (visibles, cortas)
│   │   └── 00..07_*.ipynb
│   ├── eda/                              # EDAs + construcción del panel (build_panel_union.py)
│   └── tests/                            # test_repro (dataset) + test_smoke (modelos)
│
├── docs/                                 # Papers de referencia + consigna
├── requirements.txt                      # dependencias del proyecto (A + B)
└── README.md
```

**No hay** capa de configs YAML, ni orquestador CLI, ni store de runs: los
hiperparámetros y el entrenamiento viven **a la vista en los notebooks**. `src/` es solo el
pipeline de datos + los modelos.

---

## El recorrido (notebooks de `experiments/`)

| nb | contenido |
|---|---|
| `00_datos_y_pipeline` | El panel, la etiqueta `z_rinde`, splits y normalización |
| `01_evaluacion_y_baselines` | Cómo se mide (PR-AUC, multi-seed) + Isolation Forest y One-Class SVM |
| `02_ae_y_dae` | Autoencoders AE y DAE (+ IForest/OCSVM sobre su latente) |
| `03_vae` | VAE: **búsqueda de HP** + score `recon_prob` (+ latente + t-SNE) |
| `04_ensemble` | Seed-ensemble = **el modelo final** (en ambos cultivos) |
| `05_techo_estructural` | Por qué nadie pasa de ~0.6 (cross-modelo + Cartography + PCA) |
| `06_features_nuevas` | Features agro / NDVI / ERA5 × 5 detectores → **ERA5+NDVI mejora al ensemble (modelo final, 0.64)** |
| `07_leaderboard_y_conclusiones` | Leaderboard final + conclusiones (DeepSVDD como referencia moderna) |

**Correr todo:** abrir un notebook y "Run all". Con las **10 semillas** del estudio, la
corrida completa tarda un rato (entrena de verdad). Para una corrida rápida, reducir
`SEEDS` arriba de cada notebook, o exportar `LAB_SEEDS="42,43,44"` antes de lanzar Jupyter.

---

## Los datos

- **Panel**: `data/processed/panel_union.parquet` — una fila por (departamento, campaña, cultivo).
- **Features (X)**: **54** = 7 variables NASA POWER × 7 meses (Sep–Mar) + ONI × 5 meses.
  No incluyen el rinde.
- **Etiqueta proxy**: `z_rinde < −1.5` (z-score del rinde vs media móvil de 5 años, por
  `[provincia, departamento, cultivo]`). Solo para evaluar; el modelo nunca la ve.
- **Splits temporales**: train ≤2020 (el ex-val 2018–2020 se pliega) · test ≥2021. **No hay validación** (el val era ciego). El train se filtra
  a solo-normales (se excluyen anomalías y años de sequía generalizada).

Todo esto lo arma `src.data.build_crop_dataset(panel_z, cultivo)`, que devuelve
`X_train / X_test` ya normalizados (sin val — ver limitaciones).

```python
from src import data
panel_z = data.prepare()                          # panel + etiqueta
ds = data.build_crop_dataset(panel_z, "soja")     # split + normalización
# variar el split es un parámetro, no un config:
ds2 = data.build_crop_dataset(panel_z, "soja", train_end=2015)
```

---

## Los modelos

Todos implementan la misma interfaz `AnomalyDetector`:

```python
from src.models import VAEDetector
det = VAEDetector(hidden_dims=(128,64), latent_dim=24, score_mode="recon_prob",
                  n_mc_samples=50, random_state=42)   # config ganadora (nb 3)
det.fit(ds.X_train)                 # solo campañas normales
scores = det.score_samples(ds.X_test)   # mayor = más anómalo
```

Disponibles: `IsolationForestDetector`, `OneClassSVMDetector`, `AEDetector`,
`DenoisingAEDetector`, `VAEDetector`, `EnsembleDetector`, `DeepODDetector`. Los AE/VAE
además exponen `.encode(X)` (espacio latente).

---

## Resultados (test, PR-AUC media±std)

| Modelo (soja, test) | PR-AUC |
|--------|------|
| **VAE seed-ensemble ×10 + ERA5+NDVI ⭐** (modelo final) | **0.640 ± 0.003** |
| VAE seed-ensemble ×10 (solo clima) | 0.616 ± 0.008 |
| VAE `recon_prob` single | 0.587 ± 0.039 |
| Isolation Forest (max_features=0.3) | 0.517 ± 0.034 |
| DeepSVDD (moderno) | 0.512 ± 0.072 |
| One-Class SVM (RBF, determinista) | 0.477 ± 0.000 |

> El mejor modelo es el **seed-ensemble del VAE `recon_prob`** (config ganadora de la
> búsqueda multi-seed del nb 3: `hidden_dims=(128,64), latent_dim=24, β=1`) **+ features
> ERA5+NDVI**. Tres decisiones lo explican:
> 1. el **score probabilístico** (An & Cho 2015) que aplasta al MSE plano;
> 2. el **ensemble de 10 semillas**, que baja el desvío a ±0.004 (elegido por media−desvío,
>    no por una semilla afortunada);
> 3. sumar **ERA5+NDVI** (estado de suelo + verdor), que mejora +0.024 — **una señal chica
>    pero real que solo se ve una vez que el ensemble quita el ruido entre semillas** (con
>    VAEs single, esa mejora quedaba enterrada en la varianza).
>
> Lección metodológica: reportar la varianza multi-seed fue clave — sin ella, el aporte de
> ERA5+NDVI era invisible. `agro` y `NDVI`-solo no ayudan; hace falta la combinación
> suelo+vegetación. El resto (detectores sobre el latente, deep AD moderno) no supera.

---

## Limitaciones conocidas

1. **Distribution shift temporal**: los scores y la tasa base de anomalías suben del
   período de train al de test (el clima 2021–24 se aleja del train; el test incluye la
   sequía 2022/23). Por eso la métrica principal es **PR-AUC** (ranking, libre de umbral) y
   las matrices de confusión se re-umbralizan por split (ver `01`).
2. **Sin validación / data snooping**: probamos un val 2018–2020 pero era **ciego** (sus
   anomalías no tienen firma climática → PR-AUC ≈ azar para todos los modelos, no
   discrimina). Lo plegamos al train, así que la selección de modelos e HP se **ilustra en
   test**, con el *data snooping* declarado (ver `01`, `03`).
3. **Techo estructural**: buena parte de las anomalías de rinde tienen causas no climáticas
   (plaga, granizo, manejo) invisibles a las features. Existe (~0.64, no llegamos a 0.8+),
   pero **no es infranqueable**: sumar suelo+verdor (ERA5+NDVI) lo corre un poco (`06`).
   Superarlo más requeriría otra clase de datos (sanidad, granizo, manejo).

---

## Tests

```bash
cd componente_a && python -m pytest tests/ -q
```

`test_repro.py` hashea el dataset que sale del pipeline (guarda contra cambios accidentales);
`test_smoke.py` corre todos los detectores sobre datos sintéticos.
