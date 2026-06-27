# Componente A — Detección de Anomalías Agroclimáticas

Detecta campañas agrícolas anómalas (rinde bajo) usando variables climáticas mensuales.
Modelo no supervisado: se entrena solo con años normales y puntúa por error de reconstrucción.

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_componente_a.txt
```

---

## Estructura del proyecto

```
Trabajo_final_ML/
│
├── train.py              # Entrena UN config YAML (también agente de W&B sweeps)
├── batch_train.py        # Entrena VARIOS configs en secuencia, tabla resumen
├── app_streamlit.py      # App de comparación de modelos (comparador lado a lado)
├── arch_sweep.py         # Barrido rápido de arquitecturas hardcodeadas
│
├── configs/              # Configs YAML separados por modelo
│   ├── ae/               # Autoencoder
│   │   ├── ae_v1_1capa.yaml
│   │   ├── ae_v11_cosine_deep.yaml   ← mejor AE hasta ahora
│   │   └── ...           # (17 configs en total)
│   ├── vae/              # Variational Autoencoder
│   │   ├── vae_v1_1capa.yaml
│   │   └── ...
│   ├── dae/              # Denoising Autoencoder
│   │   ├── dae_v1_base.yaml
│   │   └── ...
│   ├── hybrid/           # AE-latente + Isolation Forest (híbrido)
│   │   ├── ae_iforest_v1.yaml
│   │   └── ae_iforest_v2_latent16.yaml
│   └── iforest/          # Isolation Forest (baseline + tuning)
│       ├── iforest_v1_base.yaml
│       └── ...           # grilla max_features / n_estimators
│
├── runs/                 # Resultados guardados localmente (por modelo)
│   ├── ae/
│   ├── vae/
│   ├── dae/
│   ├── ae_iforest/
│   └── isolation_forest/
│
├── componente_a/         # Código fuente
│   ├── config.py         # Rutas, splits temporales, lista de features
│   ├── data.py           # Carga panel, etiqueta proxy, splits, normalización
│   ├── evaluate.py       # Métricas: PR-AUC, ROC-AUC, Recall@k, F1, Precision@k
│   ├── runner.py         # Orquestador: fit → score → evaluar → RunResult
│   ├── store.py          # Persistencia: LocalBackend (disco) / WandbBackend
│   ├── embeddings.py     # Proyecciones 2D con UMAP y t-SNE
│   └── models/
│       ├── ae.py         # AEDetector, DenoisingAEDetector
│       ├── vae.py        # VAEDetector
│       ├── baselines.py  # IsolationForestDetector, PCAReconDetector
│       ├── trainer.py    # Loop Adam + early stopping + LR schedule
│       └── base.py       # Interfaz AnomalyDetector
│
├── data/
│   └── processed/
│       └── panel_union.parquet   # Dataset principal (28683 filas, 54 features)
│
├── docs/                 # Papers de referencia
└── sweeps/               # Configs de W&B sweeps
```

---

## Dataset

- **Panel**: `data/processed/panel_union.parquet`
- **Filas**: 28 683 (departamento × cultivo × campaña)
- **Features**: 54 (9 variables climáticas × 6 meses de campaña Sep–Feb)
  - Variables: radiación solar, precipitación NASA, humedad, T media/max/min, viento, ONI, precipitación CHIRPS
- **Etiqueta proxy**: `z_rinde < -1.5` (z-score del rinde vs media móvil 5 años por departamento y cultivo)
- **Splits temporales**:
  - Train: campañas 1981/82 – 2017/18 (solo años normales)
  - Val:   campañas 2018/19 – 2020/21
  - Test:  campañas 2021/22 – 2024/25

---

## Métricas

| Métrica | Qué mide | Nota |
|---------|----------|------|
| **PR-AUC** | Área bajo la curva Precision-Recall | Principal — mejor que ROC para desbalance (10% anomalías) |
| **Recall@contam** | % de anomalías reales capturadas en el top-10% por score | "¿Cuántas anomalías detecto si uso el prior de contaminación?" |
| **ROC-AUC** | Área bajo la curva ROC | Referencia, más optimista por el desbalance |
| **F1** | F1 de la clase anómala al umbral calibrado | Calibrado por contaminación en val |
| **Precision@k** | Precisión en los top-k (k = n anomalías reales) | Cuántos de los k más sospechosos son verdaderos positivos |

Para comparar modelos mirá primero **PR-AUC** y **Recall@contam** en conjunto.

---

## Entrenamiento — `train.py`

Entrena **un solo config** para uno o ambos cultivos.

```bash
# Correr un config (ambos cultivos)
python train.py configs/ae/ae_v11_cosine_deep.yaml

# Solo un cultivo
python train.py configs/ae/ae_v11_cosine_deep.yaml --cultivo soja

# Override del nombre de la run
python train.py configs/ae/ae_v11_cosine_deep.yaml --name mi_experimento

# Guardar en W&B en lugar de local
python train.py configs/ae/ae_v11_cosine_deep.yaml --backend wandb
```

**Flags disponibles:**

| Flag | Descripción | Default |
|------|-------------|---------|
| `--cultivo` | `soja`, `maiz` o `ambos` | valor del YAML |
| `--name` | Nombre de la run (override) | valor del YAML |
| `--backend` | `local` o `wandb` | valor del YAML |
| `--count` | Máximo de runs para W&B sweep agent | 20 |

---

## Entrenamiento en batch — `batch_train.py`

Entrena **varios configs** en secuencia y muestra una tabla resumen comparativa al final.

```bash
# Correr todos los AE (ambos cultivos)
python batch_train.py configs/ae/*.yaml

# Solo soja, sin embeddings (más rápido, la app no mostrará proyecciones)
python batch_train.py configs/ae/*.yaml --cultivo soja --skip-emb

# Paralelo: 4 configs corriendo al mismo tiempo (recomendado para exploración)
python batch_train.py configs/ae/*.yaml --cultivo soja --skip-emb --workers 4

# Comparar AE vs iforest en soja
python batch_train.py configs/ae/ae_v11_cosine_deep.yaml configs/iforest/iforest_v1_base.yaml --cultivo soja

# Correr todos los modelos
python batch_train.py configs/ae/*.yaml configs/vae/*.yaml configs/dae/*.yaml configs/iforest/*.yaml
```

**Flags disponibles:**

| Flag | Descripción | Default |
|------|-------------|---------|
| `--cultivo` | `soja`, `maiz` o `ambos` | `ambos` |
| `--skip-emb` | Omite UMAP/t-SNE (la app no mostrará proyecciones para esas runs) | False |
| `--workers` | Procesos en paralelo. Cada worker entrena un par (config, cultivo). | 1 (secuencial) |

**Salida de ejemplo:**
```
──────────────────────────────────────────────
Config: configs/ae/ae_v11_cosine_deep.yaml
──────────────────────────────────────────────
  [SOJA] PR-AUC=0.3741±0.0291  ROC-AUC=0.6596  Rec@k=0.4800  F1=0.4354

============================================================================================
modelo                         cultivo    PR-AUC   ±std  ROC-AUC  Rec@k      F1  seeds
--------------------------------------------------------------------------------------------
ae_v11_cosine_deep             soja       0.3741 0.0291   0.6596  0.4800  0.4354     10
============================================================================================
```

> **Nota sobre `--skip-emb`**: las proyecciones UMAP/t-SNE son lentas (~1-2 min por run).
> Usá `--skip-emb` para exploración rápida de arquitecturas. Para la run final que querés ver
> en la app, corré **sin** `--skip-emb`.

---

## App de comparación — `app_streamlit.py`

Compara dos runs lado a lado: métricas, curvas PR/ROC, distribución de scores, proyecciones 2D y heatmap departamento × campaña.

```bash
streamlit run app_streamlit.py
```

**Uso de la sidebar:**
- **Backend de resultados**: `local` (disco) o `wandb` (cloud)
- **Runs dir**: directorio raíz de runs (default: `runs/`)
- **🔄 Recargar runs**: limpia el caché si acabás de correr nuevos modelos
- **Cultivo**: filtrá por `soja` o `maiz`
- **Modelo A / Modelo B**: los dos modelos a comparar
- **Métrica destacada**: la métrica que se resalta en grande con delta
- **Proyección 2D**: UMAP o t-SNE
- **Colorear por**: etiqueta real (normal/anómala) o score continuo

> **Importante**: si corriste runs con `--skip-emb`, la proyección 2D mostrará "sin proyección".
> Para ver la app con proyecciones completas corrí los modelos sin ese flag.

---

## Estructura de un config YAML

Todos los campos tienen defaults razonables; solo sobreescribí lo que cambiás.

### AE / DAE

```yaml
model: ae           # ae | dae | vae | iforest | ae_iforest
name: ae_v11        # nombre de la run (aparece en la app)
cultivo: ambos      # soja | maiz | ambos
backend: local      # local | wandb

# Arquitectura
hidden_dims: [128, 64]   # capas ocultas ([] = sin capas, solo bottleneck)
latent_dim: 16           # dimensión del espacio latente
activation: elu          # relu | leaky_relu | elu | gelu | tanh

# Score de anomalía (AE / DAE)
score_mode: mse          # mse (MSE promedio) | max (error máx por feature) | topk (suma top-k)
top_k: 5                 # solo si score_mode=topk

# Solo para DAE
corruption: 0.1          # fracción de features a corromper
noise_type: salt_pepper  # salt_pepper | gaussian

# Optimización
lr: 0.001
weight_decay: 0.0001
dropout: 0.1
use_batch_norm: true
grad_clip_norm: 1.0
lr_schedule: cosine      # null | cosine | plateau
max_epochs: 300
patience: 40
batch_size: 64

# Evaluación
n_seeds: 10              # semillas para estimación robusta
threshold_mode: contamination
eval_contamination: 0.10
random_state: 42
```

### VAE

```yaml
model: vae
# (mismos campos de arquitectura y optimización que AE, más:)
beta: 1.0                      # peso del término KL (< 1 prioriza reconstrucción)
score_mode: neg_elbo           # recon_error | recon_prob | neg_elbo
n_mc_samples: 20               # muestras Monte Carlo para score estocástico
```

### Isolation Forest

```yaml
model: iforest
n_estimators: 100        # 100 | 200 | 300 (más árboles = menos varianza)
max_samples: auto
max_features: 1.0        # 0.3 | 0.5 | 0.7 | 1.0 (bajo = más sensible a anomalías locales)
contamination: auto
random_state: 42
```

### Híbrido AE-latente + Isolation Forest

El AE proyecta a un latente de baja dimensión y el Isolation Forest scorea
sobre ese latente (no sobre el error de reconstrucción). Combina los campos de
arquitectura/optimización del AE con los del IForest.

```yaml
model: ae_iforest
# Autoencoder (proyección)
hidden_dims: [32, 16]
latent_dim: 8
max_epochs: 200
# Isolation Forest sobre el latente (scorer)
n_estimators: 200
max_features: 1.0        # el latente ya es de baja dimensión
contamination: auto
```

---

## Resultados actuales (soja, test)

Evaluación multi-seed (10 semillas, media±std, panel_union). **PR-AUC es la
métrica principal** (libre de umbral; ver nota sobre F1 degenerado más abajo).

| Modelo | PR-AUC | ROC-AUC | Rec@k | seeds |
|--------|--------|---------|-------|-------|
| **vae_v15_reconprob_deep** (recon_prob, β=1, lat=16, [128,64]) — mejor soja | **0.441 ±0.023** | **0.671** | **0.240** | 10 |
| **vae_v4_reconprob_lat16 ⭐ recomendado** (recon_prob, β=1, lat=16, [64,32]) | 0.435 ±0.031 | 0.667 | 0.237 | 10 |
| iforest_v2_mf03_n200 (mf=0.3, n=200) | 0.416 ±0.021 | 0.663 | 0.227 | 10 |
| iforest_v6_mf02_n200 (mf=0.2) | 0.409 ±0.023 | 0.653 | 0.234 | 10 |
| vae_v9_negelbo_lat16_bn (neg_elbo, β=0.148) | 0.370 ±0.031 | 0.636 | 0.181 | 10 |
| ae_score_max (lat=16, score=max) | 0.329 ±0.002 | 0.606 | 0.191 | 10 |
| ae_iforest_v2_latent16 (híbrido AE+IForest) | 0.307 ±0.062 | 0.548 | 0.162 | 10 |

> **Estado:** el **VAE con `score_mode=recon_prob`** (probabilístico, An & Cho
> 2015) es el mejor modelo del Componente A — supera al IForest tuneado en
> PR-AUC, ROC-AUC y Rec@k. El score probabilístico pondera el error de cada
> feature por su varianza esperada (no lo promedia plano) y funciona mejor
> **sin** regularización pesada (BN/dropout la degradan). `vae_v4` (lat=16,
> [64,32]) es el todoterreno: gana soja **y** maiz. `vae_v15` (más capacidad,
> [128,64]) es el mejor en soja y, con ±0.023, gana al IForest incluso en
> mean−std (0.418 > 0.416) — aunque en maiz roza por debajo del baseline.
> Negativos útiles: subir `n_mc_samples` no baja la varianza (viene del
> entrenamiento, no del MC); `latent`>16 y `beta`≠[1,1.5] la empeoran. El tuning
> de `max_features` (0.3–0.5) hace al IForest un baseline fuerte; el híbrido
> AE-latente+IForest y los scorings max/top-k del AE no superan a ninguno.

---

## W&B Sweeps

```bash
# 1. Crear el sweep (genera SWEEP_ID)
wandb sweep sweeps/ae.yaml

# 2. Lanzar agente (en la misma o diferente terminal)
python train.py --sweep SWEEP_ID --cultivo soja --count 20
```

---

## Agregar un modelo nuevo

1. Implementar la interfaz `AnomalyDetector` en `componente_a/models/`:
   ```python
   class MiDetector(AnomalyDetector):
       model_type = "mi_modelo"
       def fit(self, X): ...
       def score_samples(self, X): ...
       def get_config(self): ...
   ```
2. Exportarlo en `componente_a/models/__init__.py`
3. Agregarlo al `build_detector()` en `train.py`
4. Crear config en `configs/mi_modelo/mi_modelo_v1.yaml`
5. Las runs se guardarán en `runs/mi_modelo/` automáticamente
