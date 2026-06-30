# Componente A — Detección de Anomalías Agroclimáticas

Detecta campañas agrícolas anómalas (rinde bajo) usando variables climáticas mensuales.
Modelo no supervisado: se entrena solo con años normales y puntúa por error de reconstrucción.

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r componente_a/requirements.txt
```

> **Directorio de trabajo:** todo el Componente A vive bajo `componente_a/` y los
> comandos de abajo se ejecutan **desde ahí** (`cd componente_a`). El venv queda en
> la raíz del repo (`.venv/`), así que se activa antes de entrar.

---

## Estructura del proyecto

```
Trabajo_final_ML/
│
├── data/                     # Datos COMPARTIDOS (ambos componentes)
│   └── processed/panel_union.parquet   # Dataset principal del Componente A
├── data_sources/             # Extracción + MERGE satelital COMPARTIDO (Earth Engine)
│   ├── extract_avhrr_ndvi.py / merge_avhrr_ndvi.py   # NDVI-AVHRR 1981+
│   └── extract_era5.py / merge_era5.py               # ERA5-Land: suelo + heladas
│
├── componente_a/             # TODO el Componente A (se corre desde acá)
│   │
│   ├── src/                  # Código fuente (paquete Python `src`)
│   │   ├── config.py         #   Rutas, splits temporales, lista de features
│   │   ├── data.py           #   Carga panel, etiqueta proxy, splits, normalización
│   │   ├── evaluate.py       #   Métricas: PR-AUC, ROC-AUC, Recall@k, F1, Precision@k
│   │   ├── runner.py         #   Orquestador: fit → score → evaluar → RunResult
│   │   ├── store.py          #   Persistencia: run-dirs locales en disco
│   │   ├── embeddings.py     #   Proyecciones 2D con UMAP y t-SNE
│   │   └── models/
│   │       ├── base.py       #   Interfaz AnomalyDetector (fit/score_samples/get_config)
│   │       ├── ae.py         #   AEDetector + DenoisingAEDetector (subclase)
│   │       ├── vae.py        #   VAEDetector
│   │       ├── baselines.py  #   IsolationForestDetector, PCAReconDetector
│   │       ├── hybrid.py     #   AEIForestDetector (AE-latente + IForest)
│   │       ├── ensemble.py   #   EnsembleDetector (seed-ensemble / hetero)
│   │       ├── deep_baselines.py  # DeepODDetector (métodos modernos, deepod)
│   │       └── trainer.py    #   Loop Adam + early stopping + LR schedule
│   │
│   ├── training/             # ENTRENAMIENTO
│   │   ├── train.py          #   Entrena UN config YAML
│   │   └── batch_train.py    #   Entrena VARIOS configs en paralelo, tabla resumen
│   │
│   ├── analyze_errors.py / analyze_labels.py / analyze_datamap.py  # Análisis (→ analysis/)
│   ├── app_streamlit.py      # App de comparación de modelos (lado a lado)
│   ├── experiments/          # Notebooks didácticos del recorrido completo
│   ├── eda/                  # EDA + construcción del panel
│   │
│   ├── configs/              # Configs YAML por modelo (ae/ vae/ dae/ hybrid/ ensemble/ iforest/ deepod/)
│   ├── runs/                 # Resultados guardados localmente, por modelo (gitignored)
│   ├── analysis/             # Salidas de los analyze_*.py (gitignored)
│   └── requirements.txt
│
├── docs/                     # Papers de referencia + PDF del proyecto
└── README.md
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
python training/train.py configs/ae/ae_v11_cosine_deep.yaml

# Solo un cultivo
python training/train.py configs/ae/ae_v11_cosine_deep.yaml --cultivo soja

# Override del nombre de la run
python training/train.py configs/ae/ae_v11_cosine_deep.yaml --name mi_experimento
```

**Flags disponibles:**

| Flag | Descripción | Default |
|------|-------------|---------|
| `--cultivo` | `soja`, `maiz` o `ambos` | valor del YAML |
| `--name` | Nombre de la run (override) | valor del YAML |

---

## Entrenamiento en batch — `batch_train.py`

Entrena **varios configs** en secuencia y muestra una tabla resumen comparativa al final.

```bash
# Correr todos los AE (ambos cultivos)
python training/batch_train.py configs/ae/*.yaml

# Solo soja
python training/batch_train.py configs/ae/*.yaml --cultivo soja

# Paralelo: 4 configs corriendo al mismo tiempo (recomendado para exploración)
python training/batch_train.py configs/ae/*.yaml --cultivo soja --workers 4

# Comparar AE vs iforest en soja
python training/batch_train.py configs/ae/ae_v11_cosine_deep.yaml configs/iforest/iforest_v1_base.yaml --cultivo soja

# Correr todos los modelos
python training/batch_train.py configs/ae/*.yaml configs/vae/*.yaml configs/dae/*.yaml configs/iforest/*.yaml
```

**Flags disponibles:**

| Flag | Descripción | Default |
|------|-------------|---------|
| `--cultivo` | `soja`, `maiz` o `ambos` | `ambos` |
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

> **Proyecciones 2D:** ya **no** se calculan al entrenar (eran lentas, ~1-2 min/run).
> Se computan **on-demand** la primera vez que la app o un notebook las pide
> (`src.embeddings.compute_embeddings(run_id)`) y se cachean en el run-dir. No
> necesitan el modelo entrenado: dependen solo de las features y de los scores ya
> guardados.

---

## App de comparación — `app_streamlit.py`

Compara dos runs lado a lado: métricas, curvas PR/ROC, distribución de scores, proyecciones 2D y heatmap departamento × campaña.

```bash
streamlit run app_streamlit.py
```

**Uso de la sidebar:**
- **Runs dir**: directorio raíz de runs (default: `runs/`)
- **🔄 Recargar runs**: limpia el caché si acabás de correr nuevos modelos
- **Cultivo**: filtrá por `soja` o `maiz`
- **Modelo A / Modelo B**: los dos modelos a comparar
- **Métrica destacada**: la métrica que se resalta en grande con delta
- **Proyección 2D**: UMAP o t-SNE
- **Colorear por**: etiqueta real (normal/anómala) o score continuo

> **Proyección 2D on-demand**: la primera vez que abrís un run, la app computa UMAP/t-SNE
> (~1-2 min) y lo cachea en el run-dir; las siguientes veces es instantáneo.

---

## Estructura de un config YAML

Todos los campos tienen defaults razonables; solo sobreescribí lo que cambiás.

### AE / DAE

```yaml
model: ae           # ae | dae | vae | iforest | ae_iforest
name: ae_v11        # nombre de la run (aparece en la app)
cultivo: ambos      # soja | maiz | ambos
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
score_mode: recon_prob         # recon_error | recon_prob | neg_elbo
n_mc_samples: 50               # muestras Monte Carlo para score estocástico
decoder_dist: gaussian         # gaussian | student_t (colas pesadas, robusto a outliers)
student_t_df: 4.0              # grados de libertad ν (solo si decoder_dist=student_t)
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

### Ensemble (seed-ensemble y heterogéneo)

Combina los scores normalizados (z-score per-miembro) de varios detectores.
Dos formas: `base` + `n_members` (N copias con semillas distintas → baja la
varianza) o `members` (modelos distintos → combina señales complementarias).

```yaml
model: ensemble
normalize: zscore        # zscore | rank
combine: mean            # mean | max
# A) seed-ensemble: N copias del mismo modelo con semillas distintas
n_members: 10
base: { model: vae, score_mode: recon_prob, latent_dim: 16, ... }
# B) hetero-ensemble (alternativa a base/n_members): lista de modelos
# members:
#   - { model: vae, score_mode: recon_prob, ... }
#   - { model: iforest, max_features: 0.3, ... }
```

### Features agronómicas (flag de experimento)

Bandera a nivel experimento (no del modelo) — aplica a cualquier `model`.
Anexa features de dominio derivadas de la ventana crítica del cultivo
(precip, estrés térmico, balance hídrico Hargreaves) a la matriz X.

```yaml
use_agro_features: true   # default false; per-cultivo (Dic–Feb soja / Nov–Ene maíz)
```

---

## Resultados actuales (soja, test)

Evaluación multi-seed (media±std, **panel limpio** — dedup + clave provincia,
ver `data.py`). **PR-AUC es la métrica principal** (libre de umbral).

**Soja (test):**
| Modelo | PR-AUC | ROC-AUC | Rec@k | seeds |
|--------|--------|---------|-------|-------|
| **vae_seedens_v1 ⭐ recomendado** (seed-ens ×10 de vae_v4) | **0.592 ±0.010** | 0.738 | **0.297** | 3×10 |
| vae_seedens_deep (seed-ens ×10 de vae_v15) | 0.585 ±0.007 | **0.749** | 0.296 | 3×10 |
| vae_v4_reconprob_lat16 (single) | 0.559 ±0.035 | 0.719 | 0.279 | 10 |
| iforest_v2_era5 (IForest + ERA5 suelo/heladas) | 0.542 ±0.029 | 0.719 | 0.282 | 10 |
| iforest_v2_mf03_n200 (sin ERA5) | 0.511 ±0.031 | 0.688 | 0.266 | 10 |

**Maíz (test):**
| Modelo | PR-AUC | ROC-AUC | Rec@k | seeds |
|--------|--------|---------|-------|-------|
| **vae_seedens_v1 ⭐** | **0.508 ±0.005** | 0.679 | 0.246 | 3×10 |
| vae_seedens_deep | 0.500 ±0.003 | 0.669 | 0.237 | 3×10 |
| iforest_v2_era5 | 0.504 ±0.019 | 0.656 | 0.251 | 10 |
| iforest_v2_mf03_n200 (sin ERA5) | 0.492 ±0.030 | 0.648 | 0.244 | 10 |

> **Estado:** el mejor modelo del Componente A es el **seed-ensemble del VAE
> `recon_prob`** (`vae_seedens_v1`): soja 0.592, maíz 0.508. La mayor mejora del
> proyecto fue la **limpieza de datos** (dedup 25% + clave provincia: +0.09–0.14 a
> todos). El seed-ensemble resuelve la varianza (±0.006 vs ±0.035 del single) vía
> promedio de scores normalizados de 10 semillas; el score probabilístico (An &
> Cho 2015) funciona mejor sin regularización pesada.
>
> **Ninguna feature satelital/extra supera al base con el VAE.** ERA5-Land
> (humedad de suelo + heladas) ayuda **modestamente al IForest** (soja +0.031) pero
> **perjudica al VAE** (maíz −0.024, dilución del recon_prob), y el IForest+ERA5
> (0.542) sigue debajo del VAE base. (Lección de método: un primer resultado
> ERA5 de 0.642 resultó ser un ARTEFACTO — `frost_days`=0 en el norte sin heladas
> daba varianza-cero por depto y la normalización tiraba el 34% de los
> departamentos → test set sesgado. Se corrigió con fallback a std global en
> `_normalize_per_depto`; con test sets emparejados el efecto real es el de
> arriba.) Ablaciones NDVI / Student-t / agro / híbrido: ver abajo.

**Ablaciones (no superan al seed-ensemble, útiles para justificar el modelo final):**

- **VAE decoder Student-t** (`vae_studentt_*`): neutro-negativo. ν=4 empata soja
  PR-AUC (0.435) con mejor ROC pero no baja la varianza; ν=8 peor. Las colas
  pesadas ayudan con train *contaminado*, pero el pipeline ya excluye filas
  anómalas + años problemáticos → train curado, sin nada que robustecer.
- **Features agronómicas** (`use_agro_features`: ventana crítica Dic–Feb soja /
  Nov–Ene maíz + balance hídrico Hargreaves): **ayudan al IForest, perjudican al
  VAE**. `iforest_v2_agro` sube PR-AUC (soja 0.416→0.421, maiz 0.366→0.372),
  ROC, Rec@k y baja varianza; `vae_v4_agro` empeora (el VAE debe reconstruir más
  features → más varianza). El `recall_clima_adverso` del IForest sube con agro
  (soja 0.97→0.99) — la señal de dominio afina lo climático —, pero el
  `recall_otros` (anomalías sin firma climática) sigue bajo: **el techo
  estructural clima→rinde persiste**.
- **NDVI** (`use_ndvi`: `ndvi_anomalia_pct`, único NDVI con señal temporal — las
  otras 3 columnas son estáticas por depto): **no aporta**. Experimento de 3 vías
  con control de período (`train_start`) porque NDVI solo existe 2002+. Soja: VAE
  full 0.435 → control 2002 0.352 → +NDVI 0.344; IForest 0.416 → 0.380 → 0.383.
  El recorte a 2002+ duele (−0.03 a −0.08, el VAE el más golpeado); el efecto
  marginal de NDVI vs su control es ≈0 (dentro del ruido). Refuerza el techo
  estructural: si ni el verdor de la planta marca esas anomalías de rinde, o el
  clima ya las captura o son ruido de etiqueta.

---

## ⚠️ Limitaciones conocidas

### 1. *Distribution shift* temporal (val → test)

Los splits son **temporales** (train ≤2017/18, val 2018–2020, test 2021–2024), así
que val y test pertenecen a **períodos climáticos distintos**. Esto produce un
*distribution shift* con **dos caras**, ambas importantes para leer las métricas:

- **Los scores se corren hacia arriba.** El modelo aprende lo "normal" hasta 2017;
  el clima de 2021–24 está cada vez más lejos de ese período → **todo** reconstruye
  un poco peor → los scores de anomalía suben en bloque. Ejemplo (VAE, soja): score
  medio **val 3.76 → test 20.22**.
- **La tasa real de anomalías sube.** `z_rinde < −1.5` marca **~6% en val** pero
  **~27% en test** (el test incluye la mega-sequía 2022/23). El prior de
  contaminación del 10% **subestima** la realidad del test.

**Consecuencia — el umbral no transfiere.** El umbral se calibra como el percentil
90 de los scores de *validación* y se aplica como corte absoluto a *test*. Por el
shift, ese corte termina marcando **~70% del test** como anómalo (no porque el
modelo lo crea, sino porque casi todos los scores de test superan un corte fijado en
val). Por eso:

- La **métrica principal es PR-AUC** (y ROC-AUC), que son **libres de umbral**:
  miden el *ranking* de los scores y **no se ven afectadas** por el shift. Todas las
  conclusiones del proyecto se apoyan en PR-AUC.
- El **F1 / `y_pred` guardados en los runs usan el umbral de val y NO son
  confiables** (lo documentamos como tal; no se usan para decidir).
- Las **matrices de confusión de los notebooks NO usan ese `y_pred`**: re-umbralizan
  sobre los scores del **propio test** (sin usar etiquetas para el corte) y muestran
  **dos puntos de operación** — `top-10%` (presupuesto de alertas) y `top-tasa real`
  (~27%). Con el corte bien hecho el VAE da **80% de precisión al top-10%**; el "FP
  gigante" del corte de val era un artefacto. Ver NB 1 (`plot_confusion_grid`,
  parámetro `op` en `src/embeddings`/`explib`).

### 2. Selección en validación vs. test

El val es chico (5–9 anomalías) → ruidoso. A lo largo de los notebooks mostramos
*test* (más estable) para **ilustrar** comparaciones, pero la **decisión final** de
modelo debería apoyarse en *val* (evitar *data snooping*). El ranking en val y test
coincide (el VAE seed-ensemble gana en ambos), así que la conclusión se sostiene;
queda explícito como limitación. Ver NB 1.

### 3. Techo estructural clima → rinde

~70% de las anomalías de rinde tienen causas **no climáticas** (plaga, granizo,
manejo, ruido de etiqueta) **invisibles a cualquier feature disponible**. El modelo
solo ve clima, así que estructuralmente no puede superar la fracción de anomalías con
firma climática (~30%). Cuantificado con `stratified_recall` + `analyze_errors.py`.
Es un límite del **problema/datos**, no del modelo. Ver NB 5.

---


## Agregar un modelo nuevo

1. Implementar la interfaz `AnomalyDetector` en `src/models/`:
   ```python
   class MiDetector(AnomalyDetector):
       model_type = "mi_modelo"
       def fit(self, X): ...
       def score_samples(self, X): ...
       def get_config(self): ...
   ```
2. Exportarlo en `src/models/__init__.py`
3. Agregarlo al `build_detector()` en `training/train.py`
4. Crear config en `configs/mi_modelo/mi_modelo_v1.yaml`
5. Las runs se guardarán en `runs/mi_modelo/` automáticamente
