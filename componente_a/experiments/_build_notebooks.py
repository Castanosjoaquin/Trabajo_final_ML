"""Genera los notebooks de experiments/ — didácticos, reproducibles, con narrativa.
    python experiments/_build_notebooks.py
Leen de runs/ (no re-entrenan por defecto), pero permiten entrenar en el notebook
(force_train=True). Después se ejecutan con nbconvert para embeber salidas."""
import os
import glob
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


SETUP = """import os, explib
os.chdir(explib.ROOT)          # rutas relativas (data/, configs/, runs/) funcionan
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
plt.rcParams['figure.dpi'] = 110"""


def build(fname, title, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [md(f"# {title}")] + cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    with open(os.path.join(HERE, fname), "w") as f:
        nbf.write(nb, f)
    print("escrito:", fname)


for old in glob.glob(os.path.join(HERE, "*.ipynb")):
    os.remove(old)


# ===========================================================================
# 00 — Datos y pipeline (la base que faltaba)
# ===========================================================================
build("00_datos_y_pipeline.ipynb", "0 · Los datos y el pipeline", [
    md("""Antes de cualquier modelo: **¿qué son los datos y cómo los procesamos?** Este
notebook carga el panel real y muestra cada paso. Es la base para entender todo lo demás.

**El problema:** detectar **campañas con rinde anómalo** por departamento (soja/maíz,
Argentina, 1981–2024). Es **no supervisado**: entrenamos solo con campañas normales."""),
    code(SETUP),
    md("""## 1 · El panel
Una fila = un (departamento, campaña, cultivo). Columnas: identificadores, `rinde_kgha`
(el objetivo) y ~54 features climáticas (promedios mensuales Sep–Mar)."""),
    code("""from src.config import ExperimentConfig
from src import data as cdata
cfg = ExperimentConfig()
panel = cdata.load_panel(cfg)     # incluye el dedup (ver NB 4)
print("shape:", panel.shape)
panel[["provincia","departamento","campania","cultivo","rinde_kgha"]].head()"""),
    md("""## 2 · La etiqueta proxy `z_rinde`
No tenemos etiquetas de "anomalía". Las definimos: cuánto cae el rinde respecto de la
**media móvil del propio departamento**.

`z_rinde = (rinde − media_móvil) / desvío_móvil`  → `anomalía = z_rinde < −1.5`

Se calcula con `shift(1)` (solo pasado, sin leakage) y agrupando por
`[provincia, departamento, cultivo]`. **El modelo nunca ve esta etiqueta — solo se usa
para evaluar.**"""),
    code("""panel_z = cdata.compute_z_rinde(panel, cfg)
print("tasa de anomalías por cultivo:")
print((panel_z.groupby("cultivo")["anomalia"].mean()*100).round(1).astype(str) + " %")"""),
    md("### Ejemplo: la serie de rinde de un departamento, con sus anomalías marcadas"),
    code("""d = panel_z[(panel_z.cultivo=="soja") & (panel_z.provincia=="CORDOBA")]
d = d[d.departamento==d.departamento.value_counts().idxmax()].sort_values("campania_inicio")
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(d.campania_inicio, d.rinde_kgha, "-o", ms=3, color="#4C72B0", label="rinde")
an = d[d.anomalia==1]
ax.scatter(an.campania_inicio, an.rinde_kgha, color="red", zorder=5, s=40, label="anomalía (z<-1.5)")
ax.set_xlabel("campaña"); ax.set_ylabel("rinde (kg/ha)")
ax.set_title(f"Rinde de soja — {d.departamento.iloc[0]}, Córdoba"); ax.legend(); plt.show()"""),
    md("### Distribución de `z_rinde` y el umbral de anomalía"),
    code("""fig, ax = plt.subplots(figsize=(7,4))
ax.hist(panel_z[panel_z.cultivo=="soja"]["z_rinde"].dropna(), bins=60, color="#55A868", alpha=0.8)
ax.axvline(-1.5, color="red", ls="--", lw=2, label="umbral (-1.5)")
ax.set_xlabel("z_rinde"); ax.set_ylabel("frecuencia"); ax.set_title("Distribución de z_rinde (soja)")
ax.legend(); plt.show()"""),
    md("""## 3 · Las features (X)
54 columnas = ~9 variables climáticas × 7 meses (Sep–Mar). NO incluyen el rinde."""),
    code("""feats = cdata.build_feature_list(panel, cfg.use_ndvi)
print(f"nº de features: {len(feats)}")
prefijos = sorted(set(f.rsplit('_',1)[0] for f in feats))
print("variables:", prefijos)"""),
    md("""## 4 · Splits temporales + entrenar-solo-con-normales + normalización
- **Split temporal** (no aleatorio): train ≤2017, val 2018–2020, test ≥2021.
- **Train normal**: del train se excluyen las anomalías y los años con >30% de deptos
  anómalos → el modelo aprende solo "lo normal".
- **Normalización por departamento** (z-score), con stats calculadas SOLO en train normal
  (sin leakage), e incluyendo la provincia para no mezclar deptos homónimos."""),
    code("""ds = cdata.build_crop_dataset(panel_z, "soja", cfg)
print(f"train normal (≤2017): {len(ds.X_train):5d} filas")
print(f"val        (2018-20): {len(ds.X_val):5d} filas  ({int(ds.y_val.sum())} anomalías)")
print(f"test       (≥2021)  : {len(ds.X_test):5d} filas  ({int(ds.y_test.sum())} anomalías)")
print(f"X_train: matriz {ds.X_train.shape} (filas normales × features, ya normalizada)")"""),
    md("""**Con esto el pipeline está listo.** Cualquier detector recibe `X_train` (normales)
para `fit()` y puntúa `X_val`/`X_test` con `score_samples()` (mayor = más anómalo). Todo lo
demás (entrenar, evaluar, comparar) se construye sobre esto → próximo notebook."""),
])

# ===========================================================================
# 01 — Evaluación, reproducibilidad y baseline
# ===========================================================================
build("01_evaluacion_baseline_y_como_correrlo.ipynb", "1 · Evaluación, baseline y cómo reproducirlo", [
    md("""Acá explicamos **cómo medimos, cómo se entrena un modelo, y cómo reproducir todo**.
Después fijamos el baseline (Isolation Forest) que hay que superar."""),
    code(SETUP),
    md("""## 1 · Cómo se entrena y evalúa un modelo (y cómo correrlo vos)
Todo modelo cumple la interfaz `AnomalyDetector`: `fit(X_train)` (solo normales) +
`score_samples(X)`. El helper `explib.run_experiment` **carga** el run guardado o, con
`force_train=True`, lo **entrena de verdad** con el pipeline real y lo guarda en `runs/`.

Veamos la configuración del baseline (un YAML de `configs/`):"""),
    code('explib.show_yaml("configs/iforest/iforest_v2_mf03_n200.yaml")'),
    code('explib.describe_architecture("configs/iforest/iforest_v2_mf03_n200.yaml")'),
    md("""**Entrenarlo / cargarlo** (poné `force_train=True` para re-entrenar — tarda unos
segundos el IForest; los VAE, varios minutos):"""),
    code("""res = explib.run_experiment("configs/iforest/iforest_v2_mf03_n200.yaml", "soja",
                            force_train=False)   # ← cambiá a True para entrenarlo vos
print("run:", res.run_id)"""),
    md("""## 2 · Métricas: por qué PR-AUC (y no F1), y por qué reportamos desvío
- **PR-AUC** (principal) y **ROC-AUC**: libres de umbral.
- **F1 es engañoso acá**: el umbral se calibra en val (pocas anomalías) y en test la tasa
  base salta a ~30% → casi todos dan F1 ≈ tasa base. No lo usamos para decidir.
- **Multi-seed**: reportamos **media ± desvío** porque los modelos profundos varían entre
  semillas."""),
    code('explib.plot_all_metrics("iforest_v2_mf03_n200", "soja"); plt.show()'),
    md("""### Matriz de confusión y el punto de operación
Las métricas de arriba (PR-AUC) son **libres de umbral** — miden el *ranking*. La matriz de
confusión necesita un **corte**, y ahí hay una sutileza importante:

> ⚠️ **El umbral calibrado en validación NO transfiere a test.** Se calibra como el percentil
> 90 de los scores de *val*, pero los scores se corren mucho hacia arriba entre val (2018–20)
> y test (2021–24) —el clima se aleja del período de entrenamiento— así que ese corte termina
> marcando **~70% del test** como anómalo. No es que el modelo crea que el 70% son anómalas:
> es el *distribution shift* lavando un corte absoluto.

Por eso re-umbralizamos sobre los **scores del propio test** (sin usar etiquetas para el
corte) en **dos puntos de operación**: **top-10%** (presupuesto de alertas acotado) y
**top-tasa real** (~27%, la fracción que realmente es anómala en test). Filas = real,
columnas = predicho."""),
    code('explib.plot_confusion_grid([("iforest_v2_mf03_n200","IForest baseline")], "soja"); plt.show()'),
    code('explib.show(explib.confusion_report("iforest_v2_mf03_n200", "soja"))'),
    md("""Leído así, el punto de operación se entiende: al **top-10%** la precisión es alta
(casi todo lo que marca es anomalía real) pero el recall bajo (con 10% de presupuesto no se
puede cubrir el 27% que hay); al **top-tasa real** precision y recall se igualan. Este es el
patrón con el que comparar todos los modelos."""),
    md("""## 3 · ⚠️ Selección en validación, no en test (nota metodológica)
**Lo correcto** es *elegir* el mejor modelo mirando **validación**, y usar **test una sola
vez** para el número final del modelo ya elegido. Comparar muchos modelos en test y quedarse
con el ganador sesga la métrica (data snooping).

**Limitación de este proyecto:** nuestro val es chico (5–9 anomalías) → ruidoso. Por eso a
lo largo de los notebooks mostramos test (más estable) para *ilustrar* las comparaciones,
pero la **decisión final** debería apoyarse en val. Acá los dos, lado a lado:"""),
    code("""modelos = [("iforest_v2_mf03_n200","IForest"), ("vae_seedens_v1","VAE seed-ens"),
           ("vae_v4_reconprob_lat16","VAE single")]
print("— VALIDACIÓN —"); display(explib.show(explib.compare_table(modelos, "soja", split="val")))
print("— TEST —");       display(explib.show(explib.compare_table(modelos, "soja", split="test")))"""),
    md("""El ranking en val y en test **coincide** (el VAE seed-ensemble gana en ambos), así
que la conclusión se sostiene; pero dejamos explícita la limitación.

## 4 · El baseline a superar: Isolation Forest = **PR-AUC 0.51 (soja)**
Es el listón de toda la historia. En los próximos notebooks vamos modelo por modelo."""),
])

# ===========================================================================
# 02 — Modelos de reconstrucción
# ===========================================================================
build("02_modelos_de_reconstruccion.ipynb", "2 · Modelos de reconstrucción: AE → DAE → híbrido → VAE", [
    md("""**Autoencoders**: se entrenan a reconstruir las campañas normales; lo que
reconstruyen mal = anómalo. Recorremos los intentos en orden, siempre vs el baseline.

> *Fase de exploración*, sobre el panel original (la limpieza de datos es el NB 4). Importa
> la **comparación relativa** entre enfoques; los números absolutos suben después."""),
    code(SETUP),
    md("""## 2.1 — Autoencoder (AE)
Arquitectura: encoder comprime las 54 features a un latente chico, decoder reconstruye.
Score = error de reconstrucción (MSE). Veamos un config y su estructura:"""),
    code('explib.describe_architecture("configs/ae/ae_v11_cosine_deep.yaml")'),
    md("Entrenamos **24 variantes** (arquitectura, latente, dropout, schedules). Los 5 mejores:"),
    code('explib.show(explib.top_runs("ae", "soja", n=5, era="dirty"))'),
    md("""El rango entre las 24 fue de **~0.034**: tunear hiperparámetros casi no movió la
aguja → el cuello de botella no era ése. Curva de loss y métricas del mejor AE:"""),
    code("""best_ae = explib.top_runs("ae","soja",1,era="dirty")["name"].iloc[0]
explib.plot_loss(best_ae, "soja"); plt.show()
explib.plot_all_metrics(best_ae, "soja"); plt.show()"""),
    md("""## 2.2 — Denoising AE (DAE)
Se entrena reconstruyendo *limpio* desde input *corrupto* → más robusto al ruido. Ayuda
algo, pero sigue en el rango del AE."""),
    code('explib.show(explib.compare_table([("dae_v1_base","DAE base"), ("dae_v2_sweep_best","DAE sweep")], "soja"))'),
    md("""## 2.3 — Híbrido: AE-latente + Isolation Forest
Correr IForest sobre el **latente** del AE. En la literatura suele ganar; acá **falló** — el
bottleneck del AE destruye la estructura que el IForest necesita."""),
    code('explib.show(explib.compare_table([("ae_iforest_v1","híbrido lat8"), ("ae_iforest_v2_latent16","híbrido lat16")], "soja"))'),
    md("""## 2.4 — VAE: el salto del *score probabilístico*
El VAE usa `recon_prob` (An & Cho 2015): pondera el error de cada feature por la **varianza
que el decoder le asigna**. Ese cambio de *score* (no de arquitectura) es lo que mueve la
aguja."""),
    code('explib.describe_architecture("configs/vae/vae_v4_reconprob_lat16.yaml")'),
    code('explib.plot_loss("vae_v4_reconprob_lat16", "soja"); plt.show()'),
    md("### Comparación de toda la fase (test, media±std) — misma era (panel original)"),
    code("""recon = [("iforest_v1_base","IForest (baseline)"), (best_ae,"AE (mejor)"),
         ("ae_iforest_v2_latent16","Híbrido AE+IForest"), ("vae_v4_reconprob_lat16","VAE recon_prob")]
explib.show(explib.compare_table(recon, "soja", era="dirty"))"""),
    code('explib.plot_compare(recon, "soja", era="dirty", baseline=0.389, title="Fase reconstrucción — PR-AUC (soja)"); plt.show()'),
    md("""### Matriz de confusión por tipo de modelo (re-umbralizado sobre test)
Más allá del ranking PR-AUC, así se ve el **punto de operación** de cada tipo de modelo. Como
en el NB 1, re-umbralizamos sobre los scores del propio test en dos puntos: **top-10%**
(presupuesto) y **top-tasa real** (~27%). Filas del grid = punto de operación; columnas =
tipo de modelo. Se ve qué enfoque rankea mejor las anomalías a igual presupuesto de alertas."""),
    code("""recon_cm = [(best_ae, "AE"), ("dae_v1_base", "DAE"),
            ("ae_iforest_v2_latent16", "Híbrido"), ("vae_v4_reconprob_lat16", "VAE")]
explib.plot_confusion_grid(recon_cm, "soja", era="dirty"); plt.show()"""),
    md("""**Conclusión:** el **VAE `recon_prob`** es el mejor modelo de reconstrucción y el
único que le pelea al baseline. Pero tiene **alta varianza** (±0.03) — lo atacamos en el NB 3."""),
])

# ===========================================================================
# 03 — Ensembles y modelo final
# ===========================================================================
build("03_ensembles_y_modelo_final.ipynb", "3 · Ensembles: de la varianza al modelo final", [
    md("""El VAE `recon_prob` era el mejor, pero **cada semilla converge distinto** → alta
varianza. Solución: **ensemble de semillas**."""),
    code(SETUP),
    md("""## 3.1 — Seed-ensemble
Entrenamos N VAEs con semillas distintas y promediamos sus scores normalizados. El
`EnsembleDetector` es agnóstico al modelo. Config y estructura del ensemble final:"""),
    code('explib.describe_architecture("configs/ensemble/vae_seedens_v1.yaml")'),
    code("""ens = [("vae_v4_reconprob_lat16","VAE single"), ("vae_seedens_v1","seed-ens ×10 (lat16)"),
       ("vae_seedens_deep","seed-ens ×10 (deep)"), ("vae_iforest_ens_v1","hetero VAE+IForest")]
explib.show(explib.compare_table(ens, "soja", era="clean"))"""),
    code('explib.plot_compare(ens, "soja", era="clean", baseline=0.511, title="Ensembles — PR-AUC (soja)"); plt.show()'),
    md("""El **desvío se desploma** (±0.035 → ±0.010) y la media *sube*: el promedio cancela el
ruido idiosincrático de cada semilla. El hetero VAE+IForest queda mixto (el IForest arrastra)."""),
    code("""explib.plot_pr_curves([("vae_v4_reconprob_lat16","VAE single"),
                       ("vae_seedens_v1","seed-ensemble ×10")], "soja"); plt.show()"""),
    md("""### Matriz de confusión de los ensembles (re-umbralizado sobre test)
Una por cada ensemble, en los dos puntos de operación (top-10% y top-tasa real; ver NB 1).
Filas del grid = punto de operación; columnas = ensemble."""),
    code("""ens_cm = [("vae_seedens_v1", "VAE seed-ens (lat16)"),
          ("vae_seedens_deep", "VAE seed-ens (deep)"),
          ("vae_iforest_ens_v1", "hetero VAE+IForest")]
explib.plot_confusion_grid(ens_cm, "soja", era="clean"); plt.show()"""),
    md("""## 3.2 — La idea de *Dataset Cartography* (la del profesor)
"Lo que el modelo predice **siempre** mal es anomalía". La implementamos como **diagnóstico**:
- **Data map** (Swayamdipta 2020): error de reconstrucción de cada muestra **por época** →
  *confidence* (error medio) + *variability* (desvío entre épocas).
- **Desacuerdo del ensemble** (`score_std`): desvío entre miembros = incertidumbre.

> **El modelo final NO usa cartography para puntuar** (es el promedio simple del ensemble).
> Fue herramienta para *auditar* (NB 5): mostró que lo "siempre mal reconstruido" es rareza
> climática, no anomalía de rinde. Lo dejamos porque responde la pregunta del profesor."""),
    code("""from IPython.display import Image, display
p = os.path.join(explib.ROOT, "analysis", "datamap_soja.png")
display(Image(p)) if os.path.exists(p) else print("generar: python analyze_datamap.py --cultivo soja")"""),
    md("## 3.3 — El modelo final: `vae_seedens_v1` (todas sus métricas, ±std)"),
    code('explib.show(explib.run_metrics("vae_seedens_v1", "soja"))'),
    code('explib.plot_all_metrics("vae_seedens_v1", "soja"); plt.show()'),
    md("""### Matriz de confusión y métricas derivadas del modelo final
Las métricas libres de umbral (PR-AUC) miden el *ranking*; la matriz de confusión hace
tangible el **punto de operación**, re-umbralizado sobre el test en los dos puntos (top-10% y
top-tasa real; ver NB 1) y para ambos cultivos. La tabla resume precision/recall/F1 en cada
punto."""),
    code("""explib.plot_confusion_grid([("vae_seedens_v1", "VAE seed-ens")], "soja"); plt.show()
explib.plot_confusion_grid([("vae_seedens_v1", "VAE seed-ens")], "maiz"); plt.show()"""),
    code('explib.show(explib.confusion_report("vae_seedens_v1", "soja"))'),
    md("""Se ve el *trade-off* del detector: al **top-10%** alta precisión (lo que marca casi
siempre es anomalía real) pero recall acotado por el presupuesto; al **top-tasa real**
precision y recall se equilibran. El resto de los falsos positivos son campañas de clima raro
con rinde normal — coherente con el techo estructural del NB 5."""),
    md("**Modelo final del Componente A.** En los próximos notebooks intentamos superarlo (datos nuevos, modernos) sin éxito."),
])

# ===========================================================================
# 04 — Limpieza de datos + features
# ===========================================================================
build("04_limpieza_de_datos_y_features.ipynb", "4 · Limpieza de datos y features nuevas", [
    md("""## 4.0 — El punto de quiebre: LIMPIEZA DE DATOS
El panel tenía **25% de filas duplicadas** (explosión cartesiana de lat/lon por un join que
ignoró la provincia) y la etiqueta corrompida por mezclar departamentos homónimos. Corregirlo
(dedup + clave provincia, en `data.py`) **subió a TODOS +0.09–0.14 PR-AUC** — la mayor mejora
del proyecto."""),
    code(SETUP),
    md("La progresión completa con las dos eras (exploración → limpieza → final):"),
    code("explib.show(explib.tbl_journey())"),
    md("""## 4.1 — Features nuevas (NDVI, suelo, heladas, agro)
Para capturar el ~70% de anomalías con clima normal (NB 5), extrajimos por
departamento×campaña con Google Earth Engine: **NDVI-AVHRR** (1981+), **ERA5-Land** (humedad
de suelo + días de helada) y **agronómicas** (ventana crítica). Efecto, con test sets
emparejados:"""),
    code("explib.show(explib.tbl_features())"),
    code("""t = explib.tbl_features(); x = np.arange(len(t)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, t["VAE_seedens"], w, label="VAE seed-ens", color="#4C72B0", alpha=0.85)
ax.bar(x + w/2, t["IForest"], w, label="IForest", color="#DD8452", alpha=0.85)
ax.axhline(0.592, ls="--", color="gray", lw=1, label="VAE base (0.592)")
ax.set_xticks(x); ax.set_xticklabels(t["features"], rotation=18, ha="right", fontsize=8)
ax.set_ylabel("PR-AUC (soja, test)"); ax.legend(fontsize=8)
ax.set_title("Features nuevas: ayudan al IForest, perjudican al VAE"); plt.tight_layout(); plt.show()"""),
    md("""**Conclusión:** ninguna feature extra supera al base con el VAE (diluyen el
`recon_prob`). Ayudan modestamente al IForest, pero IForest+ERA5 (0.542) sigue debajo del VAE
base (0.592).

**Artefacto cazado (lección de método):** un primer ERA5 dio 0.642 — era artefacto:
`frost_days`=0 en el norte → varianza-cero por depto → la normalización tiraba el 34% de los
departamentos → test sesgado. Se corrigió (std global de fallback). **Siempre validar que los
test sets coincidan antes de comparar.**"""),
])

# ===========================================================================
# 05 — Modernos + techo + comparación final
# ===========================================================================
build("05_modernos_y_techo_estructural.ipynb", "5 · Métodos modernos y el techo estructural", [
    md("""## 5.1 — Benchmark contra AD tabular profundo moderno
Comparamos contra el SOTA vía `deepod`: DeepSVDD, ICL, NeuTraL, GOAD."""),
    code(SETUP),
    code("explib.show(explib.tbl_modern())"),
    md("""**Out-of-the-box, ninguno se acerca al VAE** (el mejor, DeepSVDD, 0.43, queda
debajo hasta del IForest).

> ⚠️ **Limitación honesta de esta comparación:** estos modelos se corrieron con
> hiperparámetros **por defecto**, 100 épocas y **sin tuning**, mientras que a nuestro VAE
> lo exploramos a fondo (score, arquitectura, ensemble). **No es una comparación justa**, y
> NO alcanza para afirmar "somos mejores que los métodos modernos". Lo correcto sería
> tunearlos con el mismo esfuerzo (búsqueda de hiperparámetros + más épocas + multi-seed) —
> eso queda **pendiente**. La lectura válida es: *con configuración default no superan a
> nuestro VAE ya tuneado*, no que nuestro enfoque sea intrínsecamente superior."""),
    md("""## 5.2 — El techo estructural (por qué nadie pasa de ~0.6)
Tres análisis independientes muestran que el límite **no es del modelo**:
- **Cross-modelo**: ~68% de las anomalías las fallan TODOS, y tienen **clima normal**.
- **Auditoría de etiqueta**: excluir las sospechosas de ruido **no** mejora la PR-AUC.
- **Data map**: lo "siempre mal reconstruido" es rareza climática, no anomalía de rinde.

→ ~70% de las anomalías de rinde tienen causas **no climáticas** (plaga, granizo, manejo)
invisibles a cualquier feature disponible. Es del problema, no del modelo."""),
    code("""from IPython.display import Image, display
p = os.path.join(explib.ROOT, "analysis", "consensus_heatmap_soja.png")
display(Image(p)) if os.path.exists(p) else print("generar: python analyze_errors.py --cultivo soja")"""),
    md("## 5.3 — Comparación final de TODOS los modelos (con desvío estándar)"),
    code("explib.show(explib.tbl_leaderboard())"),
    code("explib.plot_leaderboard_compare(); plt.show()"),
    md("""**Modelo final: `vae_seedens_v1`** — mejor en ambos cultivos, con el desvío más chico.
Dos palancas: **limpieza de datos** + **score probabilístico + ensemble de semillas**. Ni
features satelitales, ni AE/DAE, ni métodos modernos lo superaron; el techo de ~0.6 es
estructural."""),
])

print("\\nNotebooks generados.")
