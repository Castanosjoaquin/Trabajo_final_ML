"""Genera los notebooks de experiments/ — el recorrido experimental del Componente A:
cada hipótesis de arquitectura, el experimento que la puso a prueba y la conclusión.
Todos los números salen de los runs actuales (mismo panel, misma evaluación).

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
# 00 — El problema y los datos
# ===========================================================================
build("00_el_problema_y_los_datos.ipynb", "0 · El problema y los datos", [
    md("""Serie de notebooks del Componente A. Cada uno toma una **hipótesis de
modelado**, muestra el **experimento** que la puso a prueba y la **conclusión** que
justifica el diseño final. Se pueden leer de corrido (vienen ejecutados) o re-correr.

| nb | pregunta que responde |
|---|---|
| **0** | ¿Qué datos, qué etiqueta, qué pipeline? |
| **1** | ¿Cómo se evalúa un detector acá (y por qué así)? + el baseline |
| **2** | ¿Reconstrucción? AE → DAE → scoring → híbrido → **VAE `recon_prob`** |
| **3** | ¿Cómo se elimina la varianza de los modelos profundos? El **seed-ensemble** |
| **4** | ¿Por qué ningún modelo pasa de ~0.6? El **techo estructural** |
| **5** | ¿Más features (agro, NDVI, suelo/heladas) ayudan? |
| **6** | ¿Y los métodos modernos? Leaderboard final y conclusiones |

---

**El problema:** detectar **campañas con rinde anómalo** por departamento (soja/maíz,
Argentina, 1981–2024) usando solo variables climáticas. Es **no supervisado**: el modelo
entrena solo con campañas normales y nunca ve la etiqueta."""),
    code(SETUP),
    md("""## 0.1 · El panel
Una fila = un (departamento, campaña, cultivo). Columnas: identificadores, `rinde_kgha`
(el objetivo) y ~54 features climáticas (promedios mensuales Sep–Mar de NASA POWER,
CHIRPS y ONI). La carga (`data.load_panel`) deduplica y usa siempre la clave geográfica
completa `[provincia, departamento]`."""),
    code("""from src.config import ExperimentConfig
from src import data as cdata
cfg = ExperimentConfig()
panel = cdata.load_panel(cfg)
print("shape:", panel.shape)
panel[["provincia","departamento","campania","cultivo","rinde_kgha"]].head()"""),
    md("""## 0.2 · La etiqueta proxy `z_rinde`
No existen etiquetas de "anomalía", así que la definimos: cuánto cae el rinde respecto
de la **media móvil del propio departamento**.

`z_rinde = (rinde − media_móvil_5_años) / desvío_móvil`  → `anomalía = z_rinde < −1.5`

Decisiones de diseño (y su porqué):
- Se calcula con `shift(1)`: solo pasado, **sin leakage**.
- Se agrupa por `[provincia, departamento, cultivo]`. El `cultivo` es clave: la soja
  rinde ~2700 kg/ha y el maíz ~6700 — mezclarlos en un mismo baseline sesgaría el
  z-score de cada uno en direcciones opuestas.
- **El modelo nunca la ve** — es solo para evaluar. Es la misma para todos los modelos
  (comparación justa)."""),
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
    md("""## 0.3 · Las features (X)
54 columnas = ~9 variables climáticas × 7 meses (Sep–Mar). NO incluyen el rinde."""),
    code("""feats = cdata.build_feature_list(panel, cfg.use_ndvi)
print(f"nº de features: {len(feats)}")
prefijos = sorted(set(f.rsplit('_',1)[0] for f in feats))
print("variables:", prefijos)"""),
    md("""## 0.4 · Splits temporales + entrenar-solo-con-normales + normalización
- **Split temporal** (no aleatorio): train ≤2017, val 2018–2020, test ≥2021. Simula el
  uso real: detectar anomalías en campañas futuras.
- **Train normal**: del train se excluyen las anomalías y los años con >30% de deptos
  anómalos → el modelo aprende solo "lo normal".
- **Normalización por departamento** (z-score) con stats calculadas SOLO en train normal
  (sin leakage). Si un feature tiene desvío 0 en un departamento, se usa el desvío
  global como fallback (no se pierde la fila)."""),
    code("""ds = cdata.build_crop_dataset(panel_z, "soja", cfg)
print(f"train normal (≤2017): {len(ds.X_train):5d} filas")
print(f"val        (2018-20): {len(ds.X_val):5d} filas  ({int(ds.y_val.sum())} anomalías)")
print(f"test       (≥2021)  : {len(ds.X_test):5d} filas  ({int(ds.y_test.sum())} anomalías)")
print(f"X_train: matriz {ds.X_train.shape} (filas normales × features, ya normalizada)")"""),
    md("""**Con esto el pipeline está listo.** Cualquier detector recibe `X_train` (normales)
para `fit()` y puntúa `X_val`/`X_test` con `score_samples()` (mayor = más anómalo). Todo
lo demás (entrenar, evaluar, comparar) se construye sobre esta interfaz común
(`AnomalyDetector`) → **próximo notebook**."""),
])

# ===========================================================================
# 01 — Evaluación y baseline
# ===========================================================================
build("01_evaluacion_y_baseline.ipynb", "1 · Cómo se evalúa (y por qué así) + el baseline", [
    md("""Antes de comparar modelos hay que poder **confiar en la comparación**. Tres
propiedades de este dataset obligan a elegir las métricas con cuidado; las tres están
verificadas experimentalmente:

1. **El F1 con umbral fijo es engañoso.** El umbral se calibra en val (5–9 anomalías,
   tasa base ~6%) y en test la tasa base salta a ~27% (la sequía 2022/23 está en test):
   un corte absoluto calibrado en val marca casi todo el test como anómalo y todos los
   modelos dan F1 ≈ tasa base. → La métrica principal es **PR-AUC** (libre de umbral,
   sensible al desbalance); ROC-AUC como referencia.
2. **Una sola semilla no es representativa.** Los modelos profundos varían mucho entre
   semillas (un mismo AE puede dar ROC 0.50 o 0.68 según la seed). → **Todo se reporta
   multi-seed: media ± desvío** (10 semillas por defecto).
3. **La validación es diminuta** (5–9 anomalías): sus métricas son ruido y no permiten
   seleccionar modelos (lo mostramos abajo). → Las comparaciones se ilustran en test,
   con el riesgo de *data snooping* dejado explícito."""),
    code(SETUP),
    md("""## 1.1 · Cómo se entrena y evalúa un modelo (reproducibilidad)
Todo modelo cumple la interfaz `AnomalyDetector`: `fit(X_train)` (solo normales) +
`score_samples(X)`. Cada experimento es un YAML de `configs/`. El helper
`explib.run_experiment` **carga** el run guardado o, con `force_train=True`, lo
**entrena de verdad** con el pipeline real y lo guarda en `runs/`."""),
    code('explib.show_yaml("configs/iforest/iforest_v2_mf03_n200.yaml")'),
    code('explib.describe_architecture("configs/iforest/iforest_v2_mf03_n200.yaml")'),
    code("""res = explib.run_experiment("configs/iforest/iforest_v2_mf03_n200.yaml", "soja",
                            force_train=False)   # ← cambiá a True para entrenarlo vos
print("run:", res.run_id)"""),
    code('explib.plot_all_metrics("iforest_v2_mf03_n200", "soja"); plt.show()'),
    md("""## 1.2 · La matriz de confusión y el *distribution shift*
Las métricas de arriba miden el **ranking** de los scores. La matriz de confusión
necesita un **corte**, y acá hay un fenómeno importante del dataset:

> ⚠️ **Un umbral calibrado en validación NO transfiere a test.** Como los splits son
> temporales, el clima de test (2021–24) está más lejos del período de entrenamiento que
> el de val (2018–20): *todo* reconstruye un poco peor y los scores suben en bloque. Un
> corte absoluto fijado en val termina marcando ~70% del test.

Por eso re-umbralizamos sobre los **scores del propio test** (sin usar etiquetas para el
corte) en **dos puntos de operación**: **top-10%** (presupuesto de alertas acotado) y
**top-tasa real** (~27%, la fracción realmente anómala del test)."""),
    code('explib.plot_confusion_grid([("iforest_v2_mf03_n200","IForest baseline")], "soja"); plt.show()'),
    code('explib.show(explib.confusion_report("iforest_v2_mf03_n200", "soja"))'),
    md("""Al **top-10%** la precisión es alta pero el recall queda acotado por el presupuesto;
al **top-tasa real** precision y recall se equilibran. Este es el patrón con el que se
lee el punto de operación de cualquier modelo de la serie.

## 1.3 · ⚠️ La validación no permite seleccionar (nota metodológica)
Lo correcto es elegir el modelo en **validación** y tocar **test una sola vez**. Veamos
qué da nuestra validación:"""),
    code("""modelos = [("iforest_v2_mf03_n200","IForest"), ("vae_seedens_v1","VAE seed-ens"),
           ("vae_v4_reconprob_lat16","VAE single")]
print("— VALIDACIÓN —"); display(explib.show(explib.compare_table(modelos, "soja", split="val")))
print("— TEST —");       display(explib.show(explib.compare_table(modelos, "soja", split="test")))"""),
    md("""Con 5–9 anomalías, las métricas de val son **ruido** — PR-AUC ≈ tasa base (~0.05)
para todos los modelos, con desvíos que solapan todo. Val no alcanza para seleccionar.
Lo declaramos en vez de esconderlo: las comparaciones se ilustran en test, y la confianza
en el modelo final viene de que su ventaja es **consistente en ambos cultivos** y su
varianza entre semillas es mínima (nb. 3).

## 1.4 · El baseline: Isolation Forest
Ensamble de árboles que aísla puntos raros; no neuronal, varianza baja — el punto de
referencia natural. La búsqueda de hiperparámetros (grilla sobre `max_features`,
`n_estimators`, `max_samples`) encontró una sola palanca con efecto real:
**`max_features=0.3`** (cada árbol mira un subconjunto de las 54 features, anomalías
más "locales")."""),
    code("""explib.show(explib.compare_table([
    ("iforest_v1_base","IForest default"),
    ("iforest_v2_mf03_n200","IForest tuneado (mf=0.3, n=200)")], "soja"))"""),
    md("""**Conclusión del notebook:** métrica principal PR-AUC, todo multi-seed, umbral
re-calibrado por split, y un listón claro: **IForest tuneado, PR-AUC 0.511 (soja)**.

**Próximo notebook:** ¿puede un modelo de reconstrucción superarlo?"""),
])

# ===========================================================================
# 02 — Reconstrucción: del AE al VAE recon_prob
# ===========================================================================
build("02_reconstruccion_ae_a_vae.ipynb",
      "2 · Reconstrucción: AE → DAE → scoring → híbrido → VAE recon_prob", [
    md("""**Hipótesis de partida:** un autoencoder entrenado solo con campañas normales
reconstruye mal las anómalas; el error de reconstrucción sirve de score de anomalía
(Sakurada & Yairi 2014). Este notebook recorre esa familia completa — cada variante con
su experimento — hasta el hallazgo que define el modelo final: **lo que importa no es la
arquitectura, es el score**."""),
    code(SETUP),
    md("""## 2.1 · AE: la arquitectura no era el cuello de botella
Encoder comprime las 54 features a un latente chico, decoder reconstruye,
score = error de reconstrucción (MSE). Config representativo:"""),
    code('explib.describe_architecture("configs/ae/ae_v11_cosine_deep.yaml")'),
    code("""explib.show(explib.compare_table([
    ("iforest_v2_mf03_n200","IForest (baseline)"),
    ("ae_v11_cosine_deep","AE")], "soja"))"""),
    code('explib.plot_loss("ae_v11_cosine_deep", "soja"); plt.show()'),
    md("""El AE entrena bien (la loss converge) pero queda **por debajo del baseline y con más
varianza** — y no por falta de tuning: en la fase de exploración se probaron **24
variantes** de arquitectura y optimización (capas, latente, activaciones, dropout,
weight decay, batch norm, schedules) sin que ninguna alcanzara al IForest.
**Conclusión: los hiperparámetros del AE no eran el cuello de botella.**

## 2.2 · ¿Y si el problema es cómo se agrega el error? (score max / top-k)
Hipótesis: el MSE *promedia* sobre 54 features; si solo unas pocas reconstruyen mal, la
señal se diluye. Probamos score = **máximo error por feature** y **suma de los top-5**:"""),
    code("""explib.show(explib.compare_table([
    ("ae_v11_cosine_deep","AE (MSE)"),
    ("ae_score_max","AE score=max"),
    ("ae_score_topk5","AE score=top-5")], "soja"))"""),
    md("""**Refutado:** igual o peor. La intuición era razonable, pero un agregado *fijo*
del error no es la respuesta (guardá la idea: la versión **probabilística** de esto sí
va a funcionar, en 2.5).

## 2.3 · DAE: denoising
Entrenar reconstruyendo *limpio* desde input *corrupto* hace la representación más
robusta al ruido — otra palanca clásica de la familia:"""),
    code("""explib.show(explib.compare_table([
    ("ae_v11_cosine_deep","AE"),
    ("dae_v1_base","DAE"),
    ("dae_seedens_v1","DAE seed-ensemble ×10")], "soja"))"""),
    md("""**Tampoco:** el denoising no supera al AE y queda lejos del baseline, incluso
promediando 10 semillas. La robustez al ruido del *input* no era el problema.

## 2.4 · Híbrido de la literatura: IForest sobre el latente del AE
Hipótesis: el AE comprime, el IForest aísla en el espacio comprimido — lo mejor de los
dos mundos."""),
    code("""explib.show(explib.compare_table([
    ("iforest_v2_mf03_n200","IForest (sobre features)"),
    ("ae_iforest_v2_latent16","IForest sobre latente del AE")], "soja"))"""),
    md("""**Refutado, con ganas:** peor que cada componente por separado y con la varianza
más alta de la familia. La receta asume que el latente preserva la estructura de
anomalía; en un panel heterogéneo de ~276 departamentos, el bottleneck la destruye.

## 2.5 · VAE: el hallazgo del *score probabilístico*
El VAE modela `p(x|z)` con media **y varianza** por feature. Eso habilita el score
**`recon_prob`** (An & Cho 2015): `-E[log p(x|z)]` estimado por Monte Carlo — el error
de cada feature **ponderado por la varianza que el decoder le asigna**. Es la versión
principiada de lo que `max`/`top-k` intentaba a mano. Dos experimentos lo aíslan:

**(a) Mismo VAE, distinto score** — el salto es del score, no de la arquitectura:"""),
    code("""explib.show(explib.compare_table([
    ("vae_v9_negelbo_lat16_bn","VAE score=neg_elbo"),
    ("vae_v4_reconprob_lat16","VAE score=recon_prob")], "soja"))"""),
    md("**(b) `recon_prob` con y sin regularización pesada** — la regularización destruye la señal:"),
    code("""explib.show(explib.compare_table([
    ("vae_v4_reconprob_lat16","recon_prob limpio (v4)"),
    ("vae_v8_reconprob_reg","recon_prob + BN/dropout/β=0.5")], "soja"))"""),
    md("""La varianza per-feature del decoder ya regulariza el score; apilarle batch norm,
dropout y β bajo lo aplana. El refinamiento convergió rápido: β óptimo 1.0–1.5,
**latente 16**, hidden [64,32] (la versión profunda [128,64], `v15`, es comparable).
Config final del single:"""),
    code('explib.describe_architecture("configs/vae/vae_v4_reconprob_lat16.yaml")'),
    code('explib.plot_loss("vae_v4_reconprob_lat16", "soja"); plt.show()'),
    md("### La familia completa, contra el baseline"),
    code("""recon = [("iforest_v2_mf03_n200","IForest (baseline)"),
         ("ae_v11_cosine_deep","AE"), ("dae_v1_base","DAE"),
         ("ae_score_max","AE score=max"), ("ae_iforest_v2_latent16","Híbrido AE+IForest"),
         ("vae_v9_negelbo_lat16_bn","VAE neg_elbo"),
         ("vae_v4_reconprob_lat16","VAE recon_prob"),
         ("vae_v15_reconprob_deep","VAE recon_prob deep")]
explib.show(explib.compare_table(recon, "soja"))"""),
    code('explib.plot_compare(recon, "soja", baseline=0.511, title="Familia reconstrucción — PR-AUC (soja)"); plt.show()'),
    md("""### El score, visto en las distribuciones
Score de campañas **normales** (azul) vs **anómalas** (naranja) en test. Cuanto más
separadas, mejor rankea. Se ve por qué `recon_prob` gana: corre la cola de las anómalas
más a la derecha que el MSE del AE o el híbrido."""),
    code("""explib.plot_score_hist_grid([("ae_v11_cosine_deep", "AE (MSE)"),
    ("ae_iforest_v2_latent16", "Híbrido"),
    ("vae_v4_reconprob_lat16", "VAE recon_prob")], "soja", ncols=3); plt.show()"""),
    md("""**Conclusiones del notebook:**
- AE/DAE con score MSE quedan por debajo del baseline, y no por hiperparámetros (24
  variantes), ni por el agregado del error (max/top-k refutados), ni por denoising, ni
  por combinar con IForest (híbrido refutado).
- El **VAE con `recon_prob`** es el único de la familia que supera al IForest
  (0.559 vs 0.511, soja) — y el salto es atribuible al **score** (+0.085 sobre el mismo
  VAE con `neg_elbo`), no a la arquitectura.
- La señal de `recon_prob` requiere el modelo **sin regularización pesada** (−0.10 al
  agregar BN/dropout/β bajo).

**Pero** el single tiene desvío ±0.035 entre semillas: en una corrida mala roza el
baseline. **Próximo notebook:** cómo se elimina esa varianza."""),
])

# ===========================================================================
# 03 — Varianza y seed-ensemble
# ===========================================================================
build("03_varianza_y_seed_ensemble.ipynb", "3 · La varianza y el seed-ensemble (modelo final)", [
    md("""**El problema:** el VAE `recon_prob` gana en promedio (0.559 vs 0.511, soja) pero
con ±0.035 de desvío entre semillas. Para recomendar un modelo hace falta que gane
**siempre**, no según la suerte de la seed.

**La pregunta:** ¿de dónde viene la varianza y cómo se elimina?"""),
    code(SETUP),
    md("""## 3.1 · Descartando causas
- **¿El estimador Monte Carlo del score?** No: duplicar las muestras MC (50→100) da un
  resultado idéntico. La varianza viene del **entrenamiento** (cada semilla converge a
  un óptimo local distinto), no del score.
- **¿Falta regularización?** No: el nb. 2 mostró que la regularización pesada *destruye*
  la señal de `recon_prob`.
- **¿Colas pesadas en el decoder (Student-t)?** Tampoco: empeora la media y el desvío
  **sube**. La razón es informativa: las colas pesadas protegen cuando el train está
  *contaminado* con outliers, pero nuestro pipeline ya entrena solo con normales
  curados (nb. 0). No había nada que robustecer:"""),
    code("""explib.show(explib.compare_table([
    ("vae_v4_reconprob_lat16","VAE gaussian (ref)"),
    ("vae_studentt_v1","VAE Student-t ν=4")], "soja"))"""),
    md("""## 3.2 · La solución: promediar semillas (seed-ensemble)
Si cada semilla es un óptimo local distinto pero igualmente válido, su ruido
idiosincrático se cancela **promediando los scores** (z-normalizados per-miembro) de N
semillas. El `EnsembleDetector` es agnóstico al modelo base:"""),
    code('explib.describe_architecture("configs/ensemble/vae_seedens_v1.yaml")'),
    code("""ens = [("vae_v4_reconprob_lat16","VAE single"),
       ("vae_seedens_v1","VAE seed-ens ×10 (lat16)"),
       ("vae_seedens_deep","VAE seed-ens ×10 (deep)"),
       ("vae_iforest_ens_v1","hetero VAE+IForest")]
explib.show(explib.compare_table(ens, "soja"))"""),
    code('explib.plot_compare(ens, "soja", baseline=0.511, title="Ensembles — PR-AUC (soja)"); plt.show()'),
    md("""**El desvío se desploma (±0.035 → ±0.010) y la media SUBE** (0.559 → 0.592): el
promedio no solo estabiliza, también mejora, porque los errores de cada semilla son
independientes. Con `media − desvío = 0.582 ≫ 0.511`, el seed-ensemble supera al
baseline **también en el peor caso**, en ambos cultivos.

Dos controles que delimitan la conclusión:
- **Hetero-ensemble VAE+IForest**: queda a la altura del VAE single y por debajo del
  seed-ensemble puro — el IForest no aporta señal nueva, solo diluye. Mezclar modelos
  solo suma si el segundo trae información que el primero no tiene.
- **El ensemble no salva una base débil** — mismo tratamiento al AE y al DAE
  (base [64,32] con score MSE):"""),
    code("""explib.show(explib.compare_table([
    ("ae_seedens_v1","AE seed-ens ×10 (MSE)"),
    ("dae_seedens_v1","DAE seed-ens ×10 (MSE)"),
    ("vae_seedens_v1","VAE seed-ens ×10 (recon_prob)")], "soja"))"""),
    md("""El ensemble les elimina la varianza igual… pero la media no acompaña: **el
diferenciador es el score `recon_prob`** (nb. 2), el ensemble solo consolida.

## 3.3 · El modelo elegido: `vae_seedens_v1`
VAE `recon_prob` (lat 16, hidden [64,32], β=1) × 10 semillas. Curva PR y punto de
operación:"""),
    code("""explib.plot_pr_curves([("vae_seedens_v1","VAE seed-ensemble"),
                       ("vae_v4_reconprob_lat16","VAE single"),
                       ("iforest_v2_mf03_n200","IForest baseline")], "soja"); plt.show()"""),
    code('explib.plot_confusion_grid([("vae_seedens_v1", "VAE seed-ens")], "soja"); plt.show()'),
    md("""**Conclusiones del notebook:** la varianza del VAE viene del entrenamiento y se
resuelve con el **ensemble de semillas** (no con más MC, ni regularización, ni Student-t);
el hetero-ensemble y los seed-ensembles de AE/DAE delimitan el resultado: hace falta
**base fuerte (`recon_prob`) + promedio de semillas**. Modelo final del Componente A:
**`vae_seedens_v1` — soja 0.592 ± 0.010, maíz 0.508 ± 0.005**.

**Pero** queda una pregunta incómoda: ¿por qué *ningún* modelo — de familias totalmente
distintas — pasa de ~0.6? **Próximo notebook.**"""),
])

# ===========================================================================
# 04 — El techo estructural
# ===========================================================================
build("04_el_techo_estructural.ipynb", "4 · ¿Por qué ningún modelo pasa de ~0.6? El techo estructural", [
    md("""**La observación:** IForest, AE, DAE, híbridos, VAE y ensembles — arquitecturas
muy distintas — quedan todas en una banda acotada, y el mejor no pasa de ~0.6. Cuando
familias tan diferentes chocan contra el mismo número, la sospecha cambia de lugar: ¿y
si el límite no está en los modelos sino en la **señal**?

**Tres análisis independientes** lo confirman."""),
    code(SETUP),
    md("""## 4.1 · Análisis cross-modelo: todos fallan en las MISMAS anomalías
`analyze_errors.py` alinea las predicciones de test de las **7 familias** (IForest, AE,
DAE, híbrido, VAE single, VAE seed-ensemble, DeepSVDD) y computa el consenso de errores
(detalle en `analysis/summary_soja.md`). Resultado (soja):

- **127 de 257 anomalías (49%) las fallan TODOS los modelos a la vez.**
- Las detectadas tienen precipitación claramente deficitaria (z medio **−0.84**); las
  falladas, clima cercano a lo normal (**−0.41**) — el **55% de las falladas no tiene
  firma climática** (precip_z ≥ −0.5).

Si cada modelo fallara en cosas distintas, un ensemble lo arreglaría; fallar todos en
las mismas apunta a la señal."""),
    code("""from IPython.display import Image, display
p = os.path.join(explib.ROOT, "analysis", "consensus_heatmap_soja.png")
display(Image(p)) if os.path.exists(p) else print("generar: python analyze_errors.py --cultivo soja")"""),
    md("""El mismo fenómeno, medido dentro de cada run — el **recall estratificado** por causa
climática (anomalías con precipitación deficitaria vs el resto):"""),
    code("""explib.show(explib.summary_fields([
    ("iforest_v2_mf03_n200","IForest"),
    ("vae_seedens_v1","VAE seed-ens")], "soja",
    keys=["recall_clima_adverso","n_clima_adverso","recall_otros","n_otros"]))"""),
    md("""Las anomalías **con clima adverso** se detectan casi todas; las **"otras"** (rinde
anómalo con clima normal) casi ninguna — en todos los modelos.

## 4.2 · ¿Lo "consistentemente difícil" es anomalía? (Dataset Cartography)
Hipótesis propuesta por el docente: lo que el modelo reconstruye **siempre** mal podría
ser anomalía. La implementamos con dos herramientas: el **desacuerdo del ensemble**
(`score_std` entre los 10 miembros) y un **data map** (Swayamdipta 2020) adaptado a
reconstrucción: error por muestra **por época** → *confidence* + *variability* → mapa
easy / ambiguous / hard-to-learn."""),
    code("""from IPython.display import Image, display
p = os.path.join(explib.ROOT, "analysis", "datamap_soja.png")
display(Image(p)) if os.path.exists(p) else print("generar: python analyze_datamap.py --cultivo soja")"""),
    md("""**Resultado:** el **27% del train normal** es hard-to-learn, pero su `z_rinde`
medio (0.66) es igual al del train completo (0.69) → lo que cuesta reconstruir es
**rareza climática, no anomalía de rinde**. La hipótesis queda refutada como *detector* (y por
eso el modelo final no la usa para puntuar), pero validada como *auditoría*: confirma
que dificultad-de-reconstrucción ≠ anomalía-de-rinde, que es exactamente el techo.

## 4.3 · ¿Y si la etiqueta es ruido? (auditoría de `z_rinde`)
Última explicación posible: que las anomalías "invisibles" fueran artefactos de la
etiqueta proxy. `analyze_labels.py` clasifica cada anomalía por la fragilidad del
denominador del z-score (CV del baseline, z extremos, historia corta):

- La gran mayoría de las anomalías es **genuina** (86% en soja; 10% z-extremo, 4%
  baseline inestable).
- **Excluir las sospechosas EMPEORA la PR-AUC** (−0.021) → no son ruido removible.
- Las z-extremo (z < −5) se detectan a la misma tasa que las genuinas (recall 0.32 vs
  0.31) → son catástrofes reales, no artefactos.

**Refutado también:** el techo no es ruido de etiqueta."""),
    md("""## 4.4 · El techo, visto en el espacio de features (t-SNE)
Proyección t-SNE de las campañas de test coloreada por **etiqueta real** (izq.) y por
**score del VAE** (der.). Las anomalías (rojo) **no forman un cluster**: están mezcladas
entre las normales. Si no se distinguen en el espacio de features, ningún modelo que
mire ese espacio puede separarlas. El score del VAE (der.) marca una región — la de
**clima raro** — que solo se solapa parcialmente con el rojo."""),
    code("""fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
explib.plot_embeddings("vae_seedens_v1", "soja", method="tsne", color_by="label", ax=axs[0])
explib.plot_embeddings("vae_seedens_v1", "soja", method="tsne", color_by="score", ax=axs[1])
axs[0].set_title("t-SNE · etiqueta real (normal vs anómala)")
axs[1].set_title("t-SNE · score del VAE")
plt.tight_layout(); plt.show()"""),
    md("""**Conclusión del notebook:** el techo es **estructural** — una mayoría de las
anomalías de rinde tiene causas no climáticas (plaga, granizo, manejo, mercado)
invisibles para un detector que solo ve clima. No es la arquitectura (todas fallan en
las mismas muestras) ni la etiqueta (la auditoría lo descarta). La única salida posible:
**darle al modelo más señal** — **próximo notebook**."""),
])

# ===========================================================================
# 05 — ¿Más features ayudan?
# ===========================================================================
build("05_features_nuevas.ipynb", "5 · ¿Más señal? Features agronómicas y satelitales", [
    md("""**Hipótesis:** si el clima mensual no ve ~2/3 de las anomalías (nb. 4), quizás
otras fuentes sí: features **agronómicas** de la ventana crítica del cultivo, el
**verdor de la planta** (NDVI-AVHRR, 1981+, vía Google Earth Engine) y el **estado del
suelo + heladas** (ERA5-Land). Si una plaga arrasa un lote, el clima no lo registra —
pero el NDVI debería.

Cada adición se evalúa con el flag correspondiente del pipeline (`use_agro_features`,
`use_ndvi`, `use_era5_features`), mismo protocolo, sobre el VAE y el IForest."""),
    code(SETUP),
    md("""## 5.1 · Features agronómicas (ventana crítica + balance hídrico)
Precipitación, estrés térmico y balance hídrico de Hargreaves en la ventana crítica
del cultivo (Dic–Feb soja / Nov–Ene maíz): conocimiento de dominio destilado en
~4 features."""),
    code("""explib.show(explib.compare_table([
    ("iforest_v2_mf03_n200","IForest base"), ("iforest_v2_agro","IForest + agro"),
    ("vae_v4_reconprob_lat16","VAE base"), ("vae_v4_agro","VAE + agro")], "soja"))"""),
    md("""## 5.2 · NDVI-AVHRR (verdor, 1981+)
NDVI mensual por departamento desde 1981 (AVHRR), mergeado por
`[provincia, departamento, campaña]` — historia completa, sin recortar el train.
*(El NDVI de MODIS se descartó de plano: solo existe desde 2002 y recortar la historia
de train cuesta más de lo que el NDVI aporta.)*"""),
    code("""explib.show(explib.compare_table([
    ("vae_seedens_v1","VAE seed-ens base"), ("vae_seedens_v1_ndvilargo","VAE seed-ens + NDVI"),
    ("iforest_v2_mf03_n200","IForest base"), ("iforest_v2_ndvilargo","IForest + NDVI")], "soja"))"""),
    md("""## 5.3 · ERA5-Land: humedad de suelo + heladas
Cuatro features por (departamento, campaña): humedad de suelo en siembra y en el
invierno previo (inercia hídrica), días de helada y heladas tardías."""),
    code("""explib.show(explib.compare_table([
    ("vae_seedens_v1","VAE seed-ens base"), ("vae_seedens_v1_era5","VAE seed-ens + ERA5"),
    ("iforest_v2_mf03_n200","IForest base"), ("iforest_v2_era5","IForest + ERA5")], "soja"))"""),
    md("### Resumen visual: efecto de cada adición sobre cada modelo"),
    code("""pares = [("base", "vae_seedens_v1", "iforest_v2_mf03_n200"),
         ("+ agro", "vae_v4_agro", "iforest_v2_agro"),
         ("+ NDVI-AVHRR", "vae_seedens_v1_ndvilargo", "iforest_v2_ndvilargo"),
         ("+ ERA5", "vae_seedens_v1_era5", "iforest_v2_era5")]
def pr(name):
    rid = explib.latest_run(name, "soja")
    s = explib.load_run(rid).summary
    return s.get("test_pr_auc_mean", s.get("test_pr_auc"))
labs = [p[0] for p in pares]
vae_v = [pr(p[1]) for p in pares]; ifo_v = [pr(p[2]) for p in pares]
x = np.arange(len(labs)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, vae_v, w, label="VAE", color="#4C72B0", alpha=0.85)
ax.bar(x + w/2, ifo_v, w, label="IForest", color="#DD8452", alpha=0.85)
ax.axhline(pr("vae_seedens_v1"), ls="--", color="gray", lw=1, label="VAE base")
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_ylabel("PR-AUC (soja, test)"); ax.legend(fontsize=8)
ax.set_title("Features nuevas: ayudan al IForest, perjudican al VAE")
plt.tight_layout(); plt.show()
print("(la fila 'agro' usa el VAE single — la ablación agro se corrió sobre ese modelo)")"""),
    md("""**Conclusiones del notebook:**
- **Asimetría consistente**: las features extra ayudan (modestamente) al IForest — que
  submuestrea columnas y tolera agregados — y **perjudican al VAE**, que debe
  reconstruir TODAS las features: cada columna nueva diluye el `recon_prob` y sube la
  varianza.
- **Ninguna combinación supera al VAE base** (0.592): el mejor "aumentado" es
  IForest+ERA5, y queda abajo.
- El resultado **confirma el techo del nb. 4**: si ni el verdor de la planta marca esas
  anomalías, no son fallas biofísicas observables por satélite. Mejorar requiere otra
  clase de datos (sanidad, granizo, manejo), no más features climático-satelitales.

**Próximo notebook:** la última vara — los métodos modernos de deep anomaly detection —
y el cierre."""),
])

# ===========================================================================
# 06 — Modernos, leaderboard final y conclusiones
# ===========================================================================
build("06_modernos_leaderboard_y_conclusiones.ipynb",
      "6 · Métodos modernos, leaderboard final y conclusiones", [
    md("""**Última pregunta de exhaustividad:** ¿un método moderno de deep anomaly detection
supera lo nuestro? Benchmark vía `deepod`: **DeepSVDD** (one-class), **ICL**
(contrastive), **NeuTraL** (transformaciones aprendidas).

> **Protocolo justo:** mismas 300 épocas que nuestros modelos, capacidad equivalente
> (hidden 64,32 + rep_dim 16) y multi-seed — no los defaults de la librería. GOAD quedó
> afuera por costo prohibitivo (256 transformaciones por época)."""),
    code(SETUP),
    code("explib.show(explib.tbl_modern())"),
    code("""explib.show(explib.compare_table([
    ("deepod_deepsvdd","DeepSVDD"), ("deepod_icl","ICL"), ("deepod_neutral","NeuTraL"),
    ("iforest_v2_mf03_n200","IForest"), ("vae_seedens_v1","VAE seed-ens")], "soja"))"""),
    md("""**Ninguno supera al VAE — ni siquiera al IForest.** Lectura honesta: esto no nos
hace estado del arte del campo; dice que en *este* dataset, con presupuesto parejo, no
despegan. Es coherente con el techo estructural (nb. 4): no hay señal extra que un método
más sofisticado pueda exprimir.

## 6.1 · Leaderboard final (media ± desvío, test)"""),
    code("explib.show(explib.tbl_leaderboard())"),
    code("explib.plot_leaderboard_compare(); plt.show()"),
    md("## 6.2 · La progresión completa, con números comparables"),
    code("""prog = [("iforest_v1_base","IForest default"),
        ("iforest_v2_mf03_n200","IForest tuneado (mf=0.3)"),
        ("ae_v11_cosine_deep","AE (MSE)"),
        ("dae_v1_base","DAE"),
        ("ae_iforest_v2_latent16","Híbrido AE+IForest"),
        ("vae_v9_negelbo_lat16_bn","VAE neg_elbo"),
        ("vae_v4_reconprob_lat16","VAE recon_prob single"),
        ("deepod_deepsvdd","DeepSVDD (moderno)"),
        ("vae_seedens_v1","VAE seed-ensemble ★")]
explib.show(explib.compare_table(prog, "soja"))"""),
    code('explib.plot_compare(prog, "soja", baseline=0.511, title="Progresión del proyecto — PR-AUC (soja)"); plt.show()'),
    md("""## 6.3 · El modelo final por dentro: `vae_seedens_v1`
VAE `recon_prob` (lat 16, hidden [64,32], β=1) × 10 semillas, scores z-normalizados y
promediados. Métricas completas y punto de operación en ambos cultivos (ver nb. 1 por
qué se re-umbraliza sobre el propio test):"""),
    code('explib.show(explib.run_metrics("vae_seedens_v1", "soja"))'),
    code('explib.plot_all_metrics("vae_seedens_v1", "soja"); plt.show()'),
    code("""explib.plot_confusion_grid([("vae_seedens_v1", "VAE seed-ens")], "soja"); plt.show()
explib.plot_confusion_grid([("vae_seedens_v1", "VAE seed-ens")], "maiz"); plt.show()"""),
    code('explib.show(explib.confusion_report("vae_seedens_v1", "soja"))'),
    md("""Al **top-10%**, precisión ~0.8: casi todo lo que marca es anomalía real, con recall
acotado por el presupuesto de alertas; al **top-tasa real** se equilibran. Los falsos
positivos restantes son campañas de clima raro con rinde normal — coherente con el
techo del nb. 4.

## 6.4 · Limitaciones
1. **Distribution shift temporal**: scores y tasa base suben de val a test (el clima
   2021–24 se aleja del train; el test incluye la sequía 2022/23). Por eso todo se
   decide por PR-AUC (ranking) y los umbrales se re-calibran por split (nb. 1).
2. **Validación chica** (5–9 anomalías): sus métricas son ruido y no permiten
   seleccionar; las comparaciones se ilustran en test con el *data snooping* declarado
   (nb. 1).
3. **Techo estructural**: la mayoría de las anomalías de rinde no tiene causa climática
   ni vegetacional observable — límite del problema, no del modelo (nbs. 4–5).

## Conclusiones del Componente A
1. **Modelo final: `vae_seedens_v1`** — seed-ensemble ×10 del VAE con score
   `recon_prob` (An & Cho 2015): **soja 0.592 ± 0.010, maíz 0.508 ± 0.005**, el desvío
   más chico del leaderboard. Supera al baseline en media **y en peor caso**, en ambos
   cultivos.
2. Las **dos decisiones de arquitectura** que explican el resultado: el **score
   probabilístico** `recon_prob` sin regularización pesada (nb. 2) y el **ensemble de
   semillas** (nb. 3).
3. Las alternativas quedaron refutadas experimentalmente: tuning de HP del AE, scoring
   max/top-k, híbrido AE+IForest (nb. 2), Student-t, hetero-ensemble (nb. 3), features
   agro/NDVI/ERA5 (nb. 5) y el deep AD moderno a presupuesto parejo (nb. 6).
4. El desempeño está acotado por un **techo estructural** demostrado por tres análisis
   independientes (nb. 4): la mayoría de las anomalías de rinde no deja huella en las
   features disponibles. Mejorar de acá en adelante requiere **otra clase de datos**
   (sanidad, granizo, manejo), no otro modelo."""),
])

print("\nNotebooks generados.")
