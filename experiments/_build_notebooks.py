"""Genera los notebooks de experiments/ (vía nbformat). Correr una vez:
    python experiments/_build_notebooks.py
Los notebooks leen de runs/ guardados (no re-entrenan). Después se ejecutan con
nbconvert para embeber las salidas."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


SETUP = """import explib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 110"""


def build(fname, title, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [md(f"# {title}")] + cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    with open(os.path.join(HERE, fname), "w") as f:
        nbf.write(nb, f)
    print("escrito:", fname)


# ===========================================================================
# 00 — Overview
# ===========================================================================
build("00_overview.ipynb", "Componente A — Overview de experimentos", [
    md("""**Detección de anomalías en rendimientos agrícolas** (soja/maíz, Argentina).
Modelo no supervisado: se entrena solo con campañas *normales* y puntúa por
desviación. La etiqueta proxy es `z_rinde < -1.5` (caída de rinde vs la media móvil
del propio departamento). El detector nunca ve la etiqueta — solo se usa para evaluar.

**Métrica principal: PR-AUC** (libre de umbral). Las features son climáticas
(promedios mensuales Sep–Mar de ~9 variables = 54 features).

Estos notebooks reconstruyen los experimentos **desde los runs guardados en
`runs/`** (no re-entrenan). Índice:

| notebook | contenido |
|---|---|
| `01` | Modelos base (AE/DAE/VAE) + búsqueda de hiperparámetros |
| `02` | Ensembles (seed-ensemble, AE/DAE, heterogéneo) |
| `03` | Híbridos (AE-latente+IForest) y scoring alternativo |
| `04` | Features nuevas (limpieza de datos, NDVI, ERA5 suelo/heladas, agro) |
| `05` | Métodos modernos (deepod) + techo estructural |"""),
    code(SETUP),
    md("## Leaderboard final (panel limpio, test PR-AUC media±std)"),
    code("explib.show(explib.tbl_leaderboard())"),
    md("### Comparación final de los modelos, con desvío estándar (¿cuál es mejor?)"),
    code("explib.plot_leaderboard_compare(); plt.show()"),
    md("""**Conclusión:** el mejor modelo es el **seed-ensemble del VAE con score
`recon_prob`** (soja 0.592, maíz 0.508), y su **desvío es diminuto** (±0.005–0.010)
frente al VAE single (±0.035–0.070) — gana no solo en promedio sino en el peor caso.
Dos palancas dominaron el proyecto: la **limpieza de datos** (notebook 04) y el
**score probabilístico + ensemble de semillas** (notebooks 01–02). El resto
(features satelitales, métodos modernos) no superó al base — ver notebooks 04–05.

### Todas las métricas del modelo ganador (test, con ±std entre seeds)"""),
    code("""explib.show(explib.run_metrics("vae_seedens_v1", "soja"))"""),
    code("""explib.plot_all_metrics("vae_seedens_v1", "soja"); plt.show()"""),
])

# ===========================================================================
# 01 — Modelos base + tuning
# ===========================================================================
build("01_modelos_base_y_tuning.ipynb", "1 · Modelos base (AE / DAE / VAE) y búsqueda de hiperparámetros", [
    md("""Arrancamos con autoencoders clásicos. **Hallazgo central:** el *score* importa
más que la arquitectura. El VAE con `recon_prob` (An & Cho 2015) — que pondera el
error de cada feature por su varianza esperada — supera por lejos al MSE plano del
AE/DAE.

Tuneamos 7 variantes de AE: el rango de PR-AUC fue de solo **0.034** → el cuello de
botella no eran los hiperparámetros sino el enfoque (score y varianza)."""),
    code(SETUP),
    md("## AE / DAE / VAE: el score es lo que diferencia (soja)"),
    code("explib.show(explib.tbl_baselines())"),
    md("""- **AE/DAE con MSE plano**: 0.33–0.41, muy por debajo.
- **VAE `recon_prob`**: 0.559 — el score probabilístico es el diferenciador.
- El `score=max` (máximo error por feature) no ayudó: el problema no era el agregador.

### Distribución de scores del VAE (separación normal vs anómala)"""),
    code("""explib.plot_score_hist("vae_seedens_v1", "soja"); plt.show()"""),
    md("### Curva de loss del entrenamiento del VAE (train vs val, por época)"),
    code("""explib.plot_loss("vae_v4_reconprob_lat16", "soja"); plt.show()"""),
    md("""El early-stopping corta cuando la val_loss deja de mejorar (restaura la mejor
época). El `recon_prob` se evalúa sobre ese modelo entrenado.

### Todas las métricas del VAE (test, con desvío estándar entre seeds)"""),
    code("""explib.show(explib.run_metrics("vae_v4_reconprob_lat16", "soja"))"""),
    code("""explib.plot_all_metrics("vae_v4_reconprob_lat16", "soja"); plt.show()"""),
    md("""**Negativos útiles documentados:** subir `n_mc_samples` no baja la varianza
(viene del entrenamiento, no del estimador MC); `latent_dim`>16 y `beta` fuera de
[1, 1.5] la empeoran; el decoder Student-t fue neutro (el train ya está curado).
El camino correcto resultó ser el **ensemble** (notebook 02)."""),
])

# ===========================================================================
# 02 — Ensembles
# ===========================================================================
build("02_ensembles.ipynb", "2 · Ensembles: seed-ensemble, AE/DAE y heterogéneo", [
    md("""El problema #1 diagnosticado del VAE era la **varianza entre semillas** (cada
seed converge distinto). Solución: **seed-ensemble** — promediar los scores
normalizados (z-score per-miembro) de 10 VAEs con semillas distintas. El
`EnsembleDetector` es agnóstico al modelo, así que la misma idea aplica a AE/DAE y a
ensembles heterogéneos."""),
    code(SETUP),
    md("## El seed-ensemble resuelve la varianza (soja)"),
    code("explib.show(explib.tbl_ensembles())"),
    md("""- **VAE single → seed-ensemble**: la varianza cae (±0.035 → ±0.010) **y la
  media sube** (0.559 → 0.592). El promedio cancela los errores idiosincráticos de
  cada semilla y conserva la señal compartida.
- **Hetero VAE+IForest**: mixto — el IForest, más débil, arrastra la soja hacia abajo.
- **AE/DAE ensembles**: bajan la varianza pero no salvan un base débil (MSE).

### Curvas PR: single vs ensemble"""),
    code("""explib.plot_pr_curves([("vae_v4_reconprob_lat16","VAE single"),
                       ("vae_seedens_v1","VAE seed-ensemble ×10")], "soja"); plt.show()"""),
    md("""### PR-AUC con desvío estándar: el ensemble baja la varianza
Comparación directa (barras con ±std) — se ve cómo el ensemble achica el desvío."""),
    code("""explib.plot_metric_comparison([
    ("iforest_v2_mf03_n200","IForest baseline"),
    ("vae_v4_reconprob_lat16","VAE single"),
    ("vae_v15_reconprob_deep","VAE deep single"),
    ("vae_seedens_v1","VAE seed-ens"),
    ("vae_seedens_deep","VAE seed-ens deep"),
], "soja", metric="pr_auc"); plt.show()"""),
    md("Mismo con **Recall@contam** (cuántas anomalías reales capturamos al presupuesto del 10%):"),
    code("""explib.plot_metric_comparison([
    ("iforest_v2_mf03_n200","IForest baseline"),
    ("vae_v4_reconprob_lat16","VAE single"),
    ("vae_seedens_v1","VAE seed-ens"),
], "soja", metric="recall_at_contamination"); plt.show()"""),
    md("**Conclusión:** el seed-ensemble del VAE `recon_prob` es el modelo final (SOTA)."),
])

# ===========================================================================
# 03 — Híbridos y scoring
# ===========================================================================
build("03_hibridos_y_scoring.ipynb", "3 · Híbridos (AE-latente + IForest) y scoring alternativo", [
    md("""Probamos combinar lo mejor de cada mundo y scorings alternativos al MSE
promedio. **Todos quedaron por debajo del base** — útil para justificar el modelo
final por descarte."""),
    code(SETUP),
    md("## Híbrido AE+IForest y scoring max/top-k (soja, panel union)"),
    code("explib.show(explib.tbl_hybrids())"),
    code("""t = explib.tbl_hybrids()
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh(t["modelo"], t["soja_pr"], color="#C44E52", alpha=0.85)
for i, v in enumerate(t["soja_pr"]): ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
ax.axvline(0.435, ls="--", color="gray", lw=1, label="VAE referencia (0.435)")
ax.set_xlabel("PR-AUC (soja, test)"); ax.legend(fontsize=8)
ax.set_title("Híbridos y scoring alternativo (todos por debajo del VAE)")
plt.tight_layout(); plt.show()"""),
    md("""- **AE-latente + IForest**: correr IForest sobre el latente del AE (en vez del
  espacio original). En la literatura suele ganar, pero acá **el bottleneck del AE
  destruye la estructura** que el IForest necesita → 0.307, peor que ambos por
  separado.
- **score=max / top-k**: la señal **no** se diluía en el promedio MSE; el cuello de
  botella del AE era otro (el recon_prob del VAE, no el agregador).

El tuning de `max_features` (0.3–0.5) sí hizo al **IForest** un baseline fuerte
(0.51), pero ninguno de estos enfoques superó al VAE."""),
])

# ===========================================================================
# 04 — Features nuevas
# ===========================================================================
build("04_features_nuevas.ipynb", "4 · Features nuevas: limpieza de datos, NDVI, ERA5 (suelo/heladas), agro", [
    md("""### 4.0 — La mayor mejora del proyecto: LIMPIEZA DE DATOS
El panel tenía **25% de filas duplicadas** (explosión cartesiana de lat/lon por un
join que ignoró la provincia) y la etiqueta `z_rinde` estaba corrompida por agrupar
departamentos homónimos de distintas provincias. Corregirlo (dedup + clave
provincia, en `data.py`) **subió a TODOS los modelos +0.09 a +0.14 PR-AUC** — más
que cualquier cambio de modelo."""),
    code(SETUP),
    md("""### 4.1 — Features satelitales / reanálisis
Buscamos features que capturen el ~70% de anomalías **sin firma climática** (ver
notebook 05). Tres fuentes nuevas, todas extraídas por departamento×campaña con
Google Earth Engine:
- **NDVI-AVHRR** (1981+): vigor vegetal.
- **ERA5-Land**: humedad de suelo (estado inicial / "inercia") + días de helada.
- **Agronómicas**: ventana crítica + balance hídrico (derivadas de columnas existentes).

Efecto sobre el modelo SOTA (VAE) y el IForest, **con test sets emparejados**:"""),
    code("explib.show(explib.tbl_features())"),
    code("""import numpy as np
t = explib.tbl_features()
x = np.arange(len(t)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, t["VAE_seedens"], w, label="VAE seed-ens", color="#4C72B0", alpha=0.85)
ax.bar(x + w/2, t["IForest"], w, label="IForest", color="#DD8452", alpha=0.85)
ax.axhline(0.592, ls="--", color="gray", lw=1, label="VAE base (0.592)")
ax.set_xticks(x); ax.set_xticklabels(t["features"], rotation=18, ha="right", fontsize=8)
ax.set_ylabel("PR-AUC (soja, test)"); ax.legend(fontsize=8)
ax.set_title("Efecto de las features nuevas (ayudan al IForest, perjudican al VAE)")
plt.tight_layout(); plt.show()"""),
    md("""**Conclusión:** ninguna feature extra supera al base **con el VAE** — lo
*perjudican* (diluyen el `recon_prob`). Ayudan **modestamente al IForest** (tolera
features extra) pero IForest+ERA5 (0.542) sigue debajo del VAE base (0.592).

**Lección de método (un artefacto cazado):** un primer resultado ERA5 dio 0.642 —
era un **artefacto**: `frost_days`=0 en el norte (nunca hiela) → varianza-cero por
depto → la normalización tiraba el 34% de los departamentos → test set sesgado y más
fácil. Se corrigió con fallback a *std global* en `_normalize_per_depto`. Con test
sets emparejados, el efecto real es el de la tabla. **Siempre validar que los test
sets coincidan antes de comparar.**"""),
])

# ===========================================================================
# 05 — Modernos + techo estructural
# ===========================================================================
build("05_modernos_y_techo.ipynb", "5 · Métodos modernos (deepod) y el techo estructural", [
    md("""### 5.1 — Comparación con AD tabular profundo moderno
Benchmark contra el estado del arte vía la librería `deepod`: DeepSVDD (one-class),
ICL (contrastive), NeuTraL/GOAD (transformation-based)."""),
    code(SETUP),
    code("explib.show(explib.tbl_modern())"),
    md("""**Ninguno se acerca al VAE.** El mejor (DeepSVDD, 0.43) queda lejos del VAE
(0.592) e incluso debajo del IForest (0.51). El argumento del informe queda blindado:
probamos contra lo último y nuestro enfoque es mejor para *estos* datos.

### 5.2 — El techo estructural (por qué ningún modelo pasa de ~0.6)
Tres análisis independientes (en `analyze_errors.py`, `analyze_labels.py`,
`analyze_datamap.py`) muestran que el límite **no es del modelo**:
- **Cross-modelo:** ~68% de las anomalías las fallan TODOS los modelos, y esas tienen
  **clima normal** (sin firma de sequía). Si fuera la arquitectura, modelos distintos
  fallarían en muestras distintas; fallan en las mismas.
- **Auditoría de etiqueta:** excluir las anomalías sospechosas de ruido **no** mejora
  la PR-AUC → el techo no es etiqueta-basura, son anomalías genuinas.
- **Data map (Dataset Cartography):** las muestras "siempre mal reconstruidas" no
  coinciden con las anomalías de rinde → la dificultad es rareza climática, no rinde.

**Conclusión:** ~70% de las anomalías de rinde tienen causas **no climáticas**
(plaga, granizo, manejo, mercado) **invisibles a cualquier feature climática/satelital
disponible**. Es una propiedad del problema (features de clima ↔ etiqueta de rinde),
no una falla del modelo. Eso es un resultado fuerte y honesto para el informe."""),
    md("### Mapa depto×campaña de errores por consenso (si `analyze_errors.py` ya generó el PNG)"),
    code("""import os
from IPython.display import Image, display
p = os.path.join(explib.ROOT, "analysis", "consensus_heatmap_soja.png")
display(Image(p)) if os.path.exists(p) else print("Corré: python analyze_errors.py --cultivo soja")"""),
    md("""## Comparación final de TODOS los modelos clave (con desvío estándar)
El cierre: PR-AUC media±std de los modelos principales, soja y maíz. Permite ver de
un vistazo **cuál es mejor y con cuánta confianza**."""),
    code("explib.show(explib.tbl_leaderboard())"),
    code("explib.plot_leaderboard_compare(); plt.show()"),
    md("""**El veredicto:** `vae_seedens_v1` (seed-ensemble del VAE `recon_prob`) es el
mejor en ambos cultivos, con el desvío más chico. Le sigue su variante deep. El
IForest es un baseline fuerte. Ni features satelitales, ni AE/DAE, ni métodos
modernos lo superaron — y el techo de ~0.6 es estructural (anomalías de rinde sin
firma climática), no del modelo."""),
])

print("\\nTodos los notebooks generados en experiments/")
