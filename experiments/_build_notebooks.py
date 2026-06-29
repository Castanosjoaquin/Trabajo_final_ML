"""Genera los notebooks de experiments/ (vía nbformat) — narrativa cronológica.
    python experiments/_build_notebooks.py
Los notebooks leen de runs/ guardados (no re-entrenan). Después se ejecutan con
nbconvert para embeber las salidas."""
import os
import glob
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


# Limpiar notebooks viejos (estructura anterior)
for old in glob.glob(os.path.join(HERE, "*.ipynb")):
    os.remove(old)


# ===========================================================================
# 00 — Intro, problema, baseline (SIN spoiler del final)
# ===========================================================================
build("00_intro_problema_y_baseline.ipynb", "0 · El problema, los datos y el baseline", [
    md("""## Detección de anomalías en rendimientos agrícolas (soja / maíz, Argentina)

Queremos detectar **campañas con rinde anómalo** a nivel departamento. Es un problema
**no supervisado**: entrenamos solo con campañas *normales* y puntuamos cada muestra
por cuánto se desvía.

**La etiqueta proxy** (solo para evaluar — el modelo nunca la ve):
`z_rinde = (rinde − media_móvil_dep) / desvío_móvil_dep`, y `anomalía = z_rinde < −1.5`.
Es decir, una caída de rinde grande respecto de la propia historia del departamento.

**Las features** son climáticas: promedios mensuales (Sep–Mar) de ~9 variables
(precipitación, temperaturas, radiación, humedad, viento, ONI) = **54 features**."""),
    code(SETUP),
    md("""## Cómo evaluamos
- **Métrica principal: PR-AUC** (libre de umbral). También miramos ROC-AUC y Recall@10%.
- **F1 es engañoso acá**: el umbral se calibra en validación (pocas anomalías) y en test
  la tasa base salta a ~30% (mega-sequía 2022/23) → casi todos los modelos dan F1 ≈ tasa
  base. Por eso priorizamos métricas libres de umbral.
- **Multi-seed**: los modelos profundos tienen alta varianza, así que reportamos
  **media ± desvío** sobre varias semillas."""),
    md("## El baseline a superar: Isolation Forest\nUn ensamble no neuronal, estable. Es el listón de toda la historia."),
    code("""explib.show(explib.run_metrics("iforest_v2_mf03_n200", "soja", split="test"))"""),
    code("""explib.plot_all_metrics("iforest_v2_mf03_n200", "soja"); plt.show()"""),
    md("""## El recorrido de estos notebooks
Vamos a ir contando los intentos en orden:

1. **Modelos de reconstrucción** (NB 1): autoencoders — AE → DAE → híbrido AE+IForest →
   VAE. Buscando un buen *modelo base*.
2. **Ensembles** (NB 2): cómo atacamos la varianza hasta el modelo final (+ la idea de
   *Dataset Cartography*).
3. **Limpieza de datos y features nuevas** (NB 3): el punto de quiebre + NDVI / suelo /
   heladas.
4. **Métodos modernos y el techo** (NB 4): benchmark contra SOTA + por qué hay un límite.

Cada notebook compara **contra este baseline**. Empecemos."""),
])

# ===========================================================================
# 01 — Modelos de reconstrucción (el recorrido)
# ===========================================================================
build("01_modelos_de_reconstruccion.ipynb", "1 · Modelos de reconstrucción: AE → DAE → híbrido → VAE", [
    md("""Arrancamos con **autoencoders**: se entrenan a reconstruir las campañas normales,
y lo que reconstruyen mal = anómalo. Vamos a recorrer los intentos en orden, siempre
midiendo contra el baseline IForest.

> *Nota:* esta fue la **fase de exploración**, sobre el panel original (antes de la
> limpieza de datos del NB 3). Los números absolutos suben después de limpiar; acá lo que
> importa es la **comparación relativa entre enfoques**."""),
    code(SETUP),
    md("""## 1.1 — Autoencoder (AE): búsqueda de hiperparámetros
Entrenamos **24 variantes** de AE (arquitectura, latente, dropout, schedules…).
Los 5 mejores:"""),
    code("""explib.show(explib.top_runs("ae", "soja", n=5, era="dirty"))"""),
    md("""El rango de PR-AUC entre las 24 variantes fue de solo **~0.034**: tunear
hiperparámetros casi no movió la aguja → el cuello de botella no era ése. Curva de loss
y métricas del mejor AE:"""),
    code("""best_ae = explib.top_runs("ae","soja",1,era="dirty")["name"].iloc[0]
explib.plot_loss(best_ae, "soja"); plt.show()"""),
    code("""explib.plot_all_metrics(best_ae, "soja"); plt.show()"""),
    md("""## 1.2 — Denoising AE (DAE)
El DAE se entrena reconstruyendo *limpio* desde un input *corrupto* → más robusto al
ruido. Ayuda un poco, pero sigue en el rango del AE."""),
    code("""explib.show(explib.compare_table([
    ("dae_v1_base","DAE base"), ("dae_v2_sweep_best","DAE sweep")], "soja"))"""),
    md("""## 1.3 — Híbrido: AE-latente + Isolation Forest
Idea de la literatura: correr IForest sobre el **espacio latente** del AE (no sobre las
features originales). Acá **falló** — el bottleneck del AE destruye la estructura que el
IForest necesita:"""),
    code("""explib.show(explib.compare_table([("ae_iforest_v1","híbrido lat8"),
                                  ("ae_iforest_v2_latent16","híbrido lat16")], "soja"))"""),
    md("""## 1.4 — VAE: el salto del *score probabilístico*
El VAE no usa el MSE plano, sino `recon_prob` (An & Cho 2015): pondera el error de cada
feature **por la varianza que el decoder le asigna**. Ese cambio de *score* — no de
arquitectura — es lo que mueve la aguja."""),
    code("""explib.plot_loss("vae_v4_reconprob_lat16", "soja", ); plt.show()"""),
    md("""### Comparación de toda la fase de reconstrucción (test, media±std)
*(El DAE se exploró sobre el panel limpio (NB 3), en el mismo rango ~0.38; acá comparamos
los modelos de la misma era —panel original— para que sea justo.)*"""),
    code("""explib.show(explib.compare_table([
    ("iforest_v1_base","IForest (baseline)"),
    (best_ae,"AE (mejor)"),
    ("ae_iforest_v2_latent16","Híbrido AE+IForest"),
    ("vae_v4_reconprob_lat16","VAE recon_prob"),
], "soja", era="dirty"))"""),
    code("""explib.plot_compare([
    ("iforest_v1_base","IForest (baseline)"),
    (best_ae,"AE (mejor)"),
    ("ae_iforest_v2_latent16","Híbrido AE+IForest"),
    ("vae_v4_reconprob_lat16","VAE recon_prob"),
], "soja", era="dirty", baseline=0.389, title="Fase reconstrucción — PR-AUC (soja)"); plt.show()"""),
    md("""**Conclusión de la fase:** el **VAE con `recon_prob`** es el mejor modelo de
reconstrucción y el único que le pelea al baseline. Pero tiene un problema: **alta
varianza** (±0.03) — una sola semilla no es confiable. Eso lo atacamos en el NB 2."""),
])

# ===========================================================================
# 02 — Ensembles y modelo final
# ===========================================================================
build("02_ensembles_y_modelo_final.ipynb", "2 · Ensembles: de la varianza al modelo final", [
    md("""El VAE `recon_prob` era el mejor, pero **cada semilla converge distinto** → alta
varianza. La solución: **ensemble de semillas** (deep ensemble)."""),
    code(SETUP),
    md("""## 2.1 — Seed-ensemble: promediar varias semillas
Entrenamos N VAEs con semillas distintas y **promediamos sus scores normalizados**
(z-score per-miembro). El `EnsembleDetector` es agnóstico al modelo. Progresión:"""),
    code("""explib.show(explib.compare_table([
    ("vae_v4_reconprob_lat16","VAE single"),
    ("vae_seedens_v1","seed-ensemble ×10 (lat16)"),
    ("vae_seedens_deep","seed-ensemble ×10 (deep)"),
    ("vae_iforest_ens_v1","hetero VAE+IForest"),
], "soja", era="clean"))"""),
    code("""explib.plot_compare([
    ("vae_v4_reconprob_lat16","VAE single"),
    ("vae_seedens_v1","seed-ens ×10"),
    ("vae_seedens_deep","seed-ens deep"),
    ("vae_iforest_ens_v1","hetero VAE+IForest"),
], "soja", era="clean", metric="pr_auc", title="Ensembles — PR-AUC (soja)"); plt.show()"""),
    md("""El **desvío se desploma** (±0.035 del single → ±0.010 del ensemble) y la media
*sube*: el promedio cancela los errores idiosincráticos de cada semilla. El
hetero-ensemble VAE+IForest queda mixto (el IForest, más débil, arrastra)."""),
    code("""explib.plot_pr_curves([("vae_v4_reconprob_lat16","VAE single"),
                       ("vae_seedens_v1","seed-ensemble ×10")], "soja"); plt.show()"""),
    md("""## 2.2 — La idea de *Dataset Cartography* (la del profesor)
La filosofía "lo que el modelo predice **siempre** mal es anomalía" la implementamos como
herramienta de **diagnóstico** (no como el scorer final):

- **Data map** (Swayamdipta et al. 2020): registramos el error de reconstrucción de cada
  muestra de train **en cada época**. Con eso medimos *confidence* (error medio) y
  *variability* (desvío entre épocas). Las muestras de error alto y baja variabilidad =
  "el modelo nunca las aprende".
- **Desacuerdo del ensemble** (`score_std`): el desvío entre los miembros del ensemble por
  muestra = incertidumbre. Alto = muestra ambigua/borderline.

> **Importante:** el modelo final NO usa cartography para puntuar — es el promedio simple
> del seed-ensemble. La cartography fue una herramienta para *auditar* (ver NB 4): mostró
> que las muestras "siempre mal reconstruidas" son rareza climática, no anomalías de
> rinde. Lo dejamos documentado porque responde a la pregunta del profesor."""),
    code("""import os
from IPython.display import Image, display
p = os.path.join(explib.ROOT, "analysis", "datamap_soja.png")
display(Image(p)) if os.path.exists(p) else print("(generar con: python analyze_datamap.py --cultivo soja)")"""),
    md("## 2.3 — El modelo final: `vae_seedens_v1`\nTodas sus métricas (test, con ±std entre semillas):"),
    code("""explib.show(explib.run_metrics("vae_seedens_v1", "soja"))"""),
    code("""explib.plot_all_metrics("vae_seedens_v1", "soja"); plt.show()"""),
    md("**Este es el modelo final del Componente A.** En los próximos notebooks intentamos superarlo (datos nuevos, métodos modernos) — sin éxito."),
])

# ===========================================================================
# 03 — Limpieza de datos + features nuevas
# ===========================================================================
build("03_limpieza_de_datos_y_features.ipynb", "3 · Limpieza de datos y features nuevas", [
    md("""## 3.0 — El punto de quiebre: LIMPIEZA DE DATOS
A esta altura descubrimos que el panel tenía **25% de filas duplicadas** (explosión
cartesiana de lat/lon por un join que ignoró la provincia) y la etiqueta `z_rinde`
corrompida por mezclar departamentos homónimos de distintas provincias.

Corregirlo (dedup + clave provincia, en `data.py`) **subió a TODOS los modelos +0.09 a
+0.14 PR-AUC** — la mayor mejora del proyecto, más que cualquier cambio de modelo."""),
    code(SETUP),
    md("La progresión completa, con las dos eras (exploración → limpieza → final):"),
    code("explib.show(explib.tbl_journey())"),
    md("""## 3.1 — Features nuevas: ¿podemos romper el techo?
El ~70% de anomalías que fallan todos los modelos tiene **clima normal** (ver NB 4).
Probamos features que capturen lo que el clima mensual no ve, extraídas por
departamento×campaña con Google Earth Engine:
- **NDVI-AVHRR** (1981+): vigor vegetal.
- **ERA5-Land**: humedad de suelo (estado inicial / "inercia") + días de helada.
- **Agronómicas**: ventana crítica + balance hídrico (de columnas existentes).

Efecto sobre el VAE (SOTA) y el IForest, **con test sets emparejados**:"""),
    code("explib.show(explib.tbl_features())"),
    code("""import numpy as np
t = explib.tbl_features(); x = np.arange(len(t)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, t["VAE_seedens"], w, label="VAE seed-ens", color="#4C72B0", alpha=0.85)
ax.bar(x + w/2, t["IForest"], w, label="IForest", color="#DD8452", alpha=0.85)
ax.axhline(0.592, ls="--", color="gray", lw=1, label="VAE base (0.592)")
ax.set_xticks(x); ax.set_xticklabels(t["features"], rotation=18, ha="right", fontsize=8)
ax.set_ylabel("PR-AUC (soja, test)"); ax.legend(fontsize=8)
ax.set_title("Features nuevas: ayudan al IForest, perjudican al VAE"); plt.tight_layout(); plt.show()"""),
    md("""**Conclusión:** ninguna feature extra supera al base **con el VAE** — lo
*perjudican* (diluyen el `recon_prob`). Ayudan modestamente al IForest, pero IForest+ERA5
(0.542) sigue debajo del VAE base (0.592).

**Un artefacto cazado (lección de método):** un primer resultado ERA5 dio 0.642 — era un
**artefacto**: `frost_days`=0 en el norte (nunca hiela) → varianza-cero por depto → la
normalización tiraba el 34% de los departamentos → test set sesgado y más fácil. Se
corrigió (fallback a *std global*). **Siempre validar que los test sets coincidan antes de
comparar.**"""),
])

# ===========================================================================
# 04 — Modernos + techo + comparación final
# ===========================================================================
build("04_modernos_y_techo_estructural.ipynb", "4 · Métodos modernos y el techo estructural", [
    md("""## 4.1 — Benchmark contra AD tabular profundo moderno
Comparamos contra el estado del arte vía la librería `deepod`: DeepSVDD (one-class), ICL
(contrastive), NeuTraL/GOAD (transformation-based)."""),
    code(SETUP),
    code("explib.show(explib.tbl_modern())"),
    md("""**Ninguno se acerca al VAE.** El mejor (DeepSVDD, 0.43) queda lejos del VAE (0.592)
e incluso debajo del IForest (0.51). Probamos contra lo último y nuestro enfoque es mejor
para *estos* datos."""),
    md("""## 4.2 — ¿Por qué ningún modelo pasa de ~0.6? El techo estructural
Tres análisis independientes muestran que el límite **no es del modelo**:
- **Cross-modelo** (`analyze_errors.py`): ~68% de las anomalías las fallan TODOS los
  modelos, y esas tienen **clima normal**. Si fuera la arquitectura, modelos distintos
  fallarían en muestras distintas; fallan en las mismas.
- **Auditoría de etiqueta** (`analyze_labels.py`): excluir las anomalías sospechosas de
  ruido **no** mejora la PR-AUC → el techo no es etiqueta-basura, son anomalías genuinas.
- **Data map** (`analyze_datamap.py`): las muestras "siempre mal reconstruidas" no
  coinciden con las anomalías de rinde → la dificultad es rareza climática, no rinde.

**Conclusión:** ~70% de las anomalías de rinde tienen causas **no climáticas** (plaga,
granizo, manejo, mercado) invisibles a cualquier feature climática/satelital disponible.
Es una propiedad del problema, no una falla del modelo."""),
    code("""import os
from IPython.display import Image, display
p = os.path.join(explib.ROOT, "analysis", "consensus_heatmap_soja.png")
display(Image(p)) if os.path.exists(p) else print("(generar con: python analyze_errors.py --cultivo soja)")"""),
    md("""## 4.3 — Comparación final de TODOS los modelos (con desvío estándar)
El cierre: PR-AUC media±std de los modelos clave, soja y maíz. **El veredicto.**"""),
    code("explib.show(explib.tbl_leaderboard())"),
    code("explib.plot_leaderboard_compare(); plt.show()"),
    md("""**Modelo final: `vae_seedens_v1`** (seed-ensemble del VAE `recon_prob`) — mejor en
ambos cultivos, con el desvío más chico. Lo lograron dos palancas: la **limpieza de datos**
y el **score probabilístico + ensemble de semillas**. Ni features satelitales, ni AE/DAE,
ni métodos modernos lo superaron — y el techo de ~0.6 es estructural."""),
])

print("\\nTodos los notebooks generados.")
