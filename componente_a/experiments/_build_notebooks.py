"""Genera los notebooks de experiments/ — versión TRANSPARENTE: cada notebook
entrena los modelos de verdad (inline, con el loop de semillas a la vista) en vez
de leer resultados guardados. El profesor puede abrir cualquiera, hacer "Run all"
y reproducir todo desde cero.

    python experiments/_build_notebooks.py

Después se ejecutan con nbconvert para embeber salidas (tarda: entrena en serio).
"""
import os
import sys
import glob
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

# Si se pasan nombres de notebook por CLI, solo se (re)generan esos (y solo esos
# se borran) — así se puede regenerar UN notebook sin perder las salidas del resto.
#   python experiments/_build_notebooks.py 06_features_nuevas.ipynb
ONLY = set(sys.argv[1:])


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


# Preámbulo común: importa lab (que agrega src al path) + el pipeline de datos.
SETUP = """import lab                      # utilidades: métricas y gráficos (experiments/lab.py)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src import config, data
from src.models import (IsolationForestDetector, OneClassSVMDetector,
                        ZScoreDetector, MahalanobisDetector,
                        AEDetector, DenoisingAEDetector, VAEDetector,
                        EnsembleDetector, DeepODDetector)
plt.rcParams["figure.dpi"] = 110
SEEDS = lab.SEEDS            # 10 semillas del estudio. Bajalas (p. ej. [42,43,44])
print("semillas:", SEEDS)   # para un Run all más rápido; los números se mueven ±std"""

DATOS = """panel_z = data.prepare()                         # panel + etiqueta z_rinde
ds   = data.build_crop_dataset(panel_z, "soja")  # split + normalización (soja)
dsm  = data.build_crop_dataset(panel_z, "maiz")  # idem maíz
print("soja  train:", ds.X_train.shape, "| test anómalas:", int(ds.y_test.sum()))
print("maíz  train:", dsm.X_train.shape, "| test anómalas:", int(dsm.y_test.sum()))"""


def build(fname, title, cells):
    if ONLY and fname not in ONLY:
        return
    nb = nbf.v4.new_notebook()
    nb.cells = [md(f"# {title}")] + cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    with open(os.path.join(HERE, fname), "w") as f:
        nbf.write(nb, f)
    print("escrito:", fname)


for old in glob.glob(os.path.join(HERE, "*.ipynb")):
    if not ONLY or os.path.basename(old) in ONLY:
        os.remove(old)


# ===========================================================================
# 00 — Datos y pipeline
# ===========================================================================
build("00_datos_y_pipeline.ipynb", "0 · Los datos y el pipeline", [
    md("""**El problema:** detectar **campañas con rinde anómalo** por departamento (soja/maíz,
Argentina, 1981–2024) usando solo variables climáticas. Es **no supervisado**: el modelo
entrena solo con campañas normales y nunca ve la etiqueta.

Esta serie de notebooks es reproducible de punta a punta: cada uno **entrena los modelos
de verdad** (no lee resultados guardados). Abrí cualquiera, "Run all", y sale todo.

| nb | contenido |
|---|---|
| **0** | los datos, la etiqueta `z_rinde`, splits y normalización |
| **1** | cómo se evalúa + baselines (Isolation Forest, One-Class SVM) |
| **2** | autoencoders: **AE y DAE** (+ IForest/OCSVM sobre su latente) |
| **3** | **VAE** y el score `recon_prob` (+ latente + t-SNE) |
| **4** | **seed-ensemble** = el modelo final (+ latente + t-SNE) |
| **5** | el **techo estructural** (por qué nadie pasa de ~0.6) |
| **6** | features nuevas (agro / NDVI / ERA5) |
| **7** | métodos modernos, leaderboard y conclusiones |"""),
    code(SETUP),
    md("""## 1 · El panel
Una fila = un (departamento, campaña, cultivo). Columnas: identificadores, `rinde_kgha`
(el objetivo) y 54 features climáticas (7 variables NASA POWER × 7 meses Sep–Mar + ONI ×
5 meses)."""),
    code("""panel = data.load_panel()
print("shape:", panel.shape)
panel[["provincia","departamento","campania","cultivo","rinde_kgha"]].head()"""),
    md("""## 2 · La etiqueta proxy `z_rinde`
No hay etiquetas de "anomalía"; las definimos: cuánto cae el rinde respecto de la **media
móvil de 5 años del propio departamento**.

`z_rinde = (rinde − media_móvil) / desvío_móvil`  → `anomalía = z_rinde < −1.5`

Se calcula con `shift(1)` (solo pasado, sin leakage), agrupando por
`[provincia, departamento, cultivo]`. **El modelo nunca la ve — es solo para evaluar.**"""),
    code("""panel_z = data.compute_z_rinde(panel)
print("tasa de anomalías por cultivo:")
print((panel_z.groupby("cultivo")["anomalia"].mean()*100).round(1).astype(str) + " %")"""),
    md("### La serie de rinde de un departamento, con sus anomalías"),
    code("""d = panel_z[(panel_z.cultivo=="soja") & (panel_z.provincia=="CORDOBA")]
d = d[d.departamento==d.departamento.value_counts().idxmax()].sort_values("campania_inicio")
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(d.campania_inicio, d.rinde_kgha, "-o", ms=3, color=lab.C_NORMAL, label="rinde")
an = d[d.anomalia==1]
ax.scatter(an.campania_inicio, an.rinde_kgha, color=lab.C_ANOM, zorder=5, s=40, label="anomalía")
ax.set_xlabel("campaña"); ax.set_ylabel("rinde (kg/ha)")
ax.set_title(f"Rinde de soja — {d.departamento.iloc[0]}, Córdoba"); ax.legend(); plt.show()"""),
    md("""## 3 · Features, splits y normalización
- **Features (X):** 54 columnas climáticas. NO incluyen el rinde.
- **Split temporal** (no aleatorio): **solo train (≤2020) y test (≥2021)**. No hay
  validación — el bloque 2018–2020 se pliega al train (ver recuadro abajo).
- **Train normal:** del train se excluyen las anomalías y los años de sequía
  generalizada (`config.EXCLUDED_TRAIN_YEARS = [1988,1996,2008,2017]`) → el modelo
  aprende solo "lo normal".
- **Normalización** z-score por departamento, con stats calculadas SOLO en train normal
  (sin leakage).

Todo eso lo arma `build_crop_dataset`. Es la única "caja" del pipeline; el resto de los
notebooks entrenan a la vista sobre su salida."""),
    code(DATOS),
    md("""`ds.X_train` (campañas normales) va al `fit()` de cualquier detector; `ds.X_test`
se puntúa con `score_samples()` (mayor = más anómalo) y se compara contra `ds.y_test`.

> **¿Por qué no hay validación?** Probamos con un val 2018–2020 y resultó **ciego**: sus
> anomalías son de años sin sequía grande → sin firma climática → su PR-AUC daba ≈ azar
> para *todos* los modelos, así que no servía para seleccionar. Plegarlo al train aporta 3
> años más de campañas normales (y acerca el train al clima de test). El *early stopping*
> de los autoencoders usa un 15% interno del train, no este split. La selección de HP
> (nb 3) se ilustra en **test**, con el *data snooping* declarado."""),
])

# ===========================================================================
# 01 — Evaluación y baselines
# ===========================================================================
build("01_evaluacion_y_baselines.ipynb", "1 · Cómo se evalúa + baselines (estadísticos y de modelos)", [
    md("""Antes de los modelos profundos, fijamos **cómo se mide** y los baselines. Van en
orden de complejidad creciente: primero los **estadísticos** (z-score, Mahalanobis —
sin aprendizaje), después los **de modelos** (Isolation Forest, One-Class SVM).

**Métrica principal: PR-AUC** (área Precision-Recall, libre de umbral). Razones:
- El **F1 con umbral fijo engaña**: el umbral se calibra en val (tasa ~6%) pero en test
  la tasa salta a ~27% (sequía 2022/23) → un corte absoluto marca casi todo → F1 ≈ tasa
  base para todos.
- Todo se reporta **multi-seed (media ± std)** porque los modelos profundos varían.

**Política de semillas (para que las comparaciones sean justas):**
- Evaluar **cualquier modelo** = **10 semillas** (`SEEDS`), reportando media ± desvío.
- Los modelos **deterministas** (One-Class SVM) no tienen semilla → std = 0; los
  estocásticos (VAE, AE, DAE, **Isolation Forest**) sí varían y muestran su ±std.
- Los **barridos** (búsqueda de HP en nb 3, ablación de features en nb 6) usan **5
  semillas** — es un screening de muchas configuraciones, se declara donde se hace.

`lab.metrics(scores, y)` devuelve PR-AUC, ROC-AUC, precisión@k y recall al top-10%;
`lab.tabla(resultados)` es la **presentación estándar** del proyecto (fila = modelo,
métricas `media±std`)."""),
    code(SETUP),
    code(DATOS),
    md("""## 1.1 · Baselines estadísticos: z-score y Mahalanobis
Puramente estadísticos, sin aprendizaje ni redes; solo necesitan media (y, en Mahalanobis,
covarianza) del train normal. Ambos son **deterministas** (sin semilla → ±0.000).

- **Z-score multivariado** (`ZScoreDetector`): `z=(x−media)/desvío` por feature; el score es
  el promedio (`mean`) o el máximo (`max`) de `|z|` entre las 54 features. No ve correlaciones.
- **Mahalanobis** (`MahalanobisDetector`): distancia con la **covarianza completa**
  (encogimiento Ledoit-Wolf para invertirla estable con 54 features) — "z-score con
  correlaciones".

Los resultados se muestran con `lab.tabla`, la **presentación estándar del proyecto** (una
fila por modelo, métricas `media±std`)."""),
    code("""estadisticos = {
    "ZScore (mean)": [lab.metrics(ZScoreDetector(agg="mean").fit(ds.X_train)
                                  .score_samples(ds.X_test), ds.y_test)],
    "ZScore (max)":  [lab.metrics(ZScoreDetector(agg="max").fit(ds.X_train)
                                  .score_samples(ds.X_test), ds.y_test)],
    "Mahalanobis":   [lab.metrics(MahalanobisDetector(shrinkage=None).fit(ds.X_train)
                                  .score_samples(ds.X_test), ds.y_test)],
}
lab.tabla(estadisticos)"""),
    md("""## 1.2 · Baselines de modelos: Isolation Forest y One-Class SVM
El siguiente escalón: modelos que *aprenden* la región normal.

- **Isolation Forest** — ensamble de árboles que aísla puntos raros; estocástico, así que va
  **multi-seed** (3 semillas acá para el "Run all"; el estudio usa 10). El único HP con
  efecto real fue `max_features=0.3`.
- **One-Class SVM** — frontera que envuelve a los normales; con kernel **RBF** es no lineal.
  Comparamos RBF vs lineal para ver que la no-linealidad importa (determinista)."""),
    code("""modelos = {
    "IForest": [lab.metrics(IsolationForestDetector(n_estimators=200, max_features=0.3,
                random_state=s).fit(ds.X_train).score_samples(ds.X_test), ds.y_test)
                for s in SEEDS[:3]],
    "OCSVM (rbf)":    [lab.metrics(OneClassSVMDetector(kernel="rbf", nu=0.1)
                                   .fit(ds.X_train).score_samples(ds.X_test), ds.y_test)],
    "OCSVM (linear)": [lab.metrics(OneClassSVMDetector(kernel="linear", nu=0.1)
                                   .fit(ds.X_train).score_samples(ds.X_test), ds.y_test)],
}
lab.tabla(modelos)"""),
    md("""**El kernel lineal fracasa** y el **RBF es competitivo con el IForest**: la frontera no
lineal es imprescindible. Y el **z-score (mean)** de 1.1, sin aprender nada, ya juega en la
misma liga que el IForest (lo interpretamos en 1.5)."""),
    md("""## 1.3 · Comparación de todos los baselines (soja y maíz)
Todos los baselines juntos, en la tabla estándar. Acá **solo comparamos baselines entre sí**
(el listón a superar); los modelos de reconstrucción del nb 2 en adelante se miden contra
este piso en sus propios notebooks."""),
    code("""def baselines_dict(dset):
    return {
        "ZScore (mean)": [lab.metrics(ZScoreDetector(agg="mean").fit(dset.X_train)
                                      .score_samples(dset.X_test), dset.y_test)],
        "ZScore (max)":  [lab.metrics(ZScoreDetector(agg="max").fit(dset.X_train)
                                      .score_samples(dset.X_test), dset.y_test)],
        "Mahalanobis":   [lab.metrics(MahalanobisDetector(shrinkage=None).fit(dset.X_train)
                                      .score_samples(dset.X_test), dset.y_test)],
        "IForest": [lab.metrics(IsolationForestDetector(n_estimators=200, max_features=0.3,
                    random_state=s).fit(dset.X_train).score_samples(dset.X_test), dset.y_test)
                    for s in SEEDS[:3]],
        "OCSVM (rbf)": [lab.metrics(OneClassSVMDetector(kernel="rbf", nu=0.1)
                                    .fit(dset.X_train).score_samples(dset.X_test), dset.y_test)],
    }

res = {"soja": baselines_dict(ds), "maíz": baselines_dict(dsm)}
lab.tabla(res["soja"])"""),
    md("Y en maíz:"),
    code("""lab.tabla(res["maíz"])"""),
    code("""fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, (nombre, r) in zip(axs, res.items()):
    lab.plot_leaderboard(lab.leaderboard(r), ax=ax, titulo=f"PR-AUC (test, {nombre})")
plt.tight_layout(); plt.show()"""),
    md("""## 1.4 · Matrices de confusión de los baselines
El PR-AUC mide el *ranking* (libre de umbral). Para una matriz de confusión hace falta un
**corte**, y un umbral absoluto no transfiere entre períodos (los scores se corren hacia
arriba en 2021–24 → un corte fijo marca ~70% del test). Por eso re-umbralizamos sobre el
**propio test** al top-tasa real (~27%, `q="base"`), para **todos los baselines** (soja):"""),
    code("""scores_conf = {
    "ZScore (mean)": ZScoreDetector(agg="mean").fit(ds.X_train).score_samples(ds.X_test),
    "Mahalanobis":   MahalanobisDetector(shrinkage=None).fit(ds.X_train).score_samples(ds.X_test),
    "IForest":       IsolationForestDetector(n_estimators=200, max_features=0.3,
                                             random_state=42).fit(ds.X_train).score_samples(ds.X_test),
    "OCSVM (rbf)":   OneClassSVMDetector(kernel="rbf", nu=0.1).fit(ds.X_train).score_samples(ds.X_test),
}
fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
for ax, (nombre, sc) in zip(axs, scores_conf.items()):
    lab.confusion_top(sc, ds.y_test, "base", ax=ax, titulo=nombre)
plt.tight_layout(); plt.show()"""),
    md("""## 1.5 · Interpretación + por qué no hay validación

**El z-score (mean) queda a la par del Isolation Forest** (en soja incluso lo supera
levemente) pese a no aprender nada: la señal detectable es la **sequía**, que corre muchas
features climáticas a la vez → promediar `|z|` es casi un "índice de sequía" a mano. El `max`
se hunde (una sola feature rara ≠ anomalía de rinde), y **Mahalanobis queda por debajo**:
des-pondera justo las direcciones de mayor varianza (donde vive la firma de sequía). Lección
de siempre: *rareza climática ≠ anomalía de rinde*.

**Por qué no hay validación:** probamos un val 2018–2020 y resultó **ciego** — sus anomalías
son de años sin sequía grande (sin firma climática) → PR-AUC ≈ tasa base para todos, no
discriminaba nada. Por eso (1) el val se pliega al train y (2) las comparaciones y la
selección de HP (nb 3) se ilustran en test, con el *data snooping* declarado. La confianza en
el modelo final viene de la **consistencia entre cultivos** y la **varianza mínima entre
semillas**, no de un número puntual.

**Checklist de baselines cerrado** (IForest, OCSVM, z-score, Mahalanobis). **Baseline a
superar: el z-score (mean) / IForest**; los próximos notebooks lo intentan con modelos de
reconstrucción."""),
])

# ===========================================================================
# 02 — AE y DAE
# ===========================================================================
build("02_ae_y_dae.ipynb", "2 · Autoencoders: AE y DAE", [
    md("""**Hipótesis:** un autoencoder entrenado solo con campañas normales reconstruye mal
las anómalas; el **error de reconstrucción** es el score.

Acá entrenamos **AE** (autoencoder) y **DAE** (denoising, reconstruye limpio desde input
corrupto), multi-seed y a la vista. Al final probamos correr **IForest y One-Class SVM
sobre el espacio latente** del AE (la receta "deep representation + shallow detector")."""),
    code(SETUP),
    code(DATOS),
    md("""## 2.1 · Autoencoder (AE)
Encoder comprime las 54 features a un latente de 16, decoder reconstruye, score = MSE.
Entrenamos **una semilla por vuelta** y guardamos las métricas — el loop está a la vista:"""),
    code("""filas_ae = []
for seed in SEEDS:
    ae = AEDetector(hidden_dims=(64,32), latent_dim=16, max_epochs=300, patience=30,
                    random_state=seed).fit(ds.X_train)          # entrena de verdad
    scores = ae.score_samples(ds.X_test)                        # error de reconstrucción
    filas_ae.append(lab.metrics(scores, ds.y_test))
print("AE (soja, media±std entre semillas):")
lab.mean_std(filas_ae)"""),
    md("La curva de loss del último AE (entrena bien; el problema no es el fit):"),
    code("""h = pd.DataFrame(ae.history_)
fig, ax = plt.subplots(figsize=(6,3.5))
ax.plot(h["epoch"], h["train_loss"], label="train"); ax.plot(h["epoch"], h["val_loss"], label="val")
ax.set_xlabel("época"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("AE — curva de loss"); plt.show()"""),
    md("""## 2.2 · Denoising AE (DAE)
Se corrompe el input y se entrena a reconstruir el limpio → representación más robusta al
ruido:"""),
    code("""filas_dae = []
for seed in SEEDS:
    dae = DenoisingAEDetector(hidden_dims=(64,32), latent_dim=16, corruption=0.1,
                              noise_type="salt_pepper", max_epochs=300, patience=30,
                              random_state=seed).fit(ds.X_train)
    filas_dae.append(lab.metrics(dae.score_samples(ds.X_test), ds.y_test))
print("DAE (soja, media±std):")
lab.mean_std(filas_dae)"""),
    md("### AE vs DAE vs TODOS los baselines\nContra el conjunto completo de baselines del nb 1 (estadísticos y de modelos), no solo el IForest:"),
    code("""resumen = {
    "ZScore (mean)": [lab.metrics(ZScoreDetector(agg="mean").fit(ds.X_train)
                                  .score_samples(ds.X_test), ds.y_test)],
    "Mahalanobis":   [lab.metrics(MahalanobisDetector(shrinkage=None).fit(ds.X_train)
                                  .score_samples(ds.X_test), ds.y_test)],
    "IForest": [lab.metrics(IsolationForestDetector(n_estimators=200, max_features=0.3,
                random_state=s).fit(ds.X_train).score_samples(ds.X_test), ds.y_test) for s in SEEDS[:3]],
    "OCSVM (rbf)": [lab.metrics(OneClassSVMDetector(kernel="rbf", nu=0.1)
                                .fit(ds.X_train).score_samples(ds.X_test), ds.y_test)],
    "AE": filas_ae, "DAE": filas_dae,
}
lab.plot_leaderboard(lab.leaderboard(resumen)); plt.show()
lab.tabla(resumen)"""),
    md("""Ambos quedan por debajo del baseline: el **error de reconstrucción con MSE no alcanza**.
El problema no es la arquitectura (se probaron 24 variantes en la exploración, todas
en un rango estrecho) sino el *score* — lo resolvemos con el VAE en el nb 3.

## 2.3 · IForest y OCSVM sobre el latente del AE
Idea de la literatura: usar el AE solo para **comprimir** y correr un detector shallow en
el espacio latente (16 dim), donde el ruido de features redundantes ya se descartó.
Entrenamos un AE, sacamos el latente con `.encode()`, y corremos los detectores ahí:"""),
    code("""ae = AEDetector(hidden_dims=(64,32), latent_dim=16, max_epochs=300, patience=30,
                random_state=42).fit(ds.X_train)
Ztr, Zte = ae.encode(ds.X_train), ae.encode(ds.X_test)   # espacio latente (16 dim)
print("latente:", Ztr.shape)
latente = {
    "IForest sobre latente": [lab.metrics(IsolationForestDetector(n_estimators=200, max_features=0.3,
                              random_state=s).fit(Ztr).score_samples(Zte), ds.y_test) for s in SEEDS[:3]],
    "OCSVM-RBF sobre latente": [lab.metrics(OneClassSVMDetector(kernel="rbf", nu=0.1)
                                .fit(Ztr).score_samples(Zte), ds.y_test)],
}
lab.tabla(latente)"""),
    md("""**No mejora sobre el baseline.** El bottleneck del AE (entrenado por MSE) destruye la
estructura que el detector necesita: comprimir a un latente optimizado para reconstruir no
preserva la señal de anomalía. Guardamos la idea — con el VAE (nb 3) el latente es mejor,
pero ni así supera al score `recon_prob` directo.

**Conclusión:** AE/DAE con MSE no alcanzan, ni directo ni vía latente. **El problema es el
score, no la arquitectura** → nb 3."""),
])

# ===========================================================================
# 03 — VAE
# ===========================================================================
build("03_vae.ipynb", "3 · VAE y el score recon_prob", [
    md("""El **VAE** modela `p(x|z)` con media **y varianza** por feature. Eso habilita un score
mejor que el MSE: **`recon_prob`** (An & Cho 2015) = `-E[log p(x|z)]`, el error de cada
feature **ponderado por la confianza que el decoder le asigna**. Es la clave del proyecto.

Primero **buscamos hiperparámetros** (grilla, a la vista), después analizamos el score y la
varianza sobre la arquitectura ganadora, y al final el latente + t-SNE."""),
    code(SETUP),
    code(DATOS),
    md("""## 3.1 · Búsqueda de hiperparámetros (y por qué NO alcanza con una semilla)
Barremos configuraciones de arquitectura/regularización del VAE (`hidden_dims`,
`latent_dim`, `beta`), con `score_mode=recon_prob` fijo. **La trampa:** estos modelos
tienen alta varianza entre semillas, así que rankear por **una sola corrida** premia
semillas afortunadas, no configuraciones buenas. Por eso reportamos, lado a lado, el
PR-AUC con **1 semilla** (búsqueda ingenua) y la **media ± desvío de 5 semillas**.

> Sin un val útil (nb 1), la búsqueda se ilustra en **test** — *data snooping* declarado.
> Para no elegir por suerte, seleccionamos por **media − desvío** (premia lo bueno *y*
> estable), coherente con que el objetivo del proyecto es un detector de baja varianza."""),
    code("""candidatas = [((64,32),16,1.0), ((64,32),24,1.0), ((64,32),24,0.5),
              ((128,64),16,1.0), ((128,64),24,1.0), ((128,64),24,0.5)]
filas_hp = []
for hid, lat, beta in candidatas:
    prs = []
    for s in SEEDS[:5]:                                # 5 semillas por configuración
        v = VAEDetector(hidden_dims=hid, latent_dim=lat, beta=beta, score_mode="recon_prob",
                        n_mc_samples=50, max_epochs=300, patience=30, random_state=s).fit(ds.X_train)
        prs.append(lab.metrics(v.score_samples(ds.X_test), ds.y_test)["pr_auc"])
    prs = np.array(prs)
    filas_hp.append({"hidden_dims": str(hid), "latent_dim": lat, "beta": beta,
                     "pr_auc_1seed": round(prs[0],3), "pr_auc_mean": round(prs.mean(),3),
                     "pr_auc_std": round(prs.std(),3), "media_menos_std": round(prs.mean()-prs.std(),3)})
hp = pd.DataFrame(filas_hp).sort_values("media_menos_std", ascending=False).reset_index(drop=True)
hp"""),
    md("""**Mirá la diferencia entre columnas:** la config que gana con **1 semilla**
(`pr_auc_1seed` más alto) suele tener un **desvío grande** — ganó por suerte, y su media
real es más baja. Por eso elegimos por **`media_menos_std`** (arriba en la tabla): la
configuración que es buena *y* estable. Esa es `BEST`:"""),
    code("""best = hp.iloc[0]
BEST = dict(hidden_dims=eval(best["hidden_dims"]), latent_dim=int(best["latent_dim"]),
            beta=float(best["beta"]))
print("config final (mejor media−std):", BEST)
print(f"  PR-AUC 1 semilla = {best['pr_auc_1seed']}  vs  media±std = {best['pr_auc_mean']}±{best['pr_auc_std']}")"""),
    md("""## 3.2 · Los tres scores del VAE, con la misma red (config ganadora)
`recon_error` (MSE), `neg_elbo` (recon + KL) y `recon_prob`. Entrenamos una vez por
semilla y evaluamos los tres scores del mismo modelo — así la comparación aísla el
*score*:"""),
    code("""filas = {"recon_error": [], "neg_elbo": [], "recon_prob": []}
for seed in SEEDS:
    vae = VAEDetector(**BEST, n_mc_samples=50, score_mode="recon_prob",
                      max_epochs=300, patience=30, random_state=seed).fit(ds.X_train)
    for modo in filas:
        vae.score_mode = modo                                  # mismo modelo, distinto score
        filas[modo].append(lab.metrics(vae.score_samples(ds.X_test), ds.y_test))
pd.DataFrame({modo: lab.mean_std(f) for modo, f in filas.items()}).T[["pr_auc","roc_auc","recall_at_contam"]]"""),
    md("""**Lo que importa es usar la verosimilitud del decoder, no el MSE plano.** Los dos
scores probabilísticos —`recon_prob` (An & Cho, muestreo MC) y `neg_elbo` (recon + KL)—
**aplastan** al `recon_error` (MSE) — mirá la tabla de arriba: la diferencia es enorme. Entre ellos quedan
parejos (ambos ponderan el error por la varianza que el decoder asigna a cada feature),
así que adoptamos **`recon_prob`** por ser el más principiado. El salto es del *score*, no
de la red. Guardamos sus métricas:"""),
    code("""filas_vae = filas["recon_prob"]
print("VAE recon_prob (soja):"); lab.mean_std(filas_vae)"""),
    md("""## 3.3 · La regularización pesada destruye la señal
La varianza per-feature del decoder ya regulariza el score; agregarle batch-norm + dropout
+ β bajo lo aplana. Lo verificamos sobre la misma arquitectura ganadora:"""),
    code("""m_reg = []
for seed in SEEDS:
    vr = VAEDetector(hidden_dims=BEST["hidden_dims"], latent_dim=BEST["latent_dim"], beta=0.5,
                     n_mc_samples=50, score_mode="recon_prob", use_batch_norm=True, dropout=0.15,
                     max_epochs=300, patience=30, random_state=seed).fit(ds.X_train)
    m_reg.append(lab.metrics(vr.score_samples(ds.X_test), ds.y_test))
print("VAE recon_prob + BN/dropout/β=0.5:", lab.mean_std(m_reg)["pr_auc"],
      "  vs limpio:", lab.mean_std(filas_vae)["pr_auc"])"""),
    md("""## 3.4 · VAE vs TODOS los baselines
Contra el conjunto completo del nb 1 — estadísticos (z-score, Mahalanobis) y de modelos
(IForest, OCSVM). Todos multi-seed donde corresponde (IForest estocástico → ±std; los
deterministas → std 0):"""),
    code("""resultados = {
    "ZScore (mean)": [lab.metrics(ZScoreDetector(agg="mean").fit(ds.X_train)
                                  .score_samples(ds.X_test), ds.y_test)],
    "Mahalanobis":   [lab.metrics(MahalanobisDetector(shrinkage=None).fit(ds.X_train)
                                  .score_samples(ds.X_test), ds.y_test)],
    "IForest": [lab.metrics(IsolationForestDetector(n_estimators=200, max_features=0.3, random_state=s)
                            .fit(ds.X_train).score_samples(ds.X_test), ds.y_test) for s in SEEDS],
    "OCSVM-RBF (determinista)": [lab.metrics(OneClassSVMDetector(kernel="rbf", nu=0.1)
                              .fit(ds.X_train).score_samples(ds.X_test), ds.y_test)],
    "VAE recon_prob": filas_vae,
}
lab.plot_leaderboard(lab.leaderboard(resultados)); plt.show()
lab.tabla(resultados)"""),
    md("""## 3.5 · IForest y OCSVM sobre el latente del VAE (multi-seed)
Como con el AE, probamos correr detectores shallow sobre el latente del VAE. **En igualdad
de condiciones que el resto**: 5 semillas, media ± std. Para cada semilla entrenamos el VAE,
sacamos su latente y corremos los detectores ahí; guardamos uno como representante para el
t-SNE de abajo:"""),
    code("""filas_lat = {"VAE recon_prob (directo)": [], "IForest sobre latente": [], "OCSVM sobre latente": []}
vae_repr = Zte_repr = sc_repr = None
for s in SEEDS:
    v = VAEDetector(**BEST, n_mc_samples=50, score_mode="recon_prob",
                    max_epochs=300, patience=30, random_state=s).fit(ds.X_train)
    Ztr, Zte = v.encode(ds.X_train), v.encode(ds.X_test)
    filas_lat["VAE recon_prob (directo)"].append(lab.metrics(v.score_samples(ds.X_test), ds.y_test)["pr_auc"])
    filas_lat["IForest sobre latente"].append(lab.metrics(IsolationForestDetector(n_estimators=200,
        max_features=0.3, random_state=s).fit(Ztr).score_samples(Zte), ds.y_test)["pr_auc"])
    filas_lat["OCSVM sobre latente"].append(lab.metrics(OneClassSVMDetector(kernel="rbf", nu=0.1)
        .fit(Ztr).score_samples(Zte), ds.y_test)["pr_auc"])
    if s == SEEDS[0]:
        vae_repr, Zte_repr, sc_repr = v, Zte, v.score_samples(ds.X_test)   # para el t-SNE
pd.Series({k: f"{np.mean(v):.3f}±{np.std(v):.3f}" for k, v in filas_lat.items()})"""),
    md("""**El `recon_prob` directo gana** (comparando las medias ± std): el latente pierde la
señal que el score probabilístico captura (el error ponderado por varianza vive en el
espacio de reconstrucción, no en el latente). Un detector shallow sobre el latente no la
recupera.

## 3.6 · t-SNE del espacio latente
Proyectamos el latente del VAE representativo a 2D con t-SNE, coloreado por etiqueta real y
por score.
Si las anomalías no forman cluster, ningún modelo que mire este espacio las separa caso a
caso (anticipo del techo, nb 5):"""),
    code("""emb = lab.embed_2d(Zte_repr, method="tsne")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
lab.plot_embedding(emb, ds.y_test, kind="label", ax=axs[0], titulo="t-SNE latente · etiqueta real")
lab.plot_embedding(emb, sc_repr, kind="score", ax=axs[1], titulo="t-SNE latente · score del VAE")
plt.tight_layout(); plt.show()"""),
    md("""**Mejor modelo hasta acá: VAE `recon_prob`** — supera al IForest y al OCSVM. Pero tiene
**alta varianza entre semillas** (ver el ±std de arriba): en una corrida mala roza el baseline. Lo
resolvemos con el ensemble (nb 4), reusando la config ganadora `BEST`."""),
])

# ===========================================================================
# 04 — Ensemble (modelo final)
# ===========================================================================
build("04_ensemble.ipynb", "4 · Seed-ensemble: el modelo final", [
    md("""El VAE `recon_prob` era el mejor pero **cada semilla converge distinto** → varianza
alta. Solución: **seed-ensemble** — entrenar N VAEs con semillas distintas y **promediar
sus scores** (z-normalizados por miembro). El ruido idiosincrático se cancela."""),
    code(SETUP),
    code(DATOS),
    md("""## 4.1 · El seed-ensemble
`EnsembleDetector` toma una lista de detectores ya construidos, normaliza el score de cada
uno y los promedia. Cada miembro es un VAE con la config ganadora del nb 3. Armamos 10 con
semillas 42..51 — el ensemble está totalmente a la vista:"""),
    code("""def make_ensemble(base_seed):
    miembros = [VAEDetector(hidden_dims=(128,64), latent_dim=24, beta=1.0, n_mc_samples=50,
                            score_mode="recon_prob", max_epochs=300, patience=30,
                            random_state=base_seed+i) for i in range(10)]   # 10 semillas
    return EnsembleDetector(miembros, normalize="zscore", combine="mean")

# ±std del ensemble = 5 réplicas con semillas base distintas (50 VAEs en total; tarda).
# Guardamos la réplica base=42 para reusarla en los gráficos de más abajo.
filas_ens, ens_soja = [], None
for base in [42, 52, 62, 72, 82]:
    ens = make_ensemble(base).fit(ds.X_train)
    filas_ens.append(lab.metrics(ens.score_samples(ds.X_test), ds.y_test))
    if base == 42: ens_soja = ens
print("VAE seed-ensemble ×10 (soja) — media±std de 5 réplicas:")
lab.mean_std(filas_ens)"""),
    md("""## 4.2 · La varianza se desploma y la media sube
Comparamos el ensemble contra un VAE single (una semilla) — mismo modelo base:"""),
    code("""single = [lab.metrics(VAEDetector(hidden_dims=(128,64), latent_dim=24, beta=1.0, n_mc_samples=50,
              score_mode="recon_prob", max_epochs=300, patience=30, random_state=s)
              .fit(ds.X_train).score_samples(ds.X_test), ds.y_test) for s in SEEDS]
print("VAE single     :", lab.mean_std(single)["pr_auc"])
print("VAE seed-ens ×10:", lab.mean_std(filas_ens)["pr_auc"])"""),
    md("""El desvío cae fuerte **y la media sube**: el promedio de scores independientes gana en
media y en el peor caso.

## 4.3 · El modelo final, en ambos cultivos
Reusamos la réplica base=42 en soja y entrenamos una en maíz, para los histogramas de score.

> Ojo: el número **oficial** es la media±std de 4.1. Acá `sc_soja` es de **una réplica
> concreta** (base=42): su PR-AUC puntual cae *dentro* de ese ±std — no es un modelo
> distinto ni mejor."""),
    code("""sc_soja = ens_soja.score_samples(ds.X_test)          # réplica base=42 (ya entrenada en 4.1)
ens_maiz = make_ensemble(42).fit(dsm.X_train)
sc_maiz = ens_maiz.score_samples(dsm.X_test)
print("soja (réplica 42):", {k: round(v,3) for k,v in lab.metrics(sc_soja, ds.y_test).items()})
print("maíz (réplica 42):", {k: round(v,3) for k,v in lab.metrics(sc_maiz, dsm.y_test).items()})"""),
    code("""fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
lab.plot_scores(sc_soja, ds.y_test, ax=axs[0], titulo="seed-ensemble · soja")
lab.plot_scores(sc_maiz, dsm.y_test, ax=axs[1], titulo="seed-ensemble · maíz")
plt.tight_layout(); plt.show()"""),
    md("""## 4.4 · Sobre el latente del ensemble
El latente del ensemble **es el del miembro 0 (VAE con seed 42), idéntico al del nb 3** —
mismo modelo, misma semilla, mismos datos. Ya lo analizamos ahí: IForest/OCSVM sobre ese
latente **no** superan al score directo, y su t-SNE muestra las anomalías mezcladas (sin
cluster). No lo repetimos acá para no duplicar el mismo gráfico.

**El seed-ensemble ×10 del VAE `recon_prob`** es el **modelo final**, con el menor desvío
entre semillas (ver la tabla de 4.1). Ni el latente + shallow lo supera. Corre sobre el panel
unificado (clima + NDVI + ERA5). En el nb 5 vemos por qué cuesta subir de acá (el techo) y en
el nb 6 chequeamos si las features agronómicas de dominio agregan algo más."""),
])

# ===========================================================================
# 05 — Techo estructural (con Cartography inline)
# ===========================================================================
build("05_techo_estructural.ipynb", "5 · El techo estructural (por qué nadie pasa de ~0.6)", [
    md("""Ningún modelo — IForest, OCSVM, AE, DAE, VAE, ensemble — pasa de ~0.6. Cuando
familias tan distintas chocan contra el mismo número, el límite no está en el modelo sino
en la **señal**. Tres análisis independientes lo confirman."""),
    code(SETUP),
    code(DATOS),
    md("""## 5.1 · ¿Se separan las anomalías en el espacio de features? (PCA)
Proyectamos las 54 features del **test** a 2D con PCA y coloreamos por etiqueta real, con
los ejes anotando **cuánta varianza explica cada componente**:"""),
    code("""from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42).fit(ds.X_test)
emb = pca.transform(ds.X_test)
v1, v2 = pca.explained_variance_ratio_[:2] * 100
fig, ax = plt.subplots(figsize=(6, 5))
for lab_, color, nombre in [(0, lab.C_NORMAL, "normal"), (1, lab.C_ANOM, "anómala")]:
    m = ds.y_test == lab_
    ax.scatter(emb[m,0], emb[m,1], s=8, alpha=0.35 if lab_==0 else 0.8, color=color, label=nombre)
ax.set_xlabel(f"PC1 ({v1:.0f}% var.)"); ax.set_ylabel(f"PC2 ({v2:.0f}% var.)")
ax.set_title("PCA de las 54 features (test, soja) · por anomalía"); ax.legend()
plt.show()
print(f"PC1+PC2 explican solo {v1+v2:.0f}% de la varianza")"""),
    md("""El panorama es **matizado, y es justo lo que dice el techo**: una parte de las
anómalas (rojo) **se concentra a la derecha** (PC1 alto) — esa región es la **firma de
sequía**, las anomalías *detectables*; pero **muchas otras quedan mezcladas** entre las
normales, sin región propia — esas son las del techo estructural, sin firma climática. No
es "todo mezclado" ni "cluster limpio": es "una parte se separa, el resto no".

> **Dos advertencias honestas sobre este PCA** (por eso es ilustrativo; el argumento
> *fuerte* del techo es el cross-modelo de abajo):
> - **PC1+PC2 explican ~48%** de la varianza (lo imprime la celda): casi la mitad se pierde
>   al bajar a 2D, así que lo que se ve es una aproximación, no la geometría completa.
> - **Difiere del PCA del EDA** porque son datos distintos: acá es sobre `X_test` (test
>   ≥2021, normalizado por el pipeline con stats de train); en el EDA es sobre *todo* el
>   panel con otra normalización. El PCA se ajusta a los datos que ve → la proyección cambia.

## 5.2 · Cross-modelo: ¿todos fallan en las MISMAS?
Entrenamos varios modelos, marcamos el top-10% de cada uno, y contamos cuántas anomalías
las falla **todo el mundo**. Acá no comparamos *performance* (para eso reportamos media±std
en los otros nb) sino **qué muestras se escapan**: un modelo representativo por familia
(seed 42) alcanza — el solapamiento de errores es robusto a la semilla."""),
    code("""modelos = {
    "IForest":  IsolationForestDetector(n_estimators=200, max_features=0.3).fit(ds.X_train),
    "OCSVM":    OneClassSVMDetector(kernel="rbf", nu=0.1).fit(ds.X_train),
    "AE":       AEDetector(hidden_dims=(64,32), latent_dim=16, max_epochs=200, random_state=42).fit(ds.X_train),
    "VAE":      VAEDetector(hidden_dims=(128,64), latent_dim=24, score_mode="recon_prob",
                            n_mc_samples=50, max_epochs=200, random_state=42).fit(ds.X_train),
}
# flag = marcada anómala (top-10% de su propio score)
flags = np.vstack([ (m.score_samples(ds.X_test) >=
                     np.quantile(m.score_samples(ds.X_test), 0.9)).astype(int)
                    for m in modelos.values()])
anom = ds.y_test == 1
falladas_por_todos = anom & (flags.sum(axis=0) == 0)   # anomalía que NADIE marca
print(f"anomalías reales: {anom.sum()}")
print(f"falladas por TODOS los modelos: {falladas_por_todos.sum()} "
      f"({100*falladas_por_todos.sum()/anom.sum():.0f}%)")"""),
    md("""### ¿Qué tienen de distinto las falladas? Su clima
Comparamos la precipitación de verano (normalizada) de las anomalías detectadas vs las
falladas por todos:"""),
    code("""prec_cols = [i for i,c in enumerate(ds.feature_cols) if "prectotcorr" in c]
precip_z = ds.X_test[:, prec_cols].mean(axis=1)         # firma de sequía (más negativo = más seco)
det_alguno = anom & (flags.sum(axis=0) > 0)
print(f"precip_z medio · detectadas por alguien: {precip_z[det_alguno].mean():+.2f}")
print(f"precip_z medio · falladas por todos    : {precip_z[falladas_por_todos].mean():+.2f}")"""),
    md("""Las **detectadas tienen sequía clara** (precip_z bien negativo); las **falladas tienen
clima casi normal**. No dejan huella en las features → ningún modelo puede verlas.

## 5.3 · La idea del profesor: "lo consistentemente difícil es anomalía" (Cartography)
Adaptamos *Dataset Cartography* (Swayamdipta 2020) al VAE: registramos el error de
reconstrucción de **cada muestra de train en cada época** (flag `track_datamap`), y medimos
por muestra el error medio (¿el modelo la aprende?) y su variabilidad."""),
    code("""vae_cart = VAEDetector(hidden_dims=(128,64), latent_dim=24, score_mode="recon_prob",
                       n_mc_samples=50, max_epochs=200, patience=200,
                       track_datamap=True, random_state=42).fit(ds.X_train)
hist = vae_cart.per_sample_error_          # matriz (n_épocas × n_train)
mean_err = hist.mean(axis=0); var_err = hist.std(axis=0)
hard = mean_err >= np.quantile(mean_err, 0.66)   # "hard-to-learn": el modelo nunca las reconstruye
z_train = ds.meta_train["z_rinde"].to_numpy()
print(f"train hard-to-learn: {hard.mean()*100:.0f}%")
print(f"z_rinde medio · hard-to-learn: {np.nanmean(z_train[hard]):.2f}  |  train completo: {np.nanmean(z_train):.2f}")"""),
    code("""fig, ax = plt.subplots(figsize=(6,4.5))
sc = ax.scatter(var_err, mean_err, s=6, alpha=0.4, c=np.nan_to_num(z_train), cmap="coolwarm_r")
ax.set_xlabel("variabilidad (desvío del error entre épocas)"); ax.set_ylabel("error medio (1/confidence)")
ax.set_title("Data map del VAE (train) — color = z_rinde"); plt.colorbar(sc, ax=ax, label="z_rinde"); plt.show()"""),
    md("""**El hallazgo clave (y por qué Cartography se queda como *diagnóstico*, no como
detector):** lo *hard-to-learn* tiene el **mismo `z_rinde` medio** que el resto del train.
Lo que cuesta reconstruir es **rareza climática, no anomalía de rinde** — son cosas
distintas. La idea del profesor queda **refutada como detector** pero **validada como
auditoría**: confirma, desde otro ángulo, que dificultad-de-reconstrucción ≠
anomalía-de-rinde, que es exactamente el techo.

## 5.4 · Conclusión
El techo es **estructural**: buena parte de las anomalías de rinde tienen causas no
climáticas (plaga, granizo, manejo) invisibles a las features climáticas. No es la
arquitectura (todos fallan en las mismas muestras), no es la etiqueta (la mayoría son
caídas de rinde reales). La salida es **más señal**: el panel unificado ya suma estado de
suelo + verdor (NDVI/ERA5), que corren el techo un poco (aunque no lo rompen: seguimos en
~0.6). El nb 6 chequea si las features agronómicas de dominio aportan algo más."""),
])

# ===========================================================================
# 06 — Features nuevas
# ===========================================================================
build("06_features_nuevas.ipynb", "6 · ¿Ayudan las features agronómicas de dominio?", [
    md("""El **panel unificado ya incluye clima + NDVI-AVHRR (verdor) + ERA5-Land (suelo,
heladas)**: el análisis histórico mostró que el estado de suelo + verdor **corren el techo
un poco** (mejora chica pero real sobre el clima solo), así que quedaron como parte del
dataset por defecto.

La pregunta abierta acá es la que sigue viva: sobre ese panel, ¿aportan además las
**features agronómicas de dominio** —balance hídrico, estrés térmico de la ventana
crítica (`add_agro_features`)— algo que el modelo no saque ya de las columnas mensuales?

> Dirección futura (a diseñar): feature engineering más profundo —índices de extremos
> (días Tmax>32 en floración, racha seca máxima, déficit hídrico acumulado), NDVI integrado
> de temporada (≈ biomasa) y anomalías de NDVI vs su climatología, interacciones
> suelo×precip—, en vez de solo cuatro features agregadas."""),
    code(SETUP),
    md("""## 6.1 · La sonda: 3 detectores multi-seed sobre un dataset
Misma evaluación en igualdad de condiciones (5 semillas, media ± std) para el VAE
`recon_prob`, IForest y OCSVM-RBF."""),
    code("""SEEDS_F = SEEDS[:5]
def probe(dset):
    y = dset.y_test
    cols = {"VAE recon_prob": [], "IForest": [], "OCSVM-RBF": []}
    for s in SEEDS_F:
        v = VAEDetector(hidden_dims=(128,64), latent_dim=24, score_mode="recon_prob",
                        n_mc_samples=50, max_epochs=300, patience=30, random_state=s).fit(dset.X_train)
        cols["VAE recon_prob"].append(lab.metrics(v.score_samples(dset.X_test), y)["pr_auc"])
        cols["IForest"].append(lab.metrics(IsolationForestDetector(n_estimators=200, max_features=0.3,
            random_state=s).fit(dset.X_train).score_samples(dset.X_test), y)["pr_auc"])
        cols["OCSVM-RBF"].append(lab.metrics(OneClassSVMDetector(kernel="rbf", nu=0.1)
            .fit(dset.X_train).score_samples(dset.X_test), y)["pr_auc"])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in cols.items()}"""),
    md("## 6.2 · Dataset unificado vs. unificado + agro"),
    code("""panel_z = data.prepare()                              # panel unificado (clima+NDVI+ERA5)
ds_uni  = data.build_crop_dataset(panel_z, "soja")
ds_agro = data.build_crop_dataset(panel_z, "soja", use_agro=True)
for nombre, d in [("unificado", ds_uni), ("+ agro", ds_agro)]:
    print(f"{nombre:12s} {len(d.feature_cols):3d} features")"""),
    md("## 6.3 · Resultados (PR-AUC test ± std, soja)"),
    code("""res = {"unificado": probe(ds_uni), "+ agro": probe(ds_agro)}
pd.DataFrame({n: {k: f"{m:.3f}±{s:.3f}" for k,(m,s) in r.items()} for n,r in res.items()}).T"""),
    md("""## 6.4 · La prueba justa: seed-ensemble unificado vs. + agro
El modelo final es el **seed-ensemble** (nb 4), que baja fuerte el ±std, así que una mejora
chica se puede ver ahí mejor que en un VAE single:"""),
    code("""def ens_scores(dset, base):
    miembros = [VAEDetector(hidden_dims=(128,64), latent_dim=24, score_mode="recon_prob",
                n_mc_samples=50, max_epochs=300, patience=30, random_state=base+i) for i in range(10)]
    return EnsembleDetector(miembros).fit(dset.X_train).score_samples(dset.X_test)

for nombre, dset in [("ensemble unificado", ds_uni), ("ensemble + agro", ds_agro)]:
    prs = [lab.metrics(ens_scores(dset, b), dset.y_test)["pr_auc"] for b in [42, 52, 62]]
    print(f"{nombre:22s} {np.mean(prs):.3f}±{np.std(prs):.3f}")"""),
    md("""**Conclusión:** el panel unificado (con suelo + verdor) es la base del modelo final.
Las features agronómicas agregadas no mueven la aguja de forma clara sobre eso (comparar los
±std de arriba) — coherente con el techo estructural (nb 5): las cuatro agro son
transformaciones de las mismas columnas mensuales que el modelo ya ve. Exprimir más señal
requiere **feature engineering de dominio más rico** (ver la nota del intro), no solo cuatro
agregados — es la dirección de trabajo pendiente."""),
])

# ===========================================================================
# 07 — Modernos + leaderboard + conclusiones
# ===========================================================================
build("07_leaderboard_y_conclusiones.ipynb", "7 · Leaderboard final y conclusiones", [
    md("""Cierre: reunimos todos los modelos en un **leaderboard** y sacamos las conclusiones.
Incluimos **DeepSVDD** como referencia de un método moderno de deep anomaly detection
(librería `deepod`), a presupuesto parejo (300 épocas, capacidad como el VAE).

> **Sobre los métodos modernos:** probamos también ICL y NeuTraL, pero con presupuesto
> parejo dieron **por debajo del baseline** (ICL) o **cerca del azar** (NeuTraL) — no aportan
> a la comparación, así que dejamos solo DeepSVDD (el mejor de los tres). A DeepSVDD **sí lo
> tuneamos** (7.0) para que la comparación contra el VAE sea justa: el VAE tuvo su búsqueda de
> HP (nb 3), así que le damos a DeepSVDD la suya."""),
    code(SETUP),
    code(DATOS),
    md("""## 7.0 · Tuning de DeepSVDD (comparación justa)
Barrido chico de `lr`, `rep_dim` y `hidden_dims`, seleccionado **multi-seed** por
`media − std` (mismo criterio que el VAE en el nb 3: premia lo bueno *y* estable).
El resto (`epochs=300`, `batch_size=64`) queda a presupuesto parejo con el VAE."""),
    code("""DSVDD_GRID = [dict(lr=lr, rep_dim=rd, hidden_dims=hd)
              for lr in (1e-3, 5e-4) for rd in (16, 24) for hd in ("64,32", "128,64")]
filas_dsvdd = []
for cfg in DSVDD_GRID:
    prs = [lab.metrics(DeepODDetector(algo="deepsvdd", epochs=300, batch_size=64,
                       random_state=s, **cfg).fit(ds.X_train).score_samples(ds.X_test),
                       ds.y_test)["pr_auc"] for s in SEEDS[:3]]
    filas_dsvdd.append({**cfg, "pr_auc_mean": round(np.mean(prs), 3),
                        "pr_auc_std": round(np.std(prs), 3),
                        "media_menos_std": round(np.mean(prs) - np.std(prs), 3)})
dsvdd_hp = pd.DataFrame(filas_dsvdd).sort_values("media_menos_std", ascending=False).reset_index(drop=True)
DSVDD_BEST = dict(lr=float(dsvdd_hp.iloc[0]["lr"]), rep_dim=int(dsvdd_hp.iloc[0]["rep_dim"]),
                  hidden_dims=dsvdd_hp.iloc[0]["hidden_dims"])
print("DeepSVDD tuneado (mejor media−std):", DSVDD_BEST)
dsvdd_hp"""),
    md("""## 7.1 · Leaderboard final
Entrenamos cada modelo clave acá mismo, sobre el **panel unificado** (clima + NDVI + ERA5).
El **modelo final** es el seed-ensemble ×10 del VAE `recon_prob` (config ganadora
`(128,64) lat24`, nb 3). Todo multi-seed; OCSVM es determinista (std 0)."""),
    code("""def ens_scores(dset, base):
    miembros = [VAEDetector(hidden_dims=(128,64), latent_dim=24, score_mode="recon_prob",
                n_mc_samples=50, max_epochs=300, patience=30, random_state=base+i) for i in range(10)]
    return EnsembleDetector(miembros).fit(dset.X_train).score_samples(dset.X_test)

lb = lab.leaderboard({
  "VAE seed-ensemble ★": [lab.metrics(ens_scores(ds, b), ds.y_test) for b in [42,52,62]],
  "VAE recon_prob single": [lab.metrics(VAEDetector(hidden_dims=(128,64), latent_dim=24,
        score_mode="recon_prob", n_mc_samples=50, max_epochs=300, patience=30, random_state=s)
        .fit(ds.X_train).score_samples(ds.X_test), ds.y_test) for s in SEEDS],
  "AE (MSE)": [lab.metrics(AEDetector(hidden_dims=(128,64), latent_dim=24, score_mode="mse",
        max_epochs=300, patience=30, random_state=s)
        .fit(ds.X_train).score_samples(ds.X_test), ds.y_test) for s in SEEDS],
  "DAE (MSE)": [lab.metrics(DenoisingAEDetector(hidden_dims=(128,64), latent_dim=24, score_mode="mse",
        max_epochs=300, patience=30, random_state=s)
        .fit(ds.X_train).score_samples(ds.X_test), ds.y_test) for s in SEEDS],
  "IForest": [lab.metrics(IsolationForestDetector(n_estimators=200, max_features=0.3, random_state=s)
        .fit(ds.X_train).score_samples(ds.X_test), ds.y_test) for s in SEEDS],   # estocástico → std
  "DeepSVDD (moderno, tuneado)": [lab.metrics(DeepODDetector(algo="deepsvdd", epochs=300,
        batch_size=64, random_state=s, **DSVDD_BEST)
        .fit(ds.X_train).score_samples(ds.X_test), ds.y_test) for s in SEEDS],
  "OCSVM-RBF (determinista)": [lab.metrics(OneClassSVMDetector(kernel="rbf", nu=0.1)
        .fit(ds.X_train).score_samples(ds.X_test), ds.y_test)],                   # 1 corrida → std 0
})
lab.plot_leaderboard(lb); plt.show()
lb[["modelo","pr_auc","pr_auc_std","roc_auc"]].round(3)"""),
    md("""## 7.2 · Recapitulación del recorrido
La historia en una tabla. Los **PR-AUC salen del leaderboard vivo de arriba** (`lb`, celda
7.1) — no hay ningún número escrito a mano: si cambiás las semillas o el pipeline, esta tabla
se mueve sola con la corrida."""),
    code("""pr = dict(zip(lb["modelo"], lb["pr_auc"]))   # PR-AUC reales, tomados del leaderboard 7.1
recap = pd.DataFrame([
    ("nb1", "Isolation Forest (baseline)",        pr["IForest"],                       "listón no neuronal, varianza baja"),
    ("nb1", "One-Class SVM (RBF)",                pr["OCSVM-RBF (determinista)"],       "kernel gaussiano ≈ IForest; el lineal fracasa (nb1)"),
    ("nb2", "AE (score MSE)",                     pr["AE (MSE)"],                       "el MSE plano no alcanza"),
    ("nb2", "DAE (score MSE)",                    pr["DAE (MSE)"],                      "el denoising no cambia el cuello de botella"),
    ("nb3", "VAE recon_prob (single)",            pr["VAE recon_prob single"],          "el SCORE probabilístico es el salto vs MSE"),
    ("nb4", "VAE seed-ensemble ×10 ★",            pr["VAE seed-ensemble ★"],            "MODELO FINAL: baja la varianza y sube la media"),
    ("nb7", "DeepSVDD (moderno, tuneado)",        pr["DeepSVDD (moderno, tuneado)"],    "tuneado (7.0) y a presupuesto parejo, por debajo del VAE"),
], columns=["nb", "etapa", "PR-AUC soja (test)", "conclusión"])
recap["PR-AUC soja (test)"] = recap["PR-AUC soja (test)"].round(3)
recap"""),
    md("""## Conclusiones del Componente A
1. **Modelo final: seed-ensemble ×10 del VAE `recon_prob`** sobre el panel unificado (clima
   + NDVI + ERA5) — el mejor del leaderboard de arriba (7.1) y con el menor desvío.
2. **Dos decisiones** lo explican: el **score probabilístico** `recon_prob` (aplasta al MSE,
   nb 3) y el **ensemble de semillas** que baja fuerte la varianza (nb 4). El panel ya trae
   suelo + verdor (NDVI/ERA5), que corren el techo un poco (mejora chica pero real; por eso
   quedaron en el dataset).
3. **Lección metodológica central:** reportar la varianza multi-seed no es un detalle — sin
   ella, mejoras chicas (como la de suelo+verdor) quedan enterradas en el ruido de los
   modelos single.
4. Alternativas refutadas: AE/DAE con MSE, detectores sobre el latente (nb 2–4), las features
   `agro` de dominio (nb 6), y el deep AD moderno tuneado a presupuesto parejo (nb 7).
5. **Techo estructural (nb 5):** existe (rondamos ~0.6), la mayoría de las anomalías siguen
   sin firma observable. Superarlo más requeriría **feature engineering más rico** (nb 6) u
   **otra clase de datos** (sanidad, granizo, manejo)."""),
])

# ===========================================================================
# 08 — Baselines estadísticos: z-score multivariado y Mahalanobis
# ===========================================================================
print("\nNotebooks generados.")
