"""
validate_panel.py — Validación del panel construido por build_panel.py

Checks:
  1. Completitud: cobertura de departamentos × campañas × cultivos
  2. Split temporal: sin solapamiento, rangos correctos
  3. Campañas anómalas: detección razonable vs. eventos conocidos
  4. Lags sin look-ahead: lag1 de la campaña Y = rinde de Y-1
  5. z_rinde: rolling correcto (sin campaña actual)
  6. Distribución del rinde: valores razonables para soja/maíz región núcleo
  7. NASA POWER: cobertura y rangos de variables climáticas
  8. NDVI: cobertura y rangos
  9. ONI: campañas conocidas marcadas correctamente

Uso:
    python src/etl/validate_panel.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(r"C:\Users\Usuario\Desktop\facu\tercero\ml\Tp final")
PROC   = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

errors   = []
warnings = []

def check(cond, msg_ok, msg_fail, is_warning=False):
    if cond:
        print(f"  {PASS} {msg_ok}")
    else:
        sym = WARN if is_warning else FAIL
        print(f"  {sym} {msg_fail}")
        if is_warning:
            warnings.append(msg_fail)
        else:
            errors.append(msg_fail)

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("VALIDACIÓN DEL PANEL NÚCLEO")
print("="*60)

panel_path = PROC / "panel_nucleo.parquet"
if not panel_path.exists():
    print(f"\n{FAIL} No se encontró {panel_path}. Corré build_panel.py primero.")
    sys.exit(1)

df = pd.read_parquet(panel_path)
print(f"\nPanel cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

# ─────────────────────────────────────────────
print("\n[1] COMPLETITUD")

n_en_panel = df["departamento"].nunique()
check(n_en_panel >= 20,
      f"{n_en_panel} departamentos en panel (esperados ~26)",
      f"Solo {n_en_panel} departamentos (esperados ~26)")

cultivos_ok = set(df["cultivo"].unique()) >= {"soja", "maiz"}
check(cultivos_ok,
      f"Cultivos presentes: {sorted(df['cultivo'].unique())}",
      f"Cultivos faltantes: {set(df['cultivo'].unique())}")

min_camp = df["campania_inicio"].min()
max_camp = df["campania_inicio"].max()
check(min_camp <= 1981,
      f"Campaña inicial: {min_camp}/{min_camp+1}",
      f"Campaña inicial {min_camp} posterior a 1981 — falta historia")
check(max_camp >= 2023,
      f"Campaña final: {max_camp}/{max_camp+1}",
      f"Campaña final {max_camp} anterior a 2023 — descargá MAGyP actualizado",
      is_warning=True)

dupes = df.duplicated(subset=["cultivo","departamento","campania_inicio"]).sum()
check(dupes == 0,
      "Sin filas duplicadas",
      f"{dupes} filas duplicadas (cultivo × depto × campaña)")

# ─────────────────────────────────────────────
print("\n[2] SPLIT TEMPORAL")

splits = df.groupby("split")["campania_inicio"].agg(["min","max","count"])
print(f"  Splits:\n{splits.to_string()}")

if "train" in splits.index:
    check(splits.loc["train","max"] == 2017,
          "Train termina en 2017/18",
          f"Train termina en {splits.loc['train','max']} (esperado 2017)",
          is_warning=(splits.loc["train","max"] >= 2015))
    check(splits.loc["train","min"] == 1981,
          "Train empieza en 1981/82",
          f"Train empieza en {splits.loc['train','min']} (esperado 1981)")

if "val" in splits.index:
    check(splits.loc["val","min"] == 2018 and splits.loc["val","max"] == 2020,
          "Val: 2018/19–2020/21",
          f"Val range incorrecto: {splits.loc['val','min']}–{splits.loc['val','max']}")

if "test" in splits.index:
    check(splits.loc["test","min"] == 2021,
          "Test empieza en 2021/22",
          f"Test empieza en {splits.loc['test','min']}")
    check(2022 in df[df["split"]=="test"]["campania_inicio"].values,
          "2022/23 (sequía) está en test",
          "2022/23 NO está en test — campaña anómala clave ausente")

splits_sets = df.groupby("split")["campania_inicio"].apply(set)
for a, b in [("train","val"), ("val","test")]:
    if a in splits_sets and b in splits_sets:
        overlap = splits_sets[a] & splits_sets[b]
        check(len(overlap) == 0,
              f"Sin solapamiento {a}–{b}",
              f"Solapamiento {a}–{b} en campañas: {overlap}")

# ─────────────────────────────────────────────
print("\n[3] CAMPAÑAS ANÓMALAS (train)")

anomalas_train = (
    df[df["es_anomala_train"]]
    .groupby(["cultivo","campania_inicio"])
    .size()
    .reset_index(name="n")
)
print("  Campañas anómalas marcadas en train:")
for _, r in anomalas_train.iterrows():
    print(f"    {r['cultivo']} {r['campania_inicio']}/{r['campania_inicio']+1}")

# 2008/09: Niña fuerte, debe estar siempre
en_2008 = 2008 in anomalas_train["campania_inicio"].values
check(en_2008,
      "2008/09 (Niña fuerte) marcada como anómala",
      "2008/09 NO marcada como anómala — revisar criterio")

# Primeras 5 campañas excluidas por falta de historia
for y in range(1981, 1986):
    en = y in anomalas_train["campania_inicio"].values
    check(en, f"{y}/{y+1} marcada (sin historia rolling)",
          f"{y}/{y+1} NO marcada — debería excluirse del AE", is_warning=True)

normales_path = SPLITS / "campanias_normales_train.parquet"
if normales_path.exists():
    normales = pd.read_parquet(normales_path)
    n_normales = normales["campania_inicio"].nunique()
    n_train_total = df[df["split"]=="train"]["campania_inicio"].nunique()
    check(n_normales > 0,
          f"Campañas normales para AE: {n_normales}/{n_train_total} campañas de train",
          "Sin campañas normales para AE")
    check(2017 not in normales["campania_inicio"].values,
          "2017/18 excluida del train AE (Niña fuerte)",
          "2017/18 incluida en train AE — potencial contaminación",
          is_warning=True)

# ─────────────────────────────────────────────
print("\n[4] LAGS SIN LOOK-AHEAD")

if "rinde_lag1" in df.columns:
    sample_check = []
    for (cult, depto), grp in df.groupby(["cultivo","departamento"]):
        grp = grp.sort_values("campania_inicio")
        for i in range(1, len(grp)):
            curr = grp.iloc[i]
            prev = grp.iloc[i-1]
            if pd.notna(curr["rinde_lag1"]) and pd.notna(prev["rinde_kgha"]):
                ok = abs(curr["rinde_lag1"] - prev["rinde_kgha"]) < 0.01
                sample_check.append(ok)
        if len(sample_check) > 200:
            break
    if sample_check:
        pct_ok = sum(sample_check) / len(sample_check)
        check(pct_ok > 0.99,
              f"lag1 correcto en {pct_ok*100:.1f}% de casos verificados",
              f"lag1 incorrecto en {(1-pct_ok)*100:.1f}% de casos — posible leakage")
    else:
        print(f"  {WARN} No se pudo verificar (insuficientes datos consecutivos)")
else:
    print(f"  {WARN} rinde_lag1 no presente")

# ─────────────────────────────────────────────
print("\n[5] Z-RINDE SIN LOOK-AHEAD")

if "z_rinde" in df.columns and "rinde_ma5" in df.columns:
    # Verificar ma5 de 1986 para Pergamino (soja)
    if "soja" in df["cultivo"].values:
        row_1986 = df[(df["cultivo"]=="soja") & (df["campania_inicio"]==1986) &
                      (df["departamento"]=="Pergamino")]
        if len(row_1986) > 0:
            ma5_1986 = row_1986.iloc[0]["rinde_ma5"]
            rinde_1981_85 = df[(df["cultivo"]=="soja") &
                               (df["campania_inicio"].between(1981,1985)) &
                               (df["departamento"]=="Pergamino")]["rinde_kgha"]
            if len(rinde_1981_85) >= 3 and pd.notna(ma5_1986):
                expected = rinde_1981_85.mean()
                pct_diff = abs(ma5_1986 - expected) / max(abs(expected), 1)
                check(pct_diff < 0.01,
                      f"rinde_ma5 1986 correcto (usa solo 1981-85)",
                      f"rinde_ma5 1986 incorrecto: {ma5_1986:.3f} ≠ esperado {expected:.3f}")

    z_train = df[df["split"]=="train"]["z_rinde"].dropna()
    check(len(z_train) > 0,
          f"z_rinde disponible en train: {len(z_train)} valores",
          "z_rinde vacío en train")
    if len(z_train) > 0:
        pct_ext = (z_train.abs() > 3).mean()
        check(pct_ext < 0.10,
              f"z_rinde: {pct_ext*100:.1f}% outliers >3σ en train",
              f"z_rinde: {pct_ext*100:.1f}% outliers >3σ — distribución sospechosa",
              is_warning=True)
else:
    print(f"  {WARN} z_rinde o rinde_ma5 no presente")

# ─────────────────────────────────────────────
print("\n[6] DISTRIBUCIÓN DEL RINDE (kg/ha)")

for cult in ["soja", "maiz"]:
    sub = df[df["cultivo"]==cult]["rinde_kgha"].dropna()
    if len(sub) == 0:
        continue
    p5, p50, p95 = sub.quantile([0.05, 0.50, 0.95])
    print(f"  {cult}: p5={p5:.1f} | median={p50:.1f} | p95={p95:.1f} | n={len(sub)}")

    # Si la mediana es <20 casi seguro es un problema de separador de miles
    if p50 < 20:
        check(False,
              "",
              f"{cult} median={p50:.2f} — separador de miles mal parseado "
              f"(esperado ~2500-3000 kg/ha). Verificar sep=';' thousands='.' decimal=','")
    else:
        if cult == "soja":
            check(1500 <= p50 <= 4500,
                  f"Soja median={p50:.0f} kg/ha (rango 1500–4500 OK)",
                  f"Soja median={p50:.0f} kg/ha fuera de rango esperado")
            check(p5 > 100,
                  f"Soja p5={p5:.0f} > 100 kg/ha",
                  f"Soja p5={p5:.0f} — posibles valores inválidos")
        elif cult == "maiz":
            check(2000 <= p50 <= 10000,
                  f"Maíz median={p50:.0f} kg/ha (rango 2000–10000 OK)",
                  f"Maíz median={p50:.0f} kg/ha fuera de rango esperado")

    pct_nan = df[df["cultivo"]==cult]["rinde_kgha"].isna().mean()
    check(pct_nan < 0.05,
          f"{cult}: {pct_nan*100:.1f}% NaN en rinde (OK)",
          f"{cult}: {pct_nan*100:.1f}% NaN en rinde",
          is_warning=(pct_nan < 0.20))

# ─────────────────────────────────────────────
print("\n[7] NASA POWER")

nasa_cols = [c for c in df.columns if any(
    c.startswith(p) for p in ["tmean","precip_","gdd","dias_t","rad_solar",
                               "t2m_","prectotcorr_","allsky_"]
)]
if not nasa_cols:
    print(f"  {WARN} Columnas NASA POWER no presentes — descargá corriendo build_panel.py en tu máquina")
else:
    print(f"  Columnas NASA POWER: {len(nasa_cols)}")
    for col in ["tmean", "precip_total", "gdd"]:
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {col}: mean={s.mean():.1f} | min={s.min():.1f} | max={s.max():.1f} | "
                  f"nan%={df[col].isna().mean()*100:.1f}")
    if "tmean" in df.columns:
        check(df["tmean"].between(10, 30).mean() > 0.90,
              "tmean en rango 10–30°C",
              "tmean fuera de rango — revisar centroides o unidades")
    if "precip_total" in df.columns:
        check(df["precip_total"].between(200, 2000).mean() > 0.90,
              "precip_total en rango 200–2000 mm/campaña",
              "precip_total fuera de rango")
    norm_path = PROC / "norm_params.parquet"
    if norm_path.exists():
        norm = pd.read_parquet(norm_path)
        check(len(norm) == df["departamento"].nunique(),
              f"norm_params: {len(norm)} departamentos OK",
              f"norm_params: {len(norm)} vs {df['departamento'].nunique()} en panel",
              is_warning=True)

# ─────────────────────────────────────────────
print("\n[8] NDVI")

ndvi_cols = [c for c in df.columns if c.startswith("ndvi")]
if not ndvi_cols:
    print(f"  {WARN} Columnas NDVI no presentes — descargá corriendo build_panel.py en tu máquina")
else:
    print(f"  Columnas NDVI: {ndvi_cols}")
    if "ndvi_mean" in df.columns:
        s = df["ndvi_mean"].dropna()
        check(s.between(0, 1).mean() > 0.90,
              f"ndvi_mean en rango [0,1]: mean={s.mean():.3f}",
              "ndvi_mean fuera de rango [0,1] — revisar escala")
        pct_cover = df["ndvi_mean"].notna().mean()
        check(pct_cover > 0.70,
              f"Cobertura NDVI: {pct_cover*100:.0f}% de filas",
              f"Cobertura NDVI baja: {pct_cover*100:.0f}% (MODIS desde 2000)",
              is_warning=True)

# ─────────────────────────────────────────────
print("\n[9] ONI")

if "oni_oct_feb_mean" in df.columns:
    cov = df["oni_oct_feb_mean"].notna().mean()
    check(cov > 0.95,
          f"Cobertura ONI: {cov*100:.1f}%",
          f"Cobertura ONI baja: {cov*100:.1f}%")

    # Niñas fuertes conocidas
    for y, label in [(1988,"triple Niña"), (2007,"previa 2008/09"),
                     (2008,"2008/09"), (2010,"triple Niña"), (2011,"triple Niña"),
                     (2022,"2022/23 sequía")]:
        rows = df[(df["campania_inicio"]==y) & df["oni_oct_feb_mean"].notna()]
        if len(rows) > 0:
            val = rows["oni_oct_feb_mean"].iloc[0]
            check(val <= -0.5,
                  f"{y}/{y+1}: ONI={val:.2f} ≤ -0.5 (Niña OK)",
                  f"{y}/{y+1}: ONI={val:.2f} > -0.5 — Niña no detectada",
                  is_warning=True)

    if "oni_categoria" in df.columns:
        cats = df.drop_duplicates("campania_inicio")["oni_categoria"].value_counts()
        print(f"  Categorías ONI: {cats.to_dict()}")
else:
    print(f"  {WARN} oni_oct_feb_mean no presente")

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"  Errores:  {len(errors)}")
print(f"  Warnings: {len(warnings)}")

if errors:
    print("\nERRORES a corregir:")
    for e in errors:
        print(f"  {FAIL} {e}")

if warnings:
    print("\nWARNINGS (esperados con datos parciales):")
    for w in warnings:
        print(f"  {WARN} {w}")

if not errors:
    print(f"\n{PASS} Panel listo para modelado.")
    cols_clave = [c for c in [
        "cultivo","campania_inicio","departamento","split",
        "rinde_kgha","es_anomala_train","z_rinde","rinde_ma5","rinde_lag1",
        "oni_oct_feb_mean","oni_categoria","tmean","precip_total","ndvi_mean"
    ] if c in df.columns]
    print(f"  Shape: {df.shape}")
    print(f"  Columnas clave presentes: {cols_clave}")
else:
    print(f"\n{FAIL} Corregí los errores antes de avanzar al modelado.")
    sys.exit(1)
