"""
validate_panel.py — Validación del panel_nucleo.parquet

Checks:
  1. Completitud: departamentos, cultivos, campañas
  2. Split temporal: rangos y sin solapamiento
  3. Campañas anómalas: detección correcta
  4. Lags sin look-ahead
  5. Z-rinde sin look-ahead
  6. Distribución del rinde
  7. NASA POWER: cobertura y rangos
  8. NDVI: cobertura y rangos
  9. ONI: campañas Niña conocidas
  10. Sin columnas normalizadas (_norm)

Uso:
    python validate_panel.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT  = Path(__file__).resolve().parent
PROC  = ROOT / "data" / "processed"
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

check(df["departamento"].nunique() == 26,
      f"26 departamentos",
      f"{df['departamento'].nunique()} departamentos (esperados 26)")

check(set(df["cultivo"].unique()) >= {"soja", "maiz"},
      f"Cultivos: {sorted(df['cultivo'].unique())}",
      f"Cultivos faltantes: {df['cultivo'].unique()}")

check(df["campania_inicio"].min() <= 1981,
      f"Campaña inicial: {df['campania_inicio'].min()}",
      f"Campaña inicial {df['campania_inicio'].min()} > 1981")

check(df["campania_inicio"].max() >= 2023,
      f"Campaña final: {df['campania_inicio'].max()}",
      f"Campaña final {df['campania_inicio'].max()} < 2023",
      is_warning=True)

check(df.duplicated(subset=["cultivo","departamento","campania_inicio"]).sum() == 0,
      "Sin filas duplicadas",
      f"{df.duplicated(subset=['cultivo','departamento','campania_inicio']).sum()} duplicadas")

# ─────────────────────────────────────────────
print("\n[2] SIN COLUMNAS NORMALIZADAS")

norm_cols = [c for c in df.columns if c.endswith("_norm")]
check(len(norm_cols) == 0,
      "Sin columnas _norm (panel crudo OK)",
      f"{len(norm_cols)} columnas _norm encontradas: {norm_cols[:5]}...")

# ─────────────────────────────────────────────
print("\n[3] SPLIT TEMPORAL")

splits = df.groupby("split")["campania_inicio"].agg(["min","max","count"])
print(f"  Splits:\n{splits.to_string()}")

if "train" in splits.index:
    check(splits.loc["train","max"] == 2017,
          "Train termina en 2017/18",
          f"Train termina en {splits.loc['train','max']} (esperado 2017)")
    check(splits.loc["train","min"] == 1981,
          "Train empieza en 1981/82",
          f"Train empieza en {splits.loc['train','min']} (esperado 1981)")

if "val" in splits.index:
    check(splits.loc["val","min"] == 2018 and splits.loc["val","max"] == 2020,
          "Val: 2018/19–2020/21",
          f"Val range: {splits.loc['val','min']}–{splits.loc['val','max']}")

if "test" in splits.index:
    check(splits.loc["test","min"] == 2021,
          "Test empieza en 2021/22",
          f"Test empieza en {splits.loc['test','min']}")
    check(2022 in df[df["split"]=="test"]["campania_inicio"].values,
          "2022/23 (sequía) está en test",
          "2022/23 NO está en test")

splits_sets = df.groupby("split")["campania_inicio"].apply(set)
for a, b in [("train","val"), ("val","test")]:
    if a in splits_sets and b in splits_sets:
        overlap = splits_sets[a] & splits_sets[b]
        check(len(overlap) == 0,
              f"Sin solapamiento {a}–{b}",
              f"Solapamiento {a}–{b}: {overlap}")

# ─────────────────────────────────────────────
print("\n[4] CAMPAÑAS ANÓMALAS (train)")

anomalas = (
    df[df["es_anomala_train"]]
    .groupby(["cultivo","campania_inicio"])
    .size().reset_index(name="n")
)
print("  Campañas anómalas en train:")
for _, r in anomalas.iterrows():
    print(f"    {r['cultivo']} {r['campania_inicio']}/{r['campania_inicio']+1}")

check(2008 in anomalas["campania_inicio"].values,
      "2008/09 (Niña fuerte) marcada como anómala",
      "2008/09 NO marcada como anómala")

for y in range(1981, 1986):
    check(y in anomalas["campania_inicio"].values,
          f"{y}/{y+1} marcada (sin historia rolling)",
          f"{y}/{y+1} NO marcada", is_warning=True)

normales_path = SPLITS / "campanias_normales_train.parquet"
if normales_path.exists():
    normales = pd.read_parquet(normales_path)
    n_norm = normales["campania_inicio"].nunique()
    n_train = df[df["split"]=="train"]["campania_inicio"].nunique()
    check(n_norm > 0,
          f"Campañas normales AE: {n_norm}/{n_train} de train",
          "Sin campañas normales para el AE")
    check(2017 not in normales["campania_inicio"].values,
          "2017/18 excluida del train AE",
          "2017/18 incluida en train AE", is_warning=True)

# ─────────────────────────────────────────────
print("\n[5] LAGS SIN LOOK-AHEAD")

if "rinde_lag1" in df.columns:
    checks = []
    for (cult, depto), grp in df.groupby(["cultivo","departamento"]):
        grp = grp.sort_values("campania_inicio")
        for i in range(1, len(grp)):
            curr, prev = grp.iloc[i], grp.iloc[i-1]
            if pd.notna(curr["rinde_lag1"]) and pd.notna(prev["rinde_kgha"]):
                checks.append(abs(curr["rinde_lag1"] - prev["rinde_kgha"]) < 0.01)
        if len(checks) > 200:
            break
    if checks:
        pct = sum(checks) / len(checks)
        check(pct > 0.99,
              f"lag1 correcto en {pct*100:.1f}% de casos",
              f"lag1 incorrecto en {(1-pct)*100:.1f}% — posible leakage")
else:
    print(f"  {WARN} rinde_lag1 no presente")

# ─────────────────────────────────────────────
print("\n[6] Z-RINDE SIN LOOK-AHEAD")

if "z_rinde" in df.columns and "rinde_ma5" in df.columns:
    # ma5 de 1986 para Pergamino soja debe usar solo 1981-85
    row_86 = df[(df["cultivo"]=="soja") & (df["campania_inicio"]==1986) &
                (df["departamento"]=="Pergamino")]
    if len(row_86):
        ma5 = row_86.iloc[0]["rinde_ma5"]
        hist = df[(df["cultivo"]=="soja") & (df["campania_inicio"].between(1981,1985)) &
                  (df["departamento"]=="Pergamino")]["rinde_kgha"]
        if len(hist) >= 3 and pd.notna(ma5):
            expected = hist.mean()
            check(abs(ma5 - expected) / max(abs(expected),1) < 0.01,
                  f"rinde_ma5 1986 correcto (usa solo 1981-85): {ma5:.0f} kg/ha",
                  f"rinde_ma5 1986 incorrecto: {ma5:.0f} ≠ {expected:.0f}")

    z = df[df["split"]=="train"]["z_rinde"].dropna()
    check(len(z) > 0,
          f"z_rinde disponible en train: {len(z)} valores",
          "z_rinde vacío en train")
    if len(z):
        pct_ext = (z.abs() > 3).mean()
        check(pct_ext < 0.10,
              f"z_rinde: {pct_ext*100:.1f}% outliers >3σ en train",
              f"z_rinde: {pct_ext*100:.1f}% outliers >3σ — distribución sospechosa",
              is_warning=True)
else:
    print(f"  {WARN} z_rinde o rinde_ma5 no presente")

# ─────────────────────────────────────────────
print("\n[7] DISTRIBUCIÓN DEL RINDE (kg/ha)")

for cult in ["soja", "maiz"]:
    sub = df[df["cultivo"]==cult]["rinde_kgha"].dropna()
    if not len(sub):
        continue
    p5, p50, p95 = sub.quantile([0.05, 0.50, 0.95])
    print(f"  {cult}: p5={p5:.0f} | median={p50:.0f} | p95={p95:.0f} | n={len(sub)}")
    if cult == "soja":
        check(1500 <= p50 <= 4500,
              f"Soja median={p50:.0f} kg/ha OK",
              f"Soja median={p50:.0f} fuera de rango 1500–4500")
        check(p5 > 100,
              f"Soja p5={p5:.0f} > 100 kg/ha",
              f"Soja p5={p5:.0f} — posibles valores inválidos")
    elif cult == "maiz":
        check(2000 <= p50 <= 10000,
              f"Maíz median={p50:.0f} kg/ha OK",
              f"Maíz median={p50:.0f} fuera de rango 2000–10000")

# ─────────────────────────────────────────────
print("\n[8] NASA POWER")

nasa_cols = [c for c in df.columns if any(
    c.startswith(p) for p in ["tmean","precip_","gdd","dias_t","rad_solar",
                               "t2m_","prectotcorr_","allsky_","rh2m_","ws2m_"]
)]
if not nasa_cols:
    print(f"  {WARN} Columnas NASA POWER no presentes")
else:
    print(f"  {len(nasa_cols)} columnas NASA POWER")
    for col, lo, hi in [("tmean",10,30), ("precip_total",200,2000), ("gdd",800,3000)]:
        if col in df.columns:
            s = df[col].dropna()
            pct = s.between(lo,hi).mean()
            print(f"  {col}: mean={s.mean():.1f} | nan%={df[col].isna().mean()*100:.1f}")
            check(pct > 0.90,
                  f"{col} en rango [{lo},{hi}]: {pct*100:.0f}%",
                  f"{col} fuera de rango: {pct*100:.0f}% en [{lo},{hi}]")

# ─────────────────────────────────────────────
print("\n[9] NDVI")

ndvi_cols = [c for c in df.columns if c.startswith("ndvi")]
if not ndvi_cols:
    print(f"  {WARN} Columnas NDVI no presentes")
else:
    print(f"  Columnas: {ndvi_cols}")
    if "ndvi_mean" in df.columns:
        s = df["ndvi_mean"].dropna()
        check(s.between(0,1).mean() > 0.95,
              f"ndvi_mean en [0,1]: mean={s.mean():.3f}",
              "ndvi_mean fuera de rango [0,1]")
        cov = df["ndvi_mean"].notna().mean()
        check(cov > 0.40,
              f"Cobertura NDVI: {cov*100:.0f}% (MODIS desde 2002 — OK)",
              f"Cobertura NDVI muy baja: {cov*100:.0f}%",
              is_warning=True)

# ─────────────────────────────────────────────
print("\n[10] ONI")

if "oni_oct_feb_mean" in df.columns:
    cov = df["oni_oct_feb_mean"].notna().mean()
    check(cov > 0.95,
          f"Cobertura ONI: {cov*100:.1f}%",
          f"Cobertura ONI baja: {cov*100:.1f}%")

    for y, label in [(1988,"triple Niña"),(2008,"2008/09"),(2010,"triple Niña"),(2022,"sequía")]:
        rows = df[(df["campania_inicio"]==y) & df["oni_oct_feb_mean"].notna()]
        if len(rows):
            val = rows["oni_oct_feb_mean"].iloc[0]
            check(val <= -0.5,
                  f"{y}/{y+1} ({label}): ONI={val:.2f} ≤ -0.5",
                  f"{y}/{y+1}: ONI={val:.2f} > -0.5", is_warning=True)

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
    print("\nERRORES:")
    for e in errors:
        print(f"  {FAIL} {e}")
if warnings:
    print("\nWARNINGS:")
    for w in warnings:
        print(f"  {WARN} {w}")

if not errors:
    print(f"\n{PASS} Panel listo para modelado.")
    print(f"  Shape: {df.shape}")
    cols_clave = [c for c in [
        "cultivo","campania_inicio","departamento","split","rinde_kgha",
        "es_anomala_train","z_rinde","rinde_ma5","rinde_lag1",
        "oni_oct_feb_mean","oni_categoria","tmean","precip_total","ndvi_mean"
    ] if c in df.columns]
    print(f"  Columnas clave: {cols_clave}")
else:
    sys.exit(1)