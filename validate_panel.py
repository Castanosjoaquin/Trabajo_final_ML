"""
validate_panel.py — Validación del panel_nucleo.parquet (merge puro de fuentes)

Checks:
  1. Completitud: departamentos, cultivos, campañas
  2. Sin columnas derivadas (_norm, rinde_lag*, z_rinde, es_anomala_train)
  3. Split temporal: rangos y sin solapamiento
  4. Distribución del rinde (MAGyP)
  5. NASA POWER: cobertura y rangos
  6. NDVI: cobertura y rangos
  7. ONI: campañas Niña conocidas
  8. Consistencia geográfica

Uso:
    python validate_panel.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

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
print("VALIDACIÓN DEL PANEL NÚCLEO (merge puro de fuentes)")
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
      "26 departamentos",
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

check(df.shape[1] == 70,
      f"70 columnas (merge puro)",
      f"{df.shape[1]} columnas (esperadas 70) — revisar composición",
      is_warning=True)

# ─────────────────────────────────────────────
print("\n[2] SIN COLUMNAS DERIVADAS")

norm_cols = [c for c in df.columns if c.endswith("_norm")]
check(len(norm_cols) == 0,
      "Sin columnas _norm",
      f"{len(norm_cols)} columnas _norm: {norm_cols[:5]}")

derived_cols = [c for c in df.columns if c in (
    ["rinde_lag1","rinde_lag2","rinde_lag3","rinde_lag4","rinde_lag5",
     "rinde_ma5","rinde_std5","z_rinde","es_anomala_train"]
)]
check(len(derived_cols) == 0,
      "Sin columnas de feature engineering (lags/z_rinde/es_anomala_train)",
      f"Columnas derivadas presentes: {derived_cols}")

# ─────────────────────────────────────────────
print("\n[3] SPLIT TEMPORAL")

check("split" in df.columns,
      "Columna split presente",
      "Columna split ausente")

if "split" in df.columns:
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
print("\n[4] MAGYP — RINDE (target)")

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
    elif cult == "maiz":
        check(2000 <= p50 <= 10000,
              f"Maíz median={p50:.0f} kg/ha OK",
              f"Maíz median={p50:.0f} fuera de rango 2000–10000")

check(df["rinde_kgha"].isna().sum() == 0,
      "Sin NaN en rinde_kgha",
      f"{df['rinde_kgha'].isna().sum()} NaN en rinde_kgha")

for col in ["sup_sembrada_ha","sup_cosechada_ha","produccion_tn"]:
    check(df[col].isna().sum() == 0,
          f"Sin NaN en {col}",
          f"{df[col].isna().sum()} NaN en {col}",
          is_warning=True)

# ─────────────────────────────────────────────
print("\n[5] NASA POWER")

nasa_cols = [c for c in df.columns if any(
    c.startswith(p) for p in ["t2m_","prectotcorr_","allsky_","rh2m_","ws2m_"]
)]
# 7 variables × 7 meses (sep–mar) = 49 columnas
check(len(nasa_cols) == 49,
      f"{len(nasa_cols)} columnas NASA POWER (esperadas 49)",
      f"{len(nasa_cols)} columnas NASA POWER (esperadas 49)",
      is_warning=True)

if nasa_cols:
    pct_nan = df[nasa_cols].isna().mean().mean()
    check(pct_nan < 0.10,
          f"NaN promedio NASA POWER: {pct_nan*100:.1f}%",
          f"NaN promedio NASA POWER alto: {pct_nan*100:.1f}%",
          is_warning=True)

    # Rango sanity check sobre t2m mensual (temperatura media, esperado 10–35 °C)
    t2m_cols = [c for c in nasa_cols if c.startswith("t2m_") and not c.startswith("t2m_m")]
    if t2m_cols:
        s = df[t2m_cols].stack().dropna()
        check(s.between(5, 40).mean() > 0.95,
              f"t2m mensual en rango [5,40]°C: {s.between(5,40).mean()*100:.0f}%",
              f"t2m mensual fuera de rango: revisar unidades")

# ─────────────────────────────────────────────
print("\n[6] NDVI")

ndvi_cols = [c for c in df.columns if c.startswith("ndvi")]
check(len(ndvi_cols) == 4,
      f"4 columnas NDVI: {ndvi_cols}",
      f"{len(ndvi_cols)} columnas NDVI (esperadas 4): {ndvi_cols}")

if "ndvi_mean" in df.columns:
    s = df["ndvi_mean"].dropna()
    check(s.between(0,1).mean() > 0.95,
          f"ndvi_mean en [0,1]: mean={s.mean():.3f}",
          "ndvi_mean fuera de rango [0,1]")
    cov = df["ndvi_mean"].notna().mean()
    check(cov > 0.40,
          f"Cobertura NDVI: {cov*100:.0f}% (MODIS desde 2002 — esperado ~48%)",
          f"Cobertura NDVI muy baja: {cov*100:.0f}%",
          is_warning=True)
    # Verificar que pre-2002 es todo NaN y post-2002 mayormente no-NaN
    pre2002 = df[df["campania_inicio"] < 2002]["ndvi_mean"]
    post2002 = df[df["campania_inicio"] >= 2002]["ndvi_mean"]
    check(pre2002.notna().sum() == 0,
          "NDVI pre-2002: 100% NaN (esperado, MODIS no existe)",
          f"NDVI pre-2002: {pre2002.notna().sum()} valores no-NaN inesperados",
          is_warning=True)
    check(post2002.notna().mean() > 0.85,
          f"NDVI post-2002: {post2002.notna().mean()*100:.0f}% cobertura",
          f"NDVI post-2002: solo {post2002.notna().mean()*100:.0f}% cobertura",
          is_warning=True)

# ─────────────────────────────────────────────
print("\n[7] ONI")

oni_cols = [c for c in df.columns if c.startswith("oni_")]
check(len(oni_cols) == 5,
      f"5 columnas ONI (oni_oct…oni_feb): {oni_cols}",
      f"{len(oni_cols)} columnas ONI (esperadas 5): {oni_cols}")

if "oni_oct" in df.columns:
    cov = df["oni_oct"].notna().mean()
    check(cov > 0.95,
          f"Cobertura ONI: {cov*100:.1f}%",
          f"Cobertura ONI baja: {cov*100:.1f}%")

    # Niñas conocidas: anomalía de enero debe ser ≤ -0.5
    for y, label in [(1988,"triple Niña"),(2008,"2008/09"),(2010,"triple Niña"),(2022,"sequía")]:
        rows = df[(df["campania_inicio"]==y) & df["oni_ene"].notna()]
        if len(rows):
            val = rows["oni_ene"].iloc[0]
            check(val <= -0.5,
                  f"{y}/{y+1} ({label}): oni_ene={val:.2f} ≤ -0.5",
                  f"{y}/{y+1}: oni_ene={val:.2f} > -0.5", is_warning=True)

    # ONI global (no varía por depto/cultivo)
    oni_unique_per_camp = df.groupby("campania_inicio")["oni_oct"].nunique()
    check((oni_unique_per_camp <= 1).all(),
          "ONI constante dentro de cada campaña (no varía por depto/cultivo)",
          "ONI varía dentro de una misma campaña — inconsistencia en el merge")
else:
    print(f"  {WARN} oni_oct no presente")

# ─────────────────────────────────────────────
print("\n[8] CONSISTENCIA GEOGRÁFICA")

check(df["lat_centroide"].notna().all() and df["lon_centroide"].notna().all(),
      "Todos los departamentos tienen centroide",
      "Hay departamentos sin centroide asignado")

check(df["lat_centroide"].between(-40, -30).mean() > 0.95,
      "Latitudes en rango región núcleo (-40,-30)",
      "Latitudes fuera de rango esperado")

check(df["lon_centroide"].between(-65, -58).mean() > 0.95,
      "Longitudes en rango región núcleo (-65,-58)",
      "Longitudes fuera de rango esperado")

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
    print(f"\n{PASS} Panel (merge puro de fuentes) listo.")
    print(f"  Shape: {df.shape}")
    print(f"  Columnas: {df.columns.tolist()}")
else:
    sys.exit(1)