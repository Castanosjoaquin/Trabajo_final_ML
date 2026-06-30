"""
validate_panel_chirps.py — Validación del panel_nucleo.parquet generado por build_panel_chirps.py
(merge puro de fuentes + columnas CHIRPS).

Copia extendida de validate_panel.py. NO modifica el validador original.

Checks:
  1. Completitud: departamentos, cultivos, campañas, nº de columnas (base + CHIRPS)
  2. Sin columnas derivadas (feature engineering vive en notebooks)
  3. Source-only: el ETL no impone split (se define en notebooks)
  4. CHIRPS no agrega ni pierde filas: misma grilla depto × campaña × cultivo que MAGyP
  5. Distribución del rinde (MAGyP)
  6. NASA POWER: cobertura y rangos
  7. NDVI: cobertura y rangos
  8. ONI: campañas Niña conocidas
  9. CHIRPS diario: cobertura temporal, % NaN por año, 26 centroides, sanity físico
 10. CHIRPS panel: columnas, rangos, correlación con prectotcorr_* de NASA por mes
 11. Consistencia geográfica

Uso:
    python validate_panel_chirps.py
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

# nº de columnas base del panel (sin CHIRPS) = 69; CHIRPS agrega 7 (sep..mar) = 76
N_COLS_BASE   = 69
N_COLS_CHIRPS = 7
N_COLS_TOTAL  = N_COLS_BASE + N_COLS_CHIRPS

MESES = ["sep", "oct", "nov", "dic", "ene", "feb", "mar"]


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
print("VALIDACIÓN DEL PANEL NÚCLEO + CHIRPS")
print("="*60)

panel_path = PROC / "panel_nucleo.parquet"
if not panel_path.exists():
    print(f"\n{FAIL} No se encontró {panel_path}. Corré build_panel_chirps.py primero.")
    sys.exit(1)

df = pd.read_parquet(panel_path)
print(f"\nPanel cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

chirps_cols = [c for c in df.columns if c.startswith("chirps_")]

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

check(df.shape[1] == N_COLS_TOTAL,
      f"{N_COLS_TOTAL} columnas (base {N_COLS_BASE} + {N_COLS_CHIRPS} CHIRPS)",
      f"{df.shape[1]} columnas (esperadas {N_COLS_TOTAL}) — revisar composición",
      is_warning=True)

# ─────────────────────────────────────────────
print("\n[2] SIN COLUMNAS DERIVADAS")

norm_cols = [c for c in df.columns if c.endswith("_norm")]
check(len(norm_cols) == 0,
      "Sin columnas _norm",
      f"{len(norm_cols)} columnas _norm: {norm_cols[:5]}")

derived_cols = [c for c in df.columns if c in (
    ["rinde_lag1","rinde_lag2","rinde_lag3","rinde_lag4","rinde_lag5",
     "rinde_ma5","rinde_std5","z_rinde","es_anomala_train",
     # CHIRPS derivados que NO deben estar en el panel (van en notebooks)
     "spi","spi_3","spi_6","chirps_dias_secos","chirps_z"]
)]
check(len(derived_cols) == 0,
      "Sin columnas de feature engineering (lags/z_rinde/SPI/días secos)",
      f"Columnas derivadas presentes: {derived_cols}")

# ─────────────────────────────────────────────
print("\n[3] SOURCE-ONLY (sin split en el ETL)")

check("split" not in df.columns,
      "Sin columna split (se define en los notebooks de modelado)",
      "Columna split presente — el ETL debería ser source-only",
      is_warning=True)

# ─────────────────────────────────────────────
print("\n[4] CHIRPS NO AGREGA NI PIERDE FILAS")

magyp_path = PROC / "fuente_magyp.parquet"
if magyp_path.exists():
    magyp = pd.read_parquet(magyp_path)
    keys_panel = set(map(tuple, df[["cultivo","departamento","campania_inicio"]].values))
    keys_magyp = set(map(tuple, magyp[["cultivo","departamento","campania_inicio"]].values))
    check(len(df) == len(magyp),
          f"Mismo nº de filas que MAGyP ({len(df)})",
          f"Filas panel={len(df)} ≠ MAGyP={len(magyp)} — CHIRPS alteró la grilla")
    check(keys_panel == keys_magyp,
          "Misma grilla (cultivo×depto×campaña) que MAGyP — CHIRPS no agregó/perdió filas",
          f"Grilla difiere de MAGyP: {len(keys_panel ^ keys_magyp)} claves distintas")
else:
    print(f"  {WARN} fuente_magyp.parquet no encontrado — no se pudo comparar la grilla")
    warnings.append("fuente_magyp.parquet ausente para chequear filas")

# ─────────────────────────────────────────────
print("\n[5] MAGYP — RINDE (target)")

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
print("\n[6] NASA POWER")

nasa_cols = [c for c in df.columns if any(
    c.startswith(p) for p in ["t2m_","prectotcorr_","allsky_","rh2m_","ws2m_"]
)]
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

    t2m_cols = [c for c in nasa_cols if c.startswith("t2m_") and not c.startswith("t2m_m")]
    if t2m_cols:
        s = df[t2m_cols].stack().dropna()
        check(s.between(5, 40).mean() > 0.95,
              f"t2m mensual en rango [5,40]°C: {s.between(5,40).mean()*100:.0f}%",
              f"t2m mensual fuera de rango: revisar unidades")

# ─────────────────────────────────────────────
print("\n[7] NDVI")

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
print("\n[8] ONI")

oni_cols = [c for c in df.columns if c.startswith("oni_")]
check(len(oni_cols) == 5,
      f"5 columnas ONI (oni_oct…oni_feb): {oni_cols}",
      f"{len(oni_cols)} columnas ONI (esperadas 5): {oni_cols}")

if "oni_oct" in df.columns:
    cov = df["oni_oct"].notna().mean()
    check(cov > 0.95,
          f"Cobertura ONI: {cov*100:.1f}%",
          f"Cobertura ONI baja: {cov*100:.1f}%")

    for y, label in [(1988,"triple Niña"),(2008,"2008/09"),(2010,"triple Niña"),(2022,"sequía")]:
        rows = df[(df["campania_inicio"]==y) & df["oni_ene"].notna()]
        if len(rows):
            val = rows["oni_ene"].iloc[0]
            check(val <= -0.5,
                  f"{y}/{y+1} ({label}): oni_ene={val:.2f} ≤ -0.5",
                  f"{y}/{y+1}: oni_ene={val:.2f} > -0.5", is_warning=True)

    oni_unique_per_camp = df.groupby("campania_inicio")["oni_oct"].nunique()
    check((oni_unique_per_camp <= 1).all(),
          "ONI constante dentro de cada campaña (no varía por depto/cultivo)",
          "ONI varía dentro de una misma campaña — inconsistencia en el merge")
else:
    print(f"  {WARN} oni_oct no presente")

# ─────────────────────────────────────────────
print("\n[9] CHIRPS — fuente diaria cruda")

chirps_daily_path = PROC / "chirps_diario.parquet"
if chirps_daily_path.exists():
    cd = pd.read_parquet(chirps_daily_path)
    cd["fecha"] = pd.to_datetime(cd["fecha"])
    print(f"  chirps_diario: {len(cd)} filas | {cd['fecha'].min().date()} .. {cd['fecha'].max().date()}")

    check(cd["departamento"].nunique() == 26,
          "26 centroides en chirps_diario",
          f"{cd['departamento'].nunique()} centroides (esperados 26)")

    check(cd["fecha"].min().year <= 1981,
          f"CHIRPS arranca en {cd['fecha'].min().year} (≤1981)",
          f"CHIRPS arranca tarde: {cd['fecha'].min().year}")

    # Sanity físico: precip ≥ 0
    neg = (cd["precip_mm"] < 0).sum()
    check(neg == 0,
          "Sin precip negativa",
          f"{neg} valores de precip < 0")

    # Rango plausible para la pampa (precip diaria; máximos extremos < ~250 mm/día)
    s = cd["precip_mm"].dropna()
    check(s.max() < 400,
          f"Precip diaria máx plausible: {s.max():.1f} mm",
          f"Precip diaria máx sospechosa: {s.max():.1f} mm",
          is_warning=True)
    check(s.mean() < 15,
          f"Precip diaria media plausible: {s.mean():.2f} mm/día",
          f"Precip diaria media sospechosa: {s.mean():.2f} mm/día",
          is_warning=True)

    # % NaN por año (esperado ~0%, CHIRPS cubre desde 1981)
    nan_por_anio = cd.assign(anio=cd["fecha"].dt.year).groupby("anio")["precip_mm"].apply(
        lambda x: x.isna().mean() * 100)
    peor = nan_por_anio.max()
    print(f"  % NaN por año — máx: {peor:.2f}% (año {nan_por_anio.idxmax()})")
    check(peor < 5.0,
          f"% NaN por año bajo (máx {peor:.2f}%)",
          f"% NaN por año alto: {peor:.2f}%",
          is_warning=True)
else:
    print(f"  {WARN} chirps_diario.parquet no encontrado")
    warnings.append("chirps_diario.parquet ausente")

# ─────────────────────────────────────────────
print("\n[10] CHIRPS — agregado mensual y correlación con NASA")

expected_chirps = [f"chirps_precip_{m}" for m in MESES]
check(len(chirps_cols) == 7,
      f"7 columnas CHIRPS: {chirps_cols}",
      f"{len(chirps_cols)} columnas CHIRPS (esperadas 7): {chirps_cols}")
check(set(chirps_cols) == set(expected_chirps),
      "Naming chirps_precip_<mes> correcto (sep…mar)",
      f"Naming inesperado: {sorted(set(chirps_cols) ^ set(expected_chirps))}",
      is_warning=True)

if chirps_cols:
    s = df[chirps_cols].stack().dropna()
    check((s >= 0).all(),
          "Precip mensual CHIRPS ≥ 0",
          f"{(s < 0).sum()} valores de precip mensual < 0")
    cov = df[chirps_cols].notna().mean().mean()
    check(cov > 0.95,
          f"Cobertura CHIRPS panel: {cov*100:.1f}%",
          f"Cobertura CHIRPS panel baja: {cov*100:.1f}%",
          is_warning=True)

    # Correlación CHIRPS vs NASA por mes (esperada alta pero NO perfecta: fuentes distintas)
    print("  Correlación chirps_precip_<mes> vs prectotcorr_<mes> (fuentes independientes):")
    corrs = []
    for m in MESES:
        cc, nn = f"chirps_precip_{m}", f"prectotcorr_{m}"
        if cc in df.columns and nn in df.columns:
            pair = df[[cc, nn]].dropna()
            if len(pair) > 30:
                r = pair[cc].corr(pair[nn])
                corrs.append(r)
                print(f"    {m}: r={r:.3f}  (n={len(pair)})")
    if corrs:
        rmean = float(np.mean(corrs))
        check(0.5 < rmean < 0.999,
              f"Correlación media CHIRPS~NASA = {rmean:.3f} (alta pero no perfecta, OK)",
              f"Correlación media CHIRPS~NASA atípica: {rmean:.3f}",
              is_warning=True)

# ─────────────────────────────────────────────
print("\n[11] CONSISTENCIA GEOGRÁFICA")

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
    print(f"\n{PASS} Panel núcleo + CHIRPS listo.")
    print(f"  Shape: {df.shape}")
    print(f"  Columnas CHIRPS: {chirps_cols}")
else:
    sys.exit(1)
