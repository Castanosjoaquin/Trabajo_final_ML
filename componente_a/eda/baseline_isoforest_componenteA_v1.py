"""
Baseline Isolation Forest — Componente A (Detección de Anomalías)
=================================================================
Entrena y evalúa Isolation Forest por separado para soja y maíz.
Principio source-only: todo feature engineering, splits y normalización
se computan aquí; el panel_nucleo.parquet NO se modifica.

Supuestos de diseño:
- Unidad de observación: departamento × campaña × cultivo.
- Etiqueta proxy (z_rinde < -1.5) usada SOLO para evaluación; el modelo
  nunca la ve en entrenamiento.
- Normalización z-score por departamento, parámetros calculados sobre
  filas normales de train únicamente (sin leakage).
- NDVI excluido del set base (USE_NDVI=False); activar para ablation.
"""

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
import warnings; warnings.filterwarnings("ignore")

PANEL_PATH    = "data/processed/panel_nucleo.parquet"
OUTPUT_DIR    = "outputs/baseline_isoforest_v1"
CULTIVOS      = ["soja", "maiz"]
USE_NDVI      = False          # Activar solo para ablation post-2002

# Split temporal (sobre campania_inicio, entero)
TRAIN_END     = 2017           # hasta 2017/18 inclusive
VAL_START     = 2018; VAL_END = 2020   # 2018/19 – 2020/21
TEST_START    = 2021           # 2021/22 – 2024/25

# Campañas excluidas de entrenamiento (>30 % deptos anómalos)
EXCLUDED_TRAIN_YEARS = [1988, 1996, 2008, 2017]

# Hiperparámetros a barrer
SWEEP_N_ESTIMATORS = [100, 200]
SWEEP_MAX_SAMPLES  = [64, 128, 256]

# Hiperparámetros finales
N_ESTIMATORS  = 100
RANDOM_STATE  = 42
CONTAMINATION = "auto"

# Ventana móvil para z_rinde
ROLLING_WINDOW = 5

# Umbral z-score para etiqueta anómala
Z_THRESH = -1.5

# =============================================================================
# IMPORTS
# =============================================================================
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve,
                             f1_score, precision_score, recall_score)
from itertools import product

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. CARGA
# =============================================================================
print(f"Cargando panel: {PANEL_PATH}")
if not os.path.exists(PANEL_PATH):
    raise FileNotFoundError(f"Panel no encontrado: {PANEL_PATH}")

panel = pd.read_parquet(PANEL_PATH)
print(f"  Shape: {panel.shape}")

# Columnas climáticas base: promedios mensuales Sep–Mar (NASA POWER + CHIRPS)
# Variables: radiación, precipitación NASA, humedad, T2m, T2m_max, T2m_min,
#            viento, ONI, precipitación CHIRPS. NO incluye rinde ni superficie.
MESES = ["sep", "oct", "nov", "dic", "ene", "feb", "mar"]

CLIM_PREFIXES = [
    "allsky_sfc_sw_dwn",  # radiación solar
    "prectotcorr",        # precipitación NASA POWER
    "rh2m",               # humedad relativa
    "t2m",                # temperatura media
    "t2m_max",            # temperatura máxima
    "t2m_min",            # temperatura mínima
    "ws2m",               # viento
    "oni",                # ONI (ENSO) – solo meses disponibles
    "chirps_precip",      # precipitación CHIRPS
]

# Construir lista de columnas esperadas
expected_cols = []
for pfx in CLIM_PREFIXES:
    for mes in MESES:
        col = f"{pfx}_{mes}"
        if col in panel.columns:
            expected_cols.append(col)

# ONI tiene solo dic/ene/feb/nov/oct en el panel — lo que exista quedará incluido
BASE_FEATURE_COLS = expected_cols

if USE_NDVI:
    ndvi_cols = [c for c in ["ndvi_mean", "ndvi_min", "ndvi_max", "ndvi_anomalia_pct"]
                 if c in panel.columns]
    BASE_FEATURE_COLS = BASE_FEATURE_COLS + ndvi_cols

print(f"\nFeatures base ({len(BASE_FEATURE_COLS)} columnas):")
for c in BASE_FEATURE_COLS:
    print(f"  {c}")

# Verificar columnas clave
KEY_COLS = ["cultivo", "campania", "campania_inicio", "departamento",
            "rinde_kgha"] + BASE_FEATURE_COLS
missing = [c for c in KEY_COLS if c not in panel.columns]
if missing:
    raise ValueError(
        f"Columnas faltantes en el panel: {missing}\n"
        f"Columnas disponibles: {list(panel.columns)}"
    )

# =============================================================================
# 3. ETIQUETA PROXY (z_rinde por depto Y cultivo, ventana móvil 5 años)
#    Se calcula sobre todo el panel antes de filtrar por cultivo.
# =============================================================================
def compute_z_rinde(df, window=ROLLING_WINDOW):
    """z-score del rinde respecto a media móvil por departamento Y cultivo.
    Se agrupa por [departamento, cultivo] para no mezclar las series de soja
    (~2700 kg/ha) y maíz (~6700 kg/ha) del mismo departamento: mezclarlas
    introduce un sesgo direccional (el cultivo de mayor rinde queda siempre
    'sobre el promedio' y el de menor rinde 'debajo'), lo que sobre-marca soja
    y deja maíz con 0 anomalías."""
    df = df.sort_values(["departamento", "cultivo", "campania_inicio"]).copy()
    grp = df.groupby(["departamento", "cultivo"])["rinde_kgha"]
    roll_mean = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    roll_std  = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=3).std())
    df["z_rinde"]  = (df["rinde_kgha"] - roll_mean) / roll_std.replace(0, np.nan)
    df["anomalia"] = (df["z_rinde"] < Z_THRESH).astype(int)
    return df

panel = compute_z_rinde(panel)
print(f"\nEtiqueta proxy: {panel['anomalia'].sum()} filas anómalas / {len(panel)} total")

# =============================================================================
# Almacenamiento de resultados globales
# =============================================================================
all_results = []

# =============================================================================
# LOOP POR CULTIVO
# =============================================================================
for cultivo in CULTIVOS:
    print(f"\n{'='*60}")
    print(f"CULTIVO: {cultivo.upper()}")
    print(f"{'='*60}")

    df = panel[panel["cultivo"] == cultivo].copy()
    print(f"  Filas: {len(df)}")

    # =========================================================================
    # 4. SPLIT TEMPORAL
    # =========================================================================
    mask_train = df["campania_inicio"] <= TRAIN_END
    mask_val   = (df["campania_inicio"] >= VAL_START) & (df["campania_inicio"] <= VAL_END)
    mask_test  = df["campania_inicio"] >= TEST_START

    df_train = df[mask_train].copy()
    df_val   = df[mask_val].copy()
    df_test  = df[mask_test].copy()

    print(f"  Train: {len(df_train)} filas | Val: {len(df_val)} | Test: {len(df_test)}")

    # Filas normales de train: excluir años problemáticos Y filas con etiqueta anómala
    mask_excl = df_train["campania_inicio"].isin(EXCLUDED_TRAIN_YEARS)
    mask_anom = df_train["anomalia"] == 1
    df_train_normal = df_train[~mask_excl & ~mask_anom].copy()
    print(f"  Train (normal): {len(df_train_normal)} filas "
          f"(excluidos {mask_excl.sum()} por año + {(~mask_excl & mask_anom).sum()} anómalos)")

    # =========================================================================
    # 5. NORMALIZACIÓN z-score POR DEPARTAMENTO
    #    Parámetros calculados solo sobre train_normal → aplicados a val/test.
    # =========================================================================
    norm_stats = (df_train_normal
                  .groupby("departamento")[BASE_FEATURE_COLS]
                  .agg(["mean", "std"]))
    # Aplanar MultiIndex de columnas
    norm_stats.columns = [f"{col}_{stat}" for col, stat in norm_stats.columns]

    def normalize(df_in):
        df_out = df_in.copy()
        for col in BASE_FEATURE_COLS:
            mean_col = f"{col}_mean"
            std_col  = f"{col}_std"
            merged   = df_out[["departamento"]].merge(
                norm_stats[[mean_col, std_col]].reset_index(),
                on="departamento", how="left"
            )
            std_vals = merged[std_col].replace(0, np.nan).values
            df_out[col] = (df_out[col].values - merged[mean_col].values) / std_vals
        return df_out

    df_train_n = normalize(df_train_normal)
    df_val_n   = normalize(df_val)
    df_test_n  = normalize(df_test)

    # Dropear NaN en features (deptos sin estadísticas en train)
    df_train_n = df_train_n.dropna(subset=BASE_FEATURE_COLS)
    df_val_n   = df_val_n.dropna(subset=BASE_FEATURE_COLS)
    df_test_n  = df_test_n.dropna(subset=BASE_FEATURE_COLS)

    X_train = df_train_n[BASE_FEATURE_COLS].values
    X_val   = df_val_n[BASE_FEATURE_COLS].values
    X_test  = df_test_n[BASE_FEATURE_COLS].values
    y_val   = df_val_n["anomalia"].values
    y_test  = df_test_n["anomalia"].values

    print(f"  Anómalas — val: {y_val.sum()} | test: {y_test.sum()}")

    # =========================================================================
    # 8d. BARRIDO DE HIPERPARÁMETROS (PR-AUC en val)
    # =========================================================================
    print("\n  Barrido de hiperparámetros...")
    sweep_rows = []
    for n_est, max_s in product(SWEEP_N_ESTIMATORS, SWEEP_MAX_SAMPLES):
        clf_sw = IsolationForest(
            n_estimators=n_est,
            max_samples=min(max_s, len(X_train)),
            max_features=1.0,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf_sw.fit(X_train)
        scores_sw = -clf_sw.score_samples(X_val)
        if y_val.sum() > 0:
            prauc_sw = average_precision_score(y_val, scores_sw)
        else:
            prauc_sw = np.nan
        sweep_rows.append({"n_estimators": n_est, "max_samples": max_s,
                            "prauc_val": prauc_sw})
        print(f"    n_est={n_est}, max_samples={max_s}: PR-AUC val={prauc_sw:.4f}")

    df_sweep = pd.DataFrame(sweep_rows)

    # Guardar heatmap de barrido
    sweep_pivot = df_sweep.pivot(index="n_estimators", columns="max_samples", values="prauc_val")
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(sweep_pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=sweep_pivot.values.min() - 0.01)
    ax.set_xticks(range(len(sweep_pivot.columns)))
    ax.set_xticklabels(sweep_pivot.columns)
    ax.set_yticks(range(len(sweep_pivot.index)))
    ax.set_yticklabels(sweep_pivot.index)
    ax.set_xlabel("max_samples"); ax.set_ylabel("n_estimators")
    ax.set_title(f"PR-AUC val — {cultivo}")
    for i in range(len(sweep_pivot.index)):
        for j in range(len(sweep_pivot.columns)):
            ax.text(j, i, f"{sweep_pivot.values[i,j]:.3f}", ha="center",
                    va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sweep_{cultivo}.png", dpi=120)
    plt.close()

    # =========================================================================
    # 6. MODELO FINAL
    # =========================================================================
    max_samples_final = min(256, len(X_train))
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=max_samples_final,
        max_features=1.0,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train)
    print(f"\n  Modelo entrenado: n_estimators={N_ESTIMATORS}, "
          f"max_samples={max_samples_final}")

    scores_val  = -clf.score_samples(X_val)
    scores_test = -clf.score_samples(X_test)

    # Guardar scores en los dataframes para uso posterior
    df_val_n  = df_val_n.copy(); df_val_n["score"]  = scores_val
    df_test_n = df_test_n.copy(); df_test_n["score"] = scores_test
    df_train_n = df_train_n.copy()
    df_train_n["score"] = -clf.score_samples(X_train)

    # =========================================================================
    # 7. EVALUACIÓN
    # =========================================================================

    # --- Calibración de umbral en val (maximiza F1) ---
    if y_val.sum() > 0:
        precs, recs, thrs = precision_recall_curve(y_val, scores_val)
        f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-9)
        best_idx = np.argmax(f1s)
        best_thr = thrs[best_idx]
        print(f"  Umbral calibrado en val: {best_thr:.4f} (F1={f1s[best_idx]:.4f})")
    else:
        best_thr = np.percentile(scores_val, 85)
        print("  Sin anómalos en val; umbral fijo en percentil 85.")

    def eval_split(scores, y_true, threshold, split_name):
        results = {"cultivo": cultivo, "split": split_name}
        if y_true.sum() == 0:
            print(f"  [{split_name}] Sin anómalos reales — métricas no definidas.")
            results.update({"PR-AUC": np.nan, "ROC-AUC": np.nan, "F1": np.nan,
                            "precision@k": np.nan})
            return results
        results["PR-AUC"]   = average_precision_score(y_true, scores)
        results["ROC-AUC"]  = roc_auc_score(y_true, scores)
        y_pred = (scores >= threshold).astype(int)
        results["F1"]       = f1_score(y_true, y_pred, zero_division=0)
        # precision@k: k = nº anomalías reales
        k = int(y_true.sum())
        top_k_idx = np.argsort(scores)[::-1][:k]
        hits = y_true[top_k_idx].sum()
        results["precision@k"] = hits / k
        print(f"\n  [{split_name}] PR-AUC={results['PR-AUC']:.4f}  "
              f"ROC-AUC={results['ROC-AUC']:.4f}  F1={results['F1']:.4f}  "
              f"precision@{k}={results['precision@k']:.4f}")
        return results

    res_val  = eval_split(scores_val,  y_val,  best_thr, "val")
    res_test = eval_split(scores_test, y_test, best_thr, "test")
    all_results.extend([res_val, res_test])

    # --- Recall estratificado (test) ---
    # Firma climática adversa: precipitación CHIRPS o NASA POWER baja (z < -0.5)
    precip_cols = [c for c in BASE_FEATURE_COLS
                   if "chirps_precip" in c or "prectotcorr" in c]
    if precip_cols and y_test.sum() > 0:
        mean_precip_z = df_test_n[precip_cols].mean(axis=1).values
        mask_clima_adv = (mean_precip_z < -0.5) & (y_test == 1)
        mask_otros     = (~(mean_precip_z < -0.5)) & (y_test == 1)
        y_pred_test = (scores_test >= best_thr).astype(int)
        def recall_subset(mask):
            if mask.sum() == 0: return np.nan
            return y_pred_test[mask].sum() / mask.sum()
        print(f"\n  Recall estratificado (test):")
        print(f"    Anomalías con firma climática adversa: "
              f"n={mask_clima_adv.sum()}, recall={recall_subset(mask_clima_adv):.3f}")
        print(f"    Anomalías sin firma climática adversa: "
              f"n={mask_otros.sum()}, recall={recall_subset(mask_otros):.3f}")

    # =========================================================================
    # 8. VISUALIZACIONES
    # =========================================================================

    # a) Distribución del score: normal vs anómala
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, ls in [(0, "steelblue", "-"), (1, "tomato", "--")]:
        mask = y_test == label
        lbl = "normal" if label == 0 else "anómala"
        ax.hist(scores_test[mask], bins=30, alpha=0.6, color=color,
                label=lbl, density=True, linestyle=ls)
    ax.axvline(best_thr, color="black", linestyle=":", label=f"umbral={best_thr:.3f}")
    ax.set_xlabel("Score de anomalía"); ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución score — {cultivo} (test)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/score_dist_{cultivo}.png", dpi=120)
    plt.close()

    # b) Curvas PR y ROC
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax_idx, (split_name, scores_s, y_s) in enumerate([
        ("val",  scores_val,  y_val),
        ("test", scores_test, y_test)
    ]):
        if y_s.sum() == 0:
            continue
        # PR
        precs_s, recs_s, _ = precision_recall_curve(y_s, scores_s)
        prauc_s = average_precision_score(y_s, scores_s)
        axes[0].plot(recs_s, precs_s, label=f"{split_name} (AUC={prauc_s:.3f})")
        # ROC
        fpr_s, tpr_s, _ = roc_curve(y_s, scores_s)
        rocauc_s = roc_auc_score(y_s, scores_s)
        axes[1].plot(fpr_s, tpr_s, label=f"{split_name} (AUC={rocauc_s:.3f})")

    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Curva PR — {cultivo}"); axes[0].legend()
    axes[1].plot([0,1],[0,1],"k--", alpha=0.4)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title(f"Curva ROC — {cultivo}"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/curves_{cultivo}.png", dpi=120)
    plt.close()

    # c) Top-10 campañas por score en test
    df_top = (df_test_n[["departamento", "campania", "z_rinde", "anomalia", "score"]]
              .sort_values("score", ascending=False)
              .head(10)
              .reset_index(drop=True))
    print(f"\n  Top-10 por score en test ({cultivo}):")
    print(df_top.to_string(index=True,
                           float_format=lambda x: f"{x:.3f}"))
    df_top.to_csv(f"{OUTPUT_DIR}/top10_{cultivo}.csv", index=False)

    # d) Heatmap departamento × campaña (test)
    hm_data = df_test_n.pivot_table(
        index="departamento", columns="campania", values="score", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(max(6, len(hm_data.columns)*1.2),
                                    max(4, len(hm_data)*0.35)))
    im = ax.imshow(hm_data.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(hm_data.columns)))
    ax.set_xticklabels(hm_data.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(hm_data.index)))
    ax.set_yticklabels(hm_data.index, fontsize=7)
    ax.set_title(f"Score anomalía (test) — {cultivo}")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_{cultivo}.png", dpi=120)
    plt.close()

# =============================================================================
# 9. TABLA RESUMEN Y CSV
# =============================================================================
print(f"\n{'='*60}")
print("TABLA RESUMEN FINAL")
print(f"{'='*60}")
df_results = pd.DataFrame(all_results)
print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
df_results.to_csv(f"{OUTPUT_DIR}/metrics_summary.csv", index=False)
print(f"\nOutputs guardados en: {OUTPUT_DIR}/")
