"""Extracción de NDVI-AVHRR (1981+) por departamento — Google Earth Engine.

Saca NDVI mensual (compuesto de máximo valor — MVC, el estándar para NDVI) de las
fuentes NOAA CDR NDVI: AVHRR v5 (1981–2013) + VIIRS v1 (2014+), ~5 km, mergeadas
para tener cobertura continua 1981–presente. Promediado sobre el polígono de cada
departamento de Argentina (FAO GAUL 2015 nivel 2). Meses de campaña Sep–Mar,
1981–2025. NOTA: hay un cambio de sensor AVHRR→VIIRS en 2013/14; la normalización
por depto absorbe parte del offset, pero es una limitación a tener presente.

POR QUÉ AVHRR: MODIS arranca en 2000, así que el NDVI que ya tenía el panel
recortaba el train a 2002+ (y eso solo, sin NDVI, ya costaba −0.08 PR-AUC). AVHRR
llega a 1981 → cubre toda la historia del panel sin recortar datos. Es la prueba
JUSTA de si el verdor vegetal ayuda con las anomalías "clima normal / rinde
anómalo".

SALIDA (CSV en Google Drive) — formato LARGO, una fila por (depto, año, mes):
    ADM1_NAME, ADM2_NAME, year, month, mean
donde `mean` = NDVI medio del depto en ese mes. El año/mes son CALENDARIO; el
script del repo `merge_avhrr_ndvi.py` los mapea a campaña (Sep–Dic→año Y,
Ene–Mar→año Y+1) y pivotea a columnas ndvi_avhrr_<mes>.

REQUISITOS:
    pip install earthengine-api
    earthengine authenticate          # una vez, abre el navegador
    # y tener un proyecto de Earth Engine habilitado.

USO:
    python extract_avhrr_ndvi.py --project TU_PROYECTO_EE --drive-folder ndvi_avhrr
El export corre en la nube (varios minutos); el CSV aparece en esa carpeta de Drive.
"""
from __future__ import annotations

import argparse

import ee


# Meses de campaña (Sep–Mar), por número de mes calendario.
MONTHS = [9, 10, 11, 12, 1, 2, 3]
YEAR_MIN, YEAR_MAX = 1981, 2025          # AVHRR v5 arranca jul-1981
NDVI_SCALE = 0.0001                       # factor de escala de la banda NDVI
REDUCE_SCALE_M = 5000                     # ~resolución nativa AVHRR (0.05°)


def build_periods() -> ee.List:
    """Lista server-side de {year, month} solo para meses Sep–Mar.

    Para mes ≥9 el año va 1981..2024; para mes ≤3 va 1982..2025 (la cola Ene–Mar
    de la última campaña). Así no se pide data fuera de rango."""
    periods = []
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        for m in MONTHS:
            # Ene–Mar pertenecen a la campaña que empezó el año anterior:
            # los incluimos para y>=YEAR_MIN+1; Sep–Dic para y<=YEAR_MAX-1.
            if m >= 9 and y > YEAR_MAX - 1:
                continue
            if m <= 3 and y < YEAR_MIN + 1:
                continue
            periods.append({"year": y, "month": m})
    return ee.List(periods)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrae NDVI-AVHRR por departamento (GEE)")
    ap.add_argument("--project", required=True, help="ID de tu proyecto de Earth Engine")
    ap.add_argument("--drive-folder", default="ndvi_avhrr",
                    help="Carpeta de Google Drive donde se exporta el CSV")
    ap.add_argument("--out-name", default="ndvi_avhrr_departamentos")
    ap.add_argument("--country", default="Argentina")
    args = ap.parse_args()

    ee.Initialize(project=args.project)

    # --- Polígonos de departamentos (GAUL nivel 2) del país ---
    deptos = (ee.FeatureCollection("FAO/GAUL/2015/level2")
              .filter(ee.Filter.eq("ADM0_NAME", args.country)))

    # --- Colección NDVI: AVHRR (1981–2013) + VIIRS (2014+) ---
    # Son el mismo producto NOAA CDR (banda 'NDVI', escala 0.0001, 0.05°),
    # diseñados para ser continuos. AVHRR solo no cubre el período de test
    # (2021+); por eso se mergean para tener 1981–presente sin huecos.
    avhrr = ee.ImageCollection("NOAA/CDR/AVHRR/NDVI/V5").select("NDVI")
    viirs = ee.ImageCollection("NOAA/CDR/VIIRS/NDVI/V1").select("NDVI")
    ndvi_col = avhrr.merge(viirs)

    # Imagen fully-masked de 1 banda para los meses sin datos (la guarda evita
    # el error Image.multiply '0 vs 1 bands' que rompía el export).
    empty_img = ee.Image.constant(0).rename("ndvi").updateMask(ee.Image.constant(0))

    def reduce_period(p):
        """Para un (year, month): compuesto de MÁXIMO valor del mes (MVC, reduce
        contaminación por nubes) → NDVI medio por departamento."""
        p = ee.Dictionary(p)
        year = ee.Number(p.get("year"))
        month = ee.Number(p.get("month"))
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        col = ndvi_col.filterDate(start, end)
        img = ee.Image(ee.Algorithms.If(
            col.size().gt(0),
            col.max().multiply(NDVI_SCALE).rename("ndvi"),   # MVC mensual
            empty_img,                                        # mes sin datos
        ))
        fc = img.reduceRegions(
            collection=deptos,
            reducer=ee.Reducer.mean(),
            scale=REDUCE_SCALE_M,
        )
        # adjunta year/month a cada feature (departamento)
        return fc.map(lambda f: f.set({"year": year, "month": month}))

    # --- Aplica a todos los períodos y aplana a una sola tabla ---
    fcs = build_periods().map(reduce_period)
    result = ee.FeatureCollection(fcs).flatten()

    # --- Export a Drive como CSV (solo las columnas necesarias) ---
    task = ee.batch.Export.table.toDrive(
        collection=result,
        description=args.out_name,
        folder=args.drive_folder,
        fileFormat="CSV",
        selectors=["ADM1_NAME", "ADM2_NAME", "year", "month", "mean"],
    )
    task.start()
    print(f"Export lanzado: '{args.out_name}' → Drive/{args.drive_folder}/")
    print("Seguí el estado en https://code.earthengine.google.com/tasks "
          "(o `earthengine task list`). Tarda varios minutos.")
    print("Cuando termine, bajá el CSV y corré: "
          "python merge_avhrr_ndvi.py <csv> --out data/processed/panel_union_ndvi.parquet")


if __name__ == "__main__":
    main()
