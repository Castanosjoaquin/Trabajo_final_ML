"""Extracción de features ERA5-Land (estado de suelo + heladas) por departamento — GEE.

Saca, por (departamento, campaña), variables que los promedios mensuales de clima
NO capturan — las que el agrónomo señaló para el ~70% de anomalías sin firma
climática:
  - sm_planting     : humedad de suelo 0-100cm en la siembra (Sep–Nov).
  - sm_winter       : humedad de suelo 0-100cm en el invierno previo (Jun–Ago)
                      = condición inicial / inercia ("arrancar inundado o seco").
  - frost_days      : nº de días con helada (Tmin<0°C) en la campaña (Sep–Mar).
  - frost_days_early: nº de heladas tardías de primavera (Sep–Nov), las más dañinas.

Fuente: ECMWF/ERA5_LAND/DAILY_AGGR (reanálisis, diario, ~9 km, desde 1950).
Promediado/contado sobre el polígono de cada departamento (FAO GAUL 2015 nivel 2).

SALIDA (CSV en Drive) — una fila por (depto, campaña), `year` = campania_inicio:
    ADM1_NAME, ADM2_NAME, year, sm_planting, sm_winter, frost_days, frost_days_early

REQUISITOS / USO: idénticos a extract_avhrr_ndvi.py.
    python extract_era5.py --project TU_PROYECTO_EE --drive-folder era5
"""
from __future__ import annotations

import argparse

import ee

YEAR_MIN, YEAR_MAX = 1981, 2024
FREEZE_K = 273.15              # 0 °C en Kelvin
REDUCE_SCALE_M = 9000         # ~resolución nativa ERA5-Land
# Pesos de espesor para la humedad de raíz 0-100cm (capas de 7, 21, 72 cm).
_SW = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
       "volumetric_soil_water_layer_3"]
_W = [0.07, 0.21, 0.72]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrae features ERA5-Land por departamento (GEE)")
    ap.add_argument("--project", required=True, help="ID de tu proyecto de Earth Engine")
    ap.add_argument("--drive-folder", default="era5")
    ap.add_argument("--out-name", default="era5_departamentos")
    ap.add_argument("--country", default="Argentina")
    args = ap.parse_args()

    ee.Initialize(project=args.project)

    deptos = (ee.FeatureCollection("FAO/GAUL/2015/level2")
              .filter(ee.Filter.eq("ADM0_NAME", args.country)))
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

    def rootzone(col):
        """Humedad de suelo 0-100cm (media temporal, ponderada por espesor)."""
        img = col.select(_SW).mean()
        out = img.select(_SW[0]).multiply(_W[0])
        for b, w in zip(_SW[1:], _W[1:]):
            out = out.add(img.select(b).multiply(w))
        return out.rename("sm")

    def frost_count(col):
        """Nº de días con Tmin < 0°C (suma temporal del booleano por píxel)."""
        return col.select("temperature_2m_min").map(
            lambda im: im.lt(FREEZE_K)).sum().rename("frost")

    def campaign(year):
        y = ee.Number(year)
        d = lambda yy, m: ee.Date.fromYMD(yy, m, 1)
        sm_plant = rootzone(era5.filterDate(d(y, 9), d(y, 12))).rename("sm_planting")
        sm_wint = rootzone(era5.filterDate(d(y, 6), d(y, 9))).rename("sm_winter")
        frost = frost_count(era5.filterDate(d(y, 9), d(y.add(1), 4))).rename("frost_days")
        frost_e = frost_count(era5.filterDate(d(y, 9), d(y, 12))).rename("frost_days_early")
        img = sm_plant.addBands(sm_wint).addBands(frost).addBands(frost_e)
        fc = img.reduceRegions(collection=deptos, reducer=ee.Reducer.mean(),
                               scale=REDUCE_SCALE_M)
        return fc.map(lambda f: f.set({"year": y}))

    years = ee.List.sequence(YEAR_MIN, YEAR_MAX)
    result = ee.FeatureCollection(years.map(campaign)).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=result, description=args.out_name, folder=args.drive_folder,
        fileFormat="CSV",
        selectors=["ADM1_NAME", "ADM2_NAME", "year",
                   "sm_planting", "sm_winter", "frost_days", "frost_days_early"],
    )
    task.start()
    print(f"Export lanzado: '{args.out_name}' → Drive/{args.drive_folder}/")
    print("Estado: https://code.earthengine.google.com/tasks . Cuando termine, bajá el CSV y:")
    print("  python merge_era5.py <csv> --out data/processed/panel_union_era5.parquet")


if __name__ == "__main__":
    main()
