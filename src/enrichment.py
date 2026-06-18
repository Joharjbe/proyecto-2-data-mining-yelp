"""Descarga y normalización de las fuentes complementarias.

Fuentes (justificación de la selección en README, sección "Diseño del estudio"):
  1. Census ACS 5-year 2022 (api.census.gov)  -> demografía por ZCTA (~ZIP)
  2. COVID-19 del New York Times (GitHub)     -> casos/muertes por county y día
  3. Crosswalk ZCTA<->county del Census       -> puente geográfico ZIP -> county
  4. Feriados de EE.UU. (paquete `holidays`)  -> calendario federal + estatal
     (incluye Mardi Gras para Louisiana, relevante para New Orleans)

Todas las descargas son idempotentes: si el archivo ya existe, se omite.
Uso:  python -m src.enrichment   (o llamar descargar_todo() desde un notebook)
"""
import csv
import io
import os

import requests

from .config import EXTERNAL, ROOT, ensure_dirs

_UA = {"User-Agent": "proyecto-academico-datamining-utec/1.0"}


def _census_key() -> str:
    """API key del Census (obligatoria desde 2025 para api.census.gov).

    Se busca en: (1) variable de entorno CENSUS_API_KEY, (2) archivo
    `codigo/.census_key` (una sola línea con la key; NO se versiona).
    Registro gratuito e inmediato: https://api.census.gov/data/key_signup.html
    """
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    archivo = ROOT / ".census_key"
    if not key and archivo.exists():
        key = archivo.read_text().strip()
    if not key:
        raise RuntimeError(
            "Falta la API key del Census.\n"
            "  1) Solicítala (gratis, llega al correo): https://api.census.gov/data/key_signup.html\n"
            "  2) Guárdala en el archivo codigo/.census_key (una sola línea)\n"
            "     o exporta CENSUS_API_KEY antes de abrir Jupyter/VSCode."
        )
    return key

# --- 1. Census ACS 5-year (2022) por ZCTA --------------------------------
# B19013_001E mediana de ingreso del hogar | B01003_001E población
# B15003: universo 25+ años; 022..025 = bachelor/master/profesional/doctorado
# B25064_001E renta bruta mediana
_ACS_VARS = "B19013_001E,B01003_001E,B15003_001E,B15003_022E,B15003_023E,B15003_024E,B15003_025E,B25064_001E"
ACS_URL = (
    "https://api.census.gov/data/2022/acs/acs5"
    f"?get={_ACS_VARS}&for=zip%20code%20tabulation%20area:*"
)

# --- 2. COVID-19 NYT (un CSV por año; el dataset de Yelp corta en ene-2022)
NYT_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-{anio}.csv"

# --- 3. Relación ZCTA <-> county (Census 2020, delimitado por '|') --------
ZCTA_COUNTY_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
    "tab20_zcta520_county20_natl.txt"
)


def _existe(nombre: str) -> bool:
    return (EXTERNAL / nombre).exists()


def _guardar_stream(url: str, nombre: str) -> None:
    """Descarga por streaming (archivos grandes) hacia bronze/external."""
    with requests.get(url, stream=True, timeout=120, headers=_UA) as r:
        r.raise_for_status()
        with open(EXTERNAL / nombre, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def descargar_acs() -> str:
    """ACS por ZCTA -> acs_zcta.csv con indicadores listos para cruzar."""
    nombre = "acs_zcta.csv"
    if _existe(nombre):
        return f"{nombre} ya existe (omitido)"
    r = requests.get(ACS_URL + f"&key={_census_key()}", timeout=120, headers=_UA)
    if r.status_code != 200 or not r.text.lstrip().startswith("["):
        raise RuntimeError(
            f"El API del Census respondió {r.status_code} con contenido inesperado "
            f"(¿key inválida?): {r.text[:200]}"
        )
    filas = r.json()
    cab, datos = filas[0], filas[1:]
    idx = {c: i for i, c in enumerate(cab)}

    def val(fila, col):
        v = fila[idx[col]]
        # El API marca "sin dato" con centinelas negativos enormes
        return None if v is None or str(v).startswith("-6666") else float(v)

    with open(EXTERNAL / nombre, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zcta", "mediana_ingreso", "poblacion", "pct_universitarios", "renta_mediana"])
        for fila in datos:
            pob25 = val(fila, "B15003_001E")
            uni = sum(val(fila, c) or 0 for c in
                      ("B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"))
            w.writerow([
                fila[idx["zip code tabulation area"]],
                val(fila, "B19013_001E"),
                val(fila, "B01003_001E"),
                round(uni / pob25, 4) if pob25 else None,
                val(fila, "B25064_001E"),
            ])
    return f"{nombre} descargado ({len(datos):,} ZCTAs)"


def descargar_covid() -> list[str]:
    """COVID NYT 2020–2022 (el corte de Yelp es 19-ene-2022)."""
    out = []
    for anio in (2020, 2021, 2022):
        nombre = f"covid_nyt_{anio}.csv"
        if _existe(nombre):
            out.append(f"{nombre} ya existe (omitido)")
            continue
        _guardar_stream(NYT_URL.format(anio=anio), nombre)
        out.append(f"{nombre} descargado")
    return out


def descargar_crosswalk() -> str:
    """Relación ZCTA<->county; nos quedamos con el county de mayor solape."""
    nombre = "zcta_county.csv"
    if _existe(nombre):
        return f"{nombre} ya existe (omitido)"
    r = requests.get(ZCTA_COUNTY_URL, timeout=300, headers=_UA)
    r.raise_for_status()
    mejor: dict[str, tuple[str, float]] = {}
    lector = csv.DictReader(io.StringIO(r.text), delimiter="|")
    for fila in lector:
        z = fila["GEOID_ZCTA5_20"]
        if not z:
            continue
        area = float(fila["AREALAND_PART"] or 0)
        county = fila["GEOID_COUNTY_20"]
        if z not in mejor or area > mejor[z][1]:
            mejor[z] = (county, area)
    with open(EXTERNAL / nombre, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zcta", "county_fips"])
        for z, (county, _) in sorted(mejor.items()):
            w.writerow([z, county])
    return f"{nombre} generado ({len(mejor):,} ZCTAs)"


def generar_feriados(desde: int = 2005, hasta: int = 2022) -> str:
    """Feriados federales + estatales (PA/FL/LA) con `holidays`.
    Louisiana incluye Mardi Gras, clave para la estacionalidad de New Orleans."""
    import holidays

    nombre = "feriados.csv"
    if _existe(nombre):
        return f"{nombre} ya existe (omitido)"
    anios = range(desde, hasta + 1)
    filas = []
    for ambito, cal in [
        ("US", holidays.US(years=anios)),
        ("PA", holidays.US(subdiv="PA", years=anios)),
        ("FL", holidays.US(subdiv="FL", years=anios)),
        ("LA", holidays.US(subdiv="LA", years=anios)),
    ]:
        for fecha, nombre_f in sorted(cal.items()):
            filas.append((fecha.isoformat(), nombre_f, ambito))
    with open(EXTERNAL / nombre, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fecha", "feriado", "ambito"])
        w.writerows(filas)
    return f"{nombre} generado ({len(filas):,} filas)"


def descargar_todo() -> None:
    ensure_dirs()
    print(descargar_acs())
    for msg in descargar_covid():
        print(msg)
    print(descargar_crosswalk())
    print(generar_feriados())


if __name__ == "__main__":
    descargar_todo()
