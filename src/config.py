"""Configuración central de rutas y constantes del proyecto.

Arquitectura medallón:
    bronze: JSON crudos de Yelp + CSV crudos de fuentes externas
    silver: Parquet limpio y tipado
    gold:   tablas listas por análisis (grafos, matrices, features)
"""
from pathlib import Path

# Raíz = carpeta codigo/ (este archivo vive en codigo/src/)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
EXTERNAL = BRONZE / "external"  # ACS, COVID, feriados, crosswalk HUD

# Archivos crudos de Yelp
YELP = {
    "business": BRONZE / "yelp_academic_dataset_business.json",
    "review": BRONZE / "yelp_academic_dataset_review.json",
    "user": BRONZE / "yelp_academic_dataset_user.json",
    "checkin": BRONZE / "yelp_academic_dataset_checkin.json",
    "tip": BRONZE / "yelp_academic_dataset_tip.json",
}

# Semilla global para reproducibilidad (muestreos, k-means++, etc.)
SEED = 42

def ensure_dirs() -> None:
    """Crea las carpetas del medallón si no existen."""
    for d in (BRONZE, SILVER, GOLD, EXTERNAL):
        d.mkdir(parents=True, exist_ok=True)

def spark_session(app: str = "yelp-dm", driver_mem: str = "8g"):
    """SparkSession local optimizada para el M1 Pro (un solo nodo).

    driver_mem: en local mode todo corre en el driver; 8g deja espacio
    al resto del sistema. Subir a 12g si hay presión de memoria.
    """
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app)
        .master("local[*]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.host", "127.0.0.1")  # silencia WARN de loopback
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.debug.maxToStringFields", "200")
        .getOrCreate()
    )


# ---------------------------------------------------------------
# Mapeo estado -> área metropolitana del Yelp Open Dataset (v2022)
# (1 metro por estado; PA absorbe suburbios de NJ/DE, St. Louis los de IL)
# ---------------------------------------------------------------
METRO_POR_ESTADO = {
    "PA": "Philadelphia", "NJ": "Philadelphia", "DE": "Philadelphia",
    "FL": "Tampa",
    "AZ": "Tucson",
    "TN": "Nashville",
    "IN": "Indianapolis",
    "LA": "New Orleans",
    "MO": "Saint Louis", "IL": "Saint Louis",
    "NV": "Reno",
    "CA": "Santa Barbara",
    "ID": "Boise",
    "AB": "Edmonton (CA)",
}
