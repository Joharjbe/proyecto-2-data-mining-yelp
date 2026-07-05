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

# ---------------------------------------------------------------
# Candado de reproducibilidad. Los resultados documentados (informe,
# README, figuras) se calcularon con numpy 1.26.4 (entorno 'yelp-dm').
# Con otra version, algunos numeros "al borde" (clustering, DBSCAN,
# diametro del grafo, benchmarks) cambian. Cortamos con un mensaje
# claro en vez de dar resultados distintos en silencio.
# Para permitir otra version a proposito: export YELPDM_SKIP_VERSION_CHECK=1
# ---------------------------------------------------------------
import os as _os
if not _os.environ.get("YELPDM_SKIP_VERSION_CHECK"):
    import numpy as _np
    if not _np.__version__.startswith("1.26"):
        raise RuntimeError(
            f"Entorno equivocado: numpy {_np.__version__}. Este proyecto requiere "
            "numpy 1.26.4 (entorno 'yelp-dm'). En VSCode usa 'Select Kernel' -> "
            "yelp-dm. Crea el entorno con: conda env create -f environment.yml  (o "
            "pip install -r requirements.txt). Para omitir este chequeo a proposito: "
            "export YELPDM_SKIP_VERSION_CHECK=1"
        )

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
