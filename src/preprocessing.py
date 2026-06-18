"""Capa de ingeniería de datos: ingesta bronze -> silver (Spark -> Parquet).

NOTA METODOLÓGICA: aquí NO hay algoritmos de minería (esos van a mano en
los demás módulos). Esto es limpieza/tipado, permitido por las reglas del
curso como manipulación de datos.

Decisiones de limpieza documentadas (van al informe, Parte I):
  - Esquemas explícitos en archivos grandes: evita doble lectura por inferencia.
  - review.date / user.yelping_since / checkin -> timestamp UTC.
  - user.friends: string "a, b, c" -> array<string> (base del grafo social).
  - user.elite: corrige el bug histórico "20,20" (=2020) y castea a array<int>.
  - checkin: explode a una fila por evento (base de la Parte V, streams).
  - business.categories: string -> array<string> con trim.
"""
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from .config import SILVER, YELP, METRO_POR_ESTADO

# ----------------------- esquemas explícitos -----------------------

SCHEMA_REVIEW = T.StructType([
    T.StructField("review_id", T.StringType()),
    T.StructField("user_id", T.StringType()),
    T.StructField("business_id", T.StringType()),
    T.StructField("stars", T.DoubleType()),
    T.StructField("useful", T.LongType()),
    T.StructField("funny", T.LongType()),
    T.StructField("cool", T.LongType()),
    T.StructField("text", T.StringType()),
    T.StructField("date", T.StringType()),
])

_COMPLIMENTS = [
    "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute",
    "compliment_list", "compliment_note", "compliment_plain", "compliment_cool",
    "compliment_funny", "compliment_writer", "compliment_photos",
]

SCHEMA_USER = T.StructType(
    [
        T.StructField("user_id", T.StringType()),
        T.StructField("name", T.StringType()),
        T.StructField("review_count", T.LongType()),
        T.StructField("yelping_since", T.StringType()),
        T.StructField("useful", T.LongType()),
        T.StructField("funny", T.LongType()),
        T.StructField("cool", T.LongType()),
        T.StructField("elite", T.StringType()),
        T.StructField("friends", T.StringType()),
        T.StructField("fans", T.LongType()),
        T.StructField("average_stars", T.DoubleType()),
    ]
    + [T.StructField(c, T.LongType()) for c in _COMPLIMENTS]
)

SCHEMA_TIP = T.StructType([
    T.StructField("user_id", T.StringType()),
    T.StructField("business_id", T.StringType()),
    T.StructField("text", T.StringType()),
    T.StructField("date", T.StringType()),
    T.StructField("compliment_count", T.LongType()),
])

SCHEMA_CHECKIN = T.StructType([
    T.StructField("business_id", T.StringType()),
    T.StructField("date", T.StringType()),
])

# ----------------------- transformaciones silver -----------------------

def silver_business(spark: SparkSession) -> DataFrame:
    """business: tipado + categories a array + metro derivado del estado."""
    metro = F.create_map([F.lit(x) for kv in METRO_POR_ESTADO.items() for x in kv])
    df = spark.read.json(str(YELP["business"]))  # 114 MB: inferir es barato (attributes anidado)
    return (
        df.withColumn(
            "categories",
            F.when(
                F.col("categories").isNotNull(),
                F.transform(F.split("categories", ","), lambda c: F.trim(c)),
            ),
        )
        .withColumn("metro", F.coalesce(metro[F.col("state")], F.lit("OTROS")))
        .withColumn("es_restaurante", F.array_contains(F.coalesce("categories", F.array()), "Restaurants"))
    )


def silver_review(spark: SparkSession) -> DataFrame:
    return (
        spark.read.json(str(YELP["review"]), schema=SCHEMA_REVIEW)
        .withColumn("date", F.to_timestamp("date"))
    )


def silver_user(spark: SparkSession) -> DataFrame:
    """user: friends->array, elite->array<int> (corrigiendo '20,20'≡2020)."""
    df = spark.read.json(str(YELP["user"]), schema=SCHEMA_USER)
    sin_amigos = (F.col("friends").isNull()) | (F.col("friends") == "None") | (F.col("friends") == "")
    elite_fix = F.regexp_replace(F.coalesce("elite", F.lit("")), "20,20", "2020")
    return (
        df.withColumn("yelping_since", F.to_timestamp("yelping_since"))
        .withColumn(
            "friends",
            F.when(sin_amigos, F.array().cast("array<string>"))
            .otherwise(F.transform(F.split("friends", ","), lambda x: F.trim(x))),
        )
        .withColumn("n_amigos", F.size("friends"))
        .withColumn(
            "elite_years",
            F.when(elite_fix == "", F.array().cast("array<int>"))
            .otherwise(F.transform(F.split(elite_fix, ","), lambda x: F.trim(x).cast("int"))),
        )
        .withColumn("n_elite", F.size("elite_years"))
        .drop("elite")
    )


def silver_tip(spark: SparkSession) -> DataFrame:
    return (
        spark.read.json(str(YELP["tip"]), schema=SCHEMA_TIP)
        .withColumn("date", F.to_timestamp("date"))
    )


def silver_checkin(spark: SparkSession) -> DataFrame:
    """checkin: una fila por evento (formato 'tidy' para streams, Parte V)."""
    return (
        spark.read.json(str(YELP["checkin"]), schema=SCHEMA_CHECKIN)
        .withColumn("ts", F.explode(F.split("date", ",\\s*")))
        .withColumn("ts", F.to_timestamp("ts"))
        .select("business_id", "ts")
    )


# ----------------------- escritura -----------------------

TABLAS = {
    "business": silver_business,
    "review": silver_review,
    "user": silver_user,
    "tip": silver_tip,
    "checkin": silver_checkin,
}


def escribir_silver(spark: SparkSession, tabla: str) -> str:
    """Materializa una tabla silver en Parquet (snappy). Devuelve la ruta."""
    out = SILVER / f"{tabla}.parquet"
    TABLAS[tabla](spark).write.mode("overwrite").parquet(str(out))
    return str(out)
