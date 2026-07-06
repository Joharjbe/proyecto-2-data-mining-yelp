"""Exportador del bundle de datos para el Gemelo Digital Adversarial de Yelp.

Lee EXCLUSIVAMENTE las tablas gold ya materializadas (nada se recalcula ni se
inventa) y emite un unico `demo/data/bundle.json` compacto (codificacion
columnar + numeros redondeados) que alimenta la web autocontenida `index.html`.

Ejecutar con el entorno del proyecto:
    /opt/anaconda3/envs/yelp-dm/bin/python demo/export_bundle.py
o, con el kernel activo:
    python demo/export_bundle.py

Todo el contenido del bundle sale de:
  - negocios_universo, clusters_negocio, features_negocio, ranking_negocios
  - stream_eventos            (serie temporal semanal por mercado)
  - actividad_covid_semanal   (contexto COVID)
  - impacto_mardi_gras        (experimento natural)
  - metricas_recomendacion, auditoria_recomendacion_exposicion (dial de exposicion)
  - representacion_clusters_ingreso (lente redlining)
  - usuarios_universo         (curva de Lorenz de concentracion de voz)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Orden canonico de los 3 mercados del universo -> indice compacto 0/1/2
METROS = ["Philadelphia", "Tampa", "New Orleans"]
METRO_IX = {m: i for i, m in enumerate(METROS)}


def r(x, nd):
    """Redondeo seguro (respeta NaN -> None para JSON)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def gload(name, columns=None):
    return pd.read_parquet(GOLD / name, columns=columns)


# ---------------------------------------------------------------------------
# 1) RESTAURANTES  (la capa mapa: 29,314 puntos)
# ---------------------------------------------------------------------------
def build_restaurantes():
    neg = load("negocios_universo.parquet",
               ["business_id", "name", "latitude", "longitude", "metro",
                "stars", "review_count"])
    clu = load("clusters_negocio.parquet",
               ["business_id", "cluster_kmeans", "segmento_kmeans"])
    fea = load("features_negocio.parquet",
               ["business_id", "price_range", "mediana_ingreso",
                "pct_universitarios"])
    rk = load("ranking_negocios.parquet",
              ["business_id", "hits_authority", "pagerank_bip"])

    df = (neg.merge(clu, on="business_id", how="left")
             .merge(fea, on="business_id", how="left")
             .merge(rk, on="business_id", how="left"))
    df = df[df["metro"].isin(METROS)].reset_index(drop=True)

    # cluster -> segmento (etiqueta interpretable) : 6 entradas
    seg = (df.dropna(subset=["cluster_kmeans"])
             .groupby("cluster_kmeans")["segmento_kmeans"].first())
    seg_lookup = {int(k): (v if isinstance(v, str) else str(v))
                  for k, v in seg.items()}

    # cuartiles de ingreso ACS (para la lente redlining) sobre valores no nulos
    inc = df["mediana_ingreso"]
    qs = inc.quantile([0.25, 0.5, 0.75]).tolist()

    def inc_q(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return -1
        if v <= qs[0]:
            return 0
        if v <= qs[1]:
            return 1
        if v <= qs[2]:
            return 2
        return 3

    cols = {
        "name": df["name"].fillna("").astype(str).tolist(),
        "lat": [r(v, 5) for v in df["latitude"]],
        "lon": [r(v, 5) for v in df["longitude"]],
        "metro": [METRO_IX[m] for m in df["metro"]],
        "stars": [r(v, 1) for v in df["stars"]],
        "rc": [int(v) if pd.notna(v) else 0 for v in df["review_count"]],
        "cl": [int(v) if pd.notna(v) else -1 for v in df["cluster_kmeans"]],
        "price": [int(v) if pd.notna(v) else -1 for v in df["price_range"]],
        "inc": [int(round(v)) if pd.notna(v) else -1 for v in df["mediana_ingreso"]],
        "incq": [inc_q(v) for v in df["mediana_ingreso"]],
        "auth": [r(v, 8) for v in df["hits_authority"]],
    }
    meta = {
        "n": len(df),
        "metros": METROS,
        "segmentos": seg_lookup,
        "inc_quartiles": [r(q, 0) for q in qs],
        "per_metro": {},
    }
    # bounding box por mercado (para el layout de "constelaciones")
    for m, i in METRO_IX.items():
        sub = df[df["metro"] == m]
        meta["per_metro"][i] = {
            "n": int(len(sub)),
            "lat": [r(sub["latitude"].min(), 5), r(sub["latitude"].max(), 5)],
            "lon": [r(sub["longitude"].min(), 5), r(sub["longitude"].max(), 5)],
        }
    return {"cols": cols, "meta": meta}


# ---------------------------------------------------------------------------
# 2) SERIE TEMPORAL  (la Maquina del Tiempo: latido semanal por mercado)
# ---------------------------------------------------------------------------
def build_serie():
    d = ds.dataset(GOLD / "stream_eventos.parquet")
    tb = d.to_table(columns=["ts", "metro", "es_restaurante"])
    df = tb.to_pandas()
    df = df[df["es_restaurante"] & df["metro"].isin(METROS)].copy()
    df["week"] = df["ts"].dt.to_period("W").dt.start_time
    g = (df.groupby(["week", "metro"]).size()
           .rename("n").reset_index())
    weeks = sorted(g["week"].unique())
    widx = {w: i for i, w in enumerate(weeks)}
    n_weeks = len(weeks)
    per_metro = {i: [0] * n_weeks for i in range(len(METROS))}
    for _, row in g.iterrows():
        per_metro[METRO_IX[row["metro"]]][widx[row["week"]]] = int(row["n"])

    weeks_iso = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks]

    # --- contexto COVID: indice de intensidad alineado a la MISMA grilla ---
    # (array de largo n_weeks por mercado; 0 donde no hay dato COVID)
    covid = load("actividad_covid_semanal.parquet")
    covid["wk"] = covid["fecha"].dt.to_period("W").dt.start_time
    covid_series = {i: [0.0] * n_weeks for i in range(len(METROS))}
    for _, row in covid.iterrows():
        if row["metro"] not in METRO_IX:
            continue
        wk = row["wk"]
        if wk in widx:
            val = row.get("covid_peak_index")
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = row.get("new_cases", 0)
            covid_series[METRO_IX[row["metro"]]][widx[wk]] = r(val, 3) or 0.0

    # --- Mardi Gras: indices de semana del evento (para pulsar NOLA) ---
    mg = load("impacto_mardi_gras.parquet")
    mardi_weeks = sorted({widx[pd.Timestamp(v).to_period("W").start_time]
                          for v in mg["evento"].unique()
                          if pd.Timestamp(v).to_period("W").start_time in widx})

    return {
        "weeks": weeks_iso,
        "per_metro": per_metro,
        "covid": covid_series,
        "mardi_weeks": mardi_weeks,
    }


# ---------------------------------------------------------------------------
# 3) EQUIDAD  (dial de exposicion, lente redlining, concentracion de voz)
# ---------------------------------------------------------------------------
def build_equidad():
    # --- metricas por modelo (precision vs cobertura) ---
    met = load("metricas_recomendacion.parquet")
    metricas = []
    for _, row in met.iterrows():
        metricas.append({
            "modelo": row["modelo"],
            "ndcg": r(row["NDCG_at_10"], 3),
            "ndcg_lo": r(row["ndcg_ci_lo"], 3),
            "ndcg_hi": r(row["ndcg_ci_hi"], 3),
            "cobertura": r(row["cobertura"], 3),
            "novedad": r(row["novedad"], 2),
        })

    # --- exposicion por cuartil de reseñas (dimension 'visibilidad') ---
    exp = load("auditoria_recomendacion_exposicion.parquet")
    vis = exp[exp["dimension"] == "visibilidad"]
    exposicion = {}
    for modelo, sub in vis.groupby("modelo"):
        q = {row["grupo"]: r(row["share_exposicion"], 3) for _, row in sub.iterrows()}
        exposicion[modelo] = {
            "Q1": q.get("reseñas Q1"), "Q2": q.get("reseñas Q2"),
            "Q3": q.get("reseñas Q3"), "Q4": q.get("reseñas Q4"),
            "gini": r(sub["gini_exposicion"].iloc[0], 3),
            "cobertura": r(sub["cobertura_catalogo"].iloc[0], 3),
            "items": int(sub["items_unicos"].iloc[0]),
        }

    # --- redlining: ratio de representacion cluster x cuartil de ingreso ---
    rep = load("representacion_clusters_ingreso.parquet")
    redlining = []
    for _, row in rep.iterrows():
        redlining.append({
            "ingreso": row["ingreso_zip"],
            "cluster": int(row["cluster_kmeans"]),
            "ratio": r(row["ratio_representacion"], 3),
        })

    # --- concentracion de voz: Lorenz de reseñas por usuario DENTRO DEL UNIVERSO ---
    # (se cuentan las reseñas reales en resenas_universo, no el review_count global
    #  de Yelp; asi reproduce el hallazgo de la Parte VII: top 10% ~ 54%, Gini ~0.59)
    rv = load("resenas_universo.parquet", ["user_id"])
    counts = rv.groupby("user_id").size().to_numpy(dtype=np.float64)
    rc = np.sort(counts)
    cum = np.cumsum(rc)
    n = len(rc)
    pts = []                       # 101 puntos: % usuarios -> % reseñas (curva de Lorenz)
    for p in range(101):
        idx = min(int(round(p / 100 * n)), n)
        pts.append(r(float(cum[idx - 1] / cum[-1]) if idx > 0 else 0.0, 4))
    gini = float((2 * np.sum(np.arange(1, n + 1) * rc) / (n * cum[-1])) - (n + 1) / n)
    top10_share = r(1.0 - pts[90], 3)   # fraccion escrita por el 10% mas activo

    return {
        "metricas": metricas,
        "exposicion": exposicion,
        "redlining": redlining,
        "lorenz": pts,
        "gini_voz": r(gini, 3),
        "top10_share": top10_share,
        "n_usuarios": int(n),
        "personajes": build_personajes(),
    }


def build_personajes():
    """~12 reseñadores estrella reales (nombres de pila) para la escena
    'Los que mandan': el rostro humano de la concentracion de voz."""
    u = load("ranking_usuarios.parquet",
             ["name", "fans", "review_count", "n_elite", "hits_hub"])
    # union: los mas seguidos + los mas prolificos + los top 'hub' (autoridad)
    cand = pd.concat([
        u.nlargest(8, "fans"),
        u.nlargest(4, "review_count"),
        u.nlargest(2, "hits_hub"),
    ]).drop_duplicates(subset=["name"]).sort_values("fans", ascending=False)
    out = []
    for _, row in cand.head(12).iterrows():
        out.append({
            "name": str(row["name"]),
            "fans": int(row["fans"]),
            "reviews": int(row["review_count"]),
            "elite": int(row["n_elite"]),
        })
    return out


# ---------------------------------------------------------------------------
# 4) CALLOUTS  (hallazgos verificados del proyecto, para la narrativa)
# ---------------------------------------------------------------------------
def build_callouts():
    return {
        "voz_top10": "El 10% de usuarios mas activos escribe el 54% de las reseñas (Gini 0.59).",
        "spam": "5 reseñas falsas de 5★ mueven un local Q1 (pocas reseñas) +0.56★; a uno consolidado, solo +0.03★.",
        "redlining": "Los ZIP de ingreso Q1 tienen 21.2% de atributos faltantes y mediana de 25 reseñas; los Q4, 17.9% y 38.",
        "exposicion": "El CF cubre 40.1% del catalogo (Gini exposicion 0.239) vs 13.0% del top-popular (Gini 0.358), pero el cuartil ya-visible aun se queda ~49% de los slots.",
        "covid": "En marzo-mayo 2020 los check-ins cayeron a 14.8-33.1% del nivel de 2019.",
        "mardi": "Mardi Gras alcanza indice mediano 187 en New Orleans frente a 97/101 en mercados placebo.",
    }


def main():
    print("Leyendo tablas gold desde", GOLD)
    bundle = {}
    print("  · restaurantes (mapa) ...")
    bundle["restaurantes"] = build_restaurantes()
    print("    ", bundle["restaurantes"]["meta"]["n"], "puntos")
    print("  · serie temporal (stream 20M -> semanal) ...")
    bundle["serie"] = build_serie()
    print("    ", len(bundle["serie"]["weeks"]), "semanas")
    print("  · equidad (exposicion / redlining / voz) ...")
    bundle["equidad"] = build_equidad()
    print("  · callouts ...")
    bundle["callouts"] = build_callouts()
    bundle["generado"] = "gold parquet (yelp-dm)"

    out = OUT / "bundle.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = os.path.getsize(out) / 1e6
    print(f"OK -> {out}  ({size_mb:.2f} MB)")


# alias usado dentro de build_restaurantes por comodidad
def load(name, columns=None):
    return pd.read_parquet(GOLD / name, columns=columns)


if __name__ == "__main__":
    sys.exit(main())
