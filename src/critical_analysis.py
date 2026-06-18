"""Parte VII - análisis crítico de escalabilidad, sesgos y equidad.

Cubre la Parte VII del ``Enunciado_proyecto2.pdf`` (pág. 3):

* comparar exactitud y eficiencia, incluyendo complejidad temporal/espacial;
* auditar subrepresentación, concentración de voz y sensibilidad a spam;
* medir si rankings y recomendaciones concentran exposición;
* discutir el impacto sobre negocios pequeños frente a actores dominantes.

Las funciones de este módulo son métricas y experimentos auditables construidos
con NumPy/Pandas. No sustituyen atributos protegidos inexistentes en Yelp: las
variables de ingreso por ZIP, visibilidad y número de locales se reportan como
*proxies* y nunca como identidad individual ni causalidad.
"""
from __future__ import annotations

from time import perf_counter
from typing import Iterable, Sequence
import re
import unicodedata

import numpy as np
import pandas as pd


def gini(values: Iterable[float]) -> float:
    """Coeficiente de Gini para valores no negativos; 0=paridad, 1=concentración."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    if np.any(x < 0):
        raise ValueError("Gini requiere valores no negativos")
    total = x.sum()
    if total == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2.0 * np.dot(np.arange(1, n + 1), x) / (n * total)) - (n + 1) / n)


def lorenz_curve(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Fracción acumulada de población y valor para dibujar una curva de Lorenz."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if np.any(x < 0):
        raise ValueError("Lorenz requiere valores no negativos")
    x = np.sort(x)
    cumulative = np.r_[0.0, np.cumsum(x)]
    if cumulative[-1] > 0:
        cumulative /= cumulative[-1]
    population = np.linspace(0.0, 1.0, len(cumulative))
    return population, cumulative


def top_share(values: Iterable[float], fraction: float = 0.10) -> float:
    """Proporción del total acumulada por la fracción superior de entidades."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction debe estar en (0, 1]")
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(x) * fraction)))
    return float(np.sort(x)[-k:].sum() / x.sum())


def concentration_summary(ids: Iterable[object]) -> dict[str, float]:
    """Cobertura y concentración de una secuencia de exposiciones/recomendaciones."""
    counts = pd.Series(list(ids), dtype="object").value_counts()
    slots = int(counts.sum())
    return {
        "slots": slots,
        "items_unicos": int(len(counts)),
        "ratio_unicos": float(len(counts) / slots) if slots else float("nan"),
        "gini_exposicion": gini(counts.to_numpy()),
        "share_top10_items": top_share(counts.to_numpy(), min(1.0, 10 / max(len(counts), 1))),
    }


def _normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _quartile_proxy(series: pd.Series, prefix: str) -> pd.Series:
    """Cuartiles robustos a empates mediante percentiles promedio."""
    numeric = pd.to_numeric(series, errors="coerce")
    pct = numeric.rank(method="average", pct=True)
    labels = np.select(
        [pct <= 0.25, pct <= 0.50, pct <= 0.75, pct > 0.75],
        [f"{prefix} Q1", f"{prefix} Q2", f"{prefix} Q3", f"{prefix} Q4"],
        default=f"{prefix} sin dato",
    )
    return pd.Series(labels, index=series.index, dtype="string")


def add_business_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Añade proxies explícitos de visibilidad, contexto y presencia multi-local.

    ``visibilidad`` usa cuartiles de ``review_count``: es exposición histórica,
    no tamaño económico. ``tipo_nombre`` usa repetición del nombre normalizado:
    1 local se denomina independiente *proxy*; no prueba propiedad empresarial.
    ``ingreso_zip`` es contexto ecológico del ZIP y no ingreso del propietario.
    """
    required = {"name", "review_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    out = df.copy()
    out["name_key"] = out["name"].map(_normalize_name)
    locations = out.groupby("name_key", dropna=False)["name_key"].transform("size")
    out["n_locales_nombre"] = locations.astype(int)
    out["tipo_nombre"] = np.select(
        [locations == 1, locations.between(2, 4), locations >= 5],
        ["independiente (proxy)", "multi-local 2-4", "cadena 5+ (proxy)"],
        default="sin clasificar",
    )
    out["visibilidad"] = _quartile_proxy(out["review_count"], "reseñas")
    if "mediana_ingreso" in out:
        out["ingreso_zip"] = _quartile_proxy(out["mediana_ingreso"], "ingreso ZIP")
    return out


def representation_audit(
    df: pd.DataFrame,
    score_columns: Sequence[str],
    group_columns: Sequence[str],
    top_fractions: Sequence[float] = (0.01, 0.05),
) -> pd.DataFrame:
    """Compara representación en la cabeza de un ranking contra el catálogo.

    ``ratio_representacion=1`` implica paridad descriptiva; valores >1 indican
    sobrerrepresentación. No es por sí solo una prueba de discriminación causal.
    """
    rows: list[dict] = []
    n = len(df)
    for score in score_columns:
        ranked = df.dropna(subset=[score]).sort_values(score, ascending=False, kind="stable")
        for fraction in top_fractions:
            k = max(1, int(np.ceil(len(ranked) * fraction)))
            selected = ranked.head(k).copy()
            selected["_discount"] = 1.0 / np.log2(np.arange(2, k + 2))
            for group in group_columns:
                catalog_share = df[group].fillna("sin dato").value_counts(normalize=True)
                selected_group = selected[group].fillna("sin dato")
                selected_share = selected_group.value_counts(normalize=True)
                discount_share = selected.groupby(selected_group)["_discount"].sum()
                discount_share = discount_share / discount_share.sum()
                for value, base_share in catalog_share.items():
                    sel_share = float(selected_share.get(value, 0.0))
                    rows.append({
                        "ranking": score,
                        "top_fraction": float(fraction),
                        "top_n": k,
                        "dimension": group,
                        "grupo": str(value),
                        "n_catalogo": int((df[group].fillna("sin dato") == value).sum()),
                        "share_catalogo": float(base_share),
                        "n_seleccion": int((selected_group == value).sum()),
                        "share_seleccion": sel_share,
                        "share_exposicion_desc": float(discount_share.get(value, 0.0)),
                        "ratio_representacion": sel_share / float(base_share) if base_share else float("nan"),
                    })
    return pd.DataFrame(rows)


def review_attack_stress(
    business_stats: pd.DataFrame,
    attacks: Sequence[int] = (5, 20, 100),
    target_stars: Sequence[float] = (1.0, 5.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula campañas coordinadas y resume sensibilidad por visibilidad.

    El experimento no identifica reseñas falsas reales. Pregunta cuánto movería
    una campaña hipotética el promedio si el sistema no la detectara.
    """
    required = {"business_id", "n_reviews", "mean_stars"}
    missing = required - set(business_stats)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    base = business_stats.copy()
    base["visibilidad"] = _quartile_proxy(base["n_reviews"], "reseñas")
    detail: list[pd.DataFrame] = []
    for stars in target_stars:
        for n_fake in attacks:
            x = base.copy()
            x["tipo_ataque"] = f"{stars:.0f} estrellas"
            x["n_falsas"] = int(n_fake)
            x["mean_after"] = (
                x["mean_stars"] * x["n_reviews"] + stars * n_fake
            ) / (x["n_reviews"] + n_fake)
            x["delta"] = x["mean_after"] - x["mean_stars"]
            x["cruza_4"] = ((x["mean_stars"] < 4) & (x["mean_after"] >= 4)) | (
                (x["mean_stars"] >= 4) & (x["mean_after"] < 4)
            )
            detail.append(x)
    detailed = pd.concat(detail, ignore_index=True)
    summary = (
        detailed.groupby(["tipo_ataque", "n_falsas", "visibilidad"], observed=True)
        .agg(
            negocios=("business_id", "size"),
            delta_mediana=("delta", "median"),
            delta_abs_p90=("delta", lambda s: float(np.quantile(np.abs(s), 0.90))),
            pct_cruza_4=("cruza_4", lambda s: 100.0 * float(np.mean(s))),
        )
        .reset_index()
    )
    return detailed, summary


def benchmark_distance_kernels(
    X: np.ndarray,
    sizes: Sequence[int] = (500, 1000, 2000, 4000),
    k: int = 6,
    repeats: int = 3,
    block_size: int = 256,
) -> pd.DataFrame:
    """Mide los núcleos dominantes O(nkd) y O(n²d) sobre features reales.

    Se mide el mejor tiempo de varias repeticiones para reducir ruido de carga.
    El kernel cuadrático cuenta vecindades por bloques y nunca materializa n².
    """
    X = np.asarray(X, dtype=float)
    rows: list[dict] = []
    for requested in sizes:
        n = min(int(requested), len(X))
        A = np.ascontiguousarray(X[:n])
        C = np.ascontiguousarray(X[:k])

        linear_times = []
        for _ in range(repeats):
            start = perf_counter()
            d2 = (A * A).sum(1, keepdims=True) + (C * C).sum(1) - 2.0 * A @ C.T
            np.argmin(d2, axis=1)
            linear_times.append(perf_counter() - start)
        rows.append({"kernel": "asignación K-Means", "n": n, "segundos": min(linear_times)})

        quadratic_times = []
        for _ in range(repeats):
            start = perf_counter()
            norms = (A * A).sum(axis=1)
            neighbors = 0
            for begin in range(0, n, block_size):
                B = A[begin : begin + block_size]
                d2 = (B * B).sum(1, keepdims=True) + norms - 2.0 * B @ A.T
                neighbors += int(np.count_nonzero(d2 <= X.shape[1]))
            quadratic_times.append(perf_counter() - start)
        rows.append({
            "kernel": "vecindades DBSCAN",
            "n": n,
            "segundos": min(quadratic_times),
            "conteo_control": neighbors,
        })
    result = pd.DataFrame(rows)
    slopes = {}
    for kernel, group in result.groupby("kernel"):
        slope = np.polyfit(np.log(group["n"]), np.log(group["segundos"].clip(lower=1e-9)), 1)[0]
        slopes[kernel] = float(slope)
    result["pendiente_loglog"] = result["kernel"].map(slopes)
    return result


def projected_growth(complexity: str, factor: float) -> float:
    """Factor relativo de costo para clases usadas en la tabla de escalabilidad."""
    if factor <= 0:
        raise ValueError("factor debe ser positivo")
    rules = {
        "O(n)": factor,
        "O(n log n)": factor * np.log2(max(2.0, factor * 1_000_000)) / np.log2(1_000_000),
        "O(n²)": factor**2,
        "O(n³)": factor**3,
        "O(m²n)": factor**3,
        "O(nd²)": factor,
    }
    if complexity not in rules:
        return float("nan")
    return float(rules[complexity])
