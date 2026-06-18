"""Parte III - Clustering (K-Means++, DBSCAN y BFR).

Cubre del `Enunciado_proyecto2.pdf`, PARTE III (4 pts):
  - K-Means++ para inicializacion dispersa, codo (SSE) y silueta.
  - DBSCAN para clusters por densidad y outliers.
  - BFR para procesamiento por bloques con DS/CS/RS y distancia de Mahalanobis.

Regla del curso: los algoritmos se implementan a mano sobre numpy. Pandas se usa
para preparar features y persistir resultados; no hay scikit-learn/scipy.

Fundamento teorico principal:
  - `teoria/08 - Clustering.pdf`, pags. 11, 19-27, 31-42 y 46-49.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import GOLD, SEED


# ===========================================================================
#  Features: de la tabla gold a una matriz numerica estandarizada
# ===========================================================================
NUMERICAS_BASE = [
    "stars",
    "is_open",
    "n_categorias",
    "price_range",
    "mediana_ingreso",
    "poblacion",
    "pct_universitarios",
    "renta_mediana",
]

BOOLEANAS = [
    "RestaurantsTakeOut",
    "BusinessAcceptsCreditCards",
    "RestaurantsDelivery",
    "RestaurantsReservations",
    "OutdoorSeating",
    "HasTV",
    "RestaurantsGoodForGroups",
    "GoodForKids",
    "BikeParking",
    "Caters",
    "RestaurantsTableService",
    "parking_alguno",
    "amb_alguno",
    "meal_alguno",
]

CATEGORICAS = ["metro", "alcohol", "wifi"]


@dataclass
class FeatureMatrix:
    """Resultado del preprocesamiento para clustering."""

    df: pd.DataFrame
    X: np.ndarray
    X_raw: pd.DataFrame
    feature_names: list[str]
    ids: np.ndarray
    medias: np.ndarray
    escalas: np.ndarray
    imputaciones: dict[str, float | str]


def cargar_features() -> pd.DataFrame:
    """Lee `features_negocio.parquet` desde gold."""
    return pd.read_parquet(GOLD / "features_negocio.parquet")


def preparar_matriz_features(
    df: pd.DataFrame,
    incluir_geo: bool = False,
    incluir_mercado: bool = False,
) -> FeatureMatrix:
    """Convierte features mixtas a matriz numerica estandarizada.

    Decisiones:
      - `review_count` entra como log1p para controlar cola larga.
      - `poblacion` entra como log1p por la misma razon.
      - booleanos faltantes se llenan con 0 y se agrega indicador `_missing`
        cuando falta mas de 5%; asi no confundimos "no reportado" con "False".
      - categoricas se one-hot encodean, con categoria `missing`.
      - `metro` y `latitude/longitude` quedan fuera por defecto para evitar
        clusters definidos por geografia. `metro` se reserva como etiqueta
        proxy externa para purity/NMI; se puede activar como experimento.
    """
    base = df.copy()
    ids = base["business_id"].to_numpy()
    X = pd.DataFrame(index=base.index)
    imputaciones: dict[str, float | str] = {}

    X["stars"] = pd.to_numeric(base["stars"], errors="coerce")
    X["log_review_count"] = np.log1p(pd.to_numeric(base["review_count"], errors="coerce"))
    X["is_open"] = pd.to_numeric(base["is_open"], errors="coerce")
    X["n_categorias"] = pd.to_numeric(base["n_categorias"], errors="coerce")
    X["price_range"] = pd.to_numeric(base["price_range"], errors="coerce")
    X["mediana_ingreso"] = pd.to_numeric(base["mediana_ingreso"], errors="coerce")
    X["log_poblacion"] = np.log1p(pd.to_numeric(base["poblacion"], errors="coerce"))
    X["pct_universitarios"] = pd.to_numeric(base["pct_universitarios"], errors="coerce")
    X["renta_mediana"] = pd.to_numeric(base["renta_mediana"], errors="coerce")

    if incluir_geo:
        X["latitude"] = pd.to_numeric(base["latitude"], errors="coerce")
        X["longitude"] = pd.to_numeric(base["longitude"], errors="coerce")

    for c in BOOLEANAS:
        if c not in base:
            continue
        s = base[c].astype("boolean")
        miss = s.isna()
        X[c] = s.fillna(False).astype(float)
        if miss.mean() > 0.05:
            X[f"{c}_missing"] = miss.astype(float)

    categoricas = CATEGORICAS if incluir_mercado else [c for c in CATEGORICAS if c != "metro"]
    for c in categoricas:
        if c not in base:
            continue
        s = base[c].astype("string").fillna("missing")
        dummies = pd.get_dummies(s, prefix=c, dtype=float)
        X = pd.concat([X, dummies], axis=1)
        imputaciones[c] = "missing"

    for c in X.columns:
        if X[c].isna().any():
            med = float(X[c].median())
            X[c] = X[c].fillna(med)
            imputaciones[c] = med

    X_raw = X.copy()
    arr = X.to_numpy(dtype=float)
    medias = arr.mean(axis=0)
    escalas = arr.std(axis=0)
    escalas[escalas == 0] = 1.0
    X_std = (arr - medias) / escalas
    return FeatureMatrix(
        df=base,
        X=X_std.astype(np.float64),
        X_raw=X_raw,
        feature_names=X.columns.tolist(),
        ids=ids,
        medias=medias,
        escalas=escalas,
        imputaciones=imputaciones,
    )


# ===========================================================================
#  Utilidades numericas
# ===========================================================================
def dist2_a_centroides(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Distancia euclidiana cuadratica de cada punto a cada centroide."""
    x2 = (X * X).sum(axis=1, keepdims=True)
    c2 = (C * C).sum(axis=1)
    d2 = x2 + c2 - 2.0 * X @ C.T
    return np.maximum(d2, 0.0)


def asignar_centroides(X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Asigna cada punto a su centroide mas cercano."""
    d2 = dist2_a_centroides(X, C)
    labels = d2.argmin(axis=1)
    return labels.astype(np.int32), d2[np.arange(len(X)), labels]


def sse(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray | None = None) -> float:
    """Suma de cuadrados intra-cluster. Ignora labels negativos (ruido)."""
    labels = np.asarray(labels)
    ok = labels >= 0
    if not ok.any():
        return float("nan")
    if centroids is None:
        k = int(labels[ok].max()) + 1
        centroids = centroides_desde_labels(X, labels, k)
    d = X[ok] - centroids[labels[ok]]
    return float((d * d).sum())


def centroides_desde_labels(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Centroides para labels 0..k-1."""
    d = X.shape[1]
    C = np.zeros((k, d), dtype=float)
    counts = np.bincount(labels[labels >= 0], minlength=k).astype(float)
    for j in range(d):
        C[:, j] = np.bincount(labels[labels >= 0], weights=X[labels >= 0, j], minlength=k)
    nz = counts > 0
    C[nz] /= counts[nz, None]
    return C


# ===========================================================================
#  PCA manual para visualizacion comun
#  Deck 13, pags. 11-12 y 45-47: ejes latentes, varianza y autovectores.
# ===========================================================================
def pca_proyeccion(X: np.ndarray, n_components: int = 2) -> dict:
    """Ajusta PCA por autodescomposicion de la covarianza y proyecta X.

    Se usa como lente visual: el clustering sigue entrenado y evaluado en el
    espacio estandarizado original. No depende de sklearn ni scipy.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("X debe ser una matriz 2D con al menos dos filas")
    if not 1 <= n_components <= X.shape[1]:
        raise ValueError("n_components debe estar entre 1 y el numero de columnas")

    media = X.mean(axis=0)
    centrada = X - media
    cov = (centrada.T @ centrada) / (len(X) - 1)
    autovalores, autovectores = np.linalg.eigh(cov)
    orden = np.argsort(autovalores)[::-1]
    autovalores = np.maximum(autovalores[orden], 0.0)
    autovectores = autovectores[:, orden]

    # El signo de un autovector es arbitrario; fijarlo vuelve la figura estable.
    for j in range(autovectores.shape[1]):
        i = int(np.argmax(np.abs(autovectores[:, j])))
        if autovectores[i, j] < 0:
            autovectores[:, j] *= -1

    componentes = autovectores[:, :n_components]
    total = float(autovalores.sum())
    ratio = autovalores / total if total > 0 else np.zeros_like(autovalores)
    return {
        "scores": centrada @ componentes,
        "components": componentes.T,
        "mean": media,
        "explained_variance": autovalores,
        "explained_ratio": ratio,
        "covariance": cov,
    }


# ===========================================================================
#  K-Means++ y K-Means
#  Deck 08, pags. 19-27: inicializacion por D(x)^2 y descenso de SSE.
# ===========================================================================
def kmeans_pp_init(X: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """Inicializa centroides con K-Means++."""
    rng = np.random.default_rng(seed)
    n = len(X)
    first = int(rng.integers(n))
    centroids = [X[first].copy()]
    min_d2 = dist2_a_centroides(X, np.array(centroids))[:, 0]
    for _ in range(1, k):
        total = min_d2.sum()
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            idx = int(np.searchsorted(np.cumsum(min_d2 / total), rng.random(), side="right"))
            idx = min(idx, n - 1)
        centroids.append(X[idx].copy())
        min_d2 = np.minimum(min_d2, dist2_a_centroides(X, np.array([centroids[-1]]))[:, 0])
    return np.vstack(centroids)


def kmeans(
    X: np.ndarray,
    k: int,
    seed: int = SEED,
    init: str = "++",
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> dict:
    """K-Means con inicializacion ++ o aleatoria."""
    rng = np.random.default_rng(seed)
    if init == "++":
        C = kmeans_pp_init(X, k, seed=seed)
    elif init == "random":
        C = X[rng.choice(len(X), size=k, replace=False)].copy()
    else:
        raise ValueError("init debe ser '++' o 'random'")

    hist: list[float] = []
    labels = np.zeros(len(X), dtype=np.int32)
    dist_min = np.zeros(len(X), dtype=float)
    for it in range(1, max_iter + 1):
        labels, dist_min = asignar_centroides(X, C)
        actual = float(dist_min.sum())
        hist.append(actual)
        Cnew = C.copy()
        counts = np.bincount(labels, minlength=k)
        for c in range(k):
            if counts[c]:
                Cnew[c] = X[labels == c].mean(axis=0)
            else:
                lejos = int(np.argmax(dist_min))
                Cnew[c] = X[lejos]
                labels[lejos] = c
        mov = float(np.sqrt(((Cnew - C) ** 2).sum(axis=1)).max())
        C = Cnew
        if verbose:
            print(f"  k={k} iter={it:02d} SSE={actual:,.1f} mov={mov:.3e}")
        if mov < tol:
            break
    labels, dist_min = asignar_centroides(X, C)
    return {
        "labels": labels,
        "centroids": C,
        "sse": float(dist_min.sum()),
        "n_iter": it,
        "history": hist,
        "counts": np.bincount(labels, minlength=k),
    }


def barrido_kmeans(
    X: np.ndarray,
    ks: Iterable[int],
    seed: int = SEED,
    sample_silhouette: int = 1500,
) -> pd.DataFrame:
    """Corre K-Means++ para varios k y calcula SSE + silueta muestral."""
    filas = []
    for k in ks:
        res = kmeans(X, k, seed=seed, init="++")
        sil = silhouette_score(X, res["labels"], sample_size=sample_silhouette, seed=seed)
        filas.append({
            "k": k,
            "sse": res["sse"],
            "sse_por_punto": res["sse"] / len(X),
            "silueta": sil,
            "iters": res["n_iter"],
            "cluster_min": int(res["counts"].min()),
            "cluster_max": int(res["counts"].max()),
        })
    return pd.DataFrame(filas)


def detectar_codo(tabla: pd.DataFrame) -> int:
    """Detecta el codo como maxima distancia a la recta entre extremos.

    Usa `k` y `sse` normalizados a [0,1]. Es una ayuda reproducible; la decision
    final tambien debe considerar silueta e interpretabilidad.
    """
    ks = tabla["k"].to_numpy(dtype=float)
    ys = tabla["sse"].to_numpy(dtype=float)
    x = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
    y = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-12)
    a = np.array([x[0], y[0]])
    b = np.array([x[-1], y[-1]])
    den = np.linalg.norm(b - a)
    if den == 0:
        return int(ks[0])
    dist = np.abs((b[0] - a[0]) * (a[1] - y) - (a[0] - x) * (b[1] - a[1])) / den
    return int(ks[int(np.argmax(dist))])


# ===========================================================================
#  Silhouette, Purity y NMI (a mano)
#  Deck 08, pags. 47-49.
# ===========================================================================
def silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 1500,
    seed: int = SEED,
    block_size: int = 128,
) -> float:
    """Silueta promedio usando muestra de puntos y todos los puntos como referencia.

    Exacta para la muestra elegida; evita materializar la matriz n x n completa.
    Labels negativos (ruido DBSCAN) se excluyen del promedio.
    """
    _, vals = silhouette_samples(
        X, labels, sample_size=sample_size, seed=seed, block_size=block_size
    )
    return float(np.mean(vals)) if len(vals) else float("nan")


def silhouette_samples(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 1500,
    seed: int = SEED,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Silueta individual para una muestra y sus indices en la matriz original."""
    labels = np.asarray(labels)
    ok = labels >= 0
    idx_ok = np.flatnonzero(ok)
    if len(idx_ok) == 0 or len(np.unique(labels[ok])) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    if sample_size is not None and len(idx_ok) > sample_size:
        sample = rng.choice(idx_ok, size=sample_size, replace=False)
    else:
        sample = idx_ok

    labs_pos = np.unique(labels[ok])
    remap = {c: i for i, c in enumerate(labs_pos)}
    lab2 = np.array([remap.get(c, -1) for c in labels], dtype=np.int32)
    k = len(labs_pos)
    counts = np.bincount(lab2[ok], minlength=k).astype(float)
    vals = []
    for start in range(0, len(sample), block_size):
        ids = sample[start:start + block_size]
        D = _distancias_euclidianas(X[ids], X)
        for row, idx in enumerate(ids):
            sums = np.bincount(lab2[ok], weights=D[row, ok], minlength=k)
            means = np.full(k, np.inf)
            nonempty = counts > 0
            means[nonempty] = sums[nonempty] / counts[nonempty]
            c = lab2[idx]
            if counts[c] <= 1:
                a = 0.0
            else:
                a = sums[c] / (counts[c] - 1.0)
            means[c] = np.inf
            b = float(means.min())
            vals.append((b - a) / max(a, b, 1e-12))
    return sample.astype(np.int64), np.asarray(vals, dtype=float)


def _distancias_euclidianas(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d2 = (A * A).sum(axis=1, keepdims=True) + (B * B).sum(axis=1) - 2.0 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Pureza respecto a una etiqueta proxy."""
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    ok = labels >= 0
    if not ok.any():
        return float("nan")
    total = ok.sum()
    score = 0
    for c in np.unique(labels[ok]):
        vals, cnt = np.unique(truth[ok & (labels == c)], return_counts=True)
        if len(vals):
            score += int(cnt.max())
    return float(score / total)


def nmi(labels: np.ndarray, truth: np.ndarray) -> float:
    """Normalized Mutual Information simetrica: 2*MI/(H(C)+H(L))."""
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    ok = labels >= 0
    labels = labels[ok]
    truth = truth[ok]
    if len(labels) == 0:
        return float("nan")
    cl, ci = np.unique(labels, return_inverse=True)
    tl, ti = np.unique(truth, return_inverse=True)
    cont = np.zeros((len(cl), len(tl)), dtype=float)
    np.add.at(cont, (ci, ti), 1.0)
    n = cont.sum()
    pi = cont.sum(axis=1) / n
    pj = cont.sum(axis=0) / n
    pij = cont / n
    nz = pij > 0
    mi = float((pij[nz] * np.log(pij[nz] / (pi[:, None] * pj[None, :])[nz])).sum())
    hi = float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum())
    hj = float(-(pj[pj > 0] * np.log(pj[pj > 0])).sum())
    return 0.0 if hi + hj == 0 else float(2.0 * mi / (hi + hj))


# ===========================================================================
#  DBSCAN
#  Enunciado Parte III + deck 08, pag. 45: formas arbitrarias y outliers.
# ===========================================================================
def k_distance_values(X: np.ndarray, k: int = 10, sample_size: int = 6000,
                      seed: int = SEED, block_size: int = 256) -> np.ndarray:
    """Distancia al k-esimo vecino dentro de una muestra."""
    if k < 1:
        raise ValueError("k debe ser positivo")
    rng = np.random.default_rng(seed)
    n = len(X)
    ids = rng.choice(n, size=min(sample_size, n), replace=False)
    S = X[ids]
    if len(S) < 2:
        raise ValueError("se requieren al menos dos puntos")
    neighbor_index = min(k, len(S) - 1) - 1
    vals = np.empty(len(S), dtype=float)
    for start in range(0, len(S), block_size):
        A = S[start:start + block_size]
        D = _distancias_euclidianas(A, S)
        for i in range(len(A)):
            global_i = start + i
            D[i, global_i] = np.inf
        vals[start:start + len(A)] = np.partition(D, kth=neighbor_index, axis=1)[:, neighbor_index]
    return np.sort(vals)


def dbscan(X: np.ndarray, eps: float, min_pts: int = 10,
           block_size: int = 256, verbose: bool = False) -> dict:
    """DBSCAN exacto sobre la matriz recibida. Pensado para muestras controladas."""
    n = len(X)
    neigh: list[np.ndarray] = []
    eps2 = eps * eps
    for start in range(0, n, block_size):
        A = X[start:start + block_size]
        d2 = (A * A).sum(axis=1, keepdims=True) + (X * X).sum(axis=1) - 2.0 * A @ X.T
        for row in np.maximum(d2, 0.0):
            neigh.append(np.flatnonzero(row <= eps2))

    labels = np.full(n, -99, dtype=np.int32)   # -99 = no visitado, -1 = ruido
    cluster = 0
    for i in range(n):
        if labels[i] != -99:
            continue
        Ni = neigh[i]
        if len(Ni) < min_pts:
            labels[i] = -1
            continue
        labels[i] = cluster
        seeds = list(Ni[Ni != i])
        pos = 0
        while pos < len(seeds):
            q = int(seeds[pos])
            if labels[q] == -1:
                labels[q] = cluster
            if labels[q] != -99:
                pos += 1
                continue
            labels[q] = cluster
            Nq = neigh[q]
            if len(Nq) >= min_pts:
                seeds.extend([int(x) for x in Nq if labels[int(x)] in (-99, -1)])
            pos += 1
        cluster += 1
    labels[labels == -99] = -1
    counts = np.bincount(labels[labels >= 0], minlength=max(cluster, 1))
    if verbose:
        ruido = int((labels == -1).sum())
        print(f"DBSCAN eps={eps:.3f} minPts={min_pts}: {cluster} clusters, ruido={ruido} ({ruido/n:.1%})")
    return {"labels": labels, "n_clusters": cluster, "counts": counts, "noise": int((labels == -1).sum())}


# ===========================================================================
#  BFR: DS / CS / RS con distancia de Mahalanobis
#  Deck 08, pags. 31-42.
# ===========================================================================
@dataclass
class Summary:
    n: int
    sum: np.ndarray
    sumsq: np.ndarray
    ids: list[int] | None = None

    @classmethod
    def from_points(cls, X: np.ndarray, ids: list[int] | None = None) -> "Summary":
        return cls(n=len(X), sum=X.sum(axis=0), sumsq=(X * X).sum(axis=0), ids=ids)

    @property
    def centroid(self) -> np.ndarray:
        return self.sum / max(self.n, 1)

    @property
    def var(self) -> np.ndarray:
        v = self.sumsq / max(self.n, 1) - self.centroid ** 2
        return np.maximum(v, 1e-6)

    def add_point(self, x: np.ndarray, idx: int | None = None) -> None:
        self.n += 1
        self.sum += x
        self.sumsq += x * x
        if self.ids is not None and idx is not None:
            self.ids.append(int(idx))

    def merge(self, other: "Summary") -> None:
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq
        if self.ids is not None and other.ids is not None:
            self.ids.extend(other.ids)


def mahalanobis_a_summary(x: np.ndarray, s: Summary) -> float:
    """Distancia de Mahalanobis diagonal respecto a un summary."""
    z = (x - s.centroid) ** 2 / s.var
    return float(np.sqrt(z.sum()))


def bfr(
    X: np.ndarray,
    k: int,
    seed: int = SEED,
    chunk_size: int = 4000,
    init_size: int = 5000,
    threshold_ds_factor: float = 2.0,
    threshold_cs_factor: float = 1.5,
    threshold_merge_factor: float = 1.5,
    min_cs_size: int = 5,
    verbose: bool = False,
) -> dict:
    """BFR simplificado y reproducible.

    Inicializa DS con K-Means++ en una muestra inicial. Luego procesa chunks:
    puntos cercanos a DS se absorben definitivamente; los demas pasan por CS/RS.
    Al final, los CS cercanos se fusionan al DS mas cercano y el resto queda como
    outlier (-1). Los umbrales se expresan como factor * sqrt(d), una forma
    dimensionalmente consistente de llevar la regla de "sigmas" a d variables.
    """
    rng = np.random.default_rng(seed)
    n, dim = X.shape
    threshold_ds = threshold_ds_factor * np.sqrt(dim)
    threshold_cs = threshold_cs_factor * np.sqrt(dim)
    threshold_merge = threshold_merge_factor * np.sqrt(dim)
    order = rng.permutation(n)
    init_ids = order[:min(init_size, n)]
    rest = order[min(init_size, n):]
    km0 = kmeans(X[init_ids], k, seed=seed, init="++", max_iter=60)

    labels = np.full(n, -1, dtype=np.int32)
    ds: list[Summary] = []
    for c in range(k):
        ids_c = init_ids[km0["labels"] == c]
        if len(ids_c) == 0:
            ids_c = np.array([init_ids[c % len(init_ids)]])
        ds.append(Summary.from_points(X[ids_c]))
        labels[ids_c] = c

    cs: list[Summary] = []
    rs: list[int] = []
    hist = []

    def asignar_a_summaries(idx: int, summaries: list[Summary]) -> tuple[int, float]:
        dists = [mahalanobis_a_summary(X[idx], s) for s in summaries]
        j = int(np.argmin(dists)) if dists else -1
        return j, (float(dists[j]) if dists else float("inf"))

    def recluster_rs() -> None:
        nonlocal rs, cs
        if len(rs) < max(2 * k, min_cs_size * 2):
            return
        ids = np.array(rs, dtype=int)
        kk = min(max(k, len(ids) // 30), max(1, len(ids) // min_cs_size))
        if kk < 2:
            return
        res = kmeans(X[ids], kk, seed=seed + len(hist) + 7, init="++", max_iter=40)
        nuevos_rs: list[int] = []
        for c in range(kk):
            sub = ids[res["labels"] == c]
            if len(sub) >= min_cs_size:
                cs.append(Summary.from_points(X[sub], ids=[int(i) for i in sub]))
            else:
                nuevos_rs.extend([int(i) for i in sub])
        rs = nuevos_rs

    def distancia_summaries(a: Summary, b: Summary) -> float:
        var = np.maximum((a.var + b.var) / 2.0, 1e-6)
        return float(np.sqrt(((a.centroid - b.centroid) ** 2 / var).sum()))

    def fusionar_cs_cercanos() -> None:
        """Fusiona pares CS mientras su distancia normalizada permita hacerlo."""
        nonlocal cs
        while len(cs) > 1:
            mejor = (float("inf"), -1, -1)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    d = distancia_summaries(cs[i], cs[j])
                    if d < mejor[0]:
                        mejor = (d, i, j)
            if mejor[0] > threshold_merge:
                break
            _, i, j = mejor
            cs[i].merge(cs[j])
            cs.pop(j)

    def absorber_cs_en_ds() -> None:
        """Promueve a DS los CS suficientemente cercanos a un cluster final."""
        nonlocal cs
        pendientes: list[Summary] = []
        for s in cs:
            candidatos = [
                (i, mahalanobis_a_summary(s.centroid, dsum))
                for i, dsum in enumerate(ds)
            ]
            j, d = min(candidatos, key=lambda t: t[1])
            if d <= threshold_merge and s.ids:
                ds[j].merge(s)
                labels[s.ids] = j
            else:
                pendientes.append(s)
        cs = pendientes

    for chunk_start in range(0, len(rest), chunk_size):
        chunk = rest[chunk_start:chunk_start + chunk_size]
        for idx in chunk:
            j, d = asignar_a_summaries(int(idx), ds)
            if d <= threshold_ds:
                ds[j].add_point(X[idx])
                labels[idx] = j
                continue
            jcs, dcs = asignar_a_summaries(int(idx), cs)
            if dcs <= threshold_cs:
                cs[jcs].add_point(X[idx], int(idx))
                continue
            rs.append(int(idx))
        recluster_rs()
        fusionar_cs_cercanos()
        absorber_cs_en_ds()
        hist.append({"procesados": int(min(n, init_size + chunk_start + len(chunk))),
                     "ds": int(sum(s.n for s in ds)),
                     "cs_sets": len(cs),
                     "cs_points": int(sum(s.n for s in cs)),
                     "rs": len(rs)})
        if verbose:
            h = hist[-1]
            print(f"  BFR {h['procesados']:,}/{n:,}: DS={h['ds']:,} CS={h['cs_points']:,} RS={h['rs']:,}")

    fusionar_cs_cercanos()
    absorber_cs_en_ds()

    return {
        "labels": labels,
        "summaries": ds,
        "history": pd.DataFrame(hist),
        "threshold_ds": threshold_ds,
        "threshold_cs": threshold_cs,
        "threshold_merge": threshold_merge,
        "ds_points": int((labels >= 0).sum()),
        "outliers": int((labels < 0).sum()),
        "counts": np.bincount(labels[labels >= 0], minlength=k),
    }


# ===========================================================================
#  Reportes de clusters
# ===========================================================================
def resumen_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Tabla humana para caracterizar clusters de restaurantes."""
    out = df[[
        "business_id", "metro", "stars", "review_count", "price_range",
        "mediana_ingreso", "pct_universitarios", "renta_mediana",
        "RestaurantsDelivery", "OutdoorSeating", "HasTV", "RestaurantsReservations",
        "alcohol", "wifi",
    ]].copy()
    out["cluster"] = labels
    filas = []
    for c in sorted(x for x in np.unique(labels) if x >= 0):
        g = out[out["cluster"] == c]
        metro_counts = g["metro"].value_counts(normalize=True)
        alcohol_counts = g["alcohol"].fillna("missing").value_counts(normalize=True)
        wifi_counts = g["wifi"].fillna("missing").value_counts(normalize=True)
        filas.append({
            "cluster": int(c),
            "n": int(len(g)),
            "pct": round(100 * len(g) / len(out), 1),
            "metro_top": f"{metro_counts.index[0]} ({metro_counts.iloc[0]*100:.0f}%)",
            "stars_media": round(float(g["stars"].mean()), 2),
            "reviews_mediana": round(float(g["review_count"].median()), 0),
            "precio_medio": round(float(g["price_range"].mean()), 2),
            "ingreso_mediana": round(float(g["mediana_ingreso"].median()), 0),
            "pct_univ_media": round(float(g["pct_universitarios"].mean()), 2),
            "delivery_%": round(100 * g["RestaurantsDelivery"].astype("boolean").fillna(False).mean(), 1),
            "outdoor_%": round(100 * g["OutdoorSeating"].astype("boolean").fillna(False).mean(), 1),
            "reservas_%": round(100 * g["RestaurantsReservations"].astype("boolean").fillna(False).mean(), 1),
            "alcohol_top": f"{alcohol_counts.index[0]} ({alcohol_counts.iloc[0]*100:.0f}%)",
            "wifi_top": f"{wifi_counts.index[0]} ({wifi_counts.iloc[0]*100:.0f}%)",
        })
    if (labels < 0).any():
        filas.append({"cluster": -1, "n": int((labels < 0).sum()),
                      "pct": round(100 * (labels < 0).sum() / len(labels), 1),
                      "metro_top": "ruido/outlier"})
    return pd.DataFrame(filas).sort_values("cluster")


def comparar_metodos(filas: list[dict]) -> pd.DataFrame:
    """Convierte metricas de varios metodos en tabla ordenada."""
    return pd.DataFrame(filas)[[
        "metodo", "n_clusters", "outliers", "sse", "silueta", "purity_metro", "nmi_metro"
    ]]
