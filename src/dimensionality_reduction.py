"""Parte VI - Reduccion de dimensionalidad (PCA y SVD truncada).

Cubre del ``Enunciado_proyecto2.pdf``, Parte VI:

* PCA sobre las features estandarizadas de restaurantes: covarianza,
  autovalores/autovectores, 90% de varianza, proyeccion e interpretacion.
* SVD sobre TF-IDF de resenas: factores latentes, truncamiento, error de
  reconstruccion y compresion.

Fundamento disponible en las diapositivas del curso:

* Normalizacion y representacion vectorial: ``08 - Clustering``, pag. 11.
* SVD, factores latentes y compresion: ``10 - Recomendacion 2``, pags. 16 y 20.
* Proyeccion, error de reconstruccion y SVD: deck extra ``13. Reduccion de
  Dimensionalidad``, pags. 11, 25-26, 36-45 y 65-69.
* La receta explicita de PCA (covarianza, autovalores y umbral 90%) esta en el
  enunciado, pag. 3; el deck 13 complementa su interpretacion geometrica.

Regla del curso: los algoritmos se implementan sobre numpy. No se usa scipy,
scikit-learn ni una libreria de matrices dispersas. La SVD truncada se obtiene
con un rango aleatorio, multiplicaciones CSR explicitas y autodescomposicion de
la matriz reducida; nunca se densifica la matriz TF-IDF completa.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import datetime as dt
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import GOLD, SEED


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
@dataclass
class PCAResult:
    """Resultado completo de PCA ajustado por autodescomposicion."""

    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_ratio: np.ndarray
    scores: np.ndarray


def pca_fit(X: np.ndarray) -> PCAResult:
    """Ajusta PCA a mano mediante la covarianza y ``numpy.linalg.eigh``.

    ``components`` contiene un autovector por fila, ordenados de mayor a menor
    varianza. El signo se fija de forma determinista porque matematicamente es
    arbitrario y, sin esta regla, las figuras podrian invertirse entre corridas.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X debe ser una matriz 2D con al menos dos filas")
    if not np.isfinite(X).all():
        raise ValueError("X contiene valores no finitos")

    mean = X.mean(axis=0)
    centered = X - mean
    covariance = (centered.T @ centered) / (len(X) - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    for j in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, j])))
        if vectors[pivot, j] < 0:
            vectors[:, j] *= -1

    total = float(values.sum())
    ratio = values / total if total > 0 else np.zeros_like(values)
    components = vectors.T
    return PCAResult(
        mean=mean,
        components=components,
        eigenvalues=values,
        explained_ratio=ratio,
        scores=centered @ components.T,
    )


def n_components_for_variance(explained_ratio: np.ndarray, target: float = 0.90) -> int:
    """Menor numero de componentes cuya varianza acumulada alcanza ``target``."""
    ratio = np.asarray(explained_ratio, dtype=float)
    if ratio.ndim != 1 or not 0 < target <= 1:
        raise ValueError("ratio debe ser 1D y target debe estar en (0, 1]")
    return int(np.searchsorted(np.cumsum(ratio), target, side="left") + 1)


def pca_transform(X: np.ndarray, model: PCAResult, k: int | None = None) -> np.ndarray:
    """Proyecta observaciones sobre los primeros ``k`` componentes."""
    k = len(model.components) if k is None else int(k)
    if not 1 <= k <= len(model.components):
        raise ValueError("k fuera del rango de componentes")
    return (np.asarray(X, float) - model.mean) @ model.components[:k].T


def pca_inverse(scores: np.ndarray, model: PCAResult, k: int | None = None) -> np.ndarray:
    """Reconstruye observaciones desde sus coordenadas principales."""
    scores = np.asarray(scores, float)
    k = scores.shape[1] if k is None else int(k)
    return scores[:, :k] @ model.components[:k] + model.mean


def relative_frobenius_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Error de Frobenius relativo ||X-X_k||_F / ||X||_F."""
    original = np.asarray(original, float)
    reconstructed = np.asarray(reconstructed, float)
    denominator = np.linalg.norm(original, ord="fro")
    numerator = np.linalg.norm(original - reconstructed, ord="fro")
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


def pca_curve(model: PCAResult) -> pd.DataFrame:
    """Varianza acumulada y error teorico de reconstruccion para cada k."""
    cumulative = np.cumsum(model.explained_ratio)
    return pd.DataFrame(
        {
            "k": np.arange(1, len(cumulative) + 1),
            "varianza_acumulada": cumulative,
            "error_relativo": np.sqrt(np.maximum(0.0, 1.0 - cumulative)),
        }
    )


def pca_loadings(model: PCAResult, feature_names: Iterable[str]) -> pd.DataFrame:
    """Correlacion aproximada variable-PC para features estandarizadas."""
    names = list(feature_names)
    if len(names) != model.components.shape[1]:
        raise ValueError("feature_names no coincide con las columnas de PCA")
    loadings = model.components.T * np.sqrt(model.eigenvalues)[None, :]
    return pd.DataFrame(loadings, index=names, columns=[f"PC{i+1}" for i in range(len(names))])


def top_loadings(loadings: pd.DataFrame, n_pc: int = 5, top_n: int = 6) -> pd.DataFrame:
    """Variables con mayor contribucion absoluta a cada componente."""
    rows: list[dict] = []
    for pc in loadings.columns[:n_pc]:
        selected = loadings[pc].abs().nlargest(top_n).index
        for rank, feature in enumerate(selected, 1):
            rows.append(
                {
                    "componente": pc,
                    "rango": rank,
                    "feature": feature,
                    "loading": float(loadings.loc[feature, pc]),
                    "abs_loading": float(abs(loadings.loc[feature, pc])),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TF-IDF disperso para SVD
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z][a-z']{2,}")

# Lista pequena y auditable: elimina palabras funcionales que impedirian
# interpretar los factores. Los tokens de categoria (cat_*) se conservan.
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "was", "were", "are", "but",
    "you", "your", "they", "their", "our", "had", "has", "have", "not", "from",
    "very", "just", "all", "too", "can", "would", "could", "there", "here", "when",
    "what", "which", "who", "how", "also", "really", "been", "being", "will", "did",
    "does", "get", "got", "one", "out", "into", "about", "more", "some", "any", "than",
    "then", "them", "because", "only", "much", "even", "back", "first", "time", "place",
    "restaurant", "restaurants", "food", "good", "great", "nice", "like", "ordered",
    "order", "service", "come", "came", "went", "going", "try", "tried", "make", "made",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS]


def cargar_corpus_restaurantes(
    cutoff: dt.datetime = dt.datetime(2019, 1, 1),
    max_reviews_per_item: int = 15,
    max_chars_per_review: int = 250,
) -> tuple[list[str], list[str], dict[str, list[str]], pd.DataFrame]:
    """Agrega resenas previas a 2019 por restaurante con memoria acotada.

    Replica el corte temporal de la Parte IV. Devuelve IDs, documentos, tokens
    de categorias y metadata de negocio. La seleccion es determinista porque
    los IDs se ordenan antes de construir la matriz.
    """
    import pyarrow as pa
    import pyarrow.dataset as dsa

    dataset = dsa.dataset(str(GOLD / "resenas_universo.parquet"))
    scanner = dataset.scanner(
        columns=["business_id", "date", "text"],
        filter=dsa.field("date") < pa.scalar(cutoff),
    )
    kept: dict[str, list[str]] = {}
    for batch in scanner.to_batches():
        ids = batch.column("business_id").to_pylist()
        texts = batch.column("text").to_pylist()
        for business_id, text in zip(ids, texts):
            reviews = kept.get(business_id)
            snippet = (text or "")[:max_chars_per_review]
            if reviews is None:
                kept[business_id] = [snippet]
            elif len(reviews) < max_reviews_per_item:
                reviews.append(snippet)

    businesses = pd.read_parquet(
        GOLD / "negocios_universo.parquet",
        columns=["business_id", "name", "categories", "metro", "stars", "review_count"],
    )
    categories: dict[str, list[str]] = {}
    for business_id, values in zip(businesses["business_id"], businesses["categories"]):
        values = list(values) if values is not None else []
        categories[business_id] = [
            "cat_" + str(value).lower().replace(" ", "_").replace("&", "and").replace("/", "_")
            for value in values
            if str(value).lower() != "restaurants"
        ]

    business_ids = sorted(kept)
    documents = [" ".join(kept[business_id]) for business_id in business_ids]
    return business_ids, documents, categories, businesses


def build_tfidf_csr(
    business_ids: Iterable[str],
    documents: Iterable[str],
    extra_tokens: dict[str, list[str]] | None = None,
    max_vocab: int = 3000,
    min_df: int = 10,
    max_df_ratio: float = 0.55,
) -> dict:
    """Construye TF-IDF L2-normalizado en CSR sin scipy.

    Se hacen dos pasadas sobre los documentos para no conservar millones de
    tokens Python en memoria. TF = 1+log(conteo), IDF suavizado.
    """
    ids = list(business_ids)
    docs = list(documents)
    if len(ids) != len(docs):
        raise ValueError("business_ids y documents deben tener igual longitud")
    extra_tokens = extra_tokens or {}
    document_frequency: Counter = Counter()
    for business_id, document in zip(ids, docs):
        tokens = set(_tokenize(document))
        tokens.update(extra_tokens.get(business_id, []))
        document_frequency.update(tokens)

    n_docs = len(docs)
    candidates = [
        (term, frequency)
        for term, frequency in document_frequency.items()
        if frequency >= min_df and frequency < max_df_ratio * n_docs
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    vocab = {term: i for i, (term, _) in enumerate(candidates[:max_vocab])}
    inverse_vocab = np.empty(len(vocab), dtype=object)
    idf = np.empty(len(vocab), dtype=np.float64)
    for term, index in vocab.items():
        inverse_vocab[index] = term
        idf[index] = np.log((1 + n_docs) / (1 + document_frequency[term])) + 1.0

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for business_id, document in zip(ids, docs):
        counts = Counter(token for token in _tokenize(document) if token in vocab)
        counts.update(token for token in extra_tokens.get(business_id, []) if token in vocab)
        if counts:
            row_indices = np.fromiter((vocab[token] for token in counts), dtype=np.int32)
            row_values = np.fromiter((counts[token] for token in counts), dtype=np.float64)
            row_values = (1.0 + np.log(row_values)) * idf[row_indices]
            norm = np.linalg.norm(row_values)
            if norm > 0:
                row_values /= norm
            order = np.argsort(row_indices)
            indices.extend(row_indices[order].tolist())
            data.extend(row_values[order].tolist())
        indptr.append(len(indices))

    return {
        "business_ids": np.asarray(ids),
        "vocab": vocab,
        "inverse_vocab": inverse_vocab,
        "idf": idf,
        "indptr": np.asarray(indptr, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int32),
        "data": np.asarray(data, dtype=np.float32),
        "shape": (len(ids), len(vocab)),
    }


def dense_to_csr(matrix: np.ndarray) -> dict:
    """Convierte una matriz pequena a nuestro CSR; se usa en validaciones."""
    matrix = np.asarray(matrix, dtype=np.float32)
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for row in matrix:
        nz = np.flatnonzero(row)
        indices.extend(nz.tolist())
        data.extend(row[nz].tolist())
        indptr.append(len(indices))
    return {
        "business_ids": np.arange(len(matrix)),
        "inverse_vocab": np.asarray([f"v{i}" for i in range(matrix.shape[1])]),
        "indptr": np.asarray(indptr, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int32),
        "data": np.asarray(data, dtype=np.float32),
        "shape": matrix.shape,
    }


def _csr_block(csr: dict, start: int, end: int) -> np.ndarray:
    """Densifica solo un bloque pequeno de filas para usar BLAS local."""
    _, n_cols = csr["shape"]
    block = np.zeros((end - start, n_cols), dtype=np.float32)
    indptr, indices, data = csr["indptr"], csr["indices"], csr["data"]
    for local, row in enumerate(range(start, end)):
        a, b = indptr[row], indptr[row + 1]
        block[local, indices[a:b]] = data[a:b]
    return block


def csr_right_matmul(csr: dict, right: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Calcula A@B por bloques, siendo A nuestro CSR."""
    n_rows, n_cols = csr["shape"]
    right = np.asarray(right, dtype=np.float64)
    if right.ndim != 2 or right.shape[0] != n_cols:
        raise ValueError("right debe tener shape (n_columnas_A, p)")
    out = np.empty((n_rows, right.shape[1]), dtype=np.float64)
    for start in range(0, n_rows, block_size):
        end = min(start + block_size, n_rows)
        out[start:end] = _csr_block(csr, start, end) @ right
    return out


def csr_transpose_matmul(csr: dict, right: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Calcula A.T@B por bloques, siendo A nuestro CSR."""
    n_rows, n_cols = csr["shape"]
    right = np.asarray(right, dtype=np.float64)
    if right.ndim != 2 or right.shape[0] != n_rows:
        raise ValueError("right debe tener shape (n_filas_A, p)")
    out = np.zeros((n_cols, right.shape[1]), dtype=np.float64)
    for start in range(0, n_rows, block_size):
        end = min(start + block_size, n_rows)
        out += _csr_block(csr, start, end).T @ right[start:end]
    return out


@dataclass
class SVDResult:
    """SVD truncada A ~= U diag(s) Vt."""

    U: np.ndarray
    singular_values: np.ndarray
    Vt: np.ndarray
    total_energy: float


def randomized_svd_csr(
    csr: dict,
    k: int = 80,
    oversampling: int = 12,
    power_iterations: int = 2,
    seed: int = SEED,
) -> SVDResult:
    """SVD truncada manual sobre CSR mediante un subespacio aleatorio.

    1. Proyecta A sobre ``k+oversampling`` direcciones.
    2. Refina el subespacio con iteraciones de potencia A(A.T Q).
    3. Forma B=Q.T A, pequena, y obtiene sus singulares via eigen(B B.T).
    4. Recupera U y V sin densificar A completa.
    """
    n_rows, n_cols = csr["shape"]
    if not 1 <= k < min(n_rows, n_cols):
        raise ValueError("k debe ser menor que ambas dimensiones")
    width = min(k + oversampling, min(n_rows, n_cols))
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(n_cols, width)) / np.sqrt(width)
    Q, _ = np.linalg.qr(csr_right_matmul(csr, omega), mode="reduced")
    for _ in range(power_iterations):
        Z = csr_transpose_matmul(csr, Q)
        Q, _ = np.linalg.qr(csr_right_matmul(csr, Z), mode="reduced")

    B = csr_transpose_matmul(csr, Q).T
    gram = B @ B.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:k]
    singular_values = np.sqrt(np.maximum(values[order], 0.0))
    U_reduced = vectors[:, order]
    Vt = np.zeros((k, n_cols), dtype=np.float64)
    nonzero = singular_values > 1e-12
    Vt[nonzero] = (U_reduced[:, nonzero].T @ B) / singular_values[nonzero, None]
    U = Q @ U_reduced

    for j in range(k):
        pivot = int(np.argmax(np.abs(Vt[j])))
        if Vt[j, pivot] < 0:
            Vt[j] *= -1
            U[:, j] *= -1

    total_energy = float(np.dot(csr["data"], csr["data"]))
    return SVDResult(U=U, singular_values=singular_values, Vt=Vt, total_energy=total_energy)


def svd_curve(model: SVDResult, ks: Iterable[int] | None = None) -> pd.DataFrame:
    """Energia capturada y error de reconstruccion relativo para varios k."""
    max_k = len(model.singular_values)
    ks = list(ks) if ks is not None else list(range(1, max_k + 1))
    cumulative = np.cumsum(model.singular_values ** 2)
    rows = []
    for k in ks:
        if not 1 <= k <= max_k:
            raise ValueError("todos los k deben estar en el rango calculado")
        captured = float(cumulative[k - 1] / model.total_energy)
        rows.append(
            {
                "k": int(k),
                "energia_capturada": captured,
                "error_relativo": float(np.sqrt(max(0.0, 1.0 - captured))),
            }
        )
    return pd.DataFrame(rows)


def svd_compression_curve(csr: dict, ks: Iterable[int]) -> pd.DataFrame:
    """Compara parametros/bytes del original con factores truncados float32."""
    n_rows, n_cols = csr["shape"]
    dense_entries = n_rows * n_cols
    csr_bytes = csr["data"].nbytes + csr["indices"].nbytes + csr["indptr"].nbytes
    rows = []
    for k in ks:
        factor_entries = (n_rows + n_cols + 1) * int(k)
        factor_bytes = factor_entries * np.dtype(np.float32).itemsize
        rows.append(
            {
                "k": int(k),
                "parametros_factor": factor_entries,
                "compresion_vs_densa": dense_entries / factor_entries,
                "bytes_factores": factor_bytes,
                "ratio_bytes_vs_csr": factor_bytes / csr_bytes,
            }
        )
    return pd.DataFrame(rows)


def svd_factor_terms(model: SVDResult, inverse_vocab: Iterable[str], n_factors: int = 6, top_n: int = 8) -> pd.DataFrame:
    """Terminos positivos/negativos que permiten interpretar cada factor."""
    terms = np.asarray(list(inverse_vocab), dtype=object)
    rows: list[dict] = []
    for factor in range(min(n_factors, len(model.singular_values))):
        weights = model.Vt[factor]
        positive = np.argsort(weights)[-top_n:][::-1]
        negative = np.argsort(weights)[:top_n]
        for side, selected in (("positivo", positive), ("negativo", negative)):
            for rank, index in enumerate(selected, 1):
                rows.append(
                    {
                        "factor": factor + 1,
                        "lado": side,
                        "rango": rank,
                        "termino": str(terms[index]),
                        "peso": float(weights[index]),
                    }
                )
    return pd.DataFrame(rows)


def svd_representative_businesses(
    model: SVDResult,
    business_ids: Iterable[str],
    businesses: pd.DataFrame,
    n_factors: int = 6,
    top_n: int = 5,
) -> pd.DataFrame:
    """Restaurantes en los extremos de cada factor latente."""
    ids = np.asarray(list(business_ids))
    scores = model.U * model.singular_values[None, :]
    metadata = businesses.set_index("business_id")
    rows: list[dict] = []
    for factor in range(min(n_factors, scores.shape[1])):
        positive = np.argsort(scores[:, factor])[-top_n:][::-1]
        negative = np.argsort(scores[:, factor])[:top_n]
        for side, selected in (("positivo", positive), ("negativo", negative)):
            for rank, index in enumerate(selected, 1):
                business_id = ids[index]
                row = metadata.loc[business_id]
                rows.append(
                    {
                        "factor": factor + 1,
                        "lado": side,
                        "rango": rank,
                        "business_id": business_id,
                        "name": row["name"],
                        "metro": row["metro"],
                        "stars": float(row["stars"]),
                        "score": float(scores[index, factor]),
                    }
                )
    return pd.DataFrame(rows)
