"""Parte V - Mineria de flujos de datos.

Cubre del ``Enunciado_proyecto2.pdf``, Parte V:

* ventanas deslizantes exactas de 1h, 4h y 1 dia;
* Count-Min Sketch para frecuencias aproximadas con cota de error;
* DGIM como tecnica adicional para contar horas activas recientes con memoria
  sublineal.

Fundamento: ``teoria/06 - Mineria de DataStream.pdf``:

* modelo de una pasada y memoria acotada: pags. 4-7;
* ventanas y DGIM: pags. 16-23;
* Count-Min Sketch: pags. 33-37.

Los algoritmos se implementan a mano. DuckDB/PyArrow se usan en el notebook
solo para leer y agrupar Parquet; no sustituyen ninguna tecnica del curso.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np
import pandas as pd

from .config import SEED


# ---------------------------------------------------------------------------
# Ventanas deslizantes exactas
# ---------------------------------------------------------------------------
@dataclass
class WindowState:
    """Estado devuelto por una ventana exacta."""

    count: int
    total: float
    average: float


class SlidingWindow:
    """Ventana temporal exacta procesada elemento por elemento.

    Cada elemento es ``(timestamp, valor)``. El timestamp puede ser un entero
    en segundos/horas siempre que ``width`` use la misma unidad. La estructura
    conserva unicamente los elementos que siguen dentro de la ventana.
    """

    def __init__(self, width: int | float):
        if width <= 0:
            raise ValueError("width debe ser positivo")
        self.width = width
        self.queue: deque[tuple[float, float]] = deque()
        self.total = 0.0
        self.last_timestamp: float | None = None

    def update(self, timestamp: int | float, value: int | float = 1.0) -> WindowState:
        timestamp = float(timestamp)
        value = float(value)
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("el stream debe llegar ordenado por timestamp")
        self.last_timestamp = timestamp
        self.queue.append((timestamp, value))
        self.total += value
        boundary = timestamp - self.width
        while self.queue and self.queue[0][0] <= boundary:
            _, old_value = self.queue.popleft()
            self.total -= old_value
        count = len(self.queue)
        return WindowState(count=count, total=self.total, average=self.total / count if count else 0.0)

    @property
    def memory_items(self) -> int:
        return len(self.queue)


def rolling_window_series(
    timestamps: Iterable[int | float],
    values: Iterable[int | float],
    widths: Iterable[int | float],
) -> dict[float, dict[str, np.ndarray]]:
    """Ejecuta varias ventanas exactas en una sola pasada.

    Devuelve, para cada ancho, las series de ``count``, ``sum`` y ``average``.
    """
    ts = np.asarray(list(timestamps), dtype=float)
    vals = np.asarray(list(values), dtype=float)
    if len(ts) != len(vals):
        raise ValueError("timestamps y values deben tener la misma longitud")
    if len(ts) and np.any(np.diff(ts) < 0):
        raise ValueError("timestamps debe estar ordenado")
    windows = {float(width): SlidingWindow(width) for width in widths}
    result = {
        width: {
            "count": np.empty(len(ts), dtype=np.int32),
            "sum": np.empty(len(ts), dtype=np.float64),
            "average": np.empty(len(ts), dtype=np.float64),
        }
        for width in windows
    }
    for i, (timestamp, value) in enumerate(zip(ts, vals)):
        for width, window in windows.items():
            state = window.update(timestamp, value)
            result[width]["count"][i] = state.count
            result[width]["sum"][i] = state.total
            result[width]["average"][i] = state.average
    return result


def hourly_windows(hourly: pd.DataFrame, time_col: str = "hora", value_col: str = "eventos") -> pd.DataFrame:
    """Completa huecos horarios y calcula ventanas 1h/4h/24h a mano."""
    data = hourly[[time_col, value_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.groupby(time_col, as_index=True)[value_col].sum().sort_index()
    full_index = pd.date_range(data.index.min().floor("h"), data.index.max().floor("h"), freq="h")
    data = data.reindex(full_index, fill_value=0)
    timestamps = np.arange(len(data), dtype=np.int64)
    computed = rolling_window_series(timestamps, data.to_numpy(float), widths=(1, 4, 24))
    out = pd.DataFrame({"hora": full_index, "eventos_1h": computed[1.0]["sum"]})
    out["eventos_4h"] = computed[4.0]["sum"]
    out["eventos_24h"] = computed[24.0]["sum"]
    out["promedio_hora_4h"] = computed[4.0]["average"]
    out["promedio_hora_24h"] = computed[24.0]["average"]
    return out


# ---------------------------------------------------------------------------
# Count-Min Sketch
# ---------------------------------------------------------------------------
class CountMinSketch:
    """Count-Min Sketch con hashes universales reproducibles.

    La estimacion nunca subestima. Con ``epsilon=1/width`` y
    ``delta~=exp(-depth)``, el error es a lo sumo ``epsilon*N`` con
    probabilidad al menos ``1-delta`` para una consulta fija.
    """

    PRIME = np.uint64(4_294_967_311)

    def __init__(self, width: int, depth: int, seed: int = SEED, dtype=np.uint64):
        if width <= 0 or depth <= 0:
            raise ValueError("width y depth deben ser positivos")
        self.width = int(width)
        self.depth = int(depth)
        self.table = np.zeros((self.depth, self.width), dtype=dtype)
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(self.PRIME), size=self.depth, dtype=np.uint64)
        self.b = rng.integers(0, int(self.PRIME), size=self.depth, dtype=np.uint64)
        self.n_updates = 0

    def _columns(self, codes: np.ndarray, depth: int | None = None) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.uint64) + np.uint64(1)
        used_depth = self.depth if depth is None else int(depth)
        if not 1 <= used_depth <= self.depth:
            raise ValueError("depth de consulta fuera de rango")
        hashed = (self.a[:used_depth, None] * codes[None, :] + self.b[:used_depth, None]) % self.PRIME
        return (hashed % np.uint64(self.width)).astype(np.int64)

    def update(self, code: int, count: int = 1) -> None:
        self.update_batch(np.asarray([code]), np.asarray([count]))

    def update_batch(self, codes: np.ndarray, counts: np.ndarray | None = None) -> None:
        codes = np.asarray(codes, dtype=np.int64)
        if np.any(codes < 0):
            raise ValueError("codes debe contener enteros no negativos")
        if counts is None:
            counts = np.ones(len(codes), dtype=self.table.dtype)
        else:
            counts = np.asarray(counts, dtype=self.table.dtype)
        if len(codes) != len(counts):
            raise ValueError("codes y counts deben tener la misma longitud")
        if np.any(counts < 0):
            raise ValueError("Count-Min Sketch no acepta incrementos negativos")
        columns = self._columns(codes)
        for row in range(self.depth):
            np.add.at(self.table[row], columns[row], counts)
        self.n_updates += int(counts.sum())

    def query(self, code: int, depth: int | None = None) -> int:
        return int(self.query_batch(np.asarray([code]), depth=depth)[0])

    def query_batch(self, codes: np.ndarray, depth: int | None = None) -> np.ndarray:
        used_depth = self.depth if depth is None else int(depth)
        columns = self._columns(codes, used_depth)
        values = np.vstack([self.table[row, columns[row]] for row in range(used_depth)])
        return values.min(axis=0)

    @property
    def epsilon(self) -> float:
        return 1.0 / self.width

    @property
    def delta(self) -> float:
        return exp(-self.depth)

    @property
    def memory_bytes(self) -> int:
        return int(self.table.nbytes + self.a.nbytes + self.b.nbytes)

    def error_bound(self, total_events: int | None = None) -> float:
        total = self.n_updates if total_events is None else total_events
        return self.epsilon * total


def evaluate_cms(
    exact: np.ndarray,
    estimated: np.ndarray,
    total_events: int,
    width: int,
    depth: int,
    memory_bytes: int,
    top_k: int = 20,
) -> dict:
    """Metricas empiricas y teoricas de un Count-Min Sketch."""
    exact = np.asarray(exact, dtype=np.int64)
    estimated = np.asarray(estimated, dtype=np.int64)
    if exact.shape != estimated.shape:
        raise ValueError("exact y estimated deben tener igual shape")
    error = estimated - exact
    if np.any(error < 0):
        raise AssertionError("Count-Min Sketch nunca debe subestimar")
    exact_top = set(np.argsort(exact)[-top_k:])
    estimated_top = set(np.argsort(estimated)[-top_k:])
    bound = total_events / width
    return {
        "width": int(width),
        "depth": int(depth),
        "epsilon": 1.0 / width,
        "delta_aprox": float(exp(-depth)),
        "cota_epsilon_N": float(bound),
        "error_medio": float(error.mean()),
        "error_p95": float(np.percentile(error, 95)),
        "error_max": int(error.max()),
        "fraccion_exacta": float(np.mean(error == 0)),
        "fraccion_dentro_cota": float(np.mean(error <= bound)),
        "overlap_top_k": float(len(exact_top & estimated_top) / top_k),
        "memoria_bytes": int(memory_bytes),
    }


# ---------------------------------------------------------------------------
# DGIM
# ---------------------------------------------------------------------------
@dataclass(order=True)
class DGIMBucket:
    """Bucket: numero de unos y timestamp del uno mas reciente."""

    timestamp: int
    size: int


class DGIM:
    """DGIM para contar unos en los ultimos N bits.

    Los buckets se mantienen de mas antiguo a mas reciente. Hay como maximo dos
    buckets de cada potencia de dos. La consulta suma todos salvo el mas antiguo
    y usa la mitad de este, tal como indica el deck 06.
    """

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size debe ser positivo")
        self.window_size = int(window_size)
        self.buckets: list[DGIMBucket] = []
        self.time = -1

    def update(self, bit: int, timestamp: int | None = None) -> None:
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError("DGIM solo acepta bits 0/1")
        timestamp = self.time + 1 if timestamp is None else int(timestamp)
        if timestamp <= self.time:
            raise ValueError("timestamps deben crecer estrictamente")
        self.time = timestamp
        boundary = self.time - self.window_size + 1
        self.buckets = [bucket for bucket in self.buckets if bucket.timestamp >= boundary]
        if bit == 1:
            self.buckets.append(DGIMBucket(timestamp=timestamp, size=1))
            self._compress()

    def _compress(self) -> None:
        size = 1
        while size <= self.window_size:
            positions = [i for i, bucket in enumerate(self.buckets) if bucket.size == size]
            if len(positions) <= 2:
                size *= 2
                continue
            old, newer = positions[0], positions[1]
            merged = DGIMBucket(timestamp=self.buckets[newer].timestamp, size=size * 2)
            for index in sorted((old, newer), reverse=True):
                self.buckets.pop(index)
            insert_at = int(np.searchsorted([bucket.timestamp for bucket in self.buckets], merged.timestamp, side="right"))
            self.buckets.insert(insert_at, merged)
            # La fusion puede crear el tercer bucket del siguiente tamano.
            size *= 2

    def estimate(self, last_k: int | None = None) -> float:
        if self.time < 0 or not self.buckets:
            return 0.0
        k = self.window_size if last_k is None else int(last_k)
        if not 1 <= k <= self.window_size:
            raise ValueError("last_k debe estar entre 1 y window_size")
        boundary = self.time - k + 1
        selected = [bucket for bucket in self.buckets if bucket.timestamp >= boundary]
        if not selected:
            return 0.0
        oldest = selected[0]
        return float(sum(bucket.size for bucket in selected[1:]) + oldest.size / 2.0)

    def invariant_ok(self) -> bool:
        counts: dict[int, int] = {}
        for bucket in self.buckets:
            counts[bucket.size] = counts.get(bucket.size, 0) + 1
        ordered = all(a.timestamp <= b.timestamp for a, b in zip(self.buckets, self.buckets[1:]))
        powers = all(bucket.size > 0 and bucket.size & (bucket.size - 1) == 0 for bucket in self.buckets)
        return ordered and powers and all(value <= 2 for value in counts.values())

    @property
    def memory_buckets(self) -> int:
        return len(self.buckets)


def evaluate_dgim(bits: Iterable[int], window_size: int, query_every: int = 1) -> pd.DataFrame:
    """Compara DGIM con la cuenta exacta a lo largo de un stream binario."""
    bits = np.asarray(list(bits), dtype=np.int8)
    model = DGIM(window_size)
    prefix = np.concatenate([[0], np.cumsum(bits, dtype=np.int64)])
    rows: list[dict] = []
    for time, bit in enumerate(bits):
        model.update(int(bit), timestamp=time)
        if (time + 1) % query_every != 0:
            continue
        start = max(0, time - window_size + 1)
        exact = int(prefix[time + 1] - prefix[start])
        estimate = model.estimate()
        abs_error = abs(estimate - exact)
        rows.append(
            {
                "time": time,
                "exact": exact,
                "estimate": estimate,
                "abs_error": abs_error,
                "relative_error": abs_error / exact if exact else 0.0,
                "buckets": model.memory_buckets,
                "invariant_ok": model.invariant_ok(),
            }
        )
    return pd.DataFrame(rows)
