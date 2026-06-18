"""Parte II — Análisis de grafos y ranking (PageRank, HITS, comunidades).

Cubre del `Enunciado_proyecto2.pdf`, **PARTE II** (4 pts):
  - PageRank iterativo e HITS (hubs/authorities) -> rankings de influencia.
  - Detección de comunidades (Girvan-Newman + Modularidad Q; greedy CNM).

Regla del curso: los algoritmos se implementan **a mano sobre numpy**
(sin librerías de grafos); numpy/pandas solo manipulan datos.

Fundamento teórico (diapositivas en `teoria/`, adaptadas de Leskovec et al.,
*Mining of Massive Datasets*, 3ª ed.):
  · Grafo como matriz de transición M columna-estocástica  -> deck
    «07 - Análisis de Enlaces - PageRank», **pág. 20** (Formulación matricial r = M·r).
  · Dead-end = nodo con grado de salida 0 que rompe la estocasticidad
    -> deck 07, **págs. 27-28**.
  (Las citas de cada algoritmo se anotan junto a su función al implementarlo.)

Diseño de datos: cada grafo se guarda en formato **CSR** (Compressed Sparse Row)
con arrays de numpy (`indptr`/`indices`), la misma representación dispersa del
BFS sobre CSR del notebook 04 — necesaria en una red gigante y dispersa donde
no caben matrices densas.

Salidas de la Parte II (estructura del repo):
  - tablas de resultados  -> `data/gold/` (p.ej. pagerank_*.parquet, comunidades_*.parquet)
  - figuras para informe   -> `docs/figs/`  (vía `src.viz.guardar`)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GOLD


# ===========================================================================
#  Carga de tablas gold
#  (Spark fue la capa de ingeniería que construyó gold; aquí los algoritmos
#   la consumen con numpy/pandas, como define el diseño del proyecto.)
# ===========================================================================
def _leer_gold(nombre: str, columnas: list[str]) -> pd.DataFrame:
    """Lee solo las columnas pedidas de una tabla gold (directorio Parquet)."""
    return pd.read_parquet(GOLD / nombre, columns=columnas)


# ===========================================================================
#  Estructura CSR y constructor a mano
# ===========================================================================
@dataclass
class Grafo:
    """Grafo dirigido en CSR por nodo origen.

    `indices[indptr[i]:indptr[i+1]]` = destinos de las aristas que salen de i.
    Un grafo no dirigido se guarda con cada arista en ambos sentidos.
    """
    indptr: np.ndarray            # (n+1,)
    indices: np.ndarray           # (n_aristas_dirigidas,)
    n: int                        # nº de nodos
    etiquetas: "np.ndarray | None" = None   # idx -> id original (str)

    @property
    def grado_salida(self) -> np.ndarray:
        return np.diff(self.indptr)


def construir_csr(origen: np.ndarray, destino: np.ndarray, n: int) -> Grafo:
    """Construye CSR a partir de listas de aristas (origen -> destino).

    Ordena los destinos por nodo origen con un *counting sort* vectorizado
    (bincount + cumsum): O(n + m), sin bucles Python.
    """
    origen = np.asarray(origen, dtype=np.int64)
    destino = np.asarray(destino, dtype=np.int64)
    orden = np.argsort(origen, kind="stable")
    indices = destino[orden]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(origen, minlength=n), out=indptr[1:])
    return Grafo(indptr=indptr, indices=indices, n=n)


# ===========================================================================
#  Grafo 1: red social de amistades (usuarios)
#  Sobre este grafo PageRank rankea usuarios influyentes (deck 07).
# ===========================================================================
def cargar_aristas_sociales(verbose: bool = True):
    """Devuelve (origen, destino, etiquetas, n_drop) de `aristas_sociales`.

    El índice de nodos es el universo completo de usuarios (`usuarios_universo`,
    813,792) para que los usuarios sin amigos sean visibles como aislados.
    Aún NO se decide simetrizar: el diagnóstico revela cómo viene almacenada
    la amistad y eso se decide en el Paso 2 (PageRank).
    """
    t = time.time()
    usuarios = _leer_gold("usuarios_universo.parquet", ["user_id"])["user_id"].to_numpy()
    pos = pd.Series(np.arange(len(usuarios), dtype=np.int64), index=usuarios)
    ar = _leer_gold("aristas_sociales.parquet", ["user_id", "amigo"])
    o = pos.reindex(ar["user_id"]).to_numpy()
    d = pos.reindex(ar["amigo"]).to_numpy()
    ok = ~(np.isnan(o) | np.isnan(d))
    n_drop = int((~ok).sum())
    o, d = o[ok].astype(np.int64), d[ok].astype(np.int64)
    if verbose:
        print(f"[social] {len(usuarios):,} usuarios · {o.size:,} aristas leídas · {time.time()-t:.1f}s")
    return o, d, usuarios, n_drop


def diagnostico_social(o, d, usuarios, n_drop) -> dict:
    """Estructura del grafo social: reciprocidad, grados, aislados, dead-ends.

    La reciprocidad decide si la amistad viene en una o dos direcciones — clave
    para construir bien M en PageRank (deck 07, pág. 20) y para contar dead-ends
    (deck 07, págs. 27-28)."""
    n, m = len(usuarios), o.size
    N = np.int64(n)
    claves = o * N + d                      # codifica cada arista (o,d) en un entero
    recip = float(np.isin(d * N + o, claves).mean())   # ¿existe también (d,o)?
    gout = np.bincount(o, minlength=n)
    gin = np.bincount(d, minlength=n)
    gtot = gout + gin
    aislados = int((gtot == 0).sum())
    deadends = int((gout == 0).sum())
    pct = lambda a: [int(np.percentile(a[a > 0], p)) for p in (50, 90, 99)] if (a > 0).any() else [0, 0, 0]
    sentido = ("AMBAS direcciones (no dirigido guardado x2)" if recip > 0.99
               else "UNA dirección" if recip < 0.05 else "mixto")
    print("=== GRAFO SOCIAL (amistades usuario–usuario) ===")
    print(f"nodos (usuarios del universo)        : {n:,}")
    print(f"aristas dirigidas almacenadas        : {m:,}")
    print(f"extremos fuera de universo (drop)    : {n_drop:,}")
    print(f"reciprocidad ¿(a,u) si existe (u,a)? : {recip*100:.1f}%  -> {sentido}")
    print(f"grado de salida  p50/p90/p99         : {pct(gout)}  (máx {int(gout.max()):,})")
    print(f"usuarios sin amigos (aislados)       : {aislados:,} ({aislados/n*100:.1f}%)")
    print(f"dead-ends (grado de salida 0)        : {deadends:,}")
    return dict(n=n, m=m, reciprocidad=recip, aislados=aislados, deadends=deadends)


# ===========================================================================
#  Grafo 2: bipartito usuario -> negocio (reseñas)
#  Encaje natural de HITS: usuarios = hubs, restaurantes = authorities.
# ===========================================================================
def cargar_aristas_bipartito(verbose: bool = True):
    """Devuelve (u_idx, b_idx) de `matriz_un` (índices enteros ya materializados)."""
    t = time.time()
    df = _leer_gold("matriz_un.parquet", ["u_idx", "b_idx"])
    u = df["u_idx"].to_numpy(np.int64)
    b = df["b_idx"].to_numpy(np.int64)
    if verbose:
        print(f"[bipartito] {u.size:,} aristas (reseñas) leídas · {time.time()-t:.1f}s")
    return u, b


def diagnostico_bipartito(u, b) -> dict:
    """Estructura del bipartito: tamaños, densidad, grados, sumideros.

    En el grafo dirigido usuario->negocio TODOS los negocios son sumideros
    (0 out-links): por eso PageRank dirigido sobre el bipartito exige teleporte
    (deck 07, págs. 30-32), mientras que HITS lo aprovecha de forma natural."""
    nu, nb, m = int(u.max()) + 1, int(b.max()) + 1, u.size
    gout_u = np.bincount(u, minlength=nu)       # reseñas por usuario
    gin_b = np.bincount(b, minlength=nb)        # reseñas por negocio
    pct = lambda a: [int(np.percentile(a[a > 0], p)) for p in (50, 90, 99)]
    print("=== GRAFO BIPARTITO (usuario -> negocio) ===")
    print(f"usuarios (con ≥1 reseña)              : {nu:,}")
    print(f"negocios                             : {nb:,}")
    print(f"aristas (reseñas)                    : {m:,}")
    print(f"densidad de la matriz                : {m/(nu*nb):.2e}")
    print(f"grado usuario (out) p50/p90/p99      : {pct(gout_u)}  (máx {int(gout_u.max()):,})")
    print(f"grado negocio (in)  p50/p90/p99      : {pct(gin_b)}  (máx {int(gin_b.max()):,})")
    print(f"sumideros (negocios, 0 out-links)    : {nb:,}")
    return dict(nu=nu, nb=nb, m=m)


# ===========================================================================
#  Simetrización y bipartito unificado (preparación para PageRank)
# ===========================================================================
def simetrizar(o, d):
    """Devuelve las aristas en AMBOS sentidos (grafo no dirigido).

    La amistad de Yelp es mutua, pero `aristas_sociales` la guarda una sola vez
    (el diagnóstico midió reciprocidad 0%); para PageRank hay que simetrizarla.
    """
    o, d = np.asarray(o), np.asarray(d)
    return np.concatenate([o, d]), np.concatenate([d, o])


def bipartito_unificado(u, b, n_users, n_biz):
    """Une usuarios [0..n_users) y negocios [n_users..n_users+n_biz) en un solo
    índice, con aristas usuario<->negocio en ambos sentidos (random walk no
    dirigido). Necesario porque en el bipartito dirigido todos los negocios son
    sumideros (ver `diagnostico_bipartito`)."""
    bb = b + n_users
    o = np.concatenate([u, bb])
    d = np.concatenate([bb, u])
    return o, d, n_users + n_biz


# ===========================================================================
#  PageRank — Power Iteration sobre la Google Matrix (a mano)
#  Fundamento: deck «07 - Análisis de Enlaces - PageRank»
#    · r = β·M·r + (1−β)/N, M columna-estocástica (M_ji = 1/d_i)  -> pág. 20
#    · Power Iteration; parada ‖Δr‖₁ < ε                          -> págs. 21-24
#    · Teleporte / Google Matrix (dead-ends y spider traps)        -> págs. 30-32
# ===========================================================================
def pagerank(grafo: Grafo, beta: float = 0.85, eps: float = 1e-6,
             max_iter: int = 100, verbose: bool = True) -> np.ndarray:
    """Vector PageRank del grafo (CSR). Conserva Σr = 1.

    La masa de los nodos colgantes (grado de salida 0) se redistribuye
    uniformemente — equivale a la fila de teletransportación de la Google Matrix
    y evita la fuga de probabilidad de los dead-ends.
    """
    n = grafo.n
    deg = grafo.grado_salida
    src = np.repeat(np.arange(n, dtype=np.int64), deg)   # nodo origen de cada arista
    dst = grafo.indices                                   # nodo destino de cada arista
    inv = np.zeros(n)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    colgantes = np.flatnonzero(deg == 0)
    r = np.full(n, 1.0 / n)
    err = np.nan
    for it in range(1, max_iter + 1):
        peso = (r * inv)[src]                              # cada arista i->j lleva r_i/d_i
        rnew = np.bincount(dst, weights=peso, minlength=n)
        fuga = r[colgantes].sum()                          # masa de nodos sin salida
        rnew = beta * (rnew + fuga / n) + (1.0 - beta) / n  # teleporte
        err = np.abs(rnew - r).sum()
        r = rnew
        if err < eps:
            if verbose:
                print(f"  PageRank: convergió en {it} iters (‖Δr‖₁={err:.2e}, Σr={r.sum():.4f})")
            return r
    if verbose:
        print(f"  PageRank: {max_iter} iters, ‖Δr‖₁={err:.2e}, Σr={r.sum():.4f}")
    return r


def top_k(scores: np.ndarray, k: int = 15, etiquetas=None) -> pd.DataFrame:
    """Top-k nodos por score (PageRank u otro), con su etiqueta si se da."""
    scores = np.asarray(scores)
    if k <= 0 or scores.size == 0:
        columns = ["idx"] + (["id"] if etiquetas is not None else []) + ["score"]
        return pd.DataFrame(columns=columns)
    k = min(k, scores.size)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    cols = {"idx": idx}
    if etiquetas is not None:
        cols["id"] = np.asarray(etiquetas)[idx]
    cols["score"] = scores[idx]
    return pd.DataFrame(cols)


def cargar_indices_bipartito():
    """Mapas idx->id del bipartito: (user_ids por u_idx, business_ids por b_idx)."""
    df = _leer_gold("matriz_un.parquet", ["u_idx", "b_idx", "user_id", "business_id"])
    nu = int(df["u_idx"].max()) + 1
    nb = int(df["b_idx"].max()) + 1
    us = np.empty(nu, dtype=object); us[df["u_idx"].to_numpy()] = df["user_id"].to_numpy()
    bs = np.empty(nb, dtype=object); bs[df["b_idx"].to_numpy()] = df["business_id"].to_numpy()
    return us, bs


# ===========================================================================
#  HITS — Hubs & Authorities (a mano)
#  Fundamento: HITS aparece en deck «07 - Análisis de Enlaces - PageRank»
#  (pág. 40) como modelo complementario a PageRank; su derivación —refuerzo
#  mutuo auth = Aᵀ·hub, hub = A·auth, con normalización, que converge a los
#  eigenvectores principales de AᵀA y AAᵀ— está en MMDS Cap. 5, la fuente que
#  las propias diapositivas citan. En el bipartito usuario->negocio:
#  usuarios = hubs (emiten reseñas), negocios = authorities (las reciben).
# ===========================================================================
def hits(o, d, n, eps: float = 1e-8, max_iter: int = 200, verbose: bool = True):
    """HITS sobre el grafo dirigido dado por aristas (o -> d). Devuelve (hub, auth).

    auth = Aᵀ·hub  (buena autoridad = la apuntan buenos hubs)
    hub  = A·auth  (buen hub = apunta a buenas autoridades)
    Normalización L2 por iteración.
    """
    o = np.asarray(o, np.int64); d = np.asarray(d, np.int64)
    hub = np.ones(n); auth = np.ones(n)
    err = np.nan
    for it in range(1, max_iter + 1):
        auth_new = np.bincount(d, weights=hub[o], minlength=n)        # Aᵀ·hub
        nrm = np.linalg.norm(auth_new); auth_new /= nrm if nrm else 1.0
        hub_new = np.bincount(o, weights=auth_new[d], minlength=n)    # A·auth
        nrm = np.linalg.norm(hub_new); hub_new /= nrm if nrm else 1.0
        err = np.abs(auth_new - auth).sum() + np.abs(hub_new - hub).sum()
        auth, hub = auth_new, hub_new
        if err < eps:
            if verbose:
                print(f"  HITS: convergió en {it} iters (Δ={err:.2e})")
            return hub, auth
    if verbose:
        print(f"  HITS: {max_iter} iters (Δ={err:.2e})")
    return hub, auth


def spearman(x, y) -> float:
    """Spearman a mano con rangos promedio para empates.

    Usar ``argsort(argsort(x))`` asigna rangos distintos a valores empatados y
    sesga la correlación cuando existen muchos conteos repetidos. Esta versión
    implementa el rango promedio sin depender de SciPy.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("x e y deben ser vectores 1D de igual longitud >= 2")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Spearman no acepta valores no finitos")

    def rangos_promedio(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks

    rx = rangos_promedio(x)
    ry = rangos_promedio(y)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))


# ===========================================================================
#  PARTE II — Detección de comunidades
#  Fundamento: deck «12 - Detección de Comunidades» (Semana 12, MMDS Cap. 10)
#    · Edge betweenness por BFS + propagación de crédito  -> págs. 9-22
#    · Algoritmo de Girvan-Newman (jerárquico divisivo)   -> págs. 14-19
#    · Modelo nulo y Modularidad Q                         -> págs. 31-34
#    · nº óptimo de comunidades = máximo de Q              -> pág. 33
#  Grafo elegido: co-reseña de restaurantes (micro-mercados, deck 12). Todo
#  implementado a mano (numpy/python), sin librerías de grafos.
# ===========================================================================
def grafo_coresena(metro: str = "Philadelphia", top_n: int = 150,
                   k_min: int = 15, metric: str = "count",
                   min_jaccard: float = 0.05, target_grado=None, verbose: bool = True):
    """Grafo de co-reseña entre restaurantes de un mercado.

    Nodos = `top_n` restaurantes del `metro` con más reseñas. La arista (i,j)
    según `metric`:
      - "count":   peso = nº de reseñadores en común; se conserva si >= `k_min`.
      - "jaccard": peso = común / (reseñas_i + reseñas_j − común); se conserva si
                   >= `min_jaccard`. **Jaccard controla la popularidad** (dos
                   restaurantes muy reseñados comparten gente por su tamaño, no por
                   público común), así que revela mejor los micro-mercados (deck 12).
    Devuelve (neg_df, ei, ej, ew) con las aristas no dirigidas (i<j).
    """
    import collections
    neg = _leer_gold("negocios_universo.parquet",
                     ["business_id", "name", "metro", "review_count",
                      "categories", "stars", "city", "postal_code"])
    neg = neg[neg["metro"] == metro].nlargest(top_n, "review_count").reset_index(drop=True)
    idx = {b: i for i, b in enumerate(neg["business_id"])}
    rev = _leer_gold("matriz_un.parquet", ["user_id", "business_id"])
    rev = rev[rev["business_id"].isin(idx)].copy()
    rev["r"] = rev["business_id"].map(idx)
    multi = rev.groupby("user_id")["r"].transform("size") >= 2   # solo co-reseñadores
    co = collections.Counter()
    for _, s in rev[multi].groupby("user_id")["r"]:
        v = sorted(set(s.tolist()))
        for x in range(len(v)):
            for y in range(x + 1, len(v)):
                co[(v[x], v[y])] += 1
    rc = neg["review_count"].to_numpy(float)        # ≈ reseñadores distintos por restaurante
    n = len(neg)
    cand = [(x, y, w, w / (rc[x] + rc[y] - w)) for (x, y), w in co.items()]   # (i, j, común, Jaccard)
    if target_grado is not None and cand:
        keep_n = int(np.ceil(n * target_grado / 2))
        sel = sorted(cand, key=lambda t: t[3], reverse=True)[:keep_n]
        thr = sel[-1][3] if sel else 0.0
        ei = [t[0] for t in sel]; ej = [t[1] for t in sel]; ew = [t[3] for t in sel]
        crit = f"top {len(sel)} por Jaccard (umbral efectivo {thr:.3f})"
    else:
        ei, ej, ew = [], [], []
        for x, y, w, j in cand:
            keep = (w >= k_min) if metric == "count" else (j >= min_jaccard)
            if keep:
                ei.append(x); ej.append(y)
                ew.append(float(w) if metric == "count" else float(j))
        crit = f"≥{k_min} co-reseñadores" if metric == "count" else f"Jaccard≥{min_jaccard}"
    ei, ej, ew = np.array(ei, int), np.array(ej, int), np.array(ew, float)
    if verbose:
        deg = np.bincount(np.concatenate([ei, ej]), minlength=n) if len(ei) else np.zeros(n)
        print(f"[co-reseña {metro} · {metric}] {n} restaurantes · {len(ei)} aristas ({crit}) · "
              f"aislados {int((deg == 0).sum())} · grado medio {2*len(ei)/max(n,1):.1f}")
        if cand:
            jp = np.percentile([t[3] for t in cand], [50, 90, 99]).round(3)
            print(f"   Jaccard de pares candidatos p50/p90/p99 = {jp.tolist()}")
    return neg, ei, ej, ew


def modularidad(labels, ei, ej, n, ew=None) -> float:
    """Modularidad Q de Newman (deck 12, págs. 32-34):
    Q = Σ_c [ L_c/m − (D_c/2m)² ], con m = peso total, L_c = peso intra-comunidad,
    D_c = suma de grados ponderados de la comunidad c."""
    labels = np.asarray(labels)
    if ew is None:
        ew = np.ones(len(ei))
    m = ew.sum()
    if m == 0:
        return 0.0
    k = np.zeros(n); np.add.at(k, ei, ew); np.add.at(k, ej, ew)
    nc = int(labels.max()) + 1
    same = labels[ei] == labels[ej]
    Lc = np.bincount(labels[ei][same], weights=ew[same], minlength=nc)
    Dc = np.bincount(labels, weights=k, minlength=nc)
    return float((Lc / m - (Dc / (2 * m)) ** 2).sum())


def _componentes(adj, n) -> np.ndarray:
    """Componentes conexas (etiqueta por nodo) por DFS iterativo."""
    label = -np.ones(n, int); c = 0
    for s in range(n):
        if label[s] >= 0:
            continue
        stack = [s]; label[s] = c
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if label[y] < 0:
                    label[y] = c; stack.append(y)
        c += 1
    return label


def _edge_betweenness(adj, n) -> dict:
    """Edge betweenness por BFS + propagación de crédito (Brandes), deck 12 págs. 20-22."""
    from collections import deque
    bet: dict = {}
    for s in range(n):
        S = []; pred = [[] for _ in range(n)]
        sigma = np.zeros(n); sigma[s] = 1.0
        dist = -np.ones(n); dist[s] = 0
        Q = deque([s])
        while Q:
            v = Q.popleft(); S.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1; Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]; pred[w].append(v)
        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in pred[w]:
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                e = (v, w) if v < w else (w, v)
                bet[e] = bet.get(e, 0.0) + c
                delta[v] += c
    for e in bet:
        bet[e] /= 2.0          # cada arista se cuenta desde ambos extremos
    return bet


def girvan_newman(ei, ej, n, ew=None, max_remueve: int = 400, verbose: bool = True):
    """Girvan-Newman (deck 12): quita iterativamente la arista de mayor
    betweenness y se queda con la partición de **máxima Modularidad Q**."""
    adj = [set() for _ in range(n)]
    for a, b in zip(ei.tolist(), ej.tolist()):
        adj[a].add(b); adj[b].add(a)
    best_Q, best_lab = -1.0, _componentes(adj, n)
    hist, rem, total = [], 0, int(len(ei))
    while rem < min(max_remueve, total):
        bet = _edge_betweenness(adj, n)
        if not bet:
            break
        a, b = max(bet, key=bet.get)
        adj[a].discard(b); adj[b].discard(a)
        rem += 1
        lab = _componentes(adj, n)
        Q = modularidad(lab, ei, ej, n, ew)
        hist.append((rem, int(lab.max()) + 1, Q))
        if Q > best_Q:
            best_Q, best_lab = Q, lab.copy()
        if Q < best_Q - 0.05 and lab.max() + 1 > 3:   # ya pasamos el pico de Q
            break
    _, best_lab = np.unique(best_lab, return_inverse=True)
    if verbose:
        print(f"  Girvan-Newman: {rem} aristas removidas · mejor Q={best_Q:.3f} · "
              f"{best_lab.max()+1} comunidades")
    return best_lab, best_Q, hist


def greedy_modularidad(ei, ej, n, ew=None, verbose: bool = True):
    """Aglomerativo tipo CNM: fusiona la pareja de comunidades (conectadas por
    una arista) que da mayor Q; se queda con la partición de Q máxima. Se funda
    en la misma Modularidad Q del deck 12 (págs. 32-34) y escala mejor que GN."""
    lab = np.arange(n)
    Q_best = modularidad(lab, ei, ej, n, ew); best = lab.copy()
    while True:
        pares = set()
        ca, cb = lab[ei], lab[ej]
        for x, y in zip(ca.tolist(), cb.tolist()):
            if x != y:
                pares.add((min(x, y), max(x, y)))
        if not pares:
            break
        mejor_Q, mejor_lab = -2.0, None
        for (x, y) in pares:
            trial = lab.copy(); trial[trial == y] = x
            Q = modularidad(trial, ei, ej, n, ew)
            if Q > mejor_Q:
                mejor_Q, mejor_lab = Q, trial
        lab = mejor_lab
        if mejor_Q > Q_best:
            Q_best, best = mejor_Q, lab.copy()
        if len(set(lab.tolist())) == 1 or mejor_Q < Q_best - 0.05:
            break
    _, best = np.unique(best, return_inverse=True)
    if verbose:
        print(f"  Greedy (CNM): mejor Q={Q_best:.3f} · {best.max()+1} comunidades")
    return best, Q_best


def caracterizar(neg, labels, ei, ej, ew=None, min_size: int = 3):
    """Resumen por comunidad: tamaño, densidad interna, estrellas medias,
    categorías dominantes y restaurantes representativos (deck 12: caracterizar
    cada comunidad)."""
    labels = np.asarray(labels)
    if ew is None:
        ew = np.ones(len(ei))
    filas = []
    for c in range(int(labels.max()) + 1):
        miembros = np.flatnonzero(labels == c)
        if len(miembros) < min_size:
            continue
        intra = (labels[ei] == c) & (labels[ej] == c)
        size = len(miembros)
        dens = intra.sum() / (size * (size - 1) / 2) if size > 1 else 0.0
        sub = neg.iloc[miembros]
        cats = (sub["categories"].explode()
                .loc[lambda s: ~s.isin(["Restaurants", "Food"])]
                .value_counts().head(3).index.tolist())
        ejemplos = sub.nlargest(3, "review_count")["name"].tolist()
        filas.append({"comunidad": c, "tamaño": size,
                      "densidad_interna": round(float(dens), 2),
                      "estrellas_media": round(float(sub["stars"].mean()), 2),
                      "categorías_top": ", ".join(cats),
                      "ejemplos": " · ".join(ejemplos)})
    return pd.DataFrame(filas).sort_values("tamaño", ascending=False)


def layout_resorte(ei, ej, n, iters: int = 250, seed: int = 42) -> np.ndarray:
    """Layout force-directed (Fruchterman-Reingold) a mano para dibujar el grafo.

    Fuerza repulsiva k²/d entre todos los nodos y atractiva d²/k a lo largo de las
    aristas; enfriamiento que limita el desplazamiento. O(n²) por iteración — apto
    para grafos pequeños (la red de co-reseña). Devuelve posiciones (n,2) en [0,1].
    """
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 2)) * 0.1
    k = 1.0 / np.sqrt(max(n, 1))
    for it in range(iters):
        diff = pos[:, None, :] - pos[None, :, :]          # (n,n,2)
        dist = np.sqrt((diff ** 2).sum(-1)) + 1e-9
        disp = ((k * k / dist ** 2)[..., None] * diff).sum(axis=1)   # repulsión
        d = pos[ei] - pos[ej]
        dd = np.sqrt((d ** 2).sum(1)) + 1e-9
        f = ((dd * dd / k)[:, None]) * (d / dd[:, None])             # atracción
        np.add.at(disp, ei, -f); np.add.at(disp, ej, f)
        t = 0.1 * (1 - it / iters) + 1e-3                            # enfriamiento
        length = np.sqrt((disp ** 2).sum(1)) + 1e-9
        pos = pos + (disp / length[:, None]) * np.minimum(length, t)[:, None]
    pos -= pos.min(0); pos /= (pos.max(0) + 1e-9)
    return pos


def red_comunidades(ei, ej, n, comunidades, min_size: int = 4, seed: int = 3, sep: float = 1.5):
    """Prepara la red de comunidades para dibujarla de forma legible.

    Filtra a las comunidades grandes (>= `min_size`), descarta nodos aislados/menores
    (que afean el layout), posiciona cada comunidad como un clúster en un círculo
    (layout agrupado, más claro que el de resortes para comunidades laxas) y calcula
    un PageRank local para el tamaño de los nodos. Devuelve un dict con todo lo
    necesario para el dibujo: keep, ei, ej (remapeadas), comunidad, pos, pagerank.
    """
    comunidades = np.asarray(comunidades)
    cu, sz = np.unique(comunidades, return_counts=True)
    big = cu[sz >= min_size]
    bigset = set(int(c) for c in big)
    keep = np.flatnonzero([int(c) in bigset for c in comunidades])
    remap = -np.ones(n, int); remap[keep] = np.arange(len(keep))
    emask = np.array([(int(comunidades[a]) in bigset and int(comunidades[b]) in bigset)
                      for a, b in zip(ei, ej)], dtype=bool)
    ei2, ej2 = remap[ei[emask]], remap[ej[emask]]
    m = len(keep); lab2 = comunidades[keep]
    o, d = simetrizar(ei2, ej2)
    pr = pagerank(construir_csr(o, d, m), verbose=False) if m else np.array([])
    rng = np.random.default_rng(seed); pos = np.zeros((m, 2))
    for i, c in enumerate(big):
        ang = 2 * np.pi * i / len(big)
        cen = np.array([np.cos(ang), np.sin(ang)]) * sep
        idx = np.flatnonzero(lab2 == c)
        r = 0.13 + 0.05 * np.sqrt(len(idx))
        th = rng.uniform(0, 2 * np.pi, len(idx)); rr = r * np.sqrt(rng.uniform(0, 1, len(idx)))
        pos[idx] = cen + np.c_[rr * np.cos(th), rr * np.sin(th)]
    return dict(keep=keep, ei=ei2, ej=ej2, comunidad=lab2, pos=pos,
                pagerank=pr, comunidades_grandes=big)
