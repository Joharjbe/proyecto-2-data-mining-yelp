"""Parte IV — Sistemas de recomendación híbridos (implementados a mano, numpy).

Cubre del `Enunciado_proyecto2.pdf`, PARTE IV (4 pts): filtrado colaborativo
(item-item), content-based (TF-IDF), híbrido, y evaluación rigurosa
Precision@K / Recall@K / NDCG / RMSE / MAE vs baselines (random, top-popular).

Fundamento teórico (diapositivas en `teoria/`):
  · CF item-item (vecinos, similitud)        -> deck «09 - Recomendacion», págs. 34-41
  · Pearson / coseno / adjusted cosine       -> deck 09 págs. 27-33 · deck 10 págs. 8, 11
  · Predictor baseline (μ + b_u + b_i)        -> deck 09 pág. 40 · deck 10 pág. 9
  · TF-IDF y content-based                    -> deck 09 págs. 14-21 (TF-IDF 16-17)
  · RMSE / MAE                                -> deck 09 pág. 43 · deck 10 págs. 4-6
  · Cold-start                                -> deck 09 pág. 42
  · NDCG / Precision@K / Recall@K  -> **NO están en los decks del curso**; se fundan
       en definiciones de Recuperación de Información estándar (MMDS, cap. 9) y las
       exige el enunciado. Se cita honestamente, igual que HITS en la Parte II.
  · Híbrido CF+contenido (weighted / switching) -> práctica estándar + enunciado;
       el "blend" que sí está en los decks es baseline+vecindario (cosa distinta).

Regla del curso: los algoritmos van a mano sobre numpy; pandas solo manipula datos.

Sin fuga de datos: **split TEMPORAL** (train ≤2018 · val 2019 · test 2020-21). Los
hiperparámetros se calibran en validación (2019) y el test se evalúa UNA sola vez.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GOLD, SEED


# ===========================================================================
#  Paso 1 — Datos e interacciones (split temporal sin fuga)
# ===========================================================================
def cargar_interacciones() -> pd.DataFrame:
    """`matriz_un` deduplicada a la PRIMERA interacción por (usuario, restaurante).

    Para recomendación de restaurantes *nuevos* nos quedamos con la primera visita
    de cada par (90,520 pares se repiten). Devuelve user_id, business_id, u_idx,
    b_idx, stars, year.
    """
    df = pd.read_parquet(
        GOLD / "matriz_un.parquet",
        columns=["user_id", "business_id", "u_idx", "b_idx", "stars", "date"],
    )
    df = df.sort_values("date").drop_duplicates(["user_id", "business_id"], keep="first")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return df.reset_index(drop=True)


def split_temporal(df: pd.DataFrame):
    """train ≤2018 · val 2019 · test 2020-21 (2022 —solo 19 días— se descarta)."""
    train = df[df["year"] <= 2018].copy()
    val = df[df["year"] == 2019].copy()
    test = df[df["year"].isin([2020, 2021])].copy()
    return train, val, test


def conjuntos_warm(train: pd.DataFrame, min_user: int = 3, min_item: int = 1):
    """Usuarios e ítems 'warm' = con historial suficiente en entrenamiento.

    Un usuario es warm si tiene >= `min_user` interacciones en train (sino el CF
    no tiene vecinos fiables): con min_user=3 → 150,350 usuarios warm. Un ítem es
    warm si aparece en train (min_item=1) → 25,952 restaurantes. El resto se trata
    como cold-start (se sirve con content-based / top-popular)."""
    cu = train.groupby("user_id").size()
    ci = train.groupby("business_id").size()
    return set(cu[cu >= min_user].index), set(ci[ci >= min_item].index)


def cohortes_cold_start(test: pd.DataFrame, warm_users: set, warm_items: set) -> pd.DataFrame:
    """Marca cada interacción de test por cohorte warm/cold de usuario e ítem."""
    test = test.copy()
    test["u_warm"] = test["user_id"].isin(warm_users)
    test["i_warm"] = test["business_id"].isin(warm_items)
    test["cohorte"] = np.where(
        test["u_warm"] & test["i_warm"], "warm_user-warm_item",
        np.where(test["u_warm"] & ~test["i_warm"], "warm_user-cold_item",
                 np.where(~test["u_warm"] & test["i_warm"], "cold_user-warm_item",
                          "cold_user-cold_item")))
    return test


# ===========================================================================
#  Paso 2 — Filtrado colaborativo ITEM-ITEM (Pearson + baseline + shrinkage)
#  Fundamento: deck «09 - Recomendacion» — item-item págs. 34-41; Pearson y
#  vecinos págs. 27-33; predictor baseline pág. 40 · deck 10 págs. 8-11.
#  Predicción:  r̂_ui = b_ui + Σ_j sim(i,j)·(r_uj − b_uj) / Σ_j |sim(i,j)|,
#  con b_ui = μ + b_u + b_i (regularizado) y j ∈ vecinos(i) que u ya valoró.
# ===========================================================================
def matriz_train(train: pd.DataFrame) -> dict:
    """Indexa train a enteros contiguos (usuarios e ítems) para numpy."""
    uu = pd.unique(train["user_id"]); ii = pd.unique(train["business_id"])
    u2x = {u: k for k, u in enumerate(uu)}; i2x = {b: k for k, b in enumerate(ii)}
    ux = train["user_id"].map(u2x).to_numpy(np.int64)
    ix = train["business_id"].map(i2x).to_numpy(np.int64)
    r = train["stars"].to_numpy(np.float64)
    return dict(ux=ux, ix=ix, r=r, users=uu, items=ii, u2x=u2x, i2x=i2x,
                n_users=len(uu), n_items=len(ii))


def baseline(M: dict, l_i: float = 10.0, l_u: float = 15.0):
    """Predictor baseline regularizado μ + b_u + b_i (deck 09 pág. 40)."""
    ux, ix, r, nu, ni = M["ux"], M["ix"], M["r"], M["n_users"], M["n_items"]
    mu = float(r.mean())
    n_i = np.bincount(ix, minlength=ni)
    b_i = np.bincount(ix, weights=r - mu, minlength=ni) / (l_i + n_i)
    n_u = np.bincount(ux, minlength=nu)
    b_u = np.bincount(ux, weights=r - mu - b_i[ix], minlength=nu) / (l_u + n_u)
    return mu, b_u, b_i


def similitud_item_item(M: dict, min_coratings: int = 5, shrink: float = 50.0,
                        topk: int = 40, cap_user: int = 50):
    """Similitud Pearson item-item (centrada por media del ítem) con shrinkage
    n/(n+λ) y solo pares con ≥ `min_coratings` usuarios comunes. Devuelve CSR de
    vecinos top-k: (indptr, vecino, sim)."""
    ux, ix, r, ni = M["ux"], M["ix"], M["r"], M["n_items"]
    mean_i = np.bincount(ix, weights=r, minlength=ni) / np.maximum(np.bincount(ix, minlength=ni), 1)
    cr = r - mean_i[ix]                                  # rating centrado por ítem
    order = np.argsort(ux, kind="stable")
    ixs, crs, uxs = ix[order], cr[order], ux[order]
    cuts = np.flatnonzero(np.diff(uxs)) + 1
    starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(uxs)]])
    PI, PJ, PP, PII, PJJ = [], [], [], [], []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s < 2:
            continue
        e = min(e, s + cap_user)                         # cap a hiperactivos
        it = ixs[s:e]; c = crs[s:e]
        a, b = np.triu_indices(len(it), k=1)
        lo = np.minimum(it[a], it[b]); hi = np.maximum(it[a], it[b])
        PI.append(lo); PJ.append(hi); PP.append(c[a] * c[b])
        PII.append(c[a] ** 2); PJJ.append(c[b] ** 2)
    if not PI:
        return {
            "indptr": np.zeros(ni + 1, np.int64),
            "vecino": np.array([], dtype=np.int64),
            "sim": np.array([], dtype=np.float64),
        }
    PI = np.concatenate(PI); PJ = np.concatenate(PJ)
    PP = np.concatenate(PP); PII = np.concatenate(PII); PJJ = np.concatenate(PJJ)
    key = PI * ni + PJ
    g = pd.DataFrame({"k": key, "p": PP, "di": PII, "dj": PJJ}).groupby("k")
    agg = g.agg(num=("p", "sum"), di=("di", "sum"), dj=("dj", "sum"), n=("p", "size"))
    agg = agg[agg["n"] >= min_coratings]
    sim = (agg["n"] / (agg["n"] + shrink)) * agg["num"] / np.sqrt(agg["di"] * agg["dj"] + 1e-12)
    ki = agg.index.to_numpy(); ii_ = ki // ni; jj_ = ki % ni; sv = sim.to_numpy()
    keep = np.isfinite(sv)
    ii_, jj_, sv = ii_[keep], jj_[keep], sv[keep]
    # aristas en ambos sentidos → top-k por ítem
    src = np.concatenate([ii_, jj_]); dst = np.concatenate([jj_, ii_]); val = np.concatenate([sv, sv])
    e = pd.DataFrame({"i": src, "j": dst, "s": val}).sort_values(["i", "s"], ascending=[True, False])
    e = e.groupby("i").head(topk)
    indptr = np.zeros(ni + 1, np.int64)
    np.add.at(indptr, e["i"].to_numpy() + 1, 1); np.cumsum(indptr, out=indptr)
    return dict(indptr=indptr, vecino=e["j"].to_numpy(np.int64), sim=e["s"].to_numpy(np.float64))


def predecir(u_items_idx, u_items_r, mu, b_u_val, b_i, nbrs, i_target):
    """r̂ para (usuario, i_target) dado lo que el usuario ya valoró (índices+ratings)."""
    a, b = nbrs["indptr"][i_target], nbrs["indptr"][i_target + 1]
    vecinos = nbrs["vecino"][a:b]; sims = nbrs["sim"][a:b]
    base = mu + b_u_val + b_i[i_target]
    if len(vecinos) == 0 or len(u_items_idx) == 0:
        return base
    pos = {it: k for k, it in enumerate(u_items_idx)}
    num = den = 0.0
    for v, s in zip(vecinos.tolist(), sims.tolist()):
        k = pos.get(v)
        if k is not None:
            b_uj = mu + b_u_val + b_i[v]
            num += s * (u_items_r[k] - b_uj); den += abs(s)
    return base + (num / den if den > 0 else 0.0)


# ===========================================================================
#  Paso 3 — Content-based: TF-IDF de reseñas + categorías (a mano)
#  Fundamento: deck «09 - Recomendacion» — content-based págs. 14-21,
#  TF-IDF págs. 16-17. Recomienda ítems similares (coseno) a los que el
#  usuario valoró bien.
# ===========================================================================
import re as _re

_STOP = set(("the a an and or of to in for is on with this that it was were are be been "
             "i we you they he she my our your their me us them at by from as but not so if "
             "have has had do does did will would can could should there here out up down "
             "very really just too also more most some any all no any get got go went came "
             "place food good great nice time order ordered came back again always never").split())


def _tokenizar(t: str):
    return [w for w in _re.findall(r"[a-z]+", t.lower()) if len(w) > 2 and w not in _STOP]


def construir_tfidf(business_ids, docs, extra_tokens=None, max_vocab=5000, min_df=10):
    """TF-IDF a mano (tf log-escalada × idf, L2-normalizado) en CSR por ítem.

    `extra_tokens`: dict business_id -> tokens adicionales (p.ej. categorías).
    Vocabulario = top `max_vocab` términos por document frequency (df≥min_df,
    no demasiado comunes)."""
    from collections import Counter
    extra_tokens = extra_tokens or {}
    toks_list, df = [], Counter()
    for b, d in zip(business_ids, docs):
        toks = _tokenizar(d) + list(extra_tokens.get(b, []))
        toks_list.append(toks)
        for w in set(toks):
            df[w] += 1
    N = len(docs)
    cand = [(w, c) for w, c in df.items() if c >= min_df and c < 0.6 * N]
    cand.sort(key=lambda x: -x[1])
    vocab = {w: i for i, (w, _) in enumerate(cand[:max_vocab])}
    idf = np.zeros(len(vocab))
    for w, i in vocab.items():
        idf[i] = np.log(N / df[w])
    indptr, indices, data = [0], [], []
    for toks in toks_list:
        tc = Counter(w for w in toks if w in vocab)
        if tc:
            idx = np.array([vocab[w] for w in tc]); val = np.array([tc[w] for w in tc], float)
            val = (1 + np.log(val)) * idf[idx]
            nrm = np.linalg.norm(val)
            if nrm > 0:
                val = val / nrm
            o = np.argsort(idx)
            indices.extend(idx[o].tolist()); data.extend(val[o].tolist())
        indptr.append(len(indices))
    return dict(business_ids=np.asarray(business_ids), vocab=vocab,
                indptr=np.asarray(indptr, np.int64), indices=np.asarray(indices, np.int32),
                data=np.asarray(data, np.float32), idf=idf, n_vocab=len(vocab))


def _vec_item(tfidf, i):
    a, b = tfidf["indptr"][i], tfidf["indptr"][i + 1]
    return tfidf["indices"][a:b], tfidf["data"][a:b]


def sim_contenido(tfidf, i, j) -> float:
    """Coseno entre dos ítems (sus vectores ya están L2-normalizados)."""
    ii, vi = _vec_item(tfidf, i); jj, vj = _vec_item(tfidf, j)
    d = dict(zip(ii.tolist(), vi.tolist()))
    return float(sum(d.get(int(k), 0.0) * v for k, v in zip(jj.tolist(), vj.tolist())))


def perfil_contenido(item_idxs, pesos, tfidf) -> np.ndarray:
    """Perfil de usuario = combinación ponderada (denso) de los ítems que valoró."""
    p = np.zeros(tfidf["n_vocab"], np.float32)
    for it, w in zip(item_idxs, pesos):
        idx, val = _vec_item(tfidf, int(it)); p[idx] += w * val
    n = np.linalg.norm(p)
    return p / n if n > 0 else p


def score_contenido(perfil, tfidf, cand_idxs) -> np.ndarray:
    """Coseno perfil·ítem para una lista de candidatos (perfil ya normalizado)."""
    out = np.empty(len(cand_idxs))
    for k, it in enumerate(cand_idxs):
        idx, val = _vec_item(tfidf, int(it))
        out[k] = float(perfil[idx] @ val)
    return out


# ===========================================================================
#  Paso 5 — Baselines y estructuras de apoyo
# ===========================================================================
def usuario_csr(M: dict):
    """CSR de train por usuario: (indptr, items, ratings) para lookups O(1)."""
    ux, ix, r, nu = M["ux"], M["ix"], M["r"], M["n_users"]
    o = np.argsort(ux, kind="stable")
    indptr = np.zeros(nu + 1, np.int64)
    np.add.at(indptr, ux[o] + 1, 1); np.cumsum(indptr, out=indptr)
    return indptr, ix[o], r[o]


def popularidad(M: dict) -> np.ndarray:
    """Conteo de interacciones por ítem en train (ranking top-popular)."""
    return np.bincount(M["ix"], minlength=M["n_items"]).astype(float)


# ===========================================================================
#  Paso 6 — Métricas de evaluación (a mano)
#  RMSE / MAE: deck 09 pág. 43 · deck 10 págs. 4-6.
#  Precision@K / Recall@K / NDCG@K: NO están en los decks del curso → se fundan
#  en definiciones de Recuperación de Información estándar (MMDS, cap. 9) y las
#  exige el enunciado.
# ===========================================================================
def rmse(pred, true) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(pred - true)))


def precision_at_k(rel, k) -> float:
    rel = np.asarray(rel)
    if k <= 0:
        raise ValueError("k debe ser positivo")
    return float(rel[:k].sum()) / k


def recall_at_k(rel, k, n_rel) -> float:
    rel = np.asarray(rel)
    return float(rel[:k].sum()) / n_rel if n_rel > 0 else 0.0


def ndcg_at_k(rel, k) -> float:
    rel = np.asarray(rel, float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel[:k] * disc[:len(rel[:k])]).sum()
    ideal = np.sort(rel)[::-1][:k]
    idcg = (ideal * disc[:len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0
