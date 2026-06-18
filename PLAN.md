# Proyecto 2 — Data Mining UTEC: Plan y Bitácora

**Tema:** Influencia, comunidades y recomendación en el ecosistema de restaurantes de Yelp
**Entrega:** jueves 18 de junio de 2026 · **Ritmo:** avance diario, validando cada notebook juntos
**Regla de oro:** algoritmos de clase implementados a mano (basados en los PDFs de `teoria/`); librerías solo para manipulación de datos/gráficas (Spark/Parquet = capa de ingeniería).

---

## 🚀 Quick Start para agentes (léeme primero — 2 min)

**Qué es:** Proyecto 2 de Data Mining (UTEC, Prof. Heider Sanchez). Pipeline de minería masiva sobre el **Yelp Open Dataset**; 7 partes (I–VII) + informe + presentación. Todo el código vive en `codigo/`.

**Estado real (18-jun-2026):**
- ✅ **Parte I** — Preprocesamiento + EDA (notebooks 01–04; 7 tablas gold materializadas).
- ✅ **Parte II** — Grafos: PageRank, HITS, comunidades (notebook 05 + `src/graphs.py`; 3 tablas gold + 4 figuras persistidas).
- ✅ **Parte III** — Clustering: K-Means++, DBSCAN y BFR (notebook 06 + `src/clustering.py`; 3 tablas gold + 8 figuras persistidas).
- ✅ **Parte IV** — Recomendación híbrida: CF item-item + content TF-IDF + híbrido (notebook 07 + `src/recommenders.py`; 4 tablas gold + 6 figuras persistidas; α calibrado en validación antes de test).
- ✅ **Parte V** — Streams: ventanas 1h/4h/24h + Count-Min Sketch + DGIM + COVID/Mardi Gras (notebook 09 + `src/streaming.py`; 6 tablas gold + 6 figuras).
- ✅ **Parte VI** — PCA formal + SVD truncada sobre TF-IDF (notebook 08 + `src/dimensionality_reduction.py`; 5 tablas gold + 6 figuras persistidas).
- ✅ **Parte VII** — escalabilidad, concentración de voz, equidad de exposición, stress de spam, contexto ACS y transferibilidad (notebook 10 + `src/critical_analysis.py`; 12 tablas gold + 6 figuras).
- ✅ **Auditoría técnica/reproducibilidad** — 10 pruebas automatizadas, 10 notebooks reejecutados, `.gitignore`, matriz de rúbrica y salida pública sin datos/credenciales.
- ⏳ **Informe / presentación** pendientes.
- **Avance técnico: 7 de 7 partes completas y auditadas; faltan los dos artefactos académicos finales.**

**Próxima tarea prioritaria:** informe 6–8 páginas; después presentación, demo y ensayo.

**Notebooks:** 01–04 (Parte I, **no tocar**) · **05–10** (Partes II–VII completas, con outputs e insights bajo cada salida).
**Módulos:** `src/config.py` (rutas, `SEED=42`, SparkSession) · `src/viz.py` (estilo) · `src/graphs.py`, `src/clustering.py`, `src/recommenders.py`, `src/streaming.py`, `src/dimensionality_reduction.py` y `src/critical_analysis.py` (algoritmos y auditorías reproducibles).

**Restricciones académicas (innegociables):**
1. **Algoritmos del curso A MANO** (numpy puro). Prohibido networkx / scikit-learn / scipy para el algoritmo en sí; numpy/pandas/matplotlib solo para datos y gráficas.
2. Cada técnica se **fundamenta citando la diapositiva** de `teoria/` (deck + página) *antes* de implementarla.
3. Cada entregable responde a un punto explícito del **`Enunciado_proyecto2.pdf`**.
4. **Siempre incluir gráficas** no tradicionales que expliquen el análisis/experimentación (no decorativas).

**Decisiones técnicas obligatorias:** universo = restaurantes de **Philadelphia + Tampa + New Orleans** (ver D1–D8); los algoritmos consumen **solo `data/gold/`** (nunca JSON/silver); semilla **42**; figuras → `docs/figs/` vía `viz.guardar`; tablas de resultados → `data/gold/`.

**Pasos mínimos para continuar:**
```bash
cd codigo && source .venv-yelpdm/bin/activate     # Python 3.11 · PySpark 3.5.8 · Java 17
# En notebooks: kernel `Python (yelp-dm)`. Primera celda: %autoreload 2 + sys.path.append("..")
```
Ciclo de trabajo: leer teoría → implementar a mano en `src/<modulo>.py` → validar contra ejemplo de clase (sandbox) → correr en gold → escribir insight bajo cada salida → persistir tabla/figura → actualizar `PLAN.md` + `README.md`.

---

## 🤖 Instrucciones para futuros agentes (Claude Code / Codex)

Este repo se trabaja **alternadamente por varios agentes**. Para retomarlo sin leer conversaciones previas:

**1. Cómo entender el proyecto rápido (en orden):** (a) este Quick Start; (b) `Enunciado_proyecto2.pdf` (qué pide cada parte + rúbrica); (c) `README.md` §3 (diseño del estudio) y §7 (recorrido por notebooks); (d) `src/graphs.py` como plantilla de "cómo se implementa una parte".

**2. Fuentes de verdad (en orden de prioridad):**
- Lo **materializado**: `data/gold/*.parquet`, notebooks con outputs, `docs/figs/*`.
- `PLAN.md` (estado, decisiones, riesgos, DoD) y `README.md` (narrativa/justificación para el profesor).
- `Enunciado_proyecto2.pdf` y los PDFs de `teoria/` (qué se exige y qué métodos están permitidos).
- *Si algo se contradice, gana lo que esté materializado en gold + notebooks corridos.*

**3. Convenciones que se respetan siempre:** algoritmos a mano (numpy) + cita de diapositiva + validación contra ejemplo de clase; `from src.config import ...`, `from src import viz`; semilla 42; leer datos solo de gold; notebooks `0N_descripcion.ipynb` con `%autoreload 2` + `sys.path.append("..")` en la celda de setup, análisis en prosa bajo cada salida y resumen al cierre; figuras vía `viz.guardar(fig, 'parteN_nombre')`; tablas de resultados a `data/gold/`.

**4. Qué NO modificar:** las tablas gold de la Parte I (`negocios_universo`, `usuarios_universo`, `resenas_universo`, `aristas_sociales`, `matriz_un`, `features_negocio`, `stream_eventos`) ni los notebooks 01–04 (Parte I cerrada); las decisiones D1–D8 sin consultarlo con Johar; `.census_key` (secreto). `data/` no se versiona.

**5. Qué actualizar tras CADA avance (obligatorio):** `PLAN.md` (Quick Start estado/%, bitácora con insight+números, checklist de entregables, DoD del notebook cerrado); `README.md` (§7 recorrido con resultados + citas de página, §5 linaje si hay tablas nuevas, §8 hallazgos, §11 estado); persistir resultados (gold) y figuras (docs/figs).

**Autoridad de decisiones:** **Johar** decide alcance y tema. Un agente NO cierra decisiones metodológicas por su cuenta — propone con justificación y espera visto bueno.

---

## 📍 Dónde estamos (actualizado: 18-jun-2026 — Partes I–VII completas)

**Avance global: las 7 partes técnicas están completas. Faltan informe, presentación/demo y GitHub. El cronograma original (§3) quedó superado; confirmar la hora límite de entrega.**

| Parte del enunciado | Rúbrica | Estado | Dónde |
|---|---|---|---|
| I. Preprocesamiento y EDA | 2 pts | ✅ **Completa** | notebooks 01–04 |
| II. Grafos y ranking (PageRank, HITS, comunidades) | 4 pts | ✅ **Completa** | notebook 05 (Pasos 1–5 ✓) + `src/graphs.py` |
| III. Clustering (K-Means++ + DBSCAN + BFR) | 4 pts | ✅ **Completa** | notebook 06 + `src/clustering.py` |
| IV. Recomendación híbrida (CF + content + híbrido) | 4 pts | ✅ **Completa** | notebook 07 + `src/recommenders.py` |
| V. Flujos de datos (ventanas, CMS, +1 técnica) | 3 pts* | ✅ **Completa** | notebook 09 + `src/streaming.py` |
| VI. Reducción de dimensionalidad (PCA, SVD) | 3 pts* | ✅ **Completa** | notebook 08 + `src/dimensionality_reduction.py` |
| VII. Análisis crítico y ético | 3 pts** | ✅ **Completa** | notebook 10 + `src/critical_analysis.py` |
| Informe (6–8 págs) + presentación (30 min) + GitHub | ** | ⏳ | día 8–9 |

\* Flujos+Dim comparten 3 pts en la rúbrica · ** Ética+código+reporte comparten 3 pts

**Activos ya construidos:** entorno validado (.venv-yelpdm: Python 3.11, PySpark 3.5.8, Java 17) · silver (5 tablas, 6 GB) · gold (**40 tablas lógicas**: 28 de I–VI + 12 de VII) · 4 fuentes externas · 6 módulos analíticos + utilidades · 2 diagramas mermaid + `docs/arquitectura.excalidraw` · **10 notebooks corridos sin errores** · **36 figuras** (30 de II–VI + 6 de VII) · 10 pruebas automatizadas.

**Números clave del universo:** 29,314 restaurantes · 2,671,060 reseñas · 813,792 usuarios · 2,527,630 amistades internas · matriz 0.011% llena · grafo: gigante 100%, diámetro ≥12, distancia media 6.9 · stream completo: 20,347,155 eventos.

---

## 1. Decisiones cerradas (con su porqué, para el informe)

| # | Decisión | Justificación (resumen) |
|---|----------|--------------------------|
| D1 | Dataset: **Yelp Open Dataset** | Grafo social + bipartito + texto + temporal + geo: cubre las 7 partes. Licencia académica OK. |
| D2 | Tema: **restaurantes** | Categoría #1 del dataset (52,268 negocios, 35%): el tema apunta al corazón de la data. |
| D3 | Alcance: **3 mercados completos de EE.UU.** — Philadelphia + Tampa + New Orleans | Top-3 en reseñas de restaurantes (56.5% del total) + tres arquetipos (residencial 78 / sunbelt 91 / turístico 153 reseñas por local) + cobertura ACS/COVID. Elegir mercados completos = key-based sampling (semana 06): preserva grafos y co-ocurrencias. Validado con conteos reales (notebook 02). |
| D4 | Stack: **PySpark local + Parquet + medallón** (bronze/silver/gold); DuckDB/Polars permitidos para consultas; algoritmos SIEMPRE a mano (numpy) | Parquet acelera 10–50x; medallón documenta limpieza; Spark = argumento de escalabilidad (Parte VII). |
| D5 | EDA completo **antes** del filtro y recorte | Orden metodológico trazable: nb02 perfilado → hipótesis; nb03 EDA total sin filtros; nb04 sensibilidad → materialización. Ninguna fila se descartó sin explorarla. |
| D6 | Filtro: **categoría exacta `Restaurants`** | Sensibilidad medida (nb04): ∪Food agregaría 6,746 locales tipo grocery/farmacia con 30.8 reseñas/local vs 91.1 — ruido, no señal. |
| D7 | Complementos: **ACS + COVID NYT + feriados + crosswalk** (1ª ola); NOAA/Zillow/USDA/CDC en espera; Fusion API/SafeGraph/OSM/Perú descartados con razón | Cruces limpios (ZIP↔ZCTA ✓, county ✓, fecha ✓ en Parte V), licencias claras, peso mínimo (132 MB). Cada fuente pasó su propio EDA antes de integrarse. |
| D8 | Lima/Perú: **no al pipeline** (sin clave de cruce; el curso exige UN dataset), **sí a la discusión** de transferibilidad (Parte VII) | Honestidad metodológica + sello personal sin costo. |
| D9 | Parte VI: **PCA sobre `features_negocio`; SVD sobre TF-IDF, no sobre ratings vacíos** | PCA conecta con III. El deck 10 pág. 20 advierte que SVD clásica exige matriz completa: en `matriz_un` los huecos son desconocidos; en TF-IDF los ceros sí significan ausencia de término. |
| D10 | Parte V: **stream completo para ventanas/CMS; DGIM sobre horas activas p75-2019; COVID/Mardi Gras solo en mercados con claves externas** | Preserva la razón de ser de los algoritmos de una pasada, da significado operativo al bit de DGIM y evita cruces geográficos inválidos fuera del universo. |

## 2. Reglas de trabajo acordadas

- Notebooks como flujo: experimentar → pasar salidas → validar juntos → fijar análisis con números reales → avanzar.
- Análisis directo bajo cada salida (sin prefijo "Lectura"), lenguaje humano, voz de equipo ("encontramos"), resumen al cierre de cada notebook, referencias a los PDFs de teoría donde aplique.
- Gráficas con estilo propio (`src/viz.py`): paleta fija por mercado, títulos que afirman el hallazgo.
- Si se edita un módulo de `src/` con kernel abierto → **reiniciar kernel** (cache de imports).
- El código lo verá el profesor → al final se eliminan comentarios internos de coordinación.
- Diagrama de linaje en README se actualiza en cada hito, con fecha.
- README = narrativa completa y autocontenida del proyecto (objetivo → datos → diseño → notebook por notebook con resultados → hallazgos → reproducción). La sección 7 ("Recorrido por notebooks") se extiende al cerrar cada notebook.
- GitHub al final: `.gitignore` debe excluir `data/` y `.census_key`.

## 3. Cronograma (ORIGINAL — superado)

> ⚠️ **Este es el plan del día 1 y ya no refleja la realidad.** El trabajo se hizo en dos tandas: días 1–2 (9–10 jun, Parte I) y, tras una pausa, 17–18 jun (Parte II). El estado real está en el **Quick Start**. Se conserva como registro de la planificación inicial. **El plazo real debe re-acordarse con Johar.**

| Día | Fecha | Objetivo | Estado |
|-----|-------|----------|--------|
| 1 | mar 09 | Fase 1: tema, alcance, stack, fuentes + plan | ✅ |
| 2 | mié 10 | Entorno + ingesta silver + EDA global + externos + universo + gold (Parte I) | ✅ **(absorbió el día 3)** |
| 3 | jue 11 | **Parte II**: PageRank + HITS a mano + comparación + comunidades caracterizadas | ✅ (hecho 17–18 jun) |
| 4 | vie 12 | **Parte III**: K-Means++, DBSCAN, BFR + k óptimo + comparativa | ✅ (hecho 18 jun) |
| 5 | sáb 13 | **Parte IV**: CF user/item + content-based TF-IDF + híbrido + evaluación completa | ✅ (hecho 18 jun) |
| 6 | dom 14 | **Parte V**: ventanas + Count-Min + técnica extra + cruce COVID/feriados (stream 20.3M) | ✅ (hecho 18 jun) |
| 7 | lun 15 | **Parte VI**: PCA + SVD a mano · inicio Parte VII | ✅ (hecho 18 jun) |
| 8 | mar 16 | **Parte VII** completa + pulir repo + GitHub + limpieza de comentarios | |
| 9 | mié 17 | Informe 6–8 págs + presentación 30 min + demo + ensayo (colchón) | |
| — | jue 18 | **ENTREGA** | |

## 4. Bitácora resumida

**Día 1 (mar 09):** lectura de enunciado y 12 PDFs de teoría → resumen validado · investigación Yelp (estructura, licencia) y fuentes externas → selección justificada · decisiones D1–D8 · arquitectura y cronograma · Johar descargó el dataset (8.7 GB verificados).

**Día 2 (mié 10):** entorno validado (fix numpy 2: `trapezoid`) · nb01 smoke test ✅ · nb02 ingesta silver ✅ (64s, conteos = oficiales, 0 duplicados; terna elegida con números) · nb03 EDA global ✅ (46 celdas: diccionario, colas largas, J=67%, Gini 0.61, heatmap check-ins, calidad) · fixes de presentación (voz humana, años enteros, excalidraw corregido) · nb04 ✅ (Census key + propagación resuelta; fix pd.NA→pyarrow validado en sandbox; externos con EDA propio; sensibilidad; 19/39 atributos; 7 tablas gold; grafo: densidad 1.12e-4, diámetro ≥12) · hallazgo: 4 reseñadores fantasma · README con 2 diagramas mermaid + linaje · **PARTE I COMPLETA**.

**Día 3 (17-jun) — Parte II:** retomado el proyecto; leídos enunciado + teoría (PageRank deck 07, Comunidades deck 12) con **citas por página**. Decisiones: PageRank/HITS sobre **ambos** grafos (amistades + bipartito), elegir con evidencia; comunidades **Girvan-Newman + Q** (principal) + **greedy CNM** comparados por Q (Louvain fuera: no está en las diapos). Creado `src/graphs.py`. **Paso 1 ✓** (grafos + diagnóstico): amistad guardada en una dirección (recip. 0%) → se simetriza; 55.1% de usuarios sin amigos internos; bipartito con todos los negocios como sumideros (encaje de HITS); cross-check de los 4 'reseñadores fantasma'. **Paso 2 (PageRank)** implementado y **validado** contra el ejemplo de clase (1839: y=a=0.4, m=0.2); pendiente correr en datos reales.

**Resultados Pasos 2–3 (para el informe):** PageRank social ≈ grado refinado (Spearman con nº amigos 0.81; top usuarios casi todos Elite); PageRank bipartito ≈ popularidad (Spearman con reseñas 0.99). **HITS** reordena: authority↔reseñas baja a 0.67 y solo 2/15 negocios coinciden con PageRank → *authority = aval de reseñadores Elite* (Philadelphia) vs *popularidad de turistas* (New Orleans). En el social HITS degenera (hub=authority). Pasos 2–3 ✓.

**Paso 4 (comunidades) — calibración:** primer intento con grafo de co-reseña de Philadelphia (top-150, peso = nº co-reseñadores ≥15) salió **casi completo** (grado medio 134) y Q≈0 (GN, 228 s) / 0.06 (greedy) — el conteo crudo premia la popularidad. Ajuste (regla de experimentación): peso = **Jaccard** (normaliza por tamaño) para revelar micro-mercados reales. Pista de estructura latente ya visible: clúster de *cheesesteaks/tradicional* (Reading Terminal, Pat's, Geno's).

**Parte II ✓ (17-jun):** Paso 4 cerrado — grafo de co-reseña con Jaccard (grado 8); **CNM gana por Q=0.396** vs GN 0.294; 4 micro-mercados interpretables (clásicos-turísticos, autor/destino, trendy alta gama, barrio). Notebook 05 completo (PageRank · HITS · comunidades), todo a mano y validado contra ejemplos de clase. **Resultados persistidos (Paso 5):** 3 tablas gold (`ranking_usuarios`, `ranking_negocios`, `comunidades_coresena_philadelphia`) + 4 figuras en `docs/figs/parte2_*` (incl. la red de nodos/aristas con layout de resortes propio).

**Parte III ✓ (18-jun):** `src/clustering.py` y notebook 06 completos, ejecutados sin errores. La matriz parte de `features_negocio.parquet`: 29,314 restaurantes y **42 features estandarizadas**, combinando Yelp con ingreso, población, educación y renta de ACS; `metro` y coordenadas se excluyen para evitar circularidad. **K-Means++:** barrido `k=2…10`; silueta máxima en `k=2`, pero codo e interpretabilidad seleccionan **k=6** (SSE/punto 28.32; silueta ≈0.10). Diez semillas confirman menor SSE y variabilidad que inicialización aleatoria. **DBSCAN:** muestra reproducible de 6,000 por costo O(n²); `minPts=10`, barrido de percentiles y transición p80→p82 justifican `eps=5.17`; produce un cluster gigante + 7.8% de ruido, resultado negativo explicado por las 42 dimensiones. **BFR:** universo completo por bloques de 4,000; barrido del factor DS 1.4–2.0 elige **1.8**, con 0.51% de outliers. Pureza ≈0.58 y NMI ≈0.008 muestran que los segmentos no copian la ciudad. **PCA manual para visualización:** una proyección común de la misma muestra permite comparar los tres modelos; PC1+PC2 retienen 32.4% y BFR coincide 65% con K-Means++ tras alinear IDs. PCA no interviene en el entrenamiento ni reemplaza las métricas en 42D. Persistidos `clusters_negocio`, `perfiles_clusters_kmeans` y `comparativa_clustering`, más **8 figuras**: parámetros, perfiles, silueta por cluster, mosaico mercado×cluster, DBSCAN, progreso BFR, mapa PCA común y comparación final.

**Parte IV ✓ (18-jun):** `src/recommenders.py` y notebook 07 completos, a mano y validados. Split **temporal** sin fuga (train ≤2018 / val 2019 / test 2020-21; dedup a primera visita = 2,580,540 interacciones; warm user = ≥3 en train = 150,350). **CF item-item** Pearson+shrinkage sobre baseline μ+b_u+b_i (11,234/25,952 ítems con vecinos). **Content** TF-IDF de 1.91M reseñas agregadas por restaurante + categorías (Pat's–Geno's 0.33 vs Pat's–Zahav 0.05). La calibración se ejecuta **antes de tocar test**: validación elige **α=1** (NDCG 0.502), por lo que ningún blend mejora al CF en usuarios warm; el 50/50 queda como ablación. **Evaluación test** (1,500 usuarios, negativos in-market, IC bootstrap): CF gana (NDCG@10 0.49 [.47,.51]) sobre top-popular 0.26, content 0.20 y ablación híbrida 0.25; cobertura CF 39% vs top-popular 13%. En rating, baseline gana (RMSE 1.24 vs CF 1.30). La suma de similitudes supera al rating predicho en validación (0.502 vs 0.207). Persistidas 4 tablas (`metricas_recomendacion`, `recomendaciones_ejemplo`, `curva_hibrido_validacion`, `comparativa_score_cf_validacion`) + 6 figuras.

**Parte VI ✓ (18-jun):** `src/dimensionality_reduction.py` y notebook 08 completos, ejecutados 9/9 celdas sin errores. **PCA** sobre 29,314×42 features: PC1+PC2 retienen 32.4%, PC1–PC3 38.4% y se requieren **25/42 componentes para 90%** (error relativo 0.309); PC1 captura documentación/ausencia de atributos, PC2 servicio-precio-bar y PC3–PC4 contexto ACS. **SVD truncada manual** sobre TF-IDF 25,952×3,000 (4.07M nnz, 5.22%): CSR ocupa 31.2 MB vs 297 MB denso. Con `k=80` captura 23.4% de energía, error 0.875, compresión 33.6× vs densa y tamaño 0.28× del CSR; la cola larga es resultado, no fallo. Factores interpretables: pizza/delivery vs bar/dining; asiático/sushi vs breakfast/coffee; drive-thru/fast-food vs cafés; mexicano vs asiático. Ortogonalidad numérica ≈1e-14. Persistidos 5 resultados gold + **6 figuras** revisadas visualmente.

**Parte V ✓ (18-jun):** `src/streaming.py` + notebook 09 completos y ejecutados sin errores sobre **20,347,155 eventos / 150,346 claves**. **Ventanas:** 148,361 horas en 0.34 s; el pico de 1,044 eventos/hora contrasta con 9,054 en 24h. **CMS:** pasada completa en 7.2 s; `w=4096,d=5` usa 160 KB, p95=4,029 bajo `εN=4,968`, 99.4% de consultas dentro de la cota (teoría 99.3%) y 95% del Top-20; `w=8192` recupera 100%. **DGIM:** 10–11 buckets para N=168, MAE 1.76–2.15 horas/semana y error máximo 50%; detecta el apagón pandémico. **COVID:** en mar–may 2020 los check-ins quedan en 14.8–33.1% de 2019, por debajo de reseñas (24.7–39.8%). **Mardi Gras:** índice mediano 187 en New Orleans vs 97/101 en placebos; 2021 conserva pico relativo pero cae 71% en volumen vs 2020. Persistidos 6 resultados gold + **6 figuras** revisadas.

**Parte VII ✓ (18-jun):** creado `src/critical_analysis.py` y notebook 10; ejecutado completo (10/10 celdas, 0 errores). **Escalabilidad:** microbenchmark sobre las 42 features confirma pendiente log-log **1.12** para asignación K-Means y **2.27** para vecindades DBSCAN; se consolidan complejidad, tiempos y memoria de I–VI. **Representación:** dentro del universo, Gini de producción=**0.590** y el top 10% escribe **54.0%**; quitarlo como prueba de sensibilidad mueve el rating mediano |Δ|=0.141 (p95=0.583) en 19,228 negocios. **Recomendación:** sobre 1,500 usuarios/15,000 slots, CF expone **10,210 negocios (39.3% del catálogo)** con Gini 0.248 vs top-popular **3,374 (13.0%)**, Gini 0.358; aun así, 51.3% de slots CF van al Q4 de visibilidad. **Rankings:** no se observa ventaja general de cadenas; sí concentración en negocios ya visibles y mayor asociación HITS↔ingreso ZIP (ρ=0.280). **Spam:** 5 reseñas simuladas de 5★ mueven Q1 +0.56 estrellas vs Q4 +0.03; 5 de 1★ mueven −0.88 vs −0.07. **Contexto:** ZIP ingreso Q1 tiene 21.2% de atributos faltantes y mediana 25 reseñas vs 17.9%/38 en Q4. Persistidos **12 resultados gold + 6 figuras**, todas revisadas visualmente; incluye matriz de riesgos y transferibilidad a Lima con límites de proxies explícitos.

### Próximas ⏳
- [ ] Informe 6–8 páginas
- [ ] Presentación 30 min + demo + ensayo
- [ ] `.gitignore`, limpieza final y GitHub

---

## 5. Riesgos del proyecto

**Técnicos:**
- **Algoritmos O(n²) / O(m²n)** (DBSCAN, CURE, Girvan-Newman): no corren sobre el universo completo. En III, DBSCAN se limitó a una muestra reproducible de 6,000 y BFR procesó el universo por bloques; conservar estos tiempos y tradeoffs para Parte VII.
- **Cache de imports en notebooks**: editar `src/` con kernel vivo no recarga. Mitigado con `%autoreload 2` en la celda de setup; si falla, *Restart & Run All*.
- **Memoria del M1 Pro (16–32 GB)**: matrices densas n² (p.ej. distancias de 29k negocios ≈ 3.4 GB; del dataset completo ≈ 90 GB) revientan RAM → trabajar disperso / por bloques (BFR existe por esto).
- **TF-IDF / SVD sobre millones de reseñas:** mitigado en VI con corte temporal ya justificado, máximo 15 fragmentos por negocio, vocabulario 3,000, CSR propio y SVD por bloques; 31.2 MB y 1.6 s para la factorización, sin densificar 297 MB.
- **Garantías probabilísticas de CMS:** `εN` aplica por consulta con probabilidad `1−δ`, no simultáneamente a 150k claves. Notebook 09 muestra p95 bajo la cota y 99.4% de cumplimiento para `d=5`, sin ocultar máximos superiores.
- **Umbral DGIM fijo en 2019:** mide actividad relativa al régimen prepandemia; el *concept drift* posterior es precisamente la señal, no una recalibración olvidada.
- **Corte parcial 2022:** toda figura COVID termina en la última semana completa (16-ene-2022); no interpretar la semana incompleta como desplome.
- **Deck 13 verificado:** `teoria/diapos_extras/13. Reducción de Dimensionalidad.pdf`; las citas de proyección, reconstrucción, SVD y tradeoffs ya están incorporadas en el módulo, notebook 08 y matriz de rúbrica.
- **Reproducibilidad cross-máquina**: los números se fijaron en el Mac de Johar; otro agente debe correr *Restart & Run All* y comparar contra los valores escritos en README/PLAN (semilla 42).

**Metodológicos:**
- **Confundir popularidad con señal** (visto en II: co-reseña cruda y PageRank bipartito ≈ popularidad). III mitigó la cola larga con `log1p` y estandarización; en IV cuidar el baseline top-popular.
- **Fuga de datos en evaluación** (IV): separar train/test por usuario o temporal; no evaluar sobre lo entrenado.
- **Concentración de voz confirmada en VII:** dentro del universo, el top 10% escribe 54% (Gini 0.590); la prueba sin usuarios prolíficos mueve ratings, especialmente en la cola. No confundir actividad con fraude: usar agregación robusta e incertidumbre.
- **Proxies de equidad:** ingreso ACS es del ZIP, reseñas acumuladas son visibilidad y repetición de nombre no prueba propiedad. No realizar inferencias individuales ni causales.
- **Submuestreo que rompe relaciones** (semana 06): todo recorte adicional debe ser por clave (mercado/usuario), no aleatorio.
- **Elite como etiqueta débil** (4.6%): sirve para validar rankings de forma cualitativa, no como ground-truth duro.

## 6. Dependencias entre partes (III–VII)

- **III Clustering** ← `features_negocio` (**completa**). Sus etiquetas ya están en `clusters_negocio` y alimentan VII (equidad por cluster); también pueden enriquecer IV/VI.
- **IV Recomendación** ← `matriz_un` (ratings) + `resenas_universo` (texto TF-IDF). Independiente de III; necesita split train/test propio.
- **V Streams** ← `stream_eventos` completo + COVID/feriados, **completa**. Sus errores, memoria y tiempos alimentan VII.
- **VI Reducción dim.** ← `features_negocio` (PCA) + `resenas_universo` (TF-IDF/SVD), **completa**. PCA contextualiza III; SVD extrae factores semánticos sin imputar huecos de ratings.
- **VII Ética/escalabilidad** ← **todo lo anterior**, **completa**. Produjo auditorías de complejidad/memoria, voz, exposición, spam, contexto y transferibilidad.

**Orden recomendado desde el estado actual:** informe → presentación/demo → ensayo de sustentación.

## 7. Criterios de éxito y "Definition of Done" por notebook

**Done** = corre limpio de principio a fin (*Restart & Run All*) · cada técnica fundamentada con cita de diapositiva · validada contra un ejemplo de clase · análisis en prosa bajo cada salida · resultados en gold · figuras en docs/figs · PLAN/README actualizados.

| Parte (notebook) | Criterio de éxito (rúbrica) | DoD específico |
|---|---|---|
| III Clustering (06) | K-Means++ + 2 de {DBSCAN,CURE,BFR}; k óptimo (codo+silueta); comparativa (silueta/purity/NMI); clusters caracterizados | curva del codo + silueta graficadas · tabla comparativa de métodos · cada cluster descrito por features dominantes · `clusters_negocio.parquet` |
| IV Recomendación (07) | CF (user/item) + content-based (TF-IDF) + híbrido; eval Precision@K, Recall@K, NDCG, RMSE, MAE vs baselines | split train/test documentado · tabla métricas vs baseline (random + top-popular) · análisis cold-start · ejemplos de recomendación |
| V Streams (09) | Ventanas (1h/4h/1d) + Count-Min Sketch (garantía de error vs exacto) + 1 técnica extra justificada | CMS: error empírico vs cota teórica · series por ventana · cruce COVID/feriados (Mardi Gras como test natural) |
| VI Reducción dim. (08) | PCA (90%+ var, interpretación de PCs, viz 2D/3D) + SVD (factores latentes, error reconstrucción vs k) | scree plot · proyección 2D coloreada por cluster/mercado · curva error vs k · interpretación de factores |
| VII Ética/crítico (10) | Escalabilidad (complejidad y tiempos por algoritmo, tradeoffs) + sesgos + equidad | tabla de complejidad y **tiempos reales medidos** · equidad (clusters/rankings vs ACS) · discusión de sesgos (Gini, Elite, spam) |
| Informe (6–8 pág) | Intro, metodología, resultados con tablas/gráficas, análisis crítico, conclusiones, referencias | cada criterio de rúbrica mapeado a su evidencia (figura/tabla/notebook) |

## 8. Checklist verificable de entregables

**Código (GitHub):**
- [x] `preprocessing.py` (silver) · `enrichment.py` (externos) · `config.py` · `viz.py`
- [x] `graphs.py` (II) · `clustering.py` (III) · `recommenders.py` (IV)
- [x] `streaming.py` (V) · [x] `dimensionality_reduction.py` (VI) · [x] `critical_analysis.py` (VII)
- [x] Notebooks 01–10
- [x] README con instrucciones y dependencias
- [x] `.gitignore` (excluye `data/`, `.census_key`, `.venv-yelpdm/`, `__pycache__/`) · [x] matriz `docs/RUBRICA.md` · [x] repo público `Joharjbe/proyecto-2-data-mining-yelp`

**Resultados materializados:**
- [x] 7 tablas gold (I) + 3 tablas/4 figs (II) + 3/8 (III) + 4/6 (IV) + 6/6 (V) + 5/6 (VI)
- [x] 12 tablas / 6 figuras de VII

**Informe y presentación:**
- [ ] Informe 6–8 págs · [ ] Presentación 30 min · [ ] demo interactiva · [ ] ensayo

## 9. Oportunidades de calidad académica (documentadas, NO implementadas aún)

Detectadas contra la rúbrica y los PDFs de teoría. Mejoras opcionales para maximizar nota y blindar la sustentación.

**Experimentos / análisis comparativos faltantes:**
- **Parte II (cerrada — mejoras opcionales):** sensibilidad de PageRank a β∈{0.80,0.85,0.90} (estabilidad del top-K); curva de convergencia (‖Δr‖₁ por iteración); validar comunidades contra zip/barrio (¿coinciden?).
- **Parte III cerrada:** ya incluye *k-distance*, barrido `eps`, K-Means++ vs aleatorio, estabilidad en 10 semillas, umbrales BFR, silueta individual, composición por mercado y una proyección PCA 2D diagnóstica común. Parte VI ya formalizó PCA sin confundir el mapa con evaluación.
- **Parte IV cerrada:** ablación CF/content/blend, curva de α en validación, cobertura, novedad e intervalos bootstrap completos. Opcional: curva Precision@K para varios K y conmutador cold-start explícito.
- **Parte V cerrada:** curva CMS por ancho/profundidad, cota vs error, heavy hitters, ventanas 1/4/24h, DGIM exacto/aproximado, COVID y Mardi Gras con placebos incluidos. Opcional: Bloom Filter como extensión, no necesario para rúbrica.
- **Parte VI cerrada:** scree/90%, loadings, 2D/3D, reconstrucción, compresión densa/CSR y factores semánticos ya incluidos. Opcional: estabilidad del subespacio SVD entre semillas.

**Gráficos que fortalecen el informe:** convergencia PageRank/HITS · dendrograma de comunidades · heatmap de matriz usuario×negocio. Ya están materializados PCA 2D/3D, curva CMS, ventanas, DGIM, COVID y Mardi Gras.

**Métricas cubiertas:** III silueta/purity/NMI/SSE/outliers · IV Precision/Recall/NDCG/RMSE/MAE/cobertura · V error/cota/memoria/Top-K/DGIM · VI varianza/reconstrucción/compresión · VII complejidad, tiempos, huella de memoria, representación y Gini de exposición.

**Posibles preguntas del profesor (preparar respuesta):**
- ¿Por qué simetrizaron el grafo social para PageRank? ¿Qué pasa con β→1 y β→0?
- ¿Por qué HITS y PageRank dan rankings distintos en el bipartito? ¿Qué mide cada uno?
- ¿Por qué Jaccard y no co-reseña cruda en las comunidades?
- ¿Por qué CNM sobre Girvan-Newman? ¿GN escalaría?
- ¿Cómo evitan fuga de datos al evaluar recomendación?
- ¿Qué garantías de error tiene Count-Min Sketch y cómo las verificaron?
- ¿Cuántos componentes principales y por qué? ¿Qué representan?
- ¿Qué sesgos arrastran sus recomendaciones y a quién perjudican?
- ¿Cómo escalarían esto a los 11 mercados / al dataset completo?
