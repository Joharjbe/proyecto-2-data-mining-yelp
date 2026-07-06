# Guion — PPT técnica (11 slides)

Duración objetivo: ~9–10 min. Tono: preciso, metodológico. Cada bloque es lo que dices en voz alta; la línea *Transición* es el puente a la siguiente slide.

---

## Slide 1 — Portada · [~30 s]

Esta es la parte técnica. Montamos un pipeline completo de minería de datos masivos sobre el Yelp Open Dataset —seis millones novecientas mil reseñas— con una regla dura del curso: todas las técnicas implementadas a mano en numpy, sin librerías de machine learning. Spark y Parquet solo para mover datos. El universo intensivo son 29,314 restaurantes de tres mercados, sobre una matriz usuario-negocio llena solo al 0.011%.

*Transición:* Antes de los algoritmos, la decisión más importante del proyecto: el diseño muestral.

## Slide 2 — Diseño del estudio · [~60 s]

El enunciado pide técnicas que no escalan. DBSCAN sin índice es O(n²): sobre 150 mil negocios son 2.3×10¹⁰ distancias, unos 90 GB de RAM. Girvan-Newman es O(m²n): sobre 52 millones de aristas, inviable. La tentación sería submuestrear reseñas al azar, pero eso rompe todo —el grafo social queda con amistades a medias y la matriz de recomendación pierde las co-ocurrencias—. Así que muestreamos por clave: la clave es el mercado completo. Dentro de cada ciudad elegida conservamos el 100% de negocios, reseñas, usuarios y amistades. Elegimos tres porque concentran el 56.5% de las reseñas de restaurantes y son arquetipos distintos —78, 91 y 153 reseñas por local—. Los grafos y matrices quedan íntegros, no muestreados. Y ojo: la Parte V sí corre sobre el stream completo, 20 millones de eventos, porque esos algoritmos existen precisamente para lo que no cabe en memoria.

*Transición:* Sobre ese universo, la arquitectura de datos.

## Slide 3 — Arquitectura del pipeline · [~45 s]

Usamos arquitectura medallón. Bronze son los cinco JSON crudos e inmutables. Silver es lo mismo limpio y tipado en Parquet, con cada transformación documentada. Gold son tablas listas por análisis, con solo las columnas que ese análisis usa. La regla de oro: ningún algoritmo lee JSON ni silver, todos consumen gold —eso hace cada experimento rápido y reproducible—. Dos detalles que importan: usamos esquemas explícitos en la ingesta, así Spark no relee cinco gigas para inferir tipos y todo entra en menos de dos minutos; y todo corre con semilla global 42 y versiones fijadas.

*Transición:* Empecemos por los grafos, Parte II.

## Slide 4 — Parte II: grafos y ranking · [~60 s]

Corremos PageRank e HITS a mano, con Power Iteration y la Google Matrix, teleporte 0.85. En el grafo bipartito usuario-negocio, PageRank prácticamente recupera la popularidad: correlación de Spearman 0.99 con el número de reseñas. HITS, en cambio, reordena: la correlación de la autoridad con el volumen baja a 0.67. Interpretación: una authority es un restaurante avalado por buenos hubs —reseñadores Elite—, no por volumen; por eso el top lo domina Philadelphia y desaparecen los gigantes turísticos. En comunidades hubo una iteración con evidencia: definir la arista por número de co-reseñadores daba un grafo casi completo y modularidad casi cero, porque eso confunde solape con tamaño. Al normalizar con Jaccard el grafo se dispersa y emergen micro-mercados por estilo y precio; comparando por la misma Q, el greedy CNM gana con 0.396 frente a 0.294 de Girvan-Newman.

*Transición:* De ranking a segmentación, Parte III.

## Slide 5 — Parte III: clustering · [~60 s]

K-Means++, DBSCAN y BFR, todo a mano, sobre 42 features: atributos de Yelp más contexto socioeconómico del Census. Excluimos geografía para no fabricar clusters por ciudad. Elegimos k con codo y silueta: la silueta favorece k=2, demasiado grueso, así que el codo y la interpretabilidad deciden k=6. La silueta final es 0.10 —perfiles reales pero solapados, no especies separadas, que en un mercado real es lo esperable—. DBSCAN es el contraste honesto: en 42 dimensiones colapsa a un único cluster gigante con 7.7% de ruido, evidencia de la maldición de la dimensionalidad. BFR procesa los 29 mil por bloques resumiendo en DS, CS y RS, y escala sin matriz densa dejando menos del 1% de outliers. Un dato de validación: purity con la ciudad 0.58 y NMI 0.007, o sea los segmentos no están copiando el mercado.

*Transición:* La Parte IV es la que más cuidamos metodológicamente: recomendación.

## Slide 6 — Parte IV: recomendación híbrida · [~60 s]

El pipeline: matriz usuario-negocio dispersa, split temporal —entreno hasta 2018, validación 2019, test 2020-21—, filtrado colaborativo item-item, content-based con TF-IDF, e híbrido. Lo importante es cómo se evalúa, sin fuga: el alfa del híbrido se fija en validación 2019 y el test se toca una sola vez. Y barajamos los candidatos in-market para no premiar el orden posición-primero. Con eso, el resultado honesto: el baseline top-popular es el más preciso, NDCG 0.26, pero cubre solo el 13% del catálogo; el CF cede precisión, 0.18, pero triplica la cobertura al 40%; y la validación elige el híbrido equilibrado. En predicción de rating, el baseline regularizado gana, RMSE 1.24 contra 1.31 del CF. Dos límites: con la matriz tan dispersa el CF puro se hunde en cold-start, NDCG 0.08; y la precisión sola premia concentrar.

*Transición:* Parte VI: reducción de dimensionalidad.

## Slide 7 — Parte VI: PCA y SVD · [~50 s]

PCA formal sobre las 42 features, por covarianza y autovalores. Los dos primeros componentes retienen 32.4% de la varianza, y hacen falta 25 de 42 para superar el 90% —la señal no vive en dos o tres ejes, un mapa 2D deja fuera el 67.6%—. SVD truncada la aplicamos al TF-IDF de reseñas, no a la matriz de ratings: imputar ratings ausentes mete sesgo, mientras que en TF-IDF el cero es ausencia real. Con 80 factores capturamos 23.4% de la energía pero comprimimos 33.6 veces frente a la matriz densa —de 297 a 31 megas—. La cola larga es señal real de la diversidad del texto.

*Transición:* Parte V, el otro régimen: streams.

## Slide 8 — Parte V: minería de flujos · [~50 s]

Aquí procesamos los 20 millones de eventos en una pasada. El Count-Min Sketch con ancho 4096 y profundidad 5 usa 160 KB y logra que el 99.4% de las consultas caigan dentro de la cota teórica, recuperando el 95% del Top-20; subiendo el ancho a 8192 recupera el 100%. Recordar que CMS solo sobreestima por colisiones, nunca subestima. Como técnica adicional, DGIM mantiene solo 10 u 11 buckets para una ventana de una semana, con un error medio de dos horas. Y cruzamos con contexto exógeno: los check-ins de la primavera de 2020 caen al 15-33% de 2019, y el Mardi Gras da un índice de 187 frente a los placebos.

*Transición:* Todo lo anterior alimenta la Parte VII: el análisis crítico.

## Slide 9 — Parte VII: escalabilidad y equidad · [~50 s]

Aquí conectamos resultados con riesgos. En exposición: el CF cubre el 40% del catálogo con Gini 0.239, frente al top-popular que cubre 13% con Gini 0.358 —o sea el más preciso es también el que más concentra—. Pero el CF tampoco iguala: el cuartil más visible aún se lleva cerca del 49% de los slots. En escalabilidad, el benchmark recupera crecimiento casi lineal para K-Means y cuadrático para DBSCAN. Y el stress de spam que ya mostramos: cinco reseñas mueven al cuartil pequeño +0.56 estrellas contra +0.03 del grande. Todo esto lo organizamos en una matriz dato-método-aporte-riesgo-mitigación, para que la ética no sea una opinión aparte del pipeline.

*Transición:* Seamos explícitos con lo que estos resultados no pueden afirmar.

## Slide 10 — Límites técnicos · [~45 s]

Cinco límites honestos. Uno: con la matriz al 0.011%, el CF no tiene dónde anclarse en cold-start. Dos: en 42 dimensiones las distancias se concentran, de ahí la silueta baja y que DBSCAN no encuentre estructura. Tres: el diámetro del grafo no es exacto —un BFS desde 843 mil nodos es inviable—, lo reportamos como cota inferior por double-sweep. Cuatro: no hay ground-truth de spam ni atributos protegidos, así que auditamos proxies y vulnerabilidad, no fraude real ni discriminación causal. Y cinco: el Census tiene nulos, así que lo usamos con un umbral de cobertura, sin imputar de más.

*Transición:* Y cerramos con lo que sostiene todo el trabajo.

## Slide 11 — Reproducibilidad / cierre · [~30 s]

Las siete partes están completas y auditadas, con 40 tablas gold, 38 figuras y cero librerías de machine learning en los algoritmos. Todo es reproducible: semilla 42, versiones fijadas con un candado por versión de numpy, descargas idempotentes. Y la idea que gobierna el diseño: análisis exacto donde la escala lo permite, aproximado con garantías donde no —y cada decisión de recorte aparece después de la tabla que la sustenta—. Gracias.
