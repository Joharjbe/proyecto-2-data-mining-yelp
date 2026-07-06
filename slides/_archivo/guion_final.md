# Guion del presentador — Presentación integrada (negocio + técnica)

**Archivo:** `slides/presentacion_final.html` · 8 diapositivas + 6 anexos
**Duración objetivo:** ≤ 9:15 hablados (deja margen para pausas y cambios de expositor dentro de los 10 min).

Convención: **[tiempo]** al inicio · *Transición:* es el puente a la siguiente slide · **Q&A** es una pregunta probable del profesor con respuesta breve. Lo que va en *cursiva* es indicación de expositor, **no se lee en voz alta**.

Reparto sugerido: un expositor lleva 1–4 (diseño + estructura + grafos + clustering), otro 5–8 (recomendación + procesamiento + auditoría + cierre). El cambio ocurre en la transición de la slide 4 a la 5.

---

## Slide 1 — Portada / pregunta · [0:50]

Buenos días. En Yelp casi todo —qué se ve, qué parece bueno, qué se recomienda— se decide por una sola cosa: el número de reseñas. El problema es que el conteo mezcla tres cosas distintas: qué tan influyente es alguien, qué tan bueno es un restaurante y a quién conviene recomendarle qué. Nuestro trabajo separa esas tres cosas. Lo hicimos sobre tres mercados gastronómicos completos —Philadelphia, Tampa y New Orleans— con casi 2.7 millones de reseñas y 29 mil restaurantes. Y con una regla dura del curso: los diecisiete algoritmos, implementados a mano en numpy; Spark y Parquet solo para mover datos.

*No leer la metaline; señalarla.* **Cifra a remarcar:** 29,314 restaurantes / 2.67M reseñas.

*Transición:* Antes de cualquier algoritmo, la decisión más importante del proyecto: cómo elegimos ese universo.

**Q&A — ¿por qué solo tres ciudades y no todo Yelp?** Porque el dataset no es un país, son once mercados aislados; muestrear reseñas al azar rompería los grafos. Lo detallo en la siguiente slide.

---

## Slide 2 — Diseño del estudio y pipeline · [1:10]

El enunciado pide técnicas que no escalan. DBSCAN sin índice es O(n²): sobre 150 mil negocios son 2.3 por 10 a la diez distancias, unos 90 gigas de RAM. Girvan-Newman es O(m²n): sobre 52 millones de aristas, siglos. La tentación sería submuestrear reseñas al azar, pero eso rompe todo: el grafo social queda con amistades a medias y la matriz de recomendación pierde las co-ocurrencias. Así que muestreamos por clave, y la clave es el mercado completo: dentro de cada ciudad conservamos el cien por ciento de negocios, reseñas, usuarios y amistades internas. Elegimos tres porque concentran el 56.5% de las reseñas de restaurantes y son arquetipos distintos —78, 91 y 153 reseñas por local—. Los grafos y matrices quedan íntegros, no muestreados. Y ojo con la otra mitad: la Parte V sí corre sobre el stream completo, veinte millones de eventos, porque esos algoritmos existen precisamente para lo que no cabe en memoria. Abajo, la arquitectura medallón: ningún algoritmo lee JSON, todos consumen tablas gold; semilla 42 e ingesta en menos de dos minutos.

**Cifra a remarcar:** 56.5% con solo 3 de 11 mercados. *Señalar la tabla O(n²), no leerla entera.*

*Transición:* Con ese universo íntegro, empecemos por la pregunta de influencia.

**Q&A — ¿el muestreo no sesga los resultados?** Sesga menos que la alternativa: al conservar mercados completos no rompemos ninguna relación interna. El sesgo que sí asumimos —que son tres ciudades de EE.UU.— lo discutimos como transferibilidad en la Parte VII.

---

## Slide 3 — Grafos e influencia · [1:15]

Corremos PageRank e HITS a mano, con Power Iteration y la Google Matrix, teleporte 0.85. Miren las barras: en el grafo bipartito usuario-negocio, PageRank prácticamente recupera la popularidad —correlación de Spearman 0.99 con el número de reseñas—. HITS, en cambio, reordena: la autoridad baja a 0.67, y solo dos de los quince del top coinciden. La interpretación es la clave del proyecto: una authority es un restaurante avalado por buenos hubs, reseñadores Elite muy activos, no por volumen; por eso el top lo domina Philadelphia y desaparecen los gigantes turísticos de New Orleans. En comunidades hubo una iteración con evidencia: definir la arista por número de co-reseñadores daba un grafo casi completo y modularidad casi cero, porque eso confunde solape con tamaño. Al normalizar con Jaccard el grafo se dispersa y emergen micro-mercados por estilo y precio; comparando por la misma Q, el greedy CNM gana, 0.396 frente a 0.294 de Girvan-Newman.

**Cifra a remarcar:** 0.99 vs 0.67. *No decir "miles de fans"; decir "reseñadores Elite muy activos".*

*Transición:* Si esa es la estructura por influencia, ¿qué tipos de restaurante hay?

**Q&A — ¿por qué HITS y no solo PageRank?** Porque miden cosas distintas: PageRank sobre el bipartito degenera a popularidad; HITS separa hub de authority, que es justo lo que necesitábamos para distinguir "avalado" de "grande".

---

## Slide 4 — Segmentación / clustering · [1:10]

K-Means++, DBSCAN y BFR, todo a mano, sobre 42 features: atributos de Yelp más contexto socioeconómico del Census. Excluimos la geografía para no fabricar clusters por ciudad. Elegimos k con honestidad: el codo automático da cinco y la silueta favorece dos —demasiado grueso—, así que fijamos seis por interpretabilidad. La silueta final es 0.10; eso no es un fracaso, significa perfiles reales pero solapados, no especies separadas, que es lo esperable en un mercado. DBSCAN es el contraste honesto: en 42 dimensiones colapsa a un único cluster gigante con 7.7% de ruido —la maldición de la dimensionalidad, las distancias se concentran—. BFR procesa los 29 mil por bloques de cuatro mil, resumiendo en conjuntos de descarte, compresión y retención, y escala sin matriz densa con menos del 1% de outliers. Un dato de validación: purity con la ciudad 0.58 y NMI 0.007, o sea los segmentos no están copiando el mercado.

**Cifra a remarcar:** silueta 0.10 (solapamiento), NMI 0.007 (no copia ciudad). *Ser transparente: k=6 es decisión de interpretabilidad, el codo daba 5.*

*Transición:* Cambio de expositor. Vamos a la parte que más cuidamos: la recomendación.

**Q&A — ¿una silueta de 0.10 no invalida el clustering?** No: mide que las fronteras son graduales, no que no haya estructura. Los seis perfiles son interpretables y estables en 10 semillas; DBSCAN, que sí exige separación densa, confirma que en 42D no hay huecos limpios.

---

## Slide 5 — Recomendación (central) · [1:15]

Esta es la slide central. El pipeline: matriz usuario-negocio dispersa, split temporal —entreno hasta 2018, validación 2019, test 2020-21—, filtrado colaborativo item-item, content-based con TF-IDF e híbrido. Lo importante es cómo se evalúa, sin fuga: el alfa del híbrido se fija en validación y el test se toca una sola vez, con candidatos barajados para no premiar el orden. Y aquí está la trampa —miren el gráfico—: si eliges el recomendador solo por precisión, gana el top-popular, NDCG 0.26. Pero cubre apenas el 13% del catálogo: recomienda siempre los mismos famosos y deja al 87% sin ninguna exposición. El filtrado colaborativo acierta menos, 0.18, pero triplica la cobertura al 40%. La validación elige el híbrido equilibrado, que supera a los dos puros. En predicción de rating, el baseline regularizado gana, 1.24 contra 1.31. Dos límites honestos: con la matriz tan dispersa el CF se hunde en cold-start, NDCG 0.08; y optimizar solo la precisión premia concentrar.

**Cifra a remarcar:** 13% vs 40% de cobertura; "87% sin exposición". *Señalar los dos puntos extremos del scatter.*

*Transición:* Tanto el clustering como esto exigieron procesar datos que no convenía tener enteros en memoria.

**Q&A — ¿por qué el top-popular es "el más preciso" si es el más tonto?** Porque la precisión@K premia acertar el ítem popular que el usuario iba a visitar igual. Por eso el mensaje es que precisión sola es insuficiente: hay que medir cobertura y exposición, y eso es exactamente la Parte VII.

---

## Slide 6 — Reducción dimensional + streams · [1:15]

Dos técnicas con un mismo hilo: procesar lo que no conviene mantener entero. Primero reducir la dimensión. PCA formal por covarianza sobre las 42 features: hacen falta 25 de 42 ejes para superar el 90% de varianza —la señal no vive en dos o tres dimensiones, un mapa 2D deja fuera el 67.6%—. La SVD truncada la aplicamos al TF-IDF del texto, no a la matriz de ratings: imputar ratings ausentes mete sesgo, mientras que en TF-IDF el cero es ausencia real de un término. Con 80 factores comprimimos 33.6 veces, de 297 a 31 megas. Segundo, resumir el flujo. El Count-Min Sketch resume los veinte millones de eventos en 160 kilobytes, con el 99.4% de las consultas dentro de la cota teórica, recuperando el Top-20 completo al subir el ancho. Y es una cota de verdad porque CMS solo sobreestima, nunca subestima. DGIM mantiene solo diez u once buckets por semana. Y cruzamos con contexto exógeno: en la primavera de 2020 la actividad cae al 15-33% de lo normal, y el Mardi Gras la casi duplica.

**Cifra a remarcar:** 33.6× de compresión y 160 KB para 20 millones de eventos. *No leer las seis métricas; elegir una por columna.*

*Transición:* Todo lo anterior alimenta la auditoría crítica.

**Q&A — ¿por qué SVD al texto y no a la matriz de ratings, si es la típica factorización de recomendación?** Porque la matriz de ratings está 99.99% vacía; la SVD clásica exige matriz completa e imputar los huecos introduce sesgo (deck 10, pág. 20). En TF-IDF el cero es información real: ausencia del término.

---

## Slide 7 — Auditoría, riesgos y límites · [1:05]

Aquí conectamos los resultados con sus riesgos, no como una opinión aparte. El gráfico es una simulación de estrés: ¿qué pasa si alguien inyecta cinco reseñas falsas de cinco estrellas? A un restaurante con pocas reseñas le sube la mediana media estrella, +0.56; al consolidado, apenas +0.03. Y funciona en los dos sentidos: cinco de una estrella le bajan casi una estrella entera al pequeño, sirve para sabotear. Lo más injusto es que ese local frágil suele ser el peor documentado. En exposición, el CF cubre el 40% del catálogo con Gini 0.239 frente al 13% y 0.358 del top-popular; personalizar reparte mejor, pero no iguala: el cuartil más visible aún se lleva el 49% de los slots. Y somos explícitos con los límites: la matriz dispersa hunde el cold-start; en 42 dimensiones la silueta es baja; el diámetro del grafo lo damos como cota, no exacto; y no hay ground-truth de spam ni atributos protegidos, así que auditamos vulnerabilidad y proxies, no fraude ni discriminación causal.

**Cifra a remarcar:** +0.56 vs +0.03; cobertura 40% vs 13%. *Aclarar: medimos vulnerabilidad, no detectamos fraude.*

*Transición:* Entonces, ¿qué decisiones se desprenden de todo esto?

**Q&A — ¿el análisis de spam detecta reseñas falsas reales?** No. Yelp no publica qué reseña es falsa, así que no hay etiqueta. Medimos cuánto se movería la mediana ante una campaña simulada: es una prueba de vulnerabilidad, honesta sobre lo que puede y no puede afirmar.

---

## Slide 8 — Conclusiones y decisiones · [0:55]

Tres decisiones concretas. Una: no confiar en el promedio de un local con pocas reseñas —pesar por reputación de quien opina y vigilar ráfagas—, porque es ahí donde cinco reseñas falsas mueven media estrella. Dos: rankear por respaldo, no por volumen; PageRank sigue al conteo, HITS reordena y sube lo que avala la gente que sabe. Tres: medir la recomendación por cobertura y exposición, no solo por precisión, para no esconder al 87% del catálogo. Y el hilo que une todo el proyecto: el número de reseñas no mide ni influencia, ni calidad, ni confianza —y tratarlo como si lo hiciera es justamente lo que castiga a los locales pequeños—. Gracias.

**Cifra a remarcar:** la frase de cierre, dicha despacio. *No inventar un eslogan; cerrar con la frase tal cual.*

**Q&A — si tuvieran que quedarse con un solo hallazgo, ¿cuál?** Que popularidad, calidad e influencia son tres cosas distintas que Yelp colapsa en un solo número, y que separarlas cambia a quién se premia: del grande y turístico, al bien avalado y al pequeño con reputación real.

---

### Control de tiempo

| Slide | Tiempo | Acumulado |
|---|---|---|
| 1 Portada | 0:50 | 0:50 |
| 2 Diseño + pipeline | 1:10 | 2:00 |
| 3 Grafos | 1:15 | 3:15 |
| 4 Clustering | 1:10 | 4:25 |
| 5 Recomendación | 1:15 | 5:40 |
| 6 Reducción + streams | 1:15 | 6:55 |
| 7 Auditoría | 1:05 | 8:00 |
| 8 Conclusiones | 0:55 | **8:55** |

Margen de 65 s para pausas y el cambio de expositor. Los anexos A–F solo se abren si el profesor pregunta.
