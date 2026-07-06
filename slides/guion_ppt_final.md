# Guion del presentador — `ppt_final.html`

**Archivo:** `slides/ppt_final.html` · portada + 19 láminas + apéndice
**Duración objetivo:** ~13 min (rango cómodo 12–15). Cifras verificadas contra `data/gold/`.

**Reparto (3 expositores) — los cortes caen entre bloques, sin partir un par negocio→técnico:**
| Expositor | Bloque | Láminas | Tiempo |
|---|---|---|---|
| **E1** | Apertura · datos · influencia | 1–8 | ~5:00 |
| **E2** | Fragilidad · segmentación · recomendación | 9–14 | ~4:30 |
| **E3** | Ética · contexto · cierre | 15–19 | ~3:30 |

**Convención:** `[tiempo]` al inicio · *Transición:* = puente a la siguiente · **Cifra:** = el dato a remarcar · *Q&A:* = pregunta probable + respuesta breve. Lo que va en *cursiva* es indicación de expositor, **no se lee**. Color de lámina: 🟦 claro = negocio · 🌑 oscuro = técnico. El **apéndice** (última lámina) es backup para preguntas de parámetros.

**Hilo conductor (repetir en las 3 bocas):** *la popularidad no es lo mismo que la calidad, la influencia ni la autoridad.*

---

## BLOQUE A — Expositor 1 (láminas 1–8)

### Lámina 1 · Portada 🟦/🌑 · [0:40]
Buenos días. En Yelp casi todo —qué se ve, qué parece bueno y qué se recomienda— se decide por una sola cifra: el número de reseñas. El problema es que ese conteo mezcla tres cosas distintas: quién influye, qué tan bueno es un restaurante y a quién conviene recomendarle qué. Nuestro trabajo separa esas tres cosas sobre tres mercados gastronómicos completos —Philadelphia, Tampa y New Orleans—, casi 2.7 millones de reseñas y 29 mil restaurantes. Y con una regla dura del curso: los algoritmos, implementados **a mano** en numpy; Spark y Parquet solo para mover datos.
*Transición:* Empecemos por el problema.
**Cifra:** 29,314 restaurantes · 2.67M reseñas. *Señalar la leyenda claro/oscuro: claro = negocio, oscuro = técnico.*

### Lámina 2 · El problema 🟦 · [0:40]
El conteo de reseñas decide qué se ve, qué parece bueno y qué se recomienda. Sobre eso montamos tres preguntas: ¿quién influye de verdad?, ¿cómo se agrupan los restaurantes?, y ¿qué recomendar, y a quién? Todo sobre tres mercados completos.
*Transición:* Antes de cualquier algoritmo, la decisión más importante fue cómo elegimos ese universo.

### Lámina 3 · Datos y preprocesamiento 🌑 · [0:50]
Elegimos Yelp por su grafo de amigos, sus check-ins y sus atributos geográficos —ideal para grafos, flujos y rankings—; y ojo: el dataset no es un país, son **11 islas urbanas**, así que la unidad natural es el mercado. Muestrear reseñas al azar rompería el grafo social y las co-ocurrencias del recomendador, por eso muestreamos por **mercado completo**: tres mercados que ya concentran el **56.5%** de las reseñas de restaurantes, con grafos y matrices íntegros. Lo enriquecimos con censo ACS como contexto socioeconómico —Yelp sigue siendo el dataset primario—. Todo sobre arquitectura medallón, con semilla 42 y cero librerías de ML.
*Transición:* Y antes de recortar nada, miramos los datos completos.
**Cifra:** 56.5% con solo 3 de 11 mercados. *Señalar la tabla O(n²), no leerla.*
*Q&A — ¿el ACS no viola "un único dataset"?* Yelp es el dataset primario de modelado; ACS entra como contexto público por join geográfico, no como segundo dataset.

### Lámina 4 · Parte I — EDA 🌑 · [0:45]
El EDA muestra colas largas en todo: el **1% de usuarios escribe el 24%** de las reseñas y el top 10% el 56%, con Gini 0.61. El 67% de las reseñas son de 4 o 5 estrellas, pero las de 1 estrella traen **54% más texto** —mejor señal para el análisis de contenido—. Y el grafo usuario-restaurante es un **mundo pequeño** sobre una matriz casi vacía: densidad 0.011%, componente gigante del 100%, distancia media 6.9 saltos. Esa combinación es justo la que hace interesantes a PageRank y al filtrado colaborativo.
*Transición:* Con eso claro, primera pregunta: ¿quién habla?

### Lámina 5 · Hallazgo 1 — Quién habla 🟦 · [0:45]
El 10% de usuarios más activos escribe el 56% de las reseñas. Una minoría define lo que el resto ve, y cualquier modelo que entrenemos hereda el gusto de esa minoría. Además, el 44% de los usuarios no tiene amigos en la red. Conclusión: **contar reseñas no mide influencia** —hay que mirar a quién sigue y avala la gente, no cuánto grita—.
*Transición:* Si contar no mide influencia, ¿los más reseñados son los mejores?

### Lámina 6 · Hallazgo 2 — Popularidad ≠ autoridad 🟦 · [0:40]
No. Si ordenamos por **respaldo** —quién recomienda— en vez de por volumen, el podio cambia: los gigantes turísticos de New Orleans desaparecen y suben restaurantes de Philadelphia avalados por reseñadores serios. Ser el más reseñado casi siempre es solo ser el más grande o el más turístico. Un buen aval pesa más que mil reseñas anónimas.
*Transición:* Esto lo medimos con dos algoritmos de grafos.

### Lámina 7 · Parte II — Rankings 🌑 · [0:45]
Implementamos PageRank y HITS a mano, validados contra el ejemplo de clase *"The Web in 1839"*. En el grafo social, PageRank refina el grado —los usuarios top son casi todos Elite, correlación 0.85 con el número de amigos—. En el bipartito, PageRank recupera la popularidad, correlación 0.99 con el número de reseñas. Pero **HITS reordena**: correlación 0.67, y solo 2 de 15 coinciden con PageRank. Una *authority* es un restaurante avalado por buenos *hubs* —reseñadores serios—, no por volumen.
*Transición:* Y si agrupamos por co-reseña, aparecen mercados.
**Cifra:** solo 2 de 15 coinciden.

### Lámina 8 · Parte II — Comunidades 🌑 · [0:35]
Con co-reseña cruda el grafo colapsaba —modularidad casi cero, confunde solape con tamaño—. Al normalizar con el índice de **Jaccard** emergen micro-mercados; el greedy CNM gana con modularidad 0.396. Salen cuatro comunidades reales en Philadelphia: clásicos icónicos, alta cocina de autor, trendy de gama alta y casual de barrio —por estilo y precio, no por popularidad—. Y un matiz: la alta cocina de autor funciona como **puente** entre segmentos.
*Transición:* Con esto cierro la parte de influencia; sigue [nombre].

---

## BLOQUE B — Expositor 2 (láminas 9–14)

### Lámina 9 · Hallazgo 3 — Reputación frágil 🟦 · [0:50]
Retomo el hilo: si el promedio se calcula sobre pocas reseñas, no es confiable. Lo simulamos: **cinco reseñas falsas de 5 estrellas suben +0.56 de estrella** a un local con pocas reseñas, pero solo +0.03 a uno con muchas. Y también sirve para sabotear: cinco reseñas de 1 estrella le bajan 0.88. Lo grave es que el local frágil es el peor documentado —mediana de 25 reseñas en los barrios de menor ingreso—. Con pocas reseñas conviene pesar por reputación del reseñador y vigilar ráfagas.
*Transición:* Segunda pregunta: ¿cómo se agrupan los restaurantes?

### Lámina 10 · Hallazgo 4 — Seis perfiles 🟦 · [0:45]
Los restaurantes se organizan en **seis perfiles de operación, no por ciudad**: desde servicio completo con bar hasta casual de delivery. Y los seis cruzan las tres ciudades: un restaurante se parece más a otro de su mismo tipo en otra ciudad que a su vecino de al lado. La competencia real de un local es su perfil de operación, no su código postal.
*Transición:* Estos perfiles salen de tres algoritmos de clustering.

### Lámina 11 · Parte III — Clustering 🌑 · [0:45]
K-Means++ con **k=6** —elegido por el codo, con empate técnico entre 5 y 6, y decidido por interpretabilidad—. La silueta es baja, **0.095**: los perfiles son reales pero solapados, no especies separadas. DBSCAN colapsa en un solo cluster porque en 42 dimensiones las distancias se concentran; y BFR procesa los 29 mil por bloques con solo 0.75% de outliers. Que el purity y el NMI con la ciudad sean casi cero confirma que los segmentos **no copian la geografía**.
*Transición:* Y para manejar esas 42 dimensiones, las reducimos.

### Lámina 12 · Parte VI — Dimensionalidad 🌑 · [0:40]
PCA formal: hacen falta **25 de 42 componentes** para conservar el 90% de la varianza —un mapa 2D deja fuera el 67.6% de la señal—; el primer eje captura documentación del negocio, el segundo precio y servicio. En texto, la SVD truncada comprime **33.6 veces** capturando el 23.4% de la energía, con factores semánticos claros: pizza-delivery frente a bar-dining, sushi frente a café. Reducir ilumina, pero también borra.
*Transición:* Tercera pregunta: ¿qué recomendar, y a quién?

### Lámina 13 · Hallazgo 5 — Recomendar ≠ acertar 🟦 · [0:45]
Aquí está la trampa: el recomendador **más "acertado" es el que muestra siempre lo mismo**. Recomendar lo popular acierta más, pero deja al 87% del catálogo sin exposición —solo cubre el 13%—. Personalizar cede algo de precisión y llega al triple de locales, el 40%. Optimizar solo la precisión castiga a la cola larga; la diversidad tiene que ser un objetivo, no un accidente.
*Transición:* Veámoslo con las métricas reales.

### Lámina 14 · Parte IV — Recomendación híbrida 🌑 · [0:45]
Evaluación temporal **sin fuga**: entrenamos hasta 2018, calibramos en 2019 y el test se toca una sola vez. En NDCG, el top-popular lidera con 0.26 pero cubre solo 13%; el filtrado colaborativo baja a 0.18 pero **triplica la cobertura al 40%**; y la validación elige un híbrido equilibrado. Por eso no elegimos por precisión sola: reportamos también cobertura, novedad e intervalos bootstrap. Un detalle honesto: al barajar candidatos, rankear por suma de similitudes ya no supera al rating predicho.
*Transición:* Esa tensión precisión-diversidad abre la parte crítica; sigue [nombre].
**Cifra:** top-popular 0.26 / 13% vs CF 0.18 / 40%.
*Q&A — ¿por qué "gana" el top-popular?* Porque la precisión premia acertar en lo masivo; pero concentra. Medimos cobertura y equidad justamente para no premiar esa concentración.

---

## BLOQUE C — Expositor 3 (láminas 15–19)

### Lámina 15 · Parte VII — Escalabilidad y ética 🌑 · [0:50]
Cierro con lo crítico. En escalabilidad, el benchmark confirma lo esperado: K-Means crece casi lineal, DBSCAN cuadrático; hacemos lo exacto donde cabe y lo aproximado con garantías donde no. Y la ética la integramos al pipeline con una matriz **dato → método → riesgo → mitigación**, no como opinión aparte. En exposición: el filtrado colaborativo llega a **10,402 negocios** —40%— frente a 3,376 del top-popular, y baja el Gini de exposición; pero el cuartil más visible aún se lleva el 49% de los espacios. Y el contexto pesa: los barrios de menor ingreso tienen 21% de atributos faltantes y la mitad de reseñas. Personalizar reparte, pero no borra la historia.
*Transición:* Y no todo lo que sube o baja es culpa del restaurante.

### Lámina 16 · Hallazgo 6 — Contexto externo 🟦 · [0:45]
La actividad depende también de la ciudad. En la primavera de 2020 los check-ins se desplomaron: al **16% del nivel de 2019 en Philadelphia, 15% en New Orleans y 33% en Tampa**. Y al revés, durante el Mardi Gras las visitas en New Orleans casi se duplican. Sin ese contexto, una pandemia o un feriado se confunde con un mal restaurante: comparar negocios exige controlar el momento y el lugar.
*Transición:* Eso lo medimos sobre el stream completo, con algoritmos de flujos.

### Lámina 17 · Parte V — Minería de flujos 🌑 · [0:40]
Sobre los **20.35 millones de eventos**, Count-Min Sketch resume todo en 160 KB manteniendo el 99.4% de las consultas dentro de la cota teórica. DGIM estima horas activas con apenas 10–11 buckets y un error semanal de dos horas. Y sobre eso corren los cruces externos: la caída de COVID y el índice 187 del Mardi Gras. Convertimos memoria en error controlado.
*Transición:* Y somos honestos con lo que estos resultados no pueden sostener.

### Lámina 18 · Límites técnicos 🌑 · [0:35]
Cinco límites, dichos de frente: la matriz al 0.011% deja al filtrado colaborativo sin dónde anclarse en cold-start; en 42 dimensiones las distancias se concentran, de ahí la silueta baja; el diámetro del grafo se reporta como cota, no exacto; no tenemos etiqueta real de spam, así que auditamos vulnerabilidad, no fraude; y el censo tiene nulos, que tratamos con umbral, sin imputar de más.
*Transición:* Y con eso, qué hacer con todo esto.

### Lámina 19 · Cierre 🟦 · [0:55]
Tres decisiones se desprenden del análisis: no confiar en el promedio de un local con pocas reseñas; rankear por respaldo, no por volumen; y medir la recomendación por cobertura, no solo por precisión. El hilo común de toda la charla: **el número de reseñas no mide ni influencia, ni calidad, ni confianza** —y tratarlo como si lo hiciera castiga a los locales pequeños—. Cubrimos las siete partes, con todos los algoritmos a mano y validados contra los ejemplos de clase. Gracias.
**Cifra:** las 7 partes ✓, 0 librerías de ML. *Señalar los chips de rúbrica.*

---

## Apéndice · backup para Q&A (no se expone)
La última lámina resume cada **elección de valor** con su justificación. Respuestas rápidas si preguntan:
- **¿Por qué Yelp y no Amazon?** Grafo social + check-ins + geografía → ideal para grafos, flujos y rankings.
- **¿Por qué k=6 si la silueta favorece k=2?** El codo da empate 5/6; la interpretabilidad decide 6. Silueta baja = perfiles solapados, no separables.
- **¿Cómo eligieron eps de DBSCAN?** Rodilla del k-distance plot, percentil 80 → eps 5.19.
- **¿Cómo fijaron α del híbrido?** En validación 2019 (NDCG 0.244), antes de tocar el test.
- **¿El diámetro es exacto?** No: double-sweep con 4 semillas como cota inferior (≥10); un BFS desde 843k nodos es inviable.
- **¿El benchmark da pendiente exacta?** Mide microsegundos y varía entre corridas; reportamos el comportamiento (lineal vs cuadrático), no un número fijo.
- **¿Validaron los algoritmos?** Sí: PageRank contra *"The Web in 1839"*, y cada módulo contra ejemplos de clase con tests automatizados.
