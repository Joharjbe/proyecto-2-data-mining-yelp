# Guion — PPT de negocio (9 slides)

Duración objetivo: ~6–7 min. Tono: directo, sin jerga. Cada bloque es lo que dices en voz alta; la línea *Transición* es el puente a la siguiente slide.

---

## Slide 1 — Portada · [~25 s]

Buenos días. Vamos a mirar los restaurantes de Yelp con lupa de datos: quién influye de verdad, cómo se agrupan y qué tiene sentido recomendar. Lo hicimos sobre tres mercados completos —Philadelphia, Tampa y New Orleans— leyendo casi siete millones de reseñas.

*Transición:* Y todo arranca de una pregunta incómoda.

## Slide 2 — El punto de partida · [~45 s]

En Yelp casi todo se decide por una sola cosa: el número de reseñas. Eso decide qué se ve, qué parece bueno y qué se recomienda. El problema es que el conteo mezcla dos cosas distintas —qué tan popular es un sitio y qué tan bueno es— y deja fuera a los locales chicos. Sobre esa base nos hicimos tres preguntas: quién influye, cómo se agrupan los restaurantes y qué recomendar y a quién. Y no lo miramos sobre una muestra: analizamos a fondo 2.67 millones de reseñas y casi treinta mil restaurantes.

*Transición:* Lo primero que encontramos es quién está escribiendo todo esto.

## Slide 3 — La voz está en pocas manos · [~45 s]

Miren la desproporción: el 10% de usuarios más activos escribe el 56% de todas las reseñas. Más de la mitad de la opinión de Yelp la produce una franja pequeñísima de gente. Y para dimensionarlo, casi la mitad de los usuarios ni siquiera tiene amigos en la red. Esto importa por una razón muy práctica: cualquier sistema que aprenda de estos datos —un recomendador, un ranking— va a aprender el gusto de esa minoría, no el del usuario promedio.

*Transición:* Y si contar reseñas no mide influencia… tampoco mide calidad.

## Slide 4 — Popularidad no es calidad · [~50 s]

Cuando ordenamos los restaurantes por popularidad pura, el ranking sigue casi perfectamente al volumen de reseñas —esa barra de 0.99—. Pero cuando en vez de contar reseñas medimos quién las respalda, es decir cuando pesa el aval de los reseñadores serios, el mapa se reordena: esa coincidencia cae a 0.67. En concreto: los gigantes turísticos de New Orleans, que ganan por volumen, desaparecen del podio, y suben restaurantes de Philadelphia que la gente que sabe realmente recomienda. La lección es simple: ser el más reseñado casi siempre es solo ser el más grande o el más turístico.

*Transición:* Ese sesgo hacia el volumen tiene una víctima clara: el local pequeño. Y ahí está el hallazgo más delicado.

## Slide 5 — La reputación del pequeño es frágil · [~50 s]

Esto es una simulación de estrés —Yelp no publica qué reseña es falsa, así que no detectamos fraude, medimos vulnerabilidad—. La pregunta fue: ¿qué pasa si alguien inyecta cinco reseñas falsas de cinco estrellas? A un restaurante con pocas reseñas le sube la mediana media estrella, +0.56. Al mismo experimento, un restaurante consolidado apenas se mueve, +0.03. Y funciona en los dos sentidos: cinco reseñas de una estrella le bajan casi una estrella entera al pequeño, o sea sirve para sabotear a un competidor. Lo más injusto es que ese local frágil suele ser también el peor documentado —mediana de 25 reseñas en los barrios de menor ingreso—.

*Transición:* Y si el promedio no es de fiar, la pregunta natural es qué recomendamos entonces.

## Slide 6 — El recomendador "más preciso" muestra siempre lo mismo · [~45 s]

Aquí hay una trampa. Si eliges el recomendador solo por cuánto acierta, el ganador es recomendar lo popular. Pero ese modelo llega a apenas el 13% del catálogo: recomienda siempre los mismos restaurantes famosos y deja al 87% sin ninguna exposición. Un recomendador personalizado acierta un poco menos, pero triplica el alcance: llega al 40% de los locales. O sea, optimizar solo la precisión castiga a la cola larga. La diversidad tiene que ser un objetivo explícito, no algo que salga por accidente.

*Transición:* Cambiemos de pregunta: si dejamos de mirar rankings, ¿qué tipos de restaurante hay?

## Slide 7 — Seis tipos de restaurante, no seis ciudades · [~40 s]

Cuando dejamos que los datos agrupen solos a los restaurantes, salen seis perfiles de operación: desde servicio completo con bar y reservas hasta los pequeños de baja tracción. Y lo interesante es que esos seis perfiles cruzan las tres ciudades: un restaurante se parece más a otro de su mismo tipo en otra ciudad que a su vecino de al lado. Para el negocio esto es útil directo: sirve para segmentar campañas y para comparar cada local con su verdadera competencia, que es su perfil, no su código postal.

*Transición:* Un último matiz, y es importante para no sacar conclusiones apresuradas.

## Slide 8 — La actividad no depende solo del restaurante · [~40 s]

No todo lo que le pasa a un local es mérito o culpa suya. En la primavera de 2020 las visitas se desplomaron a entre el 15 y el 33% de lo normal, según la ciudad —eso es la pandemia, no el restaurante—. Y al revés: durante el Mardi Gras las visitas en New Orleans casi se duplican, un índice de 187 contra alrededor de 100 en ciudades que no tienen el evento. La conclusión: para comparar negocios de forma justa hay que controlar el momento y el lugar.

*Transición:* Entonces, ¿qué hacemos con todo esto?

## Slide 9 — Qué hacer · [~45 s]

Tres decisiones concretas. Una: no confiar en el promedio de un local con pocas reseñas —pesar por reputación de quien opina y vigilar ráfagas—. Dos: rankear por respaldo, no por volumen, porque el conteo premia al grande y al turístico. Tres: medir la recomendación por cobertura, no solo por precisión, para no esconder al 87% del catálogo. Y el hilo que une todo: el número de reseñas no mide ni influencia, ni calidad, ni confianza —y tratarlo como si lo hiciera es justamente lo que castiga a los locales pequeños—. Gracias.
