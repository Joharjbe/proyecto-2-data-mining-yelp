# Matriz de trazabilidad contra la rúbrica

Esta matriz enlaza cada criterio de `Enunciado_proyecto2.pdf` con evidencia reproducible. Los PDFs del curso no se redistribuyen en el repositorio público; las citas indican el deck y la página que debe contrastar el profesor.

| Criterio | Puntos | Evidencia principal | Implementación y teoría | Resultados verificables |
|---|---:|---|---|---|
| Selección y preparación | 2 | Notebooks 01–04 | `src/preprocessing.py`, `src/enrichment.py`; limpieza, EDA completo antes del recorte y muestreo por clave (deck 06, págs. 9–13) | 5 tablas silver; 7 gold base; 29,314 restaurantes; 2.67M reseñas; densidad 0.011%; diámetro ≥12 |
| Grafos y ranking | 4 | Notebook 05 | `src/graphs.py`; PageRank (deck 07, págs. 18–32), HITS (deck 07, pág. 40 + MMDS cap. 5), Girvan–Newman y modularidad (deck 12, págs. 9–34) | PageRank/HITS en ambos grafos; 33 comunidades, 4 grandes caracterizadas; Q(CNM)=0.396 vs Q(GN)=0.294; 3 tablas y 4 figuras |
| Clustering | 4 | Notebook 06 | `src/clustering.py`; K-Means++ (deck 08, págs. 19–27), BFR (31–42), DBSCAN y métricas (45–49) | Barrido k=2…10; k-distance/eps; 10 semillas; K-Means++, DBSCAN y BFR comparados con SSE, silueta, purity y NMI; 3 tablas y 8 figuras |
| Recomendación | 4 | Notebook 07 | `src/recommenders.py`; TF-IDF/content (deck 09, págs. 14–21), Pearson/item-item (27–41), RMSE/MAE/cold-start (42–43), SVD y sesgos de imputación (deck 10, pág. 20) | Split temporal sin fuga; α calibrado en 2019 antes de test; CF, contenido y blend; P@10/R@10/NDCG/RMSE/MAE; baselines; cobertura/novedad; 4 tablas y 6 figuras explicables |
| Flujos + reducción | 3 | Notebooks 09 y 08 | `src/streaming.py`, `src/dimensionality_reduction.py`; streams/CMS/DGIM (deck 06, págs. 4–7, 16–23, 33–37); proyección/SVD/error (deck 13, págs. 11, 25–26, 36–45, 65–69) | 20.35M eventos; CMS con cota y 160 KB; DGIM vs exacto; PCA 25/42 para 90%; SVD k=80, error/compresión/factores; 11 tablas y 12 figuras |
| Ética + código + reporte | 3 | Notebook 10, README y esta matriz | `src/critical_analysis.py`; complejidad, memoria, representación, exposición, spam, contexto ACS, privacidad, drift y transferibilidad | 12 tablas y 6 figuras de auditoría; tests automatizados; 10 notebooks ejecutados; reporte final 6–8 páginas aún debe exportarse como entregable separado |

## Requisitos transversales

- Los algoritmos del curso están implementados con NumPy/Python, sin NetworkX, SciPy o scikit-learn para sustituirlos.
- Cada notebook contiene fundamento teórico antes del experimento, validación controlada, resultados reales, interpretación clara y cierre.
- Los algoritmos consumen `data/gold/`; ninguna tabla cruda se publica.
- Semilla global 42, split temporal en recomendación y resultados persistidos en Parquet/PNG.
- `python -m unittest discover -s tests -v` valida ejemplos de clase y casos límite.
- `data/`, `.census_key`, entornos y cachés están excluidos por `.gitignore`.

## Pendientes fuera del repositorio técnico

El código y las siete partes están completos. Para la entrega integral todavía deben producirse el reporte final de 6–8 páginas y la presentación de 30 minutos con demo/ensayo; la propia rúbrica indica que una pregunta no respondida afecta el puntaje.
