# Visualización de Co-recomendaciones de Apps con NetworkX

Este proyecto procesa un conjunto de datos de recomendaciones de aplicaciones para generar un **Grafo de Co-recomendación**. El objetivo es identificar qué aplicaciones suelen ser recomendadas juntas por los mismos usuarios, visualizando estas relaciones mediante un layout de grafos organizado por componentes conectados (islas).

## Requisitos Previos

Antes de ejecutar el script, asegúrate de tener instalado Python 3.8+ y las siguientes bibliotecas de análisis de datos:

```bash
pip install pandas networkx matplotlib numpy