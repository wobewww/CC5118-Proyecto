import gzip
import pickle

import networkx as nx
import pandas as pd


print("Cargando grafo enriquecido...")
with gzip.open("RED-ENRIQUECIDA_100K.pkl.gz", "rb") as f:
    data = pickle.load(f)

G_enriquecido = data["grafo"]
communities = data.get("comunidades", [])


# Verificar si es conectado (ya sabemos que dará False, pero es buena práctica)
if not nx.is_connected(G_enriquecido):
    print("El grafo está desconectado. Extrayendo la componente conexa más grande...")
    
    # Obtener las componentes conectadas ordenadas por tamaño (de mayor a menor)
    components = sorted(nx.connected_components(G_enriquecido), key=len, reverse=True)
    
    # Crear un subgrafo con la componente más grande
    G_giant = G_enriquecido.subgraph(components[0]).copy()
    
    print(f"Nodos originales: {G_enriquecido.number_of_nodes()} | Nodos componente gigante: {G_giant.number_of_nodes()}")
    
    # Calcular el path length en la componente gigante
    average_path_length = nx.average_shortest_path_length(G_giant)
    print(f"Average path length (Componente Gigante): {average_path_length}")
else:
    average_path_length = nx.average_shortest_path_length(G_enriquecido)
    print(f"Average path length: {average_path_length}")