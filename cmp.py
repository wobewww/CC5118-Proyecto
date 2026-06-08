import gzip
import pickle
import networkx as nx
import math

# Cargar grafo
with gzip.open("RED-ENRIQUECIDA_100K.pkl.gz", "rb") as f:
    data = pickle.load(f)

G_enriquecido = data["grafo"]

# Extraer la componente conexa más grande
components = nx.connected_components(G_enriquecido)
largest_component = max(components, key=len)
G_eval = G_enriquecido.subgraph(largest_component)

# Obtener N y <k>
N = G_eval.number_of_nodes()
k_promedio = sum(dict(G_eval.degree()).values()) / N

# Calcular métricas del modelo ER
L_er = math.log(N) / math.log(k_promedio)
C_er = k_promedio / N 

print(f"L esperado (ER): {L_er:.2f}")
print(f"C esperado (ER): {C_er:.6f}")

# Calcular métricas reales
L_real = nx.average_shortest_path_length(G_eval)
C_real = nx.average_clustering(G_eval)

print(f"L real: {L_real:.2f}")
print(f"C real: {C_real:.6f}")