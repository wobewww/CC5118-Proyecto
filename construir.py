import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
import pickle
import gzip

# 1. Carga y limpieza de datos
df_completo = pd.read_parquet("recommendations_sorted.parquet")
df = df_completo.head(100000).copy()
df.columns = df.columns.str.strip()
if "is_recommended" in df.columns:
    df["is_recommended"] = df["is_recommended"].astype(str).str.strip().str.upper()
    df = df[df["is_recommended"] == "TRUE"]

# 2. Construcción de la red
grouped = df.groupby("user_id")["app_id"].apply(list)
G = nx.Graph()

for games in grouped:
    if 1 < len(games) <= 20:
        for g1, g2 in combinations(games, 2):
            if G.has_edge(g1, g2):
                G[g1][g2]["weight"] += 1
            else:
                G.add_edge(g1, g2, weight=1)

for u, v, datos in G.edges(data=True):
    datos["weight"] = 1.0 / datos["weight"]

# 3. Análisis de comunidades y métricas
communities = list(greedy_modularity_communities(G))
color_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        color_map[node] = i

node_colors = [color_map[n] for n in G.nodes()]

# 4. Guardar la red y comunidades comprimidas
with gzip.open("RED-INICIAL_100K.pkl.gz", "wb") as f:
    pickle.dump({"grafo": G, "comunidades": communities}, f, protocol=pickle.HIGHEST_PROTOCOL)

# 5. Visualización
plt.figure(figsize=(14, 10), facecolor='#f0f0f0')

pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

weights = [G[u][v]['weight'] for u, v in G.edges()]
inverse_weights = [1.0 / w for w in weights]
max_inv_w = max(inverse_weights) if inverse_weights else 1
edge_widths = [(w / max_inv_w) * 3 for w in inverse_weights]

nx.draw_networkx_edges(G, pos, alpha=1, edge_color="gray", width=edge_widths)

nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=plt.cm.Set3, edgecolors="white", linewidths=1)

plt.title("Red de Co-recomendaciones (Steam - Muestra 100k)", fontsize=15)
plt.axis("off")
plt.tight_layout()
plt.show()

print(f"\n--- Resumen de la Red ---")
print(f"Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
print(f"Comunidades detectadas: {len(communities)}")