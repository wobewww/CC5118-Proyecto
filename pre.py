import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

# =========================
# 1. CARGA CORRECTA
# =========================

df = pd.read_csv(
    "testMORERE.csv",
    header=2
)

# eliminar columnas basura iniciales
df = df.iloc[:, 2:]

# limpiar nombres
df.columns = df.columns.str.strip()

# =========================
# 2. LIMPIAR BOOLEANOS
# =========================

df["is_recommended"] = (
    df["is_recommended"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df = df[df["is_recommended"] == "TRUE"]

print("Filas:", len(df))


# =========================
# 3. AGRUPAR POR USUARIO
# =========================

grouped = df.groupby("user_id")["app_id"].apply(list)


# =========================
# 4. CONSTRUIR GRAFO
# =========================

G = nx.Graph()

for games in grouped:
    if len(games) < 2 or len(games) > 20:
        continue

    for g1, g2 in combinations(games, 2):
        if G.has_edge(g1, g2):
            G[g1][g2]["weight"] += 1
        else:
            G.add_edge(g1, g2, weight=1)


# =========================
# 5. MÉTRICAS BÁSICAS
# =========================

print("\nNodos:", G.number_of_nodes())
print("Aristas:", G.number_of_edges())

print("Densidad:", nx.density(G))


# =========================
# 6. BETWEENNESS (PUENTES)
# =========================

bet = nx.betweenness_centrality(G, weight="weight")

top = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop juegos puente:")
for g, s in top:
    print(g, s)


# =========================
# 7. COMUNIDADES
# =========================

from networkx.algorithms.community import greedy_modularity_communities

communities = list(greedy_modularity_communities(G))

print("\nComunidades:", len(communities))


# =========================
# 8. VISUALIZACIÓN MEJORADA
# =========================

plt.figure(figsize=(12, 8))

# layout más compacto (CLAVE para ver aristas)
pos = nx.spring_layout(G, seed=42, k=0.3, iterations=100)

# nodos
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=300,
    node_color="lightblue"
)

# aristas (más visibles)
nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.6,
    width=1.5
)

# etiquetas
nx.draw_networkx_labels(
    G,
    pos,
    font_size=8
)

plt.title("Grafo de co-recomendaciones de Steam")
plt.axis("off")
plt.show()