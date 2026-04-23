import pandas as pd
import networkx as nx
from itertools import combinations

# =========================
# 1. CARGA CORRECTA
# =========================

df = pd.read_csv(
    "testMORERE.csv",
    header=2   # 👈 salta filas basura
)

# eliminar columnas vacías iniciales
df = df.iloc[:, 2:]   # 👈 elimina los dos ",,"

# limpiar nombres
df.columns = df.columns.str.strip()

# =========================
# 2. LIMPIAR BOOLEANOS
# =========================

df["is_recommended"] = df["is_recommended"].astype(str).str.strip().str.upper()
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
# 5. ANÁLISIS
# =========================

print("Nodos:", G.number_of_nodes())
print("Aristas:", G.number_of_edges())


# 🔥 juegos puente
bet = nx.betweenness_centrality(G, weight="weight")

top = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop juegos puente:")
for g, s in top:
    print(g, s)


# =========================
# 6. COMUNIDADES
# =========================

from networkx.algorithms.community import greedy_modularity_communities

communities = list(greedy_modularity_communities(G))

print("\nComunidades:", len(communities))