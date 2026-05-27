# enrich_graph.py

import pandas as pd
import networkx as nx
import pickle
import gzip

# 1. Cargar la red ya construida
with gzip.open("red_construida.pkl.gz", "rb") as f:
    data = pickle.load(f)

G = data["grafo"]
communities = data["comunidades"]

# 2. Cargar el dataset de Steam Store
df_steam = pd.read_csv("steam_games.csv")
df_steam = df_steam.rename(columns={"appid": "app_id"})
df_steam = df_steam[["app_id", "name", "genres", "developer", "publisher"]].copy()

# 3. Cruce con los nodos del grafo
nodos = list(G.nodes())
df_nodos = pd.DataFrame({"app_id": nodos})
df_nodos = df_nodos.merge(df_steam, on="app_id", how="left")

# 4. Asignar atributos a los nodos del grafo
for _, row in df_nodos.iterrows():
    node = row["app_id"]
    if G.has_node(node):
        G.nodes[node]["name"]      = row["name"]      if pd.notna(row["name"])      else "Unknown"
        G.nodes[node]["genres"]    = row["genres"]    if pd.notna(row["genres"])    else "Unknown"
        G.nodes[node]["developer"] = row["developer"] if pd.notna(row["developer"]) else "Unknown"

# 5. Verificar cobertura
total = len(nodos)
con_info = df_nodos["name"].notna().sum()
print(f"Nodos totales: {total}")
print(f"Con info en Steam Store: {con_info} ({con_info/total*100:.1f}%)")
print(f"Sin match: {total - con_info}")
print("\nEjemplo de nodos con géneros:")
print(df_nodos[["app_id", "name", "genres"]].head(10).to_string())

# 6. Guardar el grafo enriquecido
with gzip.open("red_enriquecida.pkl.gz", "wb") as f:
    pickle.dump({"grafo": G, "comunidades": communities}, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nGrafo enriquecido guardado en red_enriquecida.pkl.gz")