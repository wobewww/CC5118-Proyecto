# enrich_graph.py

import pandas as pd
import networkx as nx
import pickle
import gzip

# 1. Cargar la red ya construida
with gzip.open("RED-INICIAL_100K.pkl.gz", "rb") as f:
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

# 3b. Quedarse solo con los nodos que sí tienen match en Steam
df_match = df_nodos[df_nodos["name"].notna()].copy()
matched_ids = set(df_match["app_id"])
G_enriquecido = G.subgraph(matched_ids).copy()


def filtrar_comunidades(comunidades, nodos_validos):
    if isinstance(comunidades, dict):
        return {
            clave: [nodo for nodo in nodos if nodo in nodos_validos]
            for clave, nodos in comunidades.items()
        }
    if isinstance(comunidades, (list, tuple)):
        filtradas = []
        for comunidad in comunidades:
            if isinstance(comunidad, (list, set, tuple)):
                filtradas.append(type(comunidad)(nodo for nodo in comunidad if nodo in nodos_validos))
            else:
                filtradas.append(comunidad)
        return type(comunidades)(filtradas)
    return comunidades


communities = filtrar_comunidades(communities, matched_ids)

# 4. Asignar atributos a los nodos del grafo
for _, row in df_match.iterrows():
    node = row["app_id"]
    if G_enriquecido.has_node(node):
        G_enriquecido.nodes[node]["name"] = row["name"]
        G_enriquecido.nodes[node]["genres"] = row["genres"]
        G_enriquecido.nodes[node]["developer"] = row["developer"]

# 5. Verificar cobertura
total = len(nodos)
con_info = df_nodos["name"].notna().sum()
total_enriquecido = len(G_enriquecido.nodes())
print(f"Nodos totales: {total}")
print(f"Con info en Steam Store: {con_info} ({con_info/total*100:.1f}%)")
print(f"Sin match: {total - con_info}")
print(f"Nodos guardados en el grafo enriquecido: {total_enriquecido}")
print("\nEjemplo de nodos con géneros:")
print(df_match[["app_id", "name", "genres"]].head(10).to_string())

# 6. Guardar el grafo enriquecido
with gzip.open("RED-ENRIQUECIDA_100K.pkl.gz", "wb") as f:
    pickle.dump({"grafo": G_enriquecido, "comunidades": communities}, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nGrafo enriquecido guardado en RED-ENRIQUECIDA_100K.pkl.gz")