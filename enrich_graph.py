# enrich_graph.py

import pandas as pd
import networkx as nx
import pickle
import gzip
import numpy as np

# 1. Cargar la red ya construida
with gzip.open("red_construida1MI.pkl.gz", "rb") as f:
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
with gzip.open("red_enriquecida.pkl.gz", "wb") as f:
    pickle.dump({"grafo": G_enriquecido, "comunidades": communities}, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nGrafo enriquecido guardado en red_enriquecida.pkl.gz")

# ---- CÁLCULO DE CENTRALIDADES ----
print('\nCalculando centralidades...')

# 1) Grado (número de vecinos directos)
degree = dict(G_enriquecido.degree())
degree_centrality = nx.degree_centrality(G_enriquecido)

# 2) Eigenvector centrality (intenta usar la versión basada en numpy)
try:
    eigen_centrality = nx.eigenvector_centrality_numpy(G_enriquecido, weight='weight')
except Exception:
    try:
        eigen_centrality = nx.eigenvector_centrality(G_enriquecido, max_iter=200, tol=1e-06, weight='weight')
    except Exception as e:
        print('No se pudo calcular eigenvector centrality:', e)
        eigen_centrality = {n: 0.0 for n in G_enriquecido.nodes()}

# 3) Alpha-centrality: usamos la implementación de Katz como aproximación/generalización
# Estimamos un alpha seguro usando el radio espectral (máximo valor absoluto propio)
alpha = 0.01
try:
    A = nx.to_numpy_array(G_enriquecido, weight='weight')
    if A.size > 0:
        eigs = np.linalg.eigvals(A)
        max_eig = max(abs(eigs))
        if max_eig > 0:
            alpha = 0.85 / max_eig
        else:
            alpha = 0.01
except Exception:
    alpha = 0.01

beta = {n: 1.0 for n in G_enriquecido.nodes()}
try:
    alpha_centrality = nx.katz_centrality_numpy(G_enriquecido, alpha=alpha, beta=beta, weight='weight')
except Exception as e:
    print('Katz/alpha centrality falló:', e)
    try:
        alpha_centrality = nx.katz_centrality(G_enriquecido, alpha=alpha, beta=1.0, max_iter=200, tol=1e-06, weight='weight')
    except Exception as e2:
        print('Fallback Katz iterative falló:', e2)
        alpha_centrality = {n: 0.0 for n in G_enriquecido.nodes()}

# Construir DataFrame con resultados
rows = []
community_map = {}
if isinstance(communities, (list, tuple)):
    for i, comm in enumerate(communities):
        try:
            for n in comm:
                community_map[n] = i
        except Exception:
            pass

for n in G_enriquecido.nodes():
    rows.append({
        'app_id': n,
        'name': G_enriquecido.nodes[n].get('name'),
        'degree': degree.get(n, 0),
        'degree_centrality': degree_centrality.get(n, 0.0),
        'eigenvector': eigen_centrality.get(n, 0.0),
        'alpha_centrality': alpha_centrality.get(n, 0.0),
        'community': community_map.get(n, None)
    })

df_cent = pd.DataFrame(rows)

# Guardar resultados
df_cent = df_cent.sort_values(by=['degree'], ascending=False)
df_cent.to_csv('centralidades_red_enriquecida.csv', index=False)

print('\nTop 10 por grado:')
print(df_cent[['app_id', 'name', 'degree']].head(10).to_string(index=False))

print('\nTop 10 por eigenvector:')
print(df_cent.sort_values(by=['eigenvector'], ascending=False)[['app_id', 'name', 'eigenvector']].head(10).to_string(index=False))

print('\nTop 10 por alpha-centrality (Katz):')
print(df_cent.sort_values(by=['alpha_centrality'], ascending=False)[['app_id', 'name', 'alpha_centrality']].head(10).to_string(index=False))

print('\nCentralidades guardadas en centralidades_red_enriquecida.csv')