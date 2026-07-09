"""Calcula las centralidades de los modelos generativos propuestos, en el
mismo formato de CSV que centralidades.py, para poder reutilizar el script
analizar_distribuciones_centralidades.py sobre el modelo.

Los grafos del modelo son NO ponderados (a diferencia de la red real), por lo
que las metricas se calculan sin argumento de peso.

Uso:
    python centralidades_modelo.py --modelo densificado
    python centralidades_modelo.py --modelo cliques
"""

import argparse
import random

import networkx as nx
import pandas as pd

N = 3272
SEED = 42


def holme_kim_densificado(n, m, q, seed=None):
    rng = random.Random(seed)
    G = nx.empty_graph(m)
    repeated = list(range(m))
    for source in range(m, n):
        targets = set()
        while len(targets) < m:
            targets.add(rng.choice(repeated))
        targets = list(targets)
        for t in targets:
            G.add_edge(source, t)
        repeated.extend(targets)
        repeated.extend([source] * m)
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                if rng.random() < q and not G.has_edge(targets[i], targets[j]):
                    G.add_edge(targets[i], targets[j])
                    repeated.append(targets[i])
                    repeated.append(targets[j])
    return G


def holme_kim_cliques(n, s, seed=None):
    rng = random.Random(seed)
    G = nx.empty_graph(s)
    for u in range(s):
        for v in range(u + 1, s):
            G.add_edge(u, v)
    repeated = [x for x in range(s) for _ in range(s - 1)]
    for source in range(s, n):
        elegidos = set()
        while len(elegidos) < s - 1:
            elegidos.add(rng.choice(repeated))
        grupo = list(elegidos) + [source]
        for i in range(len(grupo)):
            for j in range(i + 1, len(grupo)):
                if not G.has_edge(grupo[i], grupo[j]):
                    G.add_edge(grupo[i], grupo[j])
                    repeated.append(grupo[i])
                    repeated.append(grupo[j])
    return G


def calcular_centralidades(G, salida_csv):
    print("1) Grado y centralidad de grado")
    degree = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)

    print("2) Betweenness centrality (no ponderado)")
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

    print("3) Closeness centrality (no ponderado)")
    closeness_centrality = nx.closeness_centrality(G, wf_improved=True)

    print("4) Eigenvector centrality")
    try:
        eigen_centrality = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eigen_centrality = nx.eigenvector_centrality(G, max_iter=500, tol=1e-06)

    print("5) Local clustering coefficient")
    local_clustering = nx.clustering(G)

    print("6) Global clustering coefficient (transitivity)")
    global_clustering = nx.transitivity(G)

    rows = []
    for node in G.nodes():
        rows.append(
            {
                "app_id": node,
                "name": f"sim_{node}",
                "degree": degree.get(node, 0),
                "degree_centrality": degree_centrality.get(node, 0.0),
                "betweenness_centrality": betweenness_centrality.get(node, 0.0),
                "closeness_centrality": closeness_centrality.get(node, 0.0),
                "eigenvector": eigen_centrality.get(node, 0.0),
                "local_clustering_coefficient": local_clustering.get(node, 0.0),
                "global_clustering_coefficient": global_clustering,
                "community": None,
            }
        )

    df = pd.DataFrame(rows).sort_values(by=["degree"], ascending=False)
    df.to_csv(salida_csv, index=False)
    print(f"\nGlobal clustering (transitivity): {global_clustering:.6f}")
    print(f"Centralidades del modelo guardadas en {salida_csv}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=["densificado", "cliques"], default="cliques")
    args = parser.parse_args()

    if args.modelo == "densificado":
        print("Generando modelo Holme-Kim densificado (m=7, q=0.8)...")
        G = holme_kim_densificado(N, 7, 0.80, seed=SEED)
        salida = "centralidades_modelo_densificado.csv"
    else:
        print("Generando modelo de crecimiento por cliques (s=7)...")
        G = holme_kim_cliques(N, 7, seed=SEED)
        salida = "centralidades_modelo_cliques.csv"

    print(f"Grafo generado: N={G.number_of_nodes()} E={G.number_of_edges()}")
    calcular_centralidades(G, salida)


if __name__ == "__main__":
    main()
