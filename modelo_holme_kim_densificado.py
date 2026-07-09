"""[DESCARTADO] Modelo Holme-Kim densificado.

>>> ESTE MODELO FUE DESCARTADO. El modelo oficial es modelo_cliques.py.
>>> Se descarta porque infla el numero de aristas ~+11% (E=61k vs 55k real,
>>> <k>=37.5 vs 33.8 real). El de crecimiento por cliques logra el mismo
>>> clustering sin agregar aristas de mas. Se conserva solo como referencia.

Modelo generativo: Holme-Kim densificado.

El Holme-Kim clasico (nx.powerlaw_cluster_graph) solo agrega aristas
incidentes al nodo nuevo, por lo que su coeficiente de clustering se satura
en torno a 0.25 y no logra reproducir el C~0.61 de la red real.

Esta modificacion agrega un mecanismo de "densificacion local": cuando el
nodo nuevo se conecta a sus m vecinos por apego preferencial, ademas se
enlazan entre si pares de esos vecinos (con probabilidad q). Esto imita el
proceso generador real de la red de co-recomendacion: un usuario que
recomienda varios juegos conecta esos juegos preexistentes entre si,
formando cliques que son la verdadera fuente del clustering alto.

Parametros calibrados contra la componente gigante real:
  m = 7   (aristas por apego preferencial; se baja respecto a E/N=17 para
           compensar las aristas extra de la densificacion y mantener <k>~34)
  q = 0.80 (probabilidad de enlazar cada par de vecinos del nodo nuevo)
"""

import gzip
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

RUTA_GRAFO = "RED-ENRIQUECIDA_100K.pkl.gz"
SALIDA_TXT = "resumen_modelo_holme_kim_densificado.txt"
SALIDA_PNG = "comparacion_degree_holme_kim_densificado.png"
M = 7
Q = 0.80
SEED = 42


def holme_kim_densificado(n, m, q, seed=None):
    """Holme-Kim con densificacion local entre los vecinos del nodo nuevo."""
    rng = random.Random(seed)
    G = nx.empty_graph(m)
    repeated = list(range(m))  # lista para muestreo por apego preferencial
    for source in range(m, n):
        # 1) apego preferencial: elegir m vecinos distintos
        targets = set()
        while len(targets) < m:
            targets.add(rng.choice(repeated))
        targets = list(targets)
        for t in targets:
            G.add_edge(source, t)
        repeated.extend(targets)
        repeated.extend([source] * m)
        # 2) densificacion: enlazar pares de vecinos del nodo nuevo
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                if rng.random() < q and not G.has_edge(targets[i], targets[j]):
                    G.add_edge(targets[i], targets[j])
                    repeated.append(targets[i])
                    repeated.append(targets[j])
    return G


def cargar_componente_gigante(ruta):
    with gzip.open(ruta, "rb") as f:
        data = pickle.load(f)
    G = data["grafo"]
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G


def componente_gigante(G):
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def caracterizar(G, nombre):
    G_ccg = componente_gigante(G)
    grados = [d for _, d in G.degree()]
    stats = {
        "nombre": nombre,
        "N": G.number_of_nodes(),
        "E": G.number_of_edges(),
        "k_prom": float(np.mean(grados)),
        "C": nx.average_clustering(G),
        "L": nx.average_shortest_path_length(G_ccg),
        "grado_max": max(grados),
        "grado_mediana": float(np.median(grados)),
        "grados": grados,
    }
    print(f"--- {nombre} ---")
    print(f"N={stats['N']}  E={stats['E']}  <k>={stats['k_prom']:.2f}")
    print(f"C={stats['C']:.6f}  L={stats['L']:.4f}")
    print(f"grado max={stats['grado_max']}  mediana={stats['grado_mediana']:.1f}")
    print()
    return stats


def main():
    print("Cargando componente gigante de la red real...")
    G_real = cargar_componente_gigante(RUTA_GRAFO)
    stats_real = caracterizar(G_real, "Red real (CCG)")

    N = stats_real["N"]
    print(f"Generando modelo Holme-Kim densificado (N={N}, m={M}, q={Q})...")
    G_mod = holme_kim_densificado(N, M, Q, seed=SEED)
    stats_mod = caracterizar(G_mod, f"Holme-Kim densificado (m={M}, q={Q})")

    lineas = [
        "Comparacion Red real (CCG) vs Modelo Holme-Kim densificado\n",
        f"{'Metrica':<15}{'Red real':>15}{'HK densif.':>15}\n",
        f"{'N':<15}{stats_real['N']:>15}{stats_mod['N']:>15}\n",
        f"{'E':<15}{stats_real['E']:>15}{stats_mod['E']:>15}\n",
        f"{'<k>':<15}{stats_real['k_prom']:>15.2f}{stats_mod['k_prom']:>15.2f}\n",
        f"{'C':<15}{stats_real['C']:>15.6f}{stats_mod['C']:>15.6f}\n",
        f"{'L':<15}{stats_real['L']:>15.4f}{stats_mod['L']:>15.4f}\n",
        f"{'grado max':<15}{stats_real['grado_max']:>15}{stats_mod['grado_max']:>15}\n",
        f"{'grado mediana':<15}{stats_real['grado_mediana']:>15.1f}{stats_mod['grado_mediana']:>15.1f}\n",
        f"\nParametros: m={M}, q={Q}, seed={SEED}\n",
    ]
    with open(SALIDA_TXT, "w", encoding="utf-8") as f:
        f.writelines(lineas)
    print("".join(lineas))
    print(f"Resumen guardado en {SALIDA_TXT}")

    plt.figure(figsize=(7, 5))
    for stats, color in [(stats_real, "tab:blue"), (stats_mod, "tab:orange")]:
        grados = np.array(stats["grados"])
        valores, conteos = np.unique(grados, return_counts=True)
        plt.scatter(valores, conteos, label=stats["nombre"], color=color, s=15)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("grado")
    plt.ylabel("frecuencia")
    plt.title("Distribucion de grado: red real vs Holme-Kim densificado")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SALIDA_PNG, dpi=150)
    print(f"Grafico guardado en {SALIDA_PNG}")


if __name__ == "__main__":
    main()
