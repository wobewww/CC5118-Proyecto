"""Modelo generativo propuesto (OFICIAL): crecimiento por cliques con apego
preferencial.

Motivacion: la red real es la proyeccion de un grafo bipartito usuario-juego.
Cada usuario que recomienda positivamente un conjunto de juegos conecta a todos
esos juegos entre si, formando una clique. Este modelo reproduce ese mecanismo
directamente: cada nodo nuevo entra formando una clique con s-1 nodos ya
existentes (elegidos por apego preferencial).

A diferencia del Holme-Kim clasico -- que solo cierra un triangulo por arista y
se satura en C~0.25 -- y de la variante densificada -- que infla el numero de
aristas (~+11%) -- este modelo reproduce el clustering alto de la red real SIN
agregar aristas de mas, porque cada arista que crea participa en el maximo
numero de triangulos posible (es la forma mas eficiente de generar clustering).

Parametro calibrado contra la componente gigante real:
  s = 7  (tamano de la clique; da <k>~33 y E~54k, casi identicos a la red real)
"""

import gzip
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

RUTA_GRAFO = "RED-ENRIQUECIDA_100K.pkl.gz"
SALIDA_TXT = "resumen_modelo_cliques.txt"
SALIDA_PNG = "comparacion_degree_cliques.png"
SEED = 42


def crecimiento_por_cliques(n, s, seed=None):
    """Cada nodo nuevo forma una clique de tamano s con s-1 nodos existentes
    elegidos por apego preferencial."""
    rng = random.Random(seed)
    # clique inicial de s nodos
    G = nx.empty_graph(s)
    for u in range(s):
        for v in range(u + 1, s):
            G.add_edge(u, v)
    # lista para muestreo por apego preferencial (cada nodo aparece = su grado)
    repeated = [x for x in range(s) for _ in range(s - 1)]
    for source in range(s, n):
        elegidos = set()
        while len(elegidos) < s - 1:
            elegidos.add(rng.choice(repeated))
        grupo = list(elegidos) + [source]
        # conectar a todos con todos (clique)
        for i in range(len(grupo)):
            for j in range(i + 1, len(grupo)):
                if not G.has_edge(grupo[i], grupo[j]):
                    G.add_edge(grupo[i], grupo[j])
                    repeated.append(grupo[i])
                    repeated.append(grupo[j])
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

def calibrar_parametro_s(N_real, E_real):
    mejor_s = None
    menor_error = float("inf")

    # se prueba un rango de valores razonables para s
    for s_test in range(2, 10):
        # generamos la red
        G_test = crecimiento_por_cliques(N_real, s_test, seed=SEED)
        E_test = G_test.number_of_edges()

        # calculamos el error
        error = abs(E_test - E_real) / E_real

        if error < menor_error:
            menor_error = error
            mejor_s = s_test

    return mejor_s, menor_error

def main():
    print("Cargando componente gigante de la red real...")
    G_real = cargar_componente_gigante(RUTA_GRAFO)
    stats_real = caracterizar(G_real, "Red real (CCG)")

    N = stats_real["N"]
    E = stats_real["E"]

    print("Obteniendo valor óptimo de S")
    S, error = calibrar_parametro_s(N, E)
    print(f"El valor óptimo es s={S} (Error en aristas: {error*100:.2f}%)")

    print(f"Generando modelo de crecimiento por cliques (N={N}, s={S})...")
    G_mod = crecimiento_por_cliques(N, S, seed=SEED)
    stats_mod = caracterizar(G_mod, f"Crecimiento por cliques (s={S})")

    lineas = [
        "Comparacion Red real (CCG) vs Modelo de crecimiento por cliques\n",
        f"{'Metrica':<15}{'Red real':>15}{'Cliques':>15}\n",
        f"{'N':<15}{stats_real['N']:>15}{stats_mod['N']:>15}\n",
        f"{'E':<15}{stats_real['E']:>15}{stats_mod['E']:>15}\n",
        f"{'<k>':<15}{stats_real['k_prom']:>15.2f}{stats_mod['k_prom']:>15.2f}\n",
        f"{'C':<15}{stats_real['C']:>15.6f}{stats_mod['C']:>15.6f}\n",
        f"{'L':<15}{stats_real['L']:>15.4f}{stats_mod['L']:>15.4f}\n",
        f"{'grado max':<15}{stats_real['grado_max']:>15}{stats_mod['grado_max']:>15}\n",
        f"{'grado mediana':<15}{stats_real['grado_mediana']:>15.1f}{stats_mod['grado_mediana']:>15.1f}\n",
        f"\nParametro: s={S}, seed={SEED}\n",
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
    plt.title("Distribucion de grado: red real vs crecimiento por cliques")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SALIDA_PNG, dpi=150)
    print(f"Grafico guardado en {SALIDA_PNG}")


if __name__ == "__main__":
    main()
