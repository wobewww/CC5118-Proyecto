import gzip
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

RUTA_GRAFO = "RED-ENRIQUECIDA_100K.pkl.gz"
SALIDA_TXT = "resumen_modelo_holme_kim.txt"
SALIDA_PNG = "comparacion_degree_holme_kim.png"
SEED = 42


def cargar_componente_gigante(ruta):
    with gzip.open(ruta, "rb") as f:
        data = pickle.load(f)
    G = data["grafo"]
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G


def caracterizar(G, nombre):
    grados = [d for _, d in G.degree()]
    stats = {
        "nombre": nombre,
        "N": G.number_of_nodes(),
        "E": G.number_of_edges(),
        "k_prom": float(np.mean(grados)),
        "C": nx.average_clustering(G),
        "L": nx.average_shortest_path_length(G),
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


def calibrar_p(N, m, C_objetivo, p_grid=None, seed=SEED):
    if p_grid is None:
        p_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    mejor_p, mejor_diff, mejor_C = None, np.inf, None
    print(f"Calibrando p (N={N}, m={m}, C objetivo={C_objetivo:.4f})...")
    for p in p_grid:
        G_sint = nx.powerlaw_cluster_graph(N, m, p, seed=seed)
        C_sint = nx.average_clustering(G_sint)
        diff = abs(C_sint - C_objetivo)
        print(f"  p={p:.2f} -> C={C_sint:.4f}")
        if diff < mejor_diff:
            mejor_p, mejor_diff, mejor_C = p, diff, C_sint
    print(f"Mejor p encontrado: {mejor_p} (C={mejor_C:.4f})\n")
    return mejor_p, mejor_C


def main():
    print("Cargando componente gigante de la red real...")
    G_real = cargar_componente_gigante(RUTA_GRAFO)
    stats_real = caracterizar(G_real, "Red real (CCG)")

    N = stats_real["N"]
    E = stats_real["E"]
    m = max(1, round(E / N))

    p_opt, _ = calibrar_p(N, m, stats_real["C"])

    print("Generando modelo Holme-Kim final...")
    G_hk = nx.powerlaw_cluster_graph(N, m, p_opt, seed=SEED)
    stats_hk = caracterizar(G_hk, f"Holme-Kim (N={N}, m={m}, p={p_opt})")

    lineas = [
        "Comparacion Red real (CCG) vs Modelo Holme-Kim\n",
        f"{'Metrica':<15}{'Red real':>15}{'Holme-Kim':>15}\n",
        f"{'N':<15}{stats_real['N']:>15}{stats_hk['N']:>15}\n",
        f"{'E':<15}{stats_real['E']:>15}{stats_hk['E']:>15}\n",
        f"{'<k>':<15}{stats_real['k_prom']:>15.2f}{stats_hk['k_prom']:>15.2f}\n",
        f"{'C':<15}{stats_real['C']:>15.6f}{stats_hk['C']:>15.6f}\n",
        f"{'L':<15}{stats_real['L']:>15.4f}{stats_hk['L']:>15.4f}\n",
        f"{'grado max':<15}{stats_real['grado_max']:>15}{stats_hk['grado_max']:>15}\n",
        f"{'grado mediana':<15}{stats_real['grado_mediana']:>15.1f}{stats_hk['grado_mediana']:>15.1f}\n",
        f"\nParametros Holme-Kim: m={m}, p={p_opt}\n",
    ]
    with open(SALIDA_TXT, "w", encoding="utf-8") as f:
        f.writelines(lineas)
    print("".join(lineas))
    print(f"Resumen guardado en {SALIDA_TXT}")

    plt.figure(figsize=(7, 5))
    for stats, color in [(stats_real, "tab:blue"), (stats_hk, "tab:orange")]:
        grados = np.array(stats["grados"])
        valores, conteos = np.unique(grados, return_counts=True)
        plt.scatter(valores, conteos, label=stats["nombre"], color=color, s=15)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("grado")
    plt.ylabel("frecuencia")
    plt.title("Distribucion de grado: red real vs Holme-Kim")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SALIDA_PNG, dpi=150)
    print(f"Grafico guardado en {SALIDA_PNG}")


if __name__ == "__main__":
    main()
