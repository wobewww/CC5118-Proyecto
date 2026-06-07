"""Grafica la componente gigante coloreada por distintas métricas de centralidad.

Este script está pensado como complemento de `analizar_distribuciones_centralidades.py`.
Aquí no se hacen histogramas ni paneles de frecuencia: solo se dibuja la componente
conexa gigante (GC) del grafo, coloreando los nodos por una métrica y mostrando
una barra de color, al estilo del notebook `Examples.ipynb`.

Uso:
    python graficar_componente_gigante.py
    python graficar_componente_gigante.py --graph red_enriquecida.pkl.gz --centralities centralidades_red_enriquecida.csv
"""

from __future__ import annotations

import argparse
import gzip
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


DEFAULT_GRAPH = Path("red_enriquecida.pkl.gz")
DEFAULT_CENTRALITIES = Path("centralidades_red_enriquecida.csv")
DEFAULT_OUTPUT_DIR = Path("gc_centralidades")

GRAPH_METRICS = [
    ("degree_centrality", plt.cm.viridis, "Degree Centrality"),
    ("betweenness_centrality", plt.cm.Oranges, "Betweenness Centrality"),
    ("closeness_centrality", plt.cm.Greens, "Closeness Centrality"),
    ("eigenvector", plt.cm.Purples, "Eigenvector Centrality"),
    ("local_clustering_coefficient", plt.cm.Blues, "Local Clustering Coefficient"),
]


def load_graph(graph_path: Path) -> nx.Graph:
    if not graph_path.exists():
        raise FileNotFoundError(f"No existe el grafo: {graph_path}")

    with gzip.open(graph_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "grafo" in data:
        graph = data["grafo"]
    elif isinstance(data, nx.Graph):
        graph = data
    else:
        raise ValueError(f"Formato de grafo no reconocido en {graph_path}")

    if not isinstance(graph, nx.Graph):
        raise ValueError(f"El objeto cargado no es un grafo válido: {graph_path}")

    return graph


def load_centralities(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo de centralidades: {csv_path}")

    df = pd.read_csv(csv_path)
    if "app_id" not in df.columns:
        raise ValueError(f"El CSV {csv_path} no tiene columna app_id")

    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce")
    df = df.dropna(subset=["app_id"]).copy()
    df["app_id"] = df["app_id"].astype(int)

    for metric, _, _ in GRAPH_METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df


def get_giant_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        raise ValueError("El grafo está vacío")

    if nx.is_connected(graph):
        return graph.copy()

    components = list(nx.connected_components(graph))
    giant_nodes = max(components, key=len)
    return graph.subgraph(giant_nodes).copy()


def build_layout(graph: nx.Graph) -> dict[int, np.ndarray]:
    # Layout determinista para comparar gráficos entre métricas.
    return nx.spring_layout(graph, seed=42, k=None, iterations=60)


def plot_metric_gc(
    graph: nx.Graph,
    gc: nx.Graph,
    metric_df: pd.DataFrame,
    metric: str,
    cmap,
    title: str,
    pos: dict[int, np.ndarray],
    output_dir: Path,
) -> None:
    metric_map = metric_df.set_index("app_id")[metric].dropna()
    metric_map.index = metric_map.index.astype(int)

    nodes = [node for node in gc.nodes if node in metric_map.index]
    if not nodes:
        return

    values = metric_map.loc[nodes].astype(float)
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    norm = Normalize(vmin=vmin, vmax=vmax)
    node_colors = [cmap(norm(float(metric_map.loc[node]))) for node in gc.nodes if node in metric_map.index]

    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white")
    nx.draw_networkx_edges(gc, pos, ax=ax, alpha=0.14, edge_color="#333333", width=0.6)
    nx.draw_networkx_nodes(
        gc,
        pos,
        ax=ax,
        nodelist=nodes,
        node_color=node_colors,
        node_size=26,
        edgecolors="white",
        linewidths=0.15,
    )

    # Etiquetas solo para el top 20, para no saturar la figura.
    top_nodes = values.sort_values(ascending=False).head(20).index.tolist()
    labels = {node: str(node) for node in top_nodes if node in gc.nodes}
    if labels:
        nx.draw_networkx_labels(gc, pos, labels=labels, ax=ax, font_size=7, font_color="black")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(title)

    ax.set_title(f"Componente gigante coloreada por {title}", fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"gc_{metric}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grafica la componente gigante por métricas de centralidad")
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH), help="Ruta al pickle del grafo")
    parser.add_argument("--centralities", default=str(DEFAULT_CENTRALITIES), help="CSV con las centralidades")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de salida")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    centralities_path = Path(args.centralities)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(graph_path)
    gc = get_giant_component(graph)
    metric_df = load_centralities(centralities_path)
    pos = build_layout(gc)

    print(f"Grafo cargado: {graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas")
    print(f"Componente gigante: {gc.number_of_nodes()} nodos, {gc.number_of_edges()} aristas")

    for metric, cmap, title in GRAPH_METRICS:
        if metric not in metric_df.columns:
            print(f"Saltando {metric}: no existe en {centralities_path}")
            continue
        plot_metric_gc(graph, gc, metric_df, metric, cmap, title, pos, output_dir)
        print(f"Generado: {output_dir / f'gc_{metric}.png'}")


if __name__ == "__main__":
    main()
