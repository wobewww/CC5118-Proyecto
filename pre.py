import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# =========================
# 1. CARGA Y LIMPIEZA
# =========================
# Asumimos que el archivo existe según tu snippet
try:
    df = pd.read_csv("testMORERE.csv", header=2)
    df = df.iloc[:, 2:]
    df.columns = df.columns.str.strip()
    df["is_recommended"] = df["is_recommended"].astype(str).str.strip().str.upper()
    df = df[df["is_recommended"] == "TRUE"]
except FileNotFoundError:
    print("Error: No se encontró el archivo 'testMORERE.csv'")
    # Crear datos sintéticos solo para que el código sea ejecutable si no tienes el CSV a mano
    df = pd.DataFrame({'user_id': [1,1,1,2,2,3,3,4,4], 'app_id': ['A','B','C','A','B','B','C','D','A']})

# =========================
# 2. CONSTRUIR GRAFO
# =========================
grouped = df.groupby("user_id")["app_id"].apply(list)
G = nx.Graph()

for games in grouped:
    # Filtro para evitar ruido y grafos sobredimensionados
    if 1 < len(games) <= 20:
        for g1, g2 in combinations(games, 2):
            if G.has_edge(g1, g2):
                G[g1][g2]["weight"] += 1
            else:
                G.add_edge(g1, g2, weight=1)

# =========================
# 3. ANÁLISIS (COMUNIDADES Y MÉTRICAS)
# =========================
# Detectar comunidades para colorear
communities = list(greedy_modularity_communities(G))
color_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        color_map[node] = i

node_colors = [color_map[n] for n in G.nodes()]

# =========================
# 4. VISUALIZACIÓN OPTIMIZADA
# =========================
plt.figure(figsize=(14, 10), facecolor='#f0f0f0')

# Ajuste de Layout: 
# k: aumenta para separar nodos encimados, disminuye si están muy lejos.
# iterations: más iteraciones permiten que el sistema de "resortes" se estabilice.
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Dibujar aristas con ancho basado en el peso (weight)
weights = [G[u][v]['weight'] for u, v in G.edges()]
# Normalizamos los pesos para que no sean excesivamente gruesos
max_w = max(weights) if weights else 1
edge_widths = [(w / max_w) * 3 for w in weights]

nx.draw_networkx_edges(
    G, pos, 
    alpha=0.3, 
    edge_color="gray", 
    width=edge_widths
)

# Dibujar nodos coloreados por comunidad
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_size=500,
    node_color=node_colors,
    cmap=plt.cm.Set3,  # Paleta de colores suave
    edgecolors="white",
    linewidths=1
)

# Dibujar etiquetas con fondo para legibilidad
labels = nx.draw_networkx_labels(
    G, pos,
    font_size=9,
    font_family="sans-serif",
    font_weight="bold"
)

# Añadir un borde a las etiquetas (opcional, mejora lectura sobre aristas)
for t in labels.values():
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

plt.title("Red de Co-recomendaciones (Steam)", fontsize=15)
plt.axis("off")
plt.tight_layout()
plt.show()

# Resumen rápido en consola
print(f"Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
print(f"Comunidades detectadas: {len(communities)}")