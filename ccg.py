import gzip
import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Cargar grafo
with gzip.open("RED-ENRIQUECIDA_100K.pkl.gz", "rb") as f:
    data = pickle.load(f)
G = data["grafo"]

# Recoger y ordenar aristas
edges = list(G.edges(data=True))
edges.sort(key=lambda x: x[2].get('weight', float('inf')), reverse=False)

# Calcular cantidades de nodos
N_total_nodos = G.number_of_nodes()
N_total_aristas = len(edges)

# Simular la evolución de la CCG
G_evolucion = nx.Graph()
G_evolucion.add_nodes_from(G.nodes())

p_valores = []
cgg_proporciones = []

step_size = max(1, N_total_aristas // 50) 

for i in range(0, N_total_aristas, step_size):
    # Añadir el bloque de aristas
    batch = edges[i:i+step_size]
    G_evolucion.add_edges_from(batch)
    
    # Calcular la fracción de aristas añadida
    p = G_evolucion.number_of_edges() / N_total_aristas
    
    # Calcular el tamaño de la cgg en este paso
    if G_evolucion.number_of_edges() > 0:
        components = nx.connected_components(G_evolucion)
        largest_component = max(components, key=len)
        size_cgg = len(largest_component)
    else:
        size_cgg = 0
        
    proporcion_cgg = size_cgg / N_total_nodos
    
    p_valores.append(p)
    cgg_proporciones.append(proporcion_cgg)

# ==========================================
# GUARDAR Y GRAFICAR RESULTADOS MODIFICADO
# ==========================================

df_evolucion = pd.DataFrame({
    'fraccion_aristas': p_valores,
    'tamano_relativo_ccg': cgg_proporciones
})

# Aumentamos el tamaño de la figura para mejorar la visibilidad en presentaciones
plt.figure(figsize=(10, 6))

# 1. Graficar la curva principal de la CCG
plt.plot(df_evolucion['fraccion_aristas'], df_evolucion['tamano_relativo_ccg'], 
         marker='o', color='#1f77b4', linewidth=2.5, markersize=5, zorder=3, label='Progreso de la CCG')

# 2. Definir los tramos solicitados con colores pasteles para el fondo
# Puedes cambiar los textos conceptuales según los hallazgos de tu proyecto
tramos = [
    {"rango": (0, 0.15), "color": "#f8d7da", "texto": "Tramo 1\n$[0, 0.15]$\nFase Inicial"},
    {"rango": (0.15, 0.34), "color": "#d4edda", "texto": "Tramo 2\n$(0.15, 0.34]$\nTransición Crítica"},
    {"rango": (0.34, 1.0), "color": "#cce5ff", "texto": "Tramo 3\n$(0.34, 1.0]$\nSaturación / Conectividad"}
]

# 3. Aplicar el sombreado y las etiquetas para cada tramo
for tramo in tramos:
    inicio, fin = tramo["rango"]
    
    # Sombreado de fondo
    plt.axvspan(inicio, fin, color=tramo["color"], alpha=0.6, zorder=1)
    
    # Línea vertical divisoria (excepto al final en p = 1)
    if fin < 1.0:
        plt.axvline(x=fin, color='#6c757d', linestyle='--', linewidth=1.8, zorder=2)
    
    # Colocar texto explicativo en el centro de cada región
    x_centro = (inicio + fin) / 2
    # El texto se posiciona verticalmente en y = 0.5 (mitad del gráfico)
    plt.text(x_centro, 0.5, tramo["texto"], fontsize=11, fontweight='bold',
             color='#333333', horizontalalignment='center', verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='none'))

# 4. Configuración de títulos y etiquetas con fuentes grandes (ideales para diapositivas)
plt.title('Evolución de la Componente Conexa Gigante (CCG)', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Fracción de Aristas Añadidas ($p$)', fontsize=13, labelpad=10)
plt.ylabel('Tamaño Relativo de la CCG ($S$)', fontsize=13, labelpad=10)

# 5. Ajustes finales de los ejes y malla
plt.xlim(0, 1)
plt.ylim(-0.02, 1.02)
plt.xticks([0, 0.15, 0.34, 0.5, 0.75, 1.0], fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.5, zorder=0)

# Mostrar leyenda
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.show()