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

# Guardar y graficar resultados
df_evolucion = pd.DataFrame({
    'fraccion_aristas': p_valores,
    'tamano_relativo_ccg': cgg_proporciones
})

plt.figure(figsize=(8, 5))
plt.plot(df_evolucion['fraccion_aristas'], df_evolucion['tamano_relativo_ccg'], marker='o', color='b')
plt.title('Evolución de la Componente Conexa Gigante (CCG)')
plt.xlabel('Fracción de Aristas Añadidas ($p$)')
plt.ylabel('Tamaño Relativo de la CCG ($S$)')
plt.grid(True)
plt.show()