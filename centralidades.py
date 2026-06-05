# ---- CÁLCULO DE CENTRALIDADES ----
print('\nCalculando centralidades...')

print("1) Grado y centralidad de grado")
# 1) Grado y centralidad de grado
degree = dict(G_enriquecido.degree())
degree_centrality = nx.degree_centrality(G_enriquecido)



# 2) Betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G_enriquecido, normalized=True, weight='weight')

print("3) Closeness centrality")
# 3) Closeness centrality
closeness_centrality = nx.closeness_centrality(G_enriquecido, distance=None, wf_improved=True)

print("4) Eigenvector centrality")
# 4) Eigenvector centrality (intenta usar la versión basada en numpy)
try:
    eigen_centrality = nx.eigenvector_centrality_numpy(G_enriquecido, weight='weight')
except Exception:
    try:
        eigen_centrality = nx.eigenvector_centrality(G_enriquecido, max_iter=200, tol=1e-06, weight='weight')
    except Exception as e:
        print('No se pudo calcular eigenvector centrality:', e)
        eigen_centrality = {n: 0.0 for n in G_enriquecido.nodes()}

print("5) Local clustering coefficient")
# 5) Local clustering coefficient
local_clustering = nx.clustering(G_enriquecido, weight='weight')

print("6) Global clustering coefficient (transitivity)")
# 6) Global clustering coefficient (transitivity)
global_clustering = nx.transitivity(G_enriquecido)

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
        'betweenness_centrality': betweenness_centrality.get(n, 0.0),
        'closeness_centrality': closeness_centrality.get(n, 0.0),
        'eigenvector': eigen_centrality.get(n, 0.0),
        'local_clustering_coefficient': local_clustering.get(n, 0.0),
        'global_clustering_coefficient': global_clustering,
        'community': community_map.get(n, None)
    })

df_cent = pd.DataFrame(rows)

# Guardar resultados
df_cent = df_cent.sort_values(by=['degree'], ascending=False)
df_cent.to_csv('centralidades_red_enriquecida.csv', index=False)

resumen = [
    'Resumen de centralidades para red_enriquecida.pkl.gz',
    f'Global clustering coefficient (transitivity): {global_clustering:.6f}',
    '',
    'Top 10 por grado:',
    df_cent[['app_id', 'name', 'degree']].head(10).to_string(index=False),
    '',
    'Top 10 por degree centrality:',
    df_cent.sort_values(by=['degree_centrality'], ascending=False)[['app_id', 'name', 'degree_centrality']].head(10).to_string(index=False),
    '',
    'Top 10 por betweenness centrality:',
    df_cent.sort_values(by=['betweenness_centrality'], ascending=False)[['app_id', 'name', 'betweenness_centrality']].head(10).to_string(index=False),
    '',
    'Top 10 por closeness centrality:',
    df_cent.sort_values(by=['closeness_centrality'], ascending=False)[['app_id', 'name', 'closeness_centrality']].head(10).to_string(index=False),
    '',
    'Top 10 por eigenvector centrality:',
    df_cent.sort_values(by=['eigenvector'], ascending=False)[['app_id', 'name', 'eigenvector']].head(10).to_string(index=False),
    '',
    'Top 10 por local clustering coefficient:',
    df_cent.sort_values(by=['local_clustering_coefficient'], ascending=False)[['app_id', 'name', 'local_clustering_coefficient']].head(10).to_string(index=False),
]

with open('resumen_centralidades_red_enriquecida.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(resumen))

print(f'Global clustering coefficient (transitivity): {global_clustering:.6f}')
print('\nTop 10 por grado:')
print(df_cent[['app_id', 'name', 'degree']].head(10).to_string(index=False))

print('\nTop 10 por degree centrality:')
print(df_cent.sort_values(by=['degree_centrality'], ascending=False)[['app_id', 'name', 'degree_centrality']].head(10).to_string(index=False))

print('\nTop 10 por betweenness centrality:')
print(df_cent.sort_values(by=['betweenness_centrality'], ascending=False)[['app_id', 'name', 'betweenness_centrality']].head(10).to_string(index=False))


print('\nTop 10 por closeness centrality:')
print(df_cent.sort_values(by=['closeness_centrality'], ascending=False)[['app_id', 'name', 'closeness_centrality']].head(10).to_string(index=False))

print('\nTop 10 por eigenvector centrality:')
print(df_cent.sort_values(by=['eigenvector'], ascending=False)[['app_id', 'name', 'eigenvector']].head(10).to_string(index=False))

print('\nTop 10 por local clustering coefficient:')
print(df_cent.sort_values(by=['local_clustering_coefficient'], ascending=False)[['app_id', 'name', 'local_clustering_coefficient']].head(10).to_string(index=False))

print('\nCentralidades guardadas en centralidades_red_enriquecida.csv')
print('Resumen breve guardado en resumen_centralidades_red_enriquecida.txt')