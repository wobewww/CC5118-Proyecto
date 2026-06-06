"""Mostrar métricas y atributos del nodo CS:GO (appid 730).

Uso:
    python show_csgo_metrics.py [--appid 730]

Genera una salida por consola con:
 - fila de `centralidades_red_enriquecida.csv` si existe
 - atributos del nodo en `red_enriquecida.pkl.gz` si existe
 - primeros 10 vecinos con peso/strength
"""
import argparse
import gzip
import pickle
import os
import pandas as pd


def find_node_key(G, appid):
    s = str(appid)
    if appid in G:
        return appid
    if s in G:
        return s
    for n in G.nodes():
        try:
            if str(n) == s:
                return n
        except Exception:
            continue
    return None


def load_graph(path):
    if not os.path.exists(path):
        return None
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'grafo' in data:
        return data['grafo']
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--appid', type=int, default=730, help='App id a mostrar')
    args = p.parse_args()
    appid = args.appid

    print(f"Mostrando métricas para appid {appid}\n")

    # 1) centralidades CSV
    csv_path = 'centralidades_red_enriquecida.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, dtype=str, low_memory=False)
            keycols = [c for c in ['app_id','appid','appId'] if c in df.columns]
            if keycols:
                mask = False
                for col in keycols:
                    mask = mask | (df[col].astype(str) == str(appid))
                rows = df[mask]
            else:
                rows = df[df.index == -1]
            if not rows.empty:
                print('Fila en centralidades:')
                print(rows.T)
            else:
                print(f'{csv_path}: no contiene appid {appid}')
        except Exception as e:
            print(f'Error leyendo {csv_path}: {e}')
    else:
        print(f'{csv_path}: no encontrado')

    # 2) grafo enriquecido
    gpath = 'red_enriquecida.pkl.gz'
    G = load_graph(gpath)
    if G is None:
        print(f'\n{gpath}: no encontrado o no contiene grafo')
        return
    node_key = find_node_key(G, appid)
    if node_key is None:
        print(f'\nNodo {appid} no encontrado en {gpath}')
        return

    print(f'\nAtributos del nodo en {gpath}:')
    try:
        attrs = G.nodes.get(node_key)
        print(attrs)
    except Exception as e:
        print(f'Error obteniendo atributos: {e}')

    # 3) vecinos
    try:
        nbrs = []
        for nbr, edata in G[node_key].items():
            w = edata.get('weight')
            s = edata.get('strength')
            nbrs.append((nbr, w, s))
        nbrs_sorted = sorted(nbrs, key=lambda x: (float('inf') if x[1] is None else float(x[1])), reverse=False)
        print('\nPrimeros 10 vecinos (node, weight, strength):')
        for n,w,s in nbrs_sorted[:10]:
            print(f' - {n}: weight={w}, strength={s}')
    except Exception as e:
        print(f'Error listando vecinos: {e}')


if __name__ == '__main__':
    main()
