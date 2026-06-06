"""Comprobador rápido de presencia de Counter-Strike (app id 730) en artefactos del proyecto.

Ejemplo de uso:
    python check_csgo.py

Busca en:
 - red_construida100K.pkl.gz, red_construida.pkl.gz, red_construida1MI.pkl.gz
 - red_enriquecida.pkl.gz
 - steam_games.csv
 - centralidades_red_enriquecida.csv
 - recommendations_sorted.parquet (si existe)
"""
import os
import gzip
import pickle
import pandas as pd


def node_in_graph(G, appid):
    s = str(appid)
    # Fast check: direct membership
    if appid in G:
        return True
    if s in G:
        return True
    # Fallback: compare stringified node ids
    for n in G.nodes():
        try:
            if str(n) == s:
                return True
        except Exception:
            continue
    return False


def check_pickle_graph(path, appid):
    if not os.path.exists(path):
        print(f"  - {path}: not found")
        return
    try:
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  - {path}: error loading ({e})")
        return
    G = data.get('grafo') if isinstance(data, dict) else data
    if G is None:
        print(f"  - {path}: no 'grafo' found in pickle")
        return
    present = node_in_graph(G, appid)
    print(f"  - {path}: {'PRESENT' if present else 'MISSING'}")
    if present:
        try:
            attrs = G.nodes.get(appid) or G.nodes.get(str(appid))
            if attrs:
                print(f"      node attrs: {attrs}")
        except Exception:
            pass


def check_csv(path, appid, colnames=('appid','app_id')):
    if not os.path.exists(path):
        print(f"  - {path}: not found")
        return
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception as e:
        print(f"  - {path}: error reading ({e})")
        return
    s = str(appid)
    found = False
    for col in colnames:
        if col in df.columns:
            if (df[col].astype(str) == s).any():
                print(f"  - {path}: PRESENT in column '{col}'")
                found = True
    if not found:
        print(f"  - {path}: MISSING")


def check_parquet(path, appid, colname='app_id'):
    if not os.path.exists(path):
        print(f"  - {path}: not found")
        return
    try:
        # Try to read only the column if possible
        df = pd.read_parquet(path, columns=[colname])
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  - {path}: error reading ({e})")
            return
    s = str(appid)
    if colname in df.columns and (df[colname].astype(str) == s).any():
        print(f"  - {path}: PRESENT in column '{colname}'")
    else:
        print(f"  - {path}: MISSING")


def main():
    appid = 4465480
    print(f"Checking presence of app id {appid} (CS:GO) in project artifacts:\n")

    # Graph pickles to check
    pickles = [
        'red_construida100K.pkl.gz',
        'red_construida.pkl.gz',
        'red_construida1MI.pkl.gz',
        'red_enriquecida.pkl.gz',
    ]
    for p in pickles:
        check_pickle_graph(p, appid)

    # Steam CSV
    print('\nChecking steam_games.csv:')
    check_csv('steam_games.csv', appid, colnames=('appid','app_id'))

    # centralidades CSV
    print('\nChecking centralidades_red_enriquecida.csv:')
    check_csv('centralidades_red_enriquecida.csv', appid, colnames=('app_id','appId','appid'))

    # recommendations parquet (may be large)
    print('\nChecking recommendations_sorted.parquet (app_id column):')
    check_parquet('recommendations_sorted.parquet', appid, colname='app_id')

    print('\nDone.')


if __name__ == '__main__':
    main()
