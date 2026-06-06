"""Cuenta reseñas/ratings de un juego por `appid` o por nombre.

Fuentes usadas:
 - steam_games.csv: positive_ratings, negative_ratings, total y porcentaje positivo
 - recommendations_sorted.parquet: número de filas por app_id y, si existe,
   conteo de recomendaciones positivas/negativas usando is_recommended

Uso:
    python contar_reseñas_juego.py --appid 730
    python contar_reseñas_juego.py --nombre "Counter-Strike: Global Offensive"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STEAM_CSV = Path("steam_games.csv")
RECOMMENDATIONS_PARQUET = Path("recommendations_sorted.parquet")
DEFAULT_APPID = 440


def load_steam_games() -> pd.DataFrame:
    if not STEAM_CSV.exists():
        raise FileNotFoundError(f"No existe {STEAM_CSV}")
    df = pd.read_csv(STEAM_CSV, low_memory=False)
    df.columns = df.columns.str.strip()
    if "appid" in df.columns:
        df["appid"] = pd.to_numeric(df["appid"], errors="coerce").astype("Int64")
    return df


def find_game(df: pd.DataFrame, appid: int | None, nombre: str | None) -> pd.DataFrame:
    if appid is not None and "appid" in df.columns:
        return df[df["appid"] == appid]

    if nombre:
        query = nombre.strip().lower()
        if "name" not in df.columns:
            return df.iloc[0:0]
        mask = df["name"].astype(str).str.lower().str.contains(query, na=False)
        return df[mask]

    return df.iloc[0:0]


def print_steam_summary(game_row: pd.Series) -> None:
    positive = pd.to_numeric(game_row.get("positive_ratings"), errors="coerce")
    negative = pd.to_numeric(game_row.get("negative_ratings"), errors="coerce")
    total = positive + negative if pd.notna(positive) and pd.notna(negative) else pd.NA
    pct_positive = (positive / total * 100) if pd.notna(total) and total != 0 else pd.NA

    print("Fuente: steam_games.csv")
    print(f"- appid: {game_row.get('appid')}")
    print(f"- nombre: {game_row.get('name')}")
    print(f"- reseñas positivas: {int(positive) if pd.notna(positive) else 'NA'}")
    print(f"- reseñas negativas: {int(negative) if pd.notna(negative) else 'NA'}")
    print(f"- total reseñas: {int(total) if pd.notna(total) else 'NA'}")
    print(f"- porcentaje positivo: {pct_positive:.2f}%" if pd.notna(pct_positive) else "- porcentaje positivo: NA")


def print_recommendation_summary(appid: int) -> None:
    if not RECOMMENDATIONS_PARQUET.exists():
        print(f"\nFuente: {RECOMMENDATIONS_PARQUET} no existe")
        return

    try:
        df = pd.read_parquet(RECOMMENDATIONS_PARQUET)
    except Exception as exc:
        print(f"\nNo se pudo leer {RECOMMENDATIONS_PARQUET}: {exc}")
        return

    df.columns = df.columns.str.strip()
    app_cols = [c for c in ["app_id", "appid", "game_id"] if c in df.columns]
    if not app_cols:
        print(f"\nFuente: {RECOMMENDATIONS_PARQUET}")
        print("- no encontré una columna de app id (app_id/appid/game_id)")
        return

    app_col = app_cols[0]
    df[app_col] = pd.to_numeric(df[app_col], errors="coerce")
    subset = df[df[app_col] == appid]

    print(f"\nFuente: {RECOMMENDATIONS_PARQUET}")
    print(f"- columna usada: {app_col}")
    print(f"- filas totales para el juego en el parquet completo: {len(subset)}")

    if "is_recommended" in subset.columns and not subset.empty:
        rec = subset["is_recommended"].astype(str).str.strip().str.upper()
        positive = int((rec == "TRUE").sum())
        negative = int((rec == "FALSE").sum())
        print(f"- recomendaciones positivas: {positive}")
        print(f"- recomendaciones negativas: {negative}")
        if positive + negative > 0:
            print(f"- porcentaje positivo en parquet: {positive / (positive + negative) * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cuenta reseñas/ratings de un juego")
    parser.add_argument("--appid", type=int, default=DEFAULT_APPID, help="AppID del juego")
    parser.add_argument("--nombre", type=str, default=None, help="Nombre o parte del nombre del juego")
    args = parser.parse_args()

    steam_df = load_steam_games()
    matches = find_game(steam_df, args.appid, args.nombre)

    if matches.empty:
        print("No encontré el juego en steam_games.csv")
        return

    if len(matches) > 1:
        print("Encontré varias coincidencias. Mostrando la primera:\n")

    row = matches.iloc[0]
    print_steam_summary(row)

    appid = int(row["appid"])
    print_recommendation_summary(appid)


if __name__ == "__main__":
    main()
