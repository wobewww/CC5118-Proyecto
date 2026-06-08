"""Analiza las distribuciones de las medidas de centralidad.

Lee `centralidades_red_enriquecida.csv`, calcula estadísticos descriptivos y
genera gráficas para cada métrica en una carpeta de salida.

Uso:
    python analizar_distribuciones_centralidades.py
    python analizar_distribuciones_centralidades.py --input centralidades_red_enriquecida.csv --output distribuciones
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COLUMNS = [
    "degree",
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector",
    "local_clustering_coefficient",
]

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA)


def raw_frequency(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cleaned = values.dropna().to_numpy()
    if cleaned.size == 0:
        return np.array([]), np.array([])
    x, y = np.unique(cleaned, return_counts=True)
    return x, y


def binned_frequency(values: pd.Series, log_bins: bool = False, bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    cleaned = values.dropna().to_numpy()
    if cleaned.size == 0:
        return np.array([]), np.array([])

    if bins is None:
        bins = max(10, min(50, int(np.sqrt(cleaned.size))))

    if log_bins:
        positive = cleaned[cleaned > 0]
        if positive.size == 0:
            return np.array([]), np.array([])
        min_value = positive.min()
        max_value = positive.max()
        if min_value == max_value:
            edges = np.array([min_value, max_value + 1.0])
        else:
            edges = np.logspace(np.log10(min_value), np.log10(max_value), bins + 1)
        counts, edges = np.histogram(positive, bins=edges)
        x = edges[:-1]
        y = counts
    else:
        counts, edges = np.histogram(cleaned, bins=bins)
        x = edges[:-1]
        y = counts

    return x, y


def summarize_series(series: pd.Series) -> dict[str, float]:
    cleaned = series.dropna()
    if cleaned.empty:
        return {}

    percentiles = cleaned.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    summary = {
        "count": float(cleaned.count()),
        "mean": float(cleaned.mean()),
        "std": float(cleaned.std(ddof=1)),
        "min": float(cleaned.min()),
        "p25": float(percentiles.loc[0.25]),
        "median": float(percentiles.loc[0.5]),
        "p75": float(percentiles.loc[0.75]),
        "p90": float(percentiles.loc[0.9]),
        "p95": float(percentiles.loc[0.95]),
        "p99": float(percentiles.loc[0.99]),
        "max": float(cleaned.max()),
        "skew": float(cleaned.skew()),
        "kurtosis": float(cleaned.kurtosis()),
        "zeros": float((cleaned == 0).sum()),
    }
    return summary


def write_summary_table(df: pd.DataFrame, columns: list[str], output_dir: Path) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        summary = summarize_series(df[column])
        if not summary:
            continue
        summary["metric"] = column
        rows.append(summary)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df[
            [
                "metric",
                "count",
                "mean",
                "std",
                "min",
                "p25",
                "median",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
                "skew",
                "kurtosis",
                "zeros",
            ]
        ]
        summary_df.to_csv(output_dir / "resumen_distribuciones.csv", index=False)
        summary_df.to_string(index=False)
        with open(output_dir / "resumen_distribuciones.txt", "w", encoding="utf-8") as f:
            f.write(summary_df.to_string(index=False))
            f.write("\n")
    return summary_df


def plot_frequency_panels(series: pd.Series, metric: str, output_dir: Path) -> None:
    cleaned = series.dropna()
    if cleaned.empty:
        return

    x_raw, y_raw = raw_frequency(cleaned)
    x_lin, y_lin = binned_frequency(cleaned, log_bins=False)
    x_log, y_log = binned_frequency(cleaned, log_bins=True)

    if x_raw.size == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    positive_raw_mask = (x_raw > 0) & (y_raw > 0)
    positive_lin_mask = (x_lin > 0) & (y_lin > 0)
    positive_log_mask = (x_log > 0) & (y_log > 0)

    ax = axes[0, 0]
    ax.scatter(x_raw, y_raw, marker="v", c="r", s=18)
    ax.set_title("Frecuencia sin bins")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frecuencia")

    ax = axes[1, 0]
    ax.scatter(x_raw[positive_raw_mask], y_raw[positive_raw_mask], marker="^", c="r", s=18)
    ax.set_title("Frecuencia sin bins, escala logarítmica")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frecuencia")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[0, 1]
    ax.scatter(x_lin, y_lin, marker="o", c="b", s=18)
    ax.set_title("Frecuencia con bins lineales")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frecuencia")

    ax = axes[1, 1]
    ax.scatter(x_lin[positive_lin_mask], y_lin[positive_lin_mask], marker="x", c="b", s=18)
    ax.set_title("Frecuencia con bins lineales, escala logarítmica")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frecuencia")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[0, 2]
    ax.scatter(x_log[positive_log_mask], y_log[positive_log_mask], marker="o", c="g", s=18)
    ax.set_title("Frecuencia con bins logarítmicos")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frecuencia")
    ax.set_xscale("log")

    ax = axes[1, 2]
    ax.scatter(x_log[positive_log_mask], y_log[positive_log_mask], marker="o", c="g", s=18)
    ax.set_title("Frecuencia con bins logarítmicos, escala logarítmica")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frecuencia")
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.suptitle(f"{metric} vs frecuencia", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_dir / f"{metric}_panel_frecuencias.png", dpi=160)
    plt.close(fig)


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> None:
    lines = []
    lines.append("Resumen de distribuciones de centralidad")
    lines.append("")
    if summary_df.empty:
        lines.append("No se encontraron columnas numéricas válidas para analizar.")
    else:
        for _, row in summary_df.iterrows():
            lines.append(f"Métrica: {row['metric']}")
            lines.append(f"- count: {int(row['count'])}")
            lines.append(f"- mean: {row['mean']:.6f}")
            lines.append(f"- median: {row['median']:.6f}")
            lines.append(f"- std: {row['std']:.6f}")
            lines.append(f"- min/max: {row['min']:.6f} / {row['max']:.6f}")
            lines.append(f"- p90/p95/p99: {row['p90']:.6f} / {row['p95']:.6f} / {row['p99']:.6f}")
            lines.append(f"- skew: {row['skew']:.6f}")
            lines.append(f"- kurtosis: {row['kurtosis']:.6f}")
            lines.append("")
    (output_dir / "resumen_distribuciones_simple.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza distribuciones de centralidades")
    parser.add_argument(
        "--input",
        default="centralidades_red_enriquecida_100K.csv",
        help="CSV de centralidades a analizar",
    )
    parser.add_argument(
        "--output",
        default="distribuciones_centralidades",
        help="Carpeta de salida para gráficos y resúmenes",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=DEFAULT_COLUMNS,
        help="Columnas numéricas a analizar",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    if "app_id" in df.columns:
        df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce")

    available_columns = []
    for column in args.columns:
        if column in df.columns:
            df[column] = safe_numeric(df[column])
            available_columns.append(column)

    summary_df = write_summary_table(df, available_columns, output_dir)

    for column in available_columns:
        plot_frequency_panels(df[column], column, output_dir)

    build_report(summary_df, output_dir)

    print(f"Analisis completado. Archivos generados en: {output_dir.resolve()}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
