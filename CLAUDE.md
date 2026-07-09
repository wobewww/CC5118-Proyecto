# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CC5118-Proyecto** is a network analysis project that builds and analyzes a co-recommendation graph from Steam game data. The project constructs a network where nodes represent video games (apps) and edges represent co-recommendations (games that tend to be recommended together by the same users).

### Core Purpose
- **Data Source**: Steam game recommendations from the `recommendations_sorted.parquet` file (557MB dataset)
- **Network**: Co-recommendation graph where edges indicate games frequently recommended together
- **Analysis**: Graph analysis including community detection, centrality measures, and distribution analysis
- **Output**: Multiple data artifacts (pickle files, CSV files, visualizations)

## Quick Commands

### Building the Network Pipeline
The typical pipeline consists of three sequential steps:

```bash
# Step 1: Build initial network from recommendations parquet (100K rows)
python construir.py
# Output: RED-INICIAL_100K.pkl.gz (compressed graph with communities)

# Step 2: Enrich network with Steam game metadata (names, genres, developers)
python enrich_graph.py
# Output: RED-ENRIQUECIDA_100K.pkl.gz (enriched graph)

# Step 3: Calculate centrality metrics
python centralidades.py
# Output: centralidades_red_enriquecida_100K.csv, resumen_centralidades_red_enriquecida_100K.txt
```

### Analysis & Visualization
```bash
# Analyze centrality distributions and generate graphs
python analizar_distribuciones_centralidades.py
# Outputs to: distribuciones_centralidades/ directory with histograms and statistics

# Visualize giant component colored by centrality metrics
python graficar_componente_gigante.py
# Outputs to: gc_centralidades/ directory with 5 PNG visualizations

# With custom input/output:
python analizar_distribuciones_centralidades.py --input centralidades_red_enriquecida_100K.csv --output distribuciones_centralidades
python graficar_componente_gigante.py --graph RED-ENRIQUECIDA_100K.pkl.gz --centralities centralidades_red_enriquecida_100K.csv --output gc_centralidades
```

### Utility Scripts
```bash
# Check presence of a game (app ID) in all project artifacts
python check_csgo.py
# Checks: pickle files, CSVs, parquet (searches for app ID 4465480 by default)

# Show metrics and neighbors for a specific game
python show_csgo_metrics.py --appid 730
# Displays: centrality row, node attributes, and top 10 neighbors with weights

# Count reviews/recommendations for a game
python contar_reseñas_juego.py --appid 730
# Or by name:
python contar_reseñas_juego.py --nombre "Counter-Strike: Global Offensive"
```

## Architecture & Data Flow

### Data Pipeline Stages

1. **construir.py** - Network Construction
   - Loads `recommendations_sorted.parquet` (first 100K rows)
   - Filters: only rows with `is_recommended == "TRUE"`
   - Groups by `user_id` to find sets of games each user recommended
   - Creates edges between all game pairs recommended by the same user (combinations)
   - Weights edges by co-recommendation frequency, then inverts to distance (1/count)
   - Detects communities via greedy modularity optimization
   - Outputs: `RED-INICIAL_100K.pkl.gz` (dict with "grafo" and "comunidades")

2. **enrich_graph.py** - Metadata Enrichment
   - Loads initial network from `RED-INICIAL_100K.pkl.gz`
   - Loads `steam_games.csv` (metadata: appid, name, genres, developer, publisher)
   - Performs inner join on app_id to keep only nodes with Steam metadata
   - Adds node attributes: `name`, `genres`, `developer`
   - Outputs: `RED-ENRIQUECIDA_100K.pkl.gz` (same structure but enriched)

3. **centralidades.py** - Centrality Computation
   - Loads enriched network
   - Converts weight interpretation: adds `strength = 1/weight` for metrics that treat weight as connection strength
   - Computes 6+ centrality metrics per node:
     - `degree` & `degree_centrality`
     - `betweenness_centrality` (using weight as distance)
     - `closeness_centrality` (using weight as distance)
     - `eigenvector_centrality` (using strength)
     - `local_clustering_coefficient` (using strength)
     - `global_clustering_coefficient` (transitivity - graph-wide)
   - Sorts by degree (descending)
   - Outputs: `centralidades_red_enriquecida_100K.csv` + summary text

4. **analizar_distribuciones_centralidades.py** - Statistical Analysis
   - Reads centrality CSV
   - For each metric (degree, degree_centrality, betweenness, etc.):
     - Calculates: count, mean, std, min, p25, median, p75, p90, p95, p99, max, skew, kurtosis
     - Generates 6-panel visualization: raw frequency, raw log-scale, linear bins, linear bins log-scale, log bins, log bins log-scale
   - Creates summary tables and plots
   - Outputs: `distribuciones_centralidades/` with PNGs and CSVs

5. **graficar_componente_gigante.py** - Component Visualization
   - Loads graph and centralities
   - Extracts largest connected component
   - Creates 5 separate visualizations (one per metric: degree_centrality, betweenness, closeness, eigenvector, local_clustering)
   - Each shows nodes colored by metric value with consistent spring layout
   - Outputs: `gc_centralidades/gc_<metric>.png`

### Key Data Files & Formats

| File | Size | Purpose |
|------|------|---------|
| `recommendations_sorted.parquet` | 557MB | Raw data: user_id, app_id, is_recommended |
| `steam_games.csv` | 5.6MB | Game metadata: appid, name, genres, developer, publisher, ratings |
| `RED-INICIAL_100K.pkl.gz` | 1.1MB | Pickled dict with "grafo" (NetworkX Graph) and "comunidades" (list of community sets) |
| `RED-ENRIQUECIDA_100K.pkl.gz` | 547KB | Same as initial but with node attributes added |
| `centralidades_red_enriquecida_100K.csv` | 480KB | Each row is a node with all centrality metrics |
| `Examples.ipynb` | — | Educational notebook showing centrality calculations with sample graphs |

### Graph Structure

- **Nodes**: App IDs (integers) from recommendations or Steam store
- **Edges**: Co-recommendations weighted by frequency
- **Edge Direction**: Undirected (mutual recommendations)
- **Edge Weight**: Initially = co-recommendation count, then inverted to distance (1/count)
- **Additional Edge Attribute**: `strength = 1/weight` (inverse of distance)
- **Node Attributes** (after enrichment):
  - `name`: Game name from Steam store
  - `genres`: Genre string from Steam store
  - `developer`: Developer name from Steam store

## Dependencies & Environment

### Requirements (from requirements.txt)
```
pandas
networkx
numpy
matplotlib
```

### Optional (used in Examples.ipynb)
```
igraph
powerlaw
```

### Python Version
Python 3.8+ (verified working with 3.10+)

### Installation
```bash
pip install -r requirements.txt
```

## Important Notes for Development

### Weight Interpretation
The project uses inverted weights as a distance metric. This is crucial when:
- Computing betweenness/closeness centrality: NetworkX treats `weight` as distance (lower = stronger connection)
- Computing eigenvector centrality: We explicitly use `strength = 1/weight` as the connection strength attribute
- This dual interpretation exists because NetworkX algorithms have different weight conventions

### Community Detection
- Uses greedy modularity optimization (`greedy_modularity_communities` from NetworkX)
- Communities are stored as a list of sets
- Filtering logic in `enrich_graph.py` preserves community structure by removing disconnected nodes

### Network Size Considerations
- Full pipeline works with 100K rows from 557MB parquet file
- Smaller test versions exist: `red_construida.pkl.gz`, `red_construida100K.pkl.gz`
- Runtime scales with row count and graph size; 100K rows is manageable (< 5min per step)

### Output Organization
- Utility scripts check multiple versions of files (100K and non-100K variants)
- Some old artifacts remain for reference (smaller graph versions)
- Latest canonical outputs: `RED-ENRIQUECIDA_100K.pkl.gz` and `centralidades_red_enriquecida_100K.csv`

## File Locations Reference

```
C:\PROYECTO REDES\CC5118-Proyecto\
├── construir.py                              # Step 1: Build network
├── enrich_graph.py                           # Step 2: Add metadata
├── centralidades.py                          # Step 3: Calculate metrics
├── analizar_distribuciones_centralidades.py  # Analysis: distributions
├── graficar_componente_gigante.py            # Visualization: giant component
├── check_csgo.py                             # Utility: search all artifacts
├── show_csgo_metrics.py                      # Utility: show game metrics
├── contar_reseñas_juego.py                   # Utility: count reviews
├── Examples.ipynb                            # Educational notebook
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project description
├── steam_games.csv                           # Game metadata source
├── recommendations_sorted.parquet            # Raw recommendations data
├── RED-INICIAL_100K.pkl.gz                   # Initial network (output of construir.py)
├── RED-ENRIQUECIDA_100K.pkl.gz               # Enriched network (output of enrich_graph.py)
├── centralidades_red_enriquecida_100K.csv    # Centrality metrics (output of centralidades.py)
├── distribuciones_centralidades/             # Distribution analysis outputs
└── gc_centralidades/                         # Giant component visualizations
```

## Typical Workflows

### Building from Scratch
```bash
python construir.py       # ~2-3 min
python enrich_graph.py    # ~1 min
python centralidades.py   # ~1 min
```

### Quick Analysis of Existing Network
```bash
python analizar_distribuciones_centralidades.py --input centralidades_red_enriquecida_100K.csv
python graficar_componente_gigante.py --graph RED-ENRIQUECIDA_100K.pkl.gz
```

### Investigating Specific Games
```bash
python contar_reseñas_juego.py --nombre "Dota 2"
python show_csgo_metrics.py --appid 570  # Dota 2
```
