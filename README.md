# AirScan — DBSCAN Urban Pollution Zone Detection

> Density-based clustering to automatically identify air pollution hotspots from global monitoring station data — no predefined cluster count required.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-DBSCAN-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-Interactive%20Maps-77B829?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What this project does

Traditional pollution hotspot analysis relies on administrative boundaries — cities, districts, states — which don't reflect how pollution actually spreads. This project uses **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to discover pollution zones purely from the spatial density of monitoring stations and their recorded AQI values.

Key outputs:
- **372 geographically coherent pollution clusters** identified from 10,265 global stations
- **Interactive Folium map** with severity-coloured markers, heatmap layer, and ranked hotspot pins
- **Noise point analysis** — isolates high-AQI stations with no nearby monitoring infrastructure (the most actionable finding)
- **AirTrace web app** — search any place name, get live AQI + DBSCAN cluster context in the browser

---

## Why DBSCAN over k-means

| Feature | DBSCAN | k-means |
|---|---|---|
| Number of clusters | Auto-discovered | Must specify k upfront |
| Cluster shape | Any shape | Spherical only |
| Noise handling | Explicit −1 label | No noise concept |
| Geographic suitability | Haversine metric support | Euclidean only |
| Outlier sensitivity | Robust | Sensitive |

---

## Project structure

```
├── pollution_preprocessing.py       # Step 1 — load, clean, validate, scale
├── dbscan_interpret_clusters_v2.py  # Step 2 — DBSCAN, label, rank hotspots
├── india_cluster_fix.py             # Step 3 — India-only re-run at city scale
├── india_pollution_map.py           # Step 4 — Folium map with state boundaries
├── map_pollution_clusters.py        # Step 5 — Global Folium + Plotly maps
├── airtrace.html                    # Web app — search place → live AQI + cluster context
├── DBSCAN_Pollution_Report.docx     # Full project report (7 sections)
└── README.md
```

**Generated outputs (not committed):**
```
pollution_clean.csv          # Cleaned dataset
pollution_clustered.csv      # With cluster labels + severity
india_clustered.csv          # India-only clusters
pollution_map.html           # Global Folium map
india_pollution_map.html     # India Folium map
pollution_map_plotly.html    # Plotly interactive map
cluster_barchart.html        # Top 30 clusters bar chart
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn folium plotly requests
```

### 2. Get the dataset

Download from Kaggle: [Global Air Quality Index by City and Coordinates](https://www.kaggle.com/datasets/adityaramachandran27/world-air-quality-index-by-city-and-coordinates)

Save as `air_quality_data.csv` in the project root.

### 3. Run the pipeline

```bash
# Clean and preprocess
python pollution_preprocessing.py

# Run DBSCAN, label clusters, rank hotspots
python dbscan_interpret_clusters_v2.py

# Build the interactive map
python map_pollution_clusters.py
```

Open `pollution_map.html` in your browser.

### 4. India-specific analysis

```bash
python india_cluster_fix.py        # Re-run at 5 km scale for India
python india_pollution_map.py      # Render India map with state borders
```

### 5. AirTrace web app

Just open `airtrace.html` in Chrome or Firefox. No server needed.
Type any city name → get live AQI + coordinates + DBSCAN cluster context.

---

## How DBSCAN works here

```python
from sklearn.cluster import DBSCAN
import numpy as np

coords_rad = np.radians(df[["lat", "lon"]].values)

db = DBSCAN(
    eps=58.6 / 6371.0,    # 58.6 km converted to radians
    min_samples=3,
    algorithm="ball_tree",
    metric="haversine",    # real-world km, not degrees
    n_jobs=-1
)

df["cluster"] = db.fit_predict(coords_rad)
# cluster >= 0  → pollution zone
# cluster == -1 → noise (isolated station)
```

Epsilon (58.6 km) was auto-derived from the k-distance elbow plot — the mean 3rd-nearest-neighbour distance across all stations. The `suggest_epsilon()` function in `dbscan_interpret_clusters_v2.py` computes this automatically from any dataset.

---

## Key results

| Metric | Value |
|---|---|
| Dataset | 10,265 monitoring stations, 100+ countries |
| Clusters found | 372 |
| Noise points | 1,821 (17.7%) |
| Largest cluster | Cluster 1 — Western Europe (2,298 stations) |
| Top AQI cluster | Cluster 13 — Central Asia (mean AQI 95.3) |
| High-AQI noise stations | 196 stations above 90th percentile |
| Peak noise AQI | 114 — Anjangaon, Maharashtra |

### Severity tiers (US EPA AQI scale)

| Tier | AQI range | Colour |
|---|---|---|
| CRITICAL | > 300 | Red |
| HIGH | 201 – 300 | Amber |
| MODERATE | 101 – 200 | Blue |
| LOW | ≤ 100 | Teal |
| NOISE | — | Gray |

---

## The most important finding

**High-AQI noise points are not clean air — they are monitoring gaps.**

DBSCAN's `−1` label identifies stations that are geographically isolated from any cluster. When these stations also record high AQI values, it signals that pollution exists in a region with insufficient monitoring infrastructure. The 196 such stations identified in this project — concentrated in interior Maharashtra, rural Telangana, and parts of West Africa — represent the clearest policy recommendation: **deploy sensors where DBSCAN finds high-AQI noise.**

---

## Dataset limitations

- The Kaggle dataset used does **not** include Delhi, Mumbai, Kolkata or other Indian megacities. The India-level findings reflect smaller towns and should not be extrapolated to the national picture.
- Data represents a **single point-in-time snapshot** per station — not a temporal average.
- At ε = 58.6 km, clusters represent **regional zones**, not city-level hotspots. Use `india_cluster_fix.py` (ε = 5 km) for city-scale analysis.

For city-level Indian analysis, replace the dataset with [CPCB AQI data](https://cpcb.nic.in/air-quality-data/) and re-run with `epsilon_km=5.0, min_samples=3`.

---

## AirTrace — live location search

`airtrace.html` is a standalone web app requiring no installation:

- Type any place name → geocoded via OpenStreetMap Nominatim
- Live PM2.5, PM10, NO₂, O₃, CO, AQI fetched from Open-Meteo
- AQI gauge, pollutant bars, and health advisory shown instantly
- DBSCAN cluster context derived from trained model's cluster centroids
- No API keys required — both APIs are free and open

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core pipeline |
| pandas / numpy | Data processing |
| scikit-learn | DBSCAN, BallTree, StandardScaler |
| Folium | Interactive Leaflet.js maps |
| Plotly | Scatter mapbox visualisation |
| requests | GeoJSON fetch for state boundaries |
| HTML / JS | AirTrace web app |
| OpenStreetMap Nominatim | Free geocoding API |
| Open-Meteo | Free live air quality API |

---

## Report

A full 7-section project report (`DBSCAN_Pollution_Report.docx`) is included covering methodology, results, insights, recommendations, and limitations. Built with `docx` (Node.js).

---

## License

MIT — free to use, adapt and build on.

---

## Acknowledgements

- Dataset: [Kaggle — Global Air Quality Index](https://www.kaggle.com/datasets/adityaramachandran27/world-air-quality-index-by-city-and-coordinates)
- Original DBSCAN paper: Ester et al., KDD-96
- Air quality API: [Open-Meteo](https://open-meteo.com)
- Geocoding: [OpenStreetMap Nominatim](https://nominatim.openstreetmap.org)
- State boundaries GeoJSON: [geohacker/india](https://github.com/geohacker/india)
