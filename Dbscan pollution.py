"""
dbscan_pollution.py
====================
Step 4 of the DBSCAN Urban Pollution Pipeline:
  - Generate synthetic pollution data (swap in your real CSV easily)
  - Plot k-distance graph to find optimal epsilon
  - Apply DBSCAN with haversine metric
  - Summarise clusters and noise points
  - Export labelled CSV for hotspot analysis

Dependencies:
    pip install pandas numpy scikit-learn matplotlib kneed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator          # pip install kneed  (auto-detects elbow)


# ─────────────────────────────────────────────────────────────
# 0. CONFIGURATION  — edit these before running
# ─────────────────────────────────────────────────────────────

INPUT_CSV   = "pollution_clean.csv"    # set to None to use synthetic data
OUTPUT_CSV  = "pollution_clustered.csv"

# DBSCAN params (auto-tuned below, but you can override)
EPSILON_KM  = None      # None = auto-detect from k-distance plot
MIN_SAMPLES = 5         # rule of thumb: ln(n) or 2*dimensions

# Earth radius for haversine conversion
EARTH_RADIUS_KM = 6371.0


# ─────────────────────────────────────────────────────────────
# 1. DATA  — synthetic fallback so you can run this immediately
# ─────────────────────────────────────────────────────────────

def load_or_generate(filepath: str | None) -> pd.DataFrame:
    """
    Load a real cleaned CSV (needs 'lat', 'lon', 'aqi' or 'pm25' columns),
    or generate a realistic synthetic dataset for Delhi, India.
    """
    if filepath:
        try:
            df = pd.read_csv(filepath)
            assert {"lat", "lon"}.issubset(df.columns)
            print(f"Loaded {len(df):,} rows from {filepath}")
            return df
        except Exception as e:
            print(f"Could not load '{filepath}': {e}\nFalling back to synthetic data.")

    # ── Synthetic: 3 industrial hotspots + 2 traffic hotspots + background noise
    rng = np.random.default_rng(42)

    # Delhi bounding box approx: lat 28.4–28.9, lon 76.8–77.4
    centres = [
        (28.67, 77.22, 280, "industrial"),   # Anand Vihar area
        (28.55, 77.10, 220, "industrial"),   # Okhla industrial zone
        (28.82, 77.08, 195, "industrial"),   # Narela
        (28.63, 77.21, 160, "traffic"),      # ITO / ring road
        (28.50, 77.26, 140, "traffic"),      # Noida border
    ]

    rows = []
    for lat_c, lon_c, aqi_c, zone in centres:
        n = rng.integers(60, 120)
        lats = rng.normal(lat_c, 0.04, n)
        lons = rng.normal(lon_c, 0.04, n)
        aqis = np.clip(rng.normal(aqi_c, 30, n), 50, 500)
        pm25 = np.clip(aqis * rng.uniform(0.3, 0.5, n), 10, 250)
        rows.append(pd.DataFrame({"lat": lats, "lon": lons,
                                  "aqi": aqis, "pm25": pm25, "zone_true": zone}))

    # Background scatter
    n_bg = 150
    rows.append(pd.DataFrame({
        "lat":       rng.uniform(28.4,  28.9,  n_bg),
        "lon":       rng.uniform(76.85, 77.35, n_bg),
        "aqi":       np.clip(rng.normal(80, 25, n_bg), 20, 150),
        "pm25":      np.clip(rng.normal(35, 15, n_bg), 5,  80),
        "zone_true": "background",
    }))

    df = pd.concat(rows, ignore_index=True)
    print(f"Generated {len(df):,} synthetic pollution readings (Delhi region)")
    return df


# ─────────────────────────────────────────────────────────────
# 2. K-DISTANCE PLOT  — visual epsilon tuning
# ─────────────────────────────────────────────────────────────

def kdistance_plot(coords_rad: np.ndarray,
                   k: int = 5,
                   auto_detect: bool = True) -> float:
    """
    Fit k-Nearest Neighbours on radian coordinates (haversine metric),
    sort distances, and plot the k-distance graph.

    The 'elbow' (point of maximum curvature) is your optimal epsilon.
    Convert from radians back to km for interpretability.

    Args:
        coords_rad  : (n, 2) array of [lat, lon] in radians
        k           : should equal your MIN_SAMPLES value
        auto_detect : use KneeLocator to mark elbow automatically

    Returns:
        Suggested epsilon in km.
    """
    print(f"\n[Epsilon] Fitting {k}-NN on {len(coords_rad):,} points (haversine)...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="haversine")
    nbrs.fit(coords_rad)
    distances, _ = nbrs.kneighbors(coords_rad)

    # Take the distance to the k-th neighbour, convert radians → km
    kth_distances_km = np.sort(distances[:, k - 1])[::-1] * EARTH_RADIUS_KM

    # ── Auto-detect elbow
    knee_km = None
    if auto_detect:
        try:
            knee = KneeLocator(
                x=range(len(kth_distances_km)),
                y=kth_distances_km,
                curve="convex",
                direction="decreasing",
                interp_method="polynomial",
            )
            if knee.knee is not None:
                knee_km = kth_distances_km[knee.knee]
                print(f"[Epsilon] KneeLocator suggests ε ≈ {knee_km:.3f} km")
        except Exception as e:
            print(f"[Epsilon] KneeLocator failed ({e}), pick epsilon visually from the plot.")

    # ── Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(kth_distances_km, color="#2563EB", linewidth=1.8, label=f"{k}-th NN distance")
    ax.set_xlabel("Points sorted by distance (descending)", fontsize=11)
    ax.set_ylabel(f"Distance to {k}-th neighbour (km)", fontsize=11)
    ax.set_title("K-Distance Graph — find the elbow for ε", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    if knee_km is not None:
        ax.axhline(knee_km, color="#DC2626", linestyle="--", linewidth=1.4,
                   label=f"Suggested ε = {knee_km:.3f} km")
        ax.legend(fontsize=10)
        # Annotate the elbow point
        ax.annotate(
            f"  ε ≈ {knee_km:.2f} km",
            xy=(knee.knee, knee_km),
            xytext=(knee.knee + len(kth_distances_km) * 0.05, knee_km * 1.2),
            arrowprops=dict(arrowstyle="->", color="#DC2626"),
            fontsize=10, color="#DC2626",
        )

    plt.tight_layout()
    plt.savefig("kdistance_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Epsilon] Plot saved → kdistance_plot.png")

    return knee_km if knee_km is not None else 1.0   # fallback 1 km


# ─────────────────────────────────────────────────────────────
# 3. APPLY DBSCAN
# ─────────────────────────────────────────────────────────────

def apply_dbscan(df: pd.DataFrame,
                 epsilon_km: float,
                 min_samples: int) -> pd.DataFrame:
    """
    Run DBSCAN using haversine distance on lat/lon.

    Key detail: sklearn's haversine metric expects coordinates in RADIANS
    as a (n, 2) array ordered [latitude, longitude].
    epsilon must also be in radians: epsilon_km / EARTH_RADIUS_KM.

    Cluster label -1 means NOISE (not dense enough to belong to any cluster).
    """
    coords_rad = np.radians(df[["lat", "lon"]].values)
    epsilon_rad = epsilon_km / EARTH_RADIUS_KM

    print(f"\n[DBSCAN] Running with ε={epsilon_km:.3f} km ({epsilon_rad:.6f} rad), "
          f"min_samples={min_samples}")

    db = DBSCAN(
        eps=epsilon_rad,
        min_samples=min_samples,
        algorithm="ball_tree",   # efficient for haversine
        metric="haversine",
    )
    df = df.copy()
    df["cluster"] = db.fit_predict(coords_rad)

    # ── Report
    n_clusters = df["cluster"].nunique() - (1 if -1 in df["cluster"].values else 0)
    n_noise    = (df["cluster"] == -1).sum()
    n_total    = len(df)

    print(f"[DBSCAN] Clusters found  : {n_clusters}")
    print(f"[DBSCAN] Noise points    : {n_noise:,}  ({100*n_noise/n_total:.1f}%)")
    print(f"[DBSCAN] Clustered points: {n_total - n_noise:,}  ({100*(n_total-n_noise)/n_total:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────
# 4. CLUSTER STATISTICS
# ─────────────────────────────────────────────────────────────

def cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cluster summary: size, centroid, avg/max pollutant values.
    Sorted by average AQI descending — top rows are your hotspots.
    """
    poll_cols = [c for c in ["aqi", "pm25", "no2", "o3", "co"] if c in df.columns]
    clustered = df[df["cluster"] != -1]

    agg = {
        "lat":  ["mean"],
        "lon":  ["mean"],
        "cluster": ["count"],
    }
    for col in poll_cols:
        agg[col] = ["mean", "max"]

    stats = clustered.groupby("cluster").agg(agg)
    stats.columns = (
        ["centroid_lat", "centroid_lon", "n_points"] +
        [f"{col}_{stat}" for col in poll_cols for stat in ["mean", "max"]]
    )
    stats = stats.sort_values("aqi_mean" if "aqi_mean" in stats.columns
                              else stats.columns[3], ascending=False)
    stats.index.name = "cluster_id"

    print("\n── Cluster Statistics (sorted by avg AQI) ──")
    print(stats.to_string())
    return stats


# ─────────────────────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_clusters(df: pd.DataFrame, epsilon_km: float, min_samples: int) -> None:
    """
    Scatter map of clustered points.
      • Each cluster gets a distinct colour
      • Noise points (-1) are grey crosses
      • Point size scales with AQI value
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    unique_clusters = sorted(df["cluster"].unique())
    n_clusters = len([c for c in unique_clusters if c != -1])
    cmap = cm.get_cmap("tab20", max(n_clusters, 1))

    poll_col = next((c for c in ["aqi", "pm25"] if c in df.columns), None)

    for cluster_id in unique_clusters:
        mask = df["cluster"] == cluster_id
        subset = df[mask]

        if cluster_id == -1:
            ax.scatter(subset["lon"], subset["lat"],
                       s=15, c="lightgray", marker="x",
                       linewidths=0.8, alpha=0.5, label="Noise", zorder=2)
        else:
            sizes = (subset[poll_col] / subset[poll_col].max() * 120 + 20
                     if poll_col else 60)
            ax.scatter(subset["lon"], subset["lat"],
                       s=sizes, c=[cmap(cluster_id % 20)],
                       alpha=0.75, edgecolors="white", linewidths=0.4,
                       label=f"Cluster {cluster_id}", zorder=3)

            # Label centroid
            cx, cy = subset["lon"].mean(), subset["lat"].mean()
            ax.annotate(str(cluster_id), (cx, cy),
                        fontsize=9, fontweight="bold",
                        ha="center", va="center", color="white",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc=cmap(cluster_id % 20), ec="none"))

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude",  fontsize=11)
    ax.set_title(
        f"DBSCAN Pollution Clusters  |  ε={epsilon_km:.2f} km, min_samples={min_samples}\n"
        f"Point size ∝ {poll_col.upper() if poll_col else 'uniform'}  |  Grey × = noise",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # Compact legend (collapse if many clusters)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 12:
        ax.legend(handles, labels, loc="upper right", fontsize=9,
                  framealpha=0.85, markerscale=1.2)

    plt.tight_layout()
    plt.savefig("dbscan_clusters.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Plot] Cluster map saved → dbscan_clusters.png")


# ─────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    # ── Load data
    df = load_or_generate(INPUT_CSV if INPUT_CSV else None)

    # ── Convert coords to radians for NearestNeighbors + DBSCAN
    coords_rad = np.radians(df[["lat", "lon"]].values)

    # ── Tune epsilon via k-distance plot
    k = MIN_SAMPLES
    suggested_eps = kdistance_plot(coords_rad, k=k, auto_detect=True)

    epsilon_km = EPSILON_KM if EPSILON_KM is not None else suggested_eps
    print(f"\n[Config] Using ε = {epsilon_km:.3f} km | min_samples = {MIN_SAMPLES}")

    # ── Run DBSCAN
    df = apply_dbscan(df, epsilon_km=epsilon_km, min_samples=MIN_SAMPLES)

    # ── Stats
    stats = cluster_stats(df)

    # ── Plot
    plot_clusters(df, epsilon_km=epsilon_km, min_samples=MIN_SAMPLES)

    # ── Save
    df.to_csv(OUTPUT_CSV, index=False)
    stats.to_csv("cluster_stats.csv")
    print(f"\n[Done] Labelled data → {OUTPUT_CSV}")
    print("[Done] Cluster stats → cluster_stats.csv")

    return df, stats


if __name__ == "__main__":
    df_out, stats_out = main()