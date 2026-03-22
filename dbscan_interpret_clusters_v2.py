"""
dbscan_interpret_clusters_v2.py
================================
Step 5 of the DBSCAN Urban Pollution Pipeline:
  - Auto-diagnose correct epsilon from your data's spatial density
  - Apply DBSCAN with haversine (real-world km distances)
  - Classify every point as core / border / noise
  - Compute per-cluster statistics
  - Rank hotspots by pollution severity
  - Flag suspicious noise (possible hidden hotspots)
  - Export labelled CSV ready for Folium / Plotly mapping

FIXES APPLIED vs v1:
  [1] epsilon_km default raised to 50 km (suits global/country datasets)
  [2] min_samples lowered to 3 (sparse global data needs a lower bar)
  [3] FutureWarning fixed — severity/hotspot_rank cast before assignment
  [4] Auto-epsilon diagnostic added — suggests the right ε for your data
  [5] Optional country filter added before clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree


# ──────────────────────────────────────────────
# 0. AUTO-DIAGNOSE CORRECT EPSILON
# ──────────────────────────────────────────────

def suggest_epsilon(df: pd.DataFrame, min_samples: int = 3) -> float:
    """
    Computes the k-th nearest neighbour distance for every point
    (k = min_samples) using haversine, then recommends an epsilon.

    Rule: epsilon ≈ mean of the k-th nearest neighbour distances.
    This is the same value you'd read off the elbow of a k-distance plot.

    Prints a summary so you can sanity-check before clustering.
    """
    print("── Epsilon diagnostic ──────────────────────────")
    coords_rad = np.radians(df[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    # k+1 because index 0 is the point itself
    dists, _ = tree.query(coords_rad, k=min_samples + 1)
    kth_dists_km = dists[:, -1] * 6371.0   # radians → km

    mean_km   = kth_dists_km.mean()
    median_km = np.median(kth_dists_km)
    p95_km    = np.percentile(kth_dists_km, 95)

    print(f"  {min_samples}-th nearest neighbour distance:")
    print(f"    Mean   : {mean_km:.1f} km")
    print(f"    Median : {median_km:.1f} km")
    print(f"    95th % : {p95_km:.1f} km")
    print(f"  Suggested ε : {mean_km:.1f} km  (mean)")
    print(f"  If too many clusters → raise ε to {mean_km * 2:.1f} km")
    print(f"  If too few clusters  → lower ε to {mean_km * 0.5:.1f} km")
    print("────────────────────────────────────────────────")

    return round(mean_km, 1)


# ──────────────────────────────────────────────
# 1. APPLY DBSCAN  (haversine = geographic distance)
# ──────────────────────────────────────────────

def apply_dbscan(
    df: pd.DataFrame,
    epsilon_km: float = 50.0,   # FIX [1]: raised from 1.0 — suits global datasets
    min_samples: int = 3        # FIX [2]: lowered from 5 — sparse data needs less
) -> tuple:
    """
    Uses haversine metric so epsilon is in real-world kilometres, not degrees.
    Coordinates must be decimal degrees (WGS-84).

    Parameter guidance:
      epsilon_km  — run suggest_epsilon() first to get a data-driven value.
                    Global datasets (one city per row): 50–200 km
                    Country-level (stations across one country): 10–50 km
                    City-level (dense sensor network): 0.5–5 km

      min_samples — try 3 for sparse global data, 5 for dense city networks.
                    Lower = more clusters (including weak ones).
                    Higher = only tight confident hotspots survive.
    """
    coords_rad = np.radians(df[["lat", "lon"]].values)

    db = DBSCAN(
        eps=epsilon_km / 6371.0,   # km → radians (earth radius = 6371 km)
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1                  # use all CPU cores
    )

    df = df.copy()
    df["cluster"] = db.fit_predict(coords_rad)

    n_clusters = df[df["cluster"] >= 0]["cluster"].nunique()
    n_noise    = (df["cluster"] == -1).sum()

    print(f"\nDBSCAN results:")
    print(f"  ε = {epsilon_km} km  |  min_samples = {min_samples}")
    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise} ({n_noise / len(df) * 100:.1f}%)")

    if n_clusters == 0:
        print("\n  WARNING: 0 clusters found.")
        print("  → Your ε is still too small for this dataset's station spacing.")
        print("  → Run suggest_epsilon(df) to get a better starting value.")
        print("  → Or filter to one country: df = df[df['country']=='India'].copy()")

    return df, db


# ──────────────────────────────────────────────
# 2. LABEL POINT TYPES  (core / border / noise)
# ──────────────────────────────────────────────

def label_point_types(df: pd.DataFrame, db: DBSCAN) -> pd.DataFrame:
    """
    sklearn exposes db.core_sample_indices_ — the positional indices
    (not DataFrame index values) of every core point in the fitted array.

    Three roles:
      core   — ≥ min_samples neighbours within ε. Dense hotspot heart.
      border — within ε of a core but fewer neighbours. Cluster fringe.
      noise  — not within ε of any core. Isolated sensor or hidden hotspot.

    Adds column:  point_type  ∈ {"core", "border", "noise"}
    """
    if (df["cluster"] == -1).all():
        print("\nAll points are noise — skipping point type classification.")
        df = df.copy()
        df["point_type"] = "noise"
        return df

    df = df.copy()
    core_positions = set(db.core_sample_indices_)

    point_types = []
    for pos in range(len(df)):
        cluster_val = df.iloc[pos]["cluster"]
        if cluster_val == -1:
            point_types.append("noise")
        elif pos in core_positions:
            point_types.append("core")
        else:
            point_types.append("border")

    df["point_type"] = point_types

    counts = df["point_type"].value_counts()
    print(f"\nPoint type breakdown:")
    for pt, n in counts.items():
        print(f"  {pt:<8}: {n:>6,}  ({n / len(df) * 100:.1f}%)")

    return df


# ──────────────────────────────────────────────
# 3. PER-CLUSTER STATISTICS
# ──────────────────────────────────────────────

POLLUTANT_COLS = ["aqi", "pm25", "no2", "o3", "co"]


def cluster_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster (noise excluded) compute:
      station_count  — number of monitoring stations inside
      centroid_lat/lon — geographic centre (mean lat/lon)
      {poll}_mean    — average pollutant reading
      {poll}_max     — peak reading (worst-case exposure)
      core_ratio     — fraction of points that are core
                       High (>0.7) = tight reliable hotspot
                       Low  (<0.3) = loose fringe, treat with caution

    Sorted by primary pollutant descending (worst cluster first).
    """
    existing_poll = [c for c in POLLUTANT_COLS if c in df.columns]
    clustered = df[df["cluster"] >= 0]

    if clustered.empty:
        print("\nNo clustered points — returning empty statistics.")
        return pd.DataFrame()

    agg_dict = {
        "lat":     "mean",
        "lon":     "mean",
        "cluster": "count",
    }
    for col in existing_poll:
        agg_dict[col] = ["mean", "max"]

    stats = clustered.groupby("cluster").agg(agg_dict)
    stats.columns = ["_".join(c).strip("_") for c in stats.columns]
    stats = stats.rename(columns={
        "cluster_count": "station_count",
        "lat_mean":      "centroid_lat",
        "lon_mean":      "centroid_lon",
    })

    core_ratio = (
        df[df["cluster"] >= 0]
        .groupby("cluster")["point_type"]
        .apply(lambda x: (x == "core").mean())
        .rename("core_ratio")
    )
    stats = stats.join(core_ratio)

    sort_col = next(
        (f"{c}_mean" for c in ["aqi", "pm25", "no2"] if f"{c}_mean" in stats.columns),
        stats.columns[0]
    )
    stats = stats.sort_values(sort_col, ascending=False)

    print(f"\nCluster statistics (sorted by {sort_col}):")
    show = [c for c in ["station_count", "centroid_lat", "centroid_lon",
                         sort_col, "core_ratio"] if c in stats.columns]
    print(stats[show].to_string())

    return stats.reset_index()


# ──────────────────────────────────────────────
# 4. RANK HOTSPOTS
# ──────────────────────────────────────────────

def rank_hotspots(stats: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Assigns a severity tier and rank to each cluster.

    AQI tiers (US EPA):
      CRITICAL  > 300   Hazardous — public health emergency
      HIGH      > 200   Very unhealthy
      MODERATE  > 100   Unhealthy for sensitive groups
      LOW       ≤ 100   Acceptable

    PM2.5 tiers µg/m³ (WHO):
      CRITICAL  > 150
      HIGH      >  55
      MODERATE  >  15   WHO annual guideline
      LOW       ≤  15
    """
    if stats.empty:
        print("No clusters to rank.")
        return stats

    hotspots = stats.copy()

    if "aqi_mean" in hotspots.columns:
        value_col = "aqi_mean"
        def tier(v):
            if v > 300: return "CRITICAL"
            if v > 200: return "HIGH"
            if v > 100: return "MODERATE"
            return "LOW"

    elif "pm25_mean" in hotspots.columns:
        value_col = "pm25_mean"
        def tier(v):
            if v > 150: return "CRITICAL"
            if v >  55: return "HIGH"
            if v >  15: return "MODERATE"
            return "LOW"

    else:
        print("No AQI or PM2.5 column found — skipping severity tiers.")
        return hotspots

    hotspots["severity"]     = hotspots[value_col].apply(tier)
    hotspots["hotspot_rank"] = range(1, len(hotspots) + 1)

    display_cols = [c for c in [
        "hotspot_rank", "cluster", "severity",
        "station_count", "centroid_lat", "centroid_lon", value_col
    ] if c in hotspots.columns]

    print(f"\nTop {top_n} pollution hotspots:")
    print(hotspots[display_cols].head(top_n).to_string(index=False))

    return hotspots


# ──────────────────────────────────────────────
# 5. ANALYSE NOISE POINTS
# ──────────────────────────────────────────────

def analyse_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Noise (−1) is NOT synonymous with bad data. Three scenarios:

      A) High-pollution noise  → isolated hotspot DBSCAN missed
         Action: re-run with smaller ε on just the noise subset
      B) Low-pollution noise   → genuine sparse rural sensor, ignore
      C) Erratic reading noise → sensor fault, cross-check preprocessing

    Points above the 90th percentile of the full dataset are flagged
    as suspicious (scenario A). Inspect these manually.
    """
    noise = df[df["cluster"] == -1].copy()

    if noise.empty:
        print("\nNo noise points — every sensor assigned to a cluster.")
        return noise

    existing_poll = [c for c in POLLUTANT_COLS if c in noise.columns]
    if not existing_poll:
        return noise

    sort_col = next((c for c in ["aqi", "pm25"] if c in existing_poll), existing_poll[0])
    noise = noise.sort_values(sort_col, ascending=False)

    threshold = df[sort_col].quantile(0.90)
    noise["suspicious"] = noise[sort_col] > threshold
    n_suspicious = noise["suspicious"].sum()

    print(f"\nNoise analysis  (ranked by {sort_col}):")
    print(f"  Total noise points      : {len(noise):,}")
    print(f"  High-pollution suspects : {n_suspicious}  (>{threshold:.1f} {sort_col})")
    print("  Tip: re-run DBSCAN on the suspect subset with a smaller ε")
    print("       to check if a hidden micro-hotspot emerges.\n")

    if n_suspicious:
        print(noise[noise["suspicious"]][["lat", "lon", sort_col]].head(8).to_string(index=False))

    return noise


# ──────────────────────────────────────────────
# 6. EXPORT LABELLED DATASET
# ──────────────────────────────────────────────

def export_labelled(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    output_path: str = "pollution_clustered.csv"
) -> pd.DataFrame:
    """
    Merges cluster metadata (severity, rank) back onto the full station
    dataframe so every row has: cluster, point_type, severity, hotspot_rank.
    This file is consumed directly by the Folium / Plotly map step.
    """
    if stats.empty:
        df_out = df.copy()
        df_out["severity"]     = "NOISE"
        df_out["hotspot_rank"] = -1.0
        df_out.to_csv(output_path, index=False)
        print(f"\nNo clusters — exported noise-only dataset → {output_path}")
        return df_out

    merge_cols = [c for c in ["cluster", "severity", "hotspot_rank"] if c in stats.columns]
    df_out = df.merge(stats[merge_cols], on="cluster", how="left")

    # FIX [3]: cast dtypes BEFORE assignment to avoid FutureWarning
    df_out["severity"]     = df_out["severity"].astype(str)
    df_out["hotspot_rank"] = df_out["hotspot_rank"].astype(float)

    df_out.loc[df_out["cluster"] == -1, "severity"]     = "NOISE"
    df_out.loc[df_out["cluster"] == -1, "hotspot_rank"] = -1.0

    df_out.to_csv(output_path, index=False)
    print(f"\nExported {len(df_out):,} labelled rows → {output_path}")
    return df_out


# ──────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load clean data from preprocessing step ──────────────────
    df = pd.read_csv("pollution_clean.csv")
    print(f"Loaded {len(df):,} rows | columns: {list(df.columns)}\n")

    # ── FIX [5]: Optional — filter to one country for city-level clustering
    # Comment this out if you want global clustering instead.
    # Change "India" to any country name that appears in your 'country' column.
    # df = df[df["country"] == "India"].copy()
    # print(f"Filtered to {len(df):,} rows after country filter\n")

    # ── FIX [4]: Auto-diagnose the right epsilon for your dataset ─
    # This reads your data's actual station spacing and suggests ε.
    # Run this once, read the suggestion, then set epsilon_km below.
    suggested_eps = suggest_epsilon(df, min_samples=3)

    # ── Run DBSCAN ───────────────────────────────────────────────
    # Set epsilon_km to the value printed by suggest_epsilon() above.
    # Or override manually:
    #   Global dataset (one city per row)   → 50–200 km
    #   Country-level dataset               → 10–50 km
    #   City-level dense sensor network     → 0.5–5 km
    df, db_model = apply_dbscan(
        df,
        epsilon_km=suggested_eps,   # auto value; override if needed
        min_samples=3               # lower for sparse data, raise for dense
    )

    # ── Classify every point ─────────────────────────────────────
    df = label_point_types(df, db_model)

    # ── Per-cluster stats ─────────────────────────────────────────
    stats = cluster_statistics(df)

    # ── Rank hotspots ─────────────────────────────────────────────
    stats = rank_hotspots(stats, top_n=10)

    # ── Investigate noise ─────────────────────────────────────────
    noise_df = analyse_noise(df)

    # ── Export labelled CSV ───────────────────────────────────────
    df_final = export_labelled(df, stats, "pollution_clustered.csv")

    print("\nDone. Next step: visualise on a Folium / Plotly map.")