"""
map_pollution_clusters.py
==========================
Step 7 of the DBSCAN Urban Pollution Pipeline:
  - Reads  pollution_clustered.csv  (output of dbscan_interpret_clusters_v2.py)
  - Builds an interactive Folium map  → pollution_map.html
  - Builds an interactive Plotly map  → pollution_map_plotly.html
  - Both open in any browser; no server needed

Install once:
    pip install folium plotly pandas
"""

import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# SHARED CONFIG
# ──────────────────────────────────────────────

INPUT_CSV        = "pollution_clustered.csv"
FOLIUM_OUTPUT    = "pollution_map.html"
PLOTLY_OUTPUT    = "pollution_map_plotly.html"

# Colour palette — maps severity tier to a hex colour
SEVERITY_COLORS = {
    "CRITICAL": "#E24B4A",   # red
    "HIGH":     "#EF9F27",   # amber
    "MODERATE": "#378ADD",   # blue
    "LOW":      "#1D9E75",   # teal
    "NOISE":    "#888780",   # gray
}

# Radius of the circle marker for each severity (pixels, Folium only)
SEVERITY_RADIUS = {
    "CRITICAL": 10,
    "HIGH":     8,
    "MODERATE": 6,
    "LOW":      5,
    "NOISE":    3,
}


# ──────────────────────────────────────────────
# 0. LOAD & PREP DATA
# ──────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure severity column exists; fall back gracefully
    if "severity" not in df.columns:
        df["severity"] = "LOW"

    # Fill any remaining NaNs in severity (noise rows merged before fix)
    df["severity"] = df["severity"].fillna("NOISE").astype(str)
    df["cluster"]  = df["cluster"].fillna(-1).astype(int)

    # Pick primary pollutant column for tooltips
    for col in ["aqi", "pm25", "no2"]:
        if col in df.columns:
            df["_poll_value"] = df[col].round(1)
            df["_poll_name"]  = col.upper()
            break

    # City label for tooltip (use city col if present)
    if "city" in df.columns:
        df["_label"] = df["city"].fillna("Unknown")
    else:
        df["_label"] = "Station " + df.index.astype(str)

    print(f"Loaded {len(df):,} stations | "
          f"clusters: {df[df['cluster']>=0]['cluster'].nunique()} | "
          f"noise: {(df['cluster']==-1).sum():,}")
    return df


# ──────────────────────────────────────────────
# 1. FOLIUM MAP
# ──────────────────────────────────────────────

def build_folium_map(df: pd.DataFrame, output_path: str) -> folium.Map:
    """
    Layers:
      • Circle markers — one per station, coloured by severity
      • Heatmap layer  — continuous pollution intensity surface
      • Layer control  — toggle each layer on/off

    Click any marker to see city, cluster ID, severity, and AQI.
    """

    # Centre map on data centroid
    centre_lat = df["lat"].mean()
    centre_lon = df["lon"].mean()

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=3,
        tiles="CartoDB dark_matter",   # dark basemap suits pollution viz
        control_scale=True,
    )

    # ── Layer 1: Circle markers per severity tier ──────────────────
    # Add a separate FeatureGroup per tier so they can be toggled
    severity_order = ["CRITICAL", "HIGH", "MODERATE", "LOW", "NOISE"]

    for sev in severity_order:
        subset = df[df["severity"] == sev]
        if subset.empty:
            continue

        layer = folium.FeatureGroup(name=f"{sev}  ({len(subset):,} stations)", show=True)

        for _, row in subset.iterrows():
            popup_html = f"""
            <div style='font-family:monospace;font-size:13px;min-width:180px'>
                <b>{row['_label']}</b><br>
                Cluster&nbsp;&nbsp;: {row['cluster']}<br>
                Severity : <b style='color:{SEVERITY_COLORS[sev]}'>{sev}</b><br>
                {row['_poll_name']}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {row['_poll_value']}<br>
                Point type: {row.get('point_type','—')}<br>
                Lat / Lon : {row['lat']:.4f}, {row['lon']:.4f}
            </div>"""

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=SEVERITY_RADIUS[sev],
                color=SEVERITY_COLORS[sev],
                fill=True,
                fill_color=SEVERITY_COLORS[sev],
                fill_opacity=0.75 if sev != "NOISE" else 0.35,
                weight=0.5,
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"{row['_label']} | {row['_poll_name']}: {row['_poll_value']}",
            ).add_to(layer)

        layer.add_to(m)

    # ── Layer 2: Heatmap of pollution intensity ────────────────────
    heat_data = (
        df[["lat", "lon", "_poll_value"]]
        .dropna()
        .values
        .tolist()
    )

    heatmap_layer = folium.FeatureGroup(name="Heatmap (intensity)", show=False)
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=6,
        radius=18,
        blur=25,
        gradient={
            "0.0": "#1D9E75",   # teal = low
            "0.4": "#378ADD",   # blue = moderate
            "0.7": "#EF9F27",   # amber = high
            "1.0": "#E24B4A",   # red = critical
        },
    ).add_to(heatmap_layer)
    heatmap_layer.add_to(m)

    # ── Layer 3: Hotspot centroids (top 20 clusters) ───────────────
    if "hotspot_rank" in df.columns:
        top_clusters = (
            df[df["hotspot_rank"] > 0]
            .sort_values("hotspot_rank")
            .drop_duplicates("cluster")
            .head(20)
        )

        centroid_layer = folium.FeatureGroup(name="Top 20 hotspot centroids", show=True)

        for _, row in top_clusters.iterrows():
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(
                    html=f"""<div style='
                        background:{SEVERITY_COLORS.get(row["severity"],"#888780")};
                        color:#fff;
                        border-radius:50%;
                        width:22px;height:22px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:10px;font-weight:bold;font-family:monospace;
                        border:1.5px solid rgba(255,255,255,0.6);
                    '>#{int(row["hotspot_rank"])}</div>""",
                    icon_size=(22, 22),
                    icon_anchor=(11, 11),
                ),
                popup=f"Rank #{int(row['hotspot_rank'])} | Cluster {row['cluster']} | {row['severity']}",
            ).add_to(centroid_layer)

        centroid_layer.add_to(m)

    # ── Legend ─────────────────────────────────────────────────────
    legend_html = """
    <div style='
        position:fixed;bottom:30px;right:10px;z-index:9999;
        background:rgba(20,20,20,0.88);
        color:#ddd;padding:12px 16px;border-radius:8px;
        font-family:monospace;font-size:12px;line-height:1.9;
        border:1px solid rgba(255,255,255,0.15);
    '>
        <b style='font-size:13px;color:#fff'>Pollution severity</b><br>
        <span style='color:#E24B4A'>&#9679;</span> CRITICAL  (AQI &gt; 300)<br>
        <span style='color:#EF9F27'>&#9679;</span> HIGH      (AQI 201–300)<br>
        <span style='color:#378ADD'>&#9679;</span> MODERATE  (AQI 101–200)<br>
        <span style='color:#1D9E75'>&#9679;</span> LOW       (AQI ≤ 100)<br>
        <span style='color:#888780'>&#9679;</span> NOISE     (isolated)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Layer control ──────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_path)
    print(f"Folium map saved → {output_path}  (open in any browser)")
    return m


# ──────────────────────────────────────────────
# 2. PLOTLY MAP
# ──────────────────────────────────────────────

def build_plotly_map(df: pd.DataFrame, output_path: str):
    """
    Plotly scatter_mapbox:
      • Each point sized by AQI value
      • Coloured by severity tier
      • Hover shows city, cluster, AQI, point type
      • Dropdown to filter by severity

    Best used inside Jupyter with fig.show(), or saved as standalone HTML.
    """

    # Assign numeric size scaled from poll value (min 4, max 18)
    poll_vals = df["_poll_value"].fillna(0)
    size_scaled = 4 + (poll_vals - poll_vals.min()) / (poll_vals.max() - poll_vals.min() + 1e-9) * 14
    df = df.copy()
    df["_marker_size"] = size_scaled.round(1)

    # Cluster label as string for hover
    df["_cluster_str"] = df["cluster"].apply(
        lambda c: f"Cluster {c}" if c >= 0 else "Noise"
    )

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="severity",
        size="_marker_size",
        size_max=18,
        hover_name="_label",
        hover_data={
            "_cluster_str": True,
            "_poll_name":   False,
            "_poll_value":  True,
            "point_type":   True,
            "severity":     False,
            "_marker_size": False,
            "lat":          ":.4f",
            "lon":          ":.4f",
        },
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"severity": ["CRITICAL","HIGH","MODERATE","LOW","NOISE"]},
        mapbox_style="carto-darkmatter",
        zoom=2,
        center={"lat": df["lat"].mean(), "lon": df["lon"].mean()},
        title="DBSCAN Urban Pollution Clusters",
        labels={
            "_poll_value":  "AQI",
            "_cluster_str": "Cluster",
            "point_type":   "Point type",
        },
    )

    fig.update_layout(
        margin={"r": 0, "t": 48, "l": 0, "b": 0},
        legend=dict(
            title="Severity",
            bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="#ddd", size=12),
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        paper_bgcolor="#111",
        font_color="#ddd",
        title_font=dict(size=16, color="#fff"),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Plotly map saved → {output_path}  (open in any browser)")
    return fig


# ──────────────────────────────────────────────
# 3. BONUS — CLUSTER SUMMARY BAR CHART
# ──────────────────────────────────────────────

def build_cluster_barchart(df: pd.DataFrame, output_path: str = "cluster_barchart.html"):
    """
    Horizontal bar chart of top 30 clusters by mean AQI.
    Bars coloured by severity. Good for the report / presentation.
    """
    poll_col = "_poll_value"

    summary = (
        df[df["cluster"] >= 0]
        .groupby(["cluster", "severity"])
        .agg(
            mean_aqi=(poll_col, "mean"),
            stations=("lat", "count"),
        )
        .reset_index()
        .sort_values("mean_aqi", ascending=False)
        .head(30)
    )

    summary["label"] = summary.apply(
        lambda r: f"Cluster {r['cluster']}  ({r['stations']} stations)", axis=1
    )

    fig = px.bar(
        summary,
        x="mean_aqi",
        y="label",
        orientation="h",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"severity": ["CRITICAL","HIGH","MODERATE","LOW"]},
        text="mean_aqi",
        title="Top 30 Pollution Clusters by Mean AQI",
        labels={"mean_aqi": "Mean AQI", "label": ""},
    )

    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Mean AQI",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#111",
        plot_bgcolor="#1a1a1a",
        font_color="#ddd",
        title_font=dict(size=16, color="#fff"),
        legend_title="Severity",
        margin=dict(l=240, r=40, t=60, b=40),
        height=700,
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Bar chart saved → {output_path}")
    return fig


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data(INPUT_CSV)

    # 1. Folium interactive map (best for sharing / embedding)
    build_folium_map(df, FOLIUM_OUTPUT)

    # 2. Plotly interactive map (best for Jupyter / presentations)
    build_plotly_map(df, PLOTLY_OUTPUT)

    # 3. Bonus bar chart of top clusters
    build_cluster_barchart(df, "cluster_barchart.html")

    print("\nAll done. Open the .html files in your browser.")
    print(f"  {FOLIUM_OUTPUT}         ← dark map, toggleable layers, heatmap")
    print(f"  {PLOTLY_OUTPUT}   ← zoomable, hover tooltips, legend filter")
    print(f"  cluster_barchart.html         ← top 30 clusters ranked by AQI")