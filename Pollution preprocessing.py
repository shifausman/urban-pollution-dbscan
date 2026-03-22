"""
pollution_preprocessing.py
===========================
Step 2 of the DBSCAN Urban Pollution Pipeline:
  - Load raw pollution CSV
  - Handle missing values
  - Validate lat/lon coordinates
  - Remove outliers (IQR + Z-score)
  - Scale features for DBSCAN
  - Save a clean, cluster-ready CSV

Compatible datasets:
  • Kaggle "Global Air Quality Index by City & Coordinates"
  • OpenAQ CSV export
  • EPA AQS daily/annual CSV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV pollution dataset.
    Adjust column renaming below to match your file's headers.
    """
    df = pd.read_csv(filepath)

    # ── Normalise column names to lowercase + underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    print(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────
# 2. STANDARDISE COLUMN NAMES
# ──────────────────────────────────────────────

# Map YOUR dataset's actual column names → standard internal names.
# Edit the keys to match what's in your CSV file.
COLUMN_MAP = {
    # Kaggle "Global AQI by City & Coordinates" example:
    "lat":        "lat",
    "lng":        "lon",         # some files use 'lng' instead of 'lon'
    "aqi_value":  "aqi",
    "pm2.5_aqi_value": "pm25",
    "no2_aqi_value":   "no2",
    "ozone_aqi_value": "o3",
    "co_aqi_value":    "co",
    "city":       "city",
    "country":    "country",

    # OpenAQ CSV example (uncomment if using OpenAQ):
    # "latitude":  "lat",
    # "longitude": "lon",
    # "value":     "pm25",
    # "location":  "city",
}

POLLUTANT_COLS = ["aqi", "pm25", "no2", "o3", "co"]   # keep only the ones that exist


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to internal standard names, drop unmapped extras."""
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    return df


# ──────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ──────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      • Drop rows with no lat/lon  → can't cluster without coordinates
      • Drop rows where ALL pollutant columns are null
      • Fill remaining pollutant NaNs with column median (conservative imputation)
    """
    before = len(df)

    # 3a. Must have coordinates
    df = df.dropna(subset=["lat", "lon"])
    print(f"[Missing] Dropped {before - len(df):,} rows with null lat/lon")

    # 3b. Must have at least one pollutant reading
    existing_poll = [c for c in POLLUTANT_COLS if c in df.columns]
    df = df.dropna(subset=existing_poll, how="all")
    print(f"[Missing] Dropped rows with no pollutant data → {len(df):,} rows remain")

    # 3c. Median-fill remaining pollutant NaNs
    for col in existing_poll:
        n_null = df[col].isna().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[Missing] Filled {n_null} NaNs in '{col}' with median={median_val:.2f}")

    return df


# ──────────────────────────────────────────────
# 4. VALIDATE LAT / LON
# ──────────────────────────────────────────────

def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valid ranges:
      latitude  : -90  to  90
      longitude : -180 to 180

    Also catches:
      • Both lat & lon == 0  (null-island, almost certainly a data error)
      • Non-numeric strings snuck in as coordinates
    """
    # 4a. Coerce to numeric (bad strings → NaN, then drop)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    before = len(df)

    # 4b. Range check
    valid_range = (
        df["lat"].between(-90, 90) &
        df["lon"].between(-180, 180)
    )
    df = df[valid_range]
    print(f"[Coords] Removed {before - len(df):,} rows outside valid lat/lon range")

    # 4c. Null-island check (0.0, 0.0) — likely placeholder
    null_island = (df["lat"] == 0.0) & (df["lon"] == 0.0)
    n_ni = null_island.sum()
    if n_ni:
        df = df[~null_island]
        print(f"[Coords] Removed {n_ni} null-island (0,0) entries")

    # 4d. Duplicate (lat, lon) rows: keep one with the highest AQI reading
    poll_col = next((c for c in ["aqi", "pm25"] if c in df.columns), None)
    if poll_col:
        df = df.sort_values(poll_col, ascending=False)
        df = df.drop_duplicates(subset=["lat", "lon"], keep="first")
        print(f"[Coords] Deduplicated to {len(df):,} unique stations")

    return df


# ──────────────────────────────────────────────
# 5. REMOVE OUTLIERS
# ──────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """
    Two methods available — pick one:

    'iqr'     : flag values outside  Q1 - 1.5*IQR … Q3 + 1.5*IQR
                Robust, recommended for skewed pollution distributions.

    'zscore'  : flag values where |z| > 3
                Works better when data is roughly normal.

    Extreme AQI (>500) and negative pollutant values are always dropped
    regardless of method, as they are physically impossible.
    """
    existing_poll = [c for c in POLLUTANT_COLS if c in df.columns]
    before = len(df)

    # 5a. Hard physical limits
    for col in existing_poll:
        df = df[df[col] >= 0]           # no negative pollution values
    if "aqi" in df.columns:
        df = df[df["aqi"] <= 500]       # AQI scale tops at 500

    print(f"[Outliers] Hard limits removed {before - len(df):,} impossible values")
    before = len(df)

    # 5b. Statistical outlier removal
    mask = pd.Series(True, index=df.index)  # True = keep

    for col in existing_poll:
        if method == "iqr":
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            col_mask = df[col].between(lower, upper)

        elif method == "zscore":
            z = (df[col] - df[col].mean()) / df[col].std()
            col_mask = z.abs() <= 3

        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        n_flagged = (~col_mask).sum()
        print(f"[Outliers] '{col}' ({method}): flagged {n_flagged} rows")
        mask &= col_mask

    df = df[mask]
    print(f"[Outliers] Removed {before - len(df):,} total outlier rows → {len(df):,} remain")
    return df


# ──────────────────────────────────────────────
# 6. SCALE FEATURES FOR DBSCAN
# ──────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScaler: zero mean, unit variance.

    NOTE: For pure geographic DBSCAN (clustering only by lat/lon with
    haversine distance), do NOT scale lat/lon — pass raw radians to
    sklearn with metric='haversine'. Scale only if you're combining
    lat/lon WITH pollutant values in a multi-feature cluster.

    This function scales the pollutant columns only.
    """
    existing_poll = [c for c in POLLUTANT_COLS if c in df.columns]
    scaler = StandardScaler()
    df[existing_poll] = scaler.fit_transform(df[existing_poll])
    print(f"[Scale] Scaled columns: {existing_poll}")
    return df, scaler


# ──────────────────────────────────────────────
# 7. SUMMARY REPORT
# ──────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    existing_poll = [c for c in POLLUTANT_COLS if c in df.columns]
    print("\n" + "="*50)
    print("CLEAN DATASET SUMMARY")
    print("="*50)
    print(f"  Rows              : {len(df):,}")
    print(f"  Lat range         : {df['lat'].min():.4f} → {df['lat'].max():.4f}")
    print(f"  Lon range         : {df['lon'].min():.4f} → {df['lon'].max():.4f}")
    for col in existing_poll:
        print(f"  {col:<8} range  : {df[col].min():.2f} → {df[col].max():.2f}  "
              f"(mean={df[col].mean():.2f})")
    print(f"  Null count        : {df.isnull().sum().sum()}")
    print("="*50 + "\n")


# ──────────────────────────────────────────────
# 8. FULL PIPELINE
# ──────────────────────────────────────────────

def preprocess_pipeline(
    filepath: str,
    outlier_method: str = "iqr",
    scale: bool = False,              # set True only for multi-feature DBSCAN
    output_path: str = "pollution_clean.csv"
) -> pd.DataFrame:

    print(f"\n{'='*50}\nStarting preprocessing: {filepath}\n{'='*50}")

    df = load_data(filepath)
    df = rename_columns(df)
    df = handle_missing(df)
    df = validate_coordinates(df)
    df = remove_outliers(df, method=outlier_method)

    scaler = None
    if scale:
        df, scaler = scale_features(df)

    print_summary(df)

    df.to_csv(output_path, index=False)
    print(f"Saved clean data → {output_path}")

    return df   # (also return scaler if you need to inverse_transform later)


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Change this path to your downloaded dataset
    INPUT_FILE  = "air_quality_data.csv"
    OUTPUT_FILE = "pollution_clean.csv"

    df_clean = preprocess_pipeline(
        filepath=INPUT_FILE,
        outlier_method="iqr",   # or "zscore"
        scale=False,            # True for multi-feature DBSCAN
        output_path=OUTPUT_FILE
    )

    # Quick peek
    print(df_clean[["lat", "lon", "aqi", "pm25"]].head(10))