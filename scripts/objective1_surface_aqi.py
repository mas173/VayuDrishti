"""
=============================================================================
Objective 1: Development of Surface Air Quality Index (AQI) Using Satellite Data
=============================================================================

This script implements the complete pipeline for:
1. Building a multi-pollutant surface measurement database from CPCB ground stations
2. Integrating ERA5 reanalysis meteorological data
3. Integrating VIIRS fire count data
4. Extracting satellite columnar pollutant data (HCHO, NO2, SO2, CO) from TROPOMI GeoTIFFs
5. Merging all datasets into a unified monthly multi-pollutant database
6. Training a Deep Learning model (Dense NN) to predict surface PM2.5
7. Computing AQI from predicted PM2.5 using CPCB breakpoints
8. Generating interactive spatial AQI maps over India using Folium

Datasets Used:
- CPCB 15-min city-wise air quality data (28 cities, 2024)
- ERA5 reanalysis (u10, v10, t2m, d2m, sp, blh)
- VIIRS JPSS-1 fire detection data (2024, India)
- Sentinel-5P TROPOMI monthly GeoTIFFs (HCHO, NO2, SO2, CO)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import joblib
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================
# Resolve paths relative to the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Raw data directories
CPCB_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "cpcb")
ERA5_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "era5")
FIRMS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "firms")
TROPOMI_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "tropomi")

# Processed and Model directories
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Outputs directories
MAPS_DIR = os.path.join(PROJECT_ROOT, "outputs", "maps")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# City coordinates for all 28 CPCB monitoring cities
CITY_COORDS = {
    "Agartala": (23.8315, 91.2868),
    "Ahmedabad": (23.0225, 72.5714),
    "Amaravati": (16.5062, 80.6480),
    "Amritsar": (31.6340, 74.8723),
    "Baddi": (30.9578, 76.7914),
    "Bengaluru": (12.9716, 77.5946),
    "Bhubaneswar": (20.2961, 85.8245),
    "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707),
    "Dehradun": (30.3165, 78.0322),
    "Delhi": (28.6139, 77.2090),
    "Gangtok": (27.3389, 88.6065),
    "Guwahati": (26.1445, 91.7362),
    "Gurugram": (28.4595, 77.0266),
    "Hyderabad": (17.3850, 78.4867),
    "Imphal": (24.8170, 93.9368),
    "Jaipur": (26.9124, 75.7873),
    "Kohima": (25.6751, 94.1086),
    "Kolkata": (22.5726, 88.3639),
    "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777),
    "Nagpur": (21.1458, 79.0882),
    "Naharlagun": (27.1047, 93.6952),
    "Patna": (25.5941, 85.1376),
    "Raipur": (21.2514, 81.6296),
    "Shillong": (25.5788, 91.8933),
    "Srinagar": (34.0837, 74.7973),
    "Thiruvananthapuram": (8.5241, 76.9366),
}

# Columns to drop (too many missing values across cities)
SPARSE_COLUMNS = [
    "O Xylene (µg/m³)",
    "Xylene (µg/m³)",
    "Toluene (µg/m³)",
    "Eth-Benzene (µg/m³)",
    "MP-Xylene (µg/m³)",
    "VWS (m/s)",
]

# Satellite pollutant folder names
SATELLITE_POLLUTANTS = {
    "HCHO": {"folder": "HCHO", "prefix": "HCHO_2024_Month_"},
    "NO2":  {"folder": "NO2",  "prefix": "NO2_2024_Month_"},
    "SO2":  {"folder": "SO2",  "prefix": "SO2_2024_Month_"},
    "CO":   {"folder": "CO",   "prefix": "CO_2024_Month_"},
}


# =============================================================================
# STEP 1: Load and Combine CPCB Ground Station Data (15-min → Daily)
# =============================================================================
def load_cpcb_data():
    """Load all 28 city CPCB 15-min CSVs from data/raw/cpcb, combine, and aggregate to daily."""
    print("=" * 60)
    print("STEP 1: Loading CPCB ground station data...")

    files = glob.glob(os.path.join(CPCB_DIR, "City_wise_raw_data_15Min_2024_*_15Min.csv"))
    print(f"  Found {len(files)} city CSV files")

    dfs = []
    for file in files:
        city = (os.path.basename(file)
                .replace("City_wise_raw_data_15Min_2024_", "")
                .replace("_15Min.csv", ""))
        df = pd.read_csv(file)
        df["City"] = city
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined shape (15-min): {combined_df.shape}")

    # Convert to daily averages
    combined_df["Timestamp"] = pd.to_datetime(combined_df["Timestamp"])
    combined_df["Date"] = combined_df["Timestamp"].dt.date

    daily_df = (
        combined_df
        .groupby(["City", "Date"])
        .mean(numeric_only=True)
        .reset_index()
    )
    print(f"  Daily aggregated shape: {daily_df.shape}")

    return daily_df


# =============================================================================
# STEP 2: Clean CPCB Data
# =============================================================================
def clean_cpcb_data(daily_df):
    """Fill NaN with city median, drop sparse columns, add temporal features."""
    print("=" * 60)
    print("STEP 2: Cleaning CPCB data...")

    # Fill NaN with per-city median
    daily_df = daily_df.groupby("City").apply(
        lambda x: x.fillna(x.median(numeric_only=True)),
        include_groups=False
    ).reset_index(level=0)

    # Drop sparse columns
    cols_to_drop = [c for c in SPARSE_COLUMNS if c in daily_df.columns]
    daily_df = daily_df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} sparse columns")

    # Add temporal features
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df["month"] = daily_df["Date"].dt.month
    daily_df["dayofyear"] = daily_df["Date"].dt.dayofyear

    # Encode city as numeric
    le = LabelEncoder()
    daily_df["city_id"] = le.fit_transform(daily_df["City"])

    # Save city mapping for later use in models directory
    joblib.dump(dict(zip(le.classes_, le.transform(le.classes_))), os.path.join(MODELS_DIR, "city_mapping.pkl"))

    print(f"  Remaining NaN: {daily_df.isnull().sum().sum()}")
    print(f"  Clean shape: {daily_df.shape}")

    return daily_df


# =============================================================================
# STEP 3: Extract and Merge ERA5 Meteorological Data
# =============================================================================
def merge_era5_data(daily_df):
    """Extract ERA5 variables per city per day and merge with CPCB data."""
    print("=" * 60)
    print("STEP 3: Merging ERA5 meteorological data...")

    era5_file = os.path.join(ERA5_DIR, "data_stream-oper_stepType-instant.nc")
    data1 = xr.open_dataset(era5_file)
    print(f"  ERA5 variables: {list(data1.data_vars)}")

    era5_all = []
    for city, (lat, lon) in CITY_COORDS.items():
        city_ds = data1.sel(latitude=lat, longitude=lon, method="nearest")
        city_df = city_ds.to_dataframe().reset_index()

        city_daily = (
            city_df
            .groupby(city_df["valid_time"].dt.date)
            .agg({"u10": "mean", "v10": "mean", "t2m": "mean",
                   "d2m": "mean", "sp": "mean", "blh": "mean"})
            .reset_index()
        )
        city_daily["City"] = city
        era5_all.append(city_daily)

    era5_df = pd.concat(era5_all, ignore_index=True)
    era5_df.rename(columns={"valid_time": "Date"}, inplace=True)
    era5_df["Date"] = pd.to_datetime(era5_df["Date"])
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])

    merged_df = daily_df.merge(era5_df, on=["City", "Date"], how="left")
    print(f"  Merged shape: {merged_df.shape}")

    return merged_df


# =============================================================================
# STEP 4: Integrate VIIRS Fire Count Data
# =============================================================================
def merge_fire_data(merged_df):
    """Load VIIRS fire data, aggregate daily counts, merge into dataset."""
    print("=" * 60)
    print("STEP 4: Integrating VIIRS fire count data...")

    fire_df = pd.read_csv(os.path.join(FIRMS_DIR, "viirs-jpss1_2024_India.csv"))
    fire_df["acq_date"] = pd.to_datetime(fire_df["acq_date"])

    daily_fire = (
        fire_df.groupby("acq_date")
        .size()
        .reset_index(name="fire_count")
    )
    daily_fire.rename(columns={"acq_date": "Date"}, inplace=True)
    daily_fire["Date"] = pd.to_datetime(daily_fire["Date"])

    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    merged_df = merged_df.merge(daily_fire, on="Date", how="left")
    merged_df["fire_count"] = merged_df["fire_count"].fillna(0)

    print(f"  Shape after fire merge: {merged_df.shape}")

    # Add lagged PM2.5 feature
    merged_df = merged_df.sort_values(["City", "Date"])
    merged_df["pm25_lag1"] = merged_df.groupby("City")["PM2.5 (µg/m³)"].shift(1)
    merged_df["pm25_lag1"] = merged_df.groupby("City")["pm25_lag1"].transform(
        lambda x: x.fillna(x.median())
    )

    # Save the merged daily dataset to data/processed
    merged_df.to_csv(os.path.join(PROCESSED_DIR, "india_pm25_era5_2024.csv"), index=False)
    print("  Saved: data/processed/india_pm25_era5_2024.csv")

    return merged_df


# =============================================================================
# STEP 5: Extract Satellite Pollutant Data from GeoTIFFs
# =============================================================================
def extract_satellite_data():
    """Extract HCHO, NO2, SO2, CO values at each city for each month from GeoTIFFs."""
    print("=" * 60)
    print("STEP 5: Extracting satellite pollutant data from GeoTIFFs...")

    all_pollutant_dfs = {}

    for pollutant, config in SATELLITE_POLLUTANTS.items():
        results = []
        folder = os.path.join(TROPOMI_DIR, config["folder"])

        for month in range(1, 13):
            tif_file = os.path.join(folder, f"{config['prefix']}{month}.tif")

            if not os.path.exists(tif_file):
                print(f"  WARNING: {tif_file} not found, skipping")
                continue

            with rasterio.open(tif_file) as src:
                data = src.read(1)
                for city, (lat, lon) in CITY_COORDS.items():
                    try:
                        row, col = src.index(lon, lat)
                        value = float(data[row, col])
                        results.append({"City": city, "Month": month, pollutant: value})
                    except Exception:
                        pass

        df = pd.DataFrame(results)
        all_pollutant_dfs[pollutant] = df
        print(f"  {pollutant}: {df.shape[0]} records extracted")

    # Merge all pollutants
    satellite_db = all_pollutant_dfs["HCHO"]
    for pol in ["NO2", "SO2", "CO"]:
        satellite_db = satellite_db.merge(all_pollutant_dfs[pol], on=["City", "Month"])

    satellite_db.to_csv(os.path.join(PROCESSED_DIR, "satellite_multi_pollutant_database.csv"), index=False)
    print(f"  Satellite DB shape: {satellite_db.shape}")
    print("  Saved: data/processed/satellite_multi_pollutant_database.csv")

    return satellite_db


# =============================================================================
# STEP 6: Build Monthly Multi-Pollutant Database
# =============================================================================
def build_monthly_database(merged_df, satellite_db):
    """Aggregate CPCB daily data to monthly, merge with satellite data."""
    print("=" * 60)
    print("STEP 6: Building monthly multi-pollutant database...")

    required_cols = [
        "City", "Date", "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)",
        "NO2 (µg/m³)", "NOx (ppb)", "NH3 (µg/m³)", "SO2 (µg/m³)",
        "CO (mg/m³)", "Ozone (µg/m³)", "AT (°C)", "RH (%)",
        "WS (m/s)", "WD (deg)", "RF (mm)", "BP (mmHg)", "SR (W/mt2)",
    ]

    available_cols = [c for c in required_cols if c in merged_df.columns]
    surface_db = merged_df[available_cols].copy()
    surface_db["Date"] = pd.to_datetime(surface_db["Date"])
    surface_db["Month"] = surface_db["Date"].dt.month

    # Fill remaining NaN with column median
    numeric_cols = surface_db.select_dtypes(include="number").columns
    for col in numeric_cols:
        surface_db[col] = surface_db[col].fillna(surface_db[col].median())

    # Monthly aggregation
    monthly_surface = (
        surface_db
        .groupby(["City", "Month"])
        .mean(numeric_only=True)
        .reset_index()
    )

    # Merge with satellite data
    final_db = monthly_surface.merge(satellite_db, on=["City", "Month"], how="inner")

    # Fill any remaining NaN (numeric columns only)
    for col in final_db.select_dtypes(include="number").columns:
        final_db[col] = final_db[col].fillna(final_db[col].median())

    print(f"  Final multi-pollutant DB shape: {final_db.shape}")
    return final_db


# =============================================================================
# STEP 7: Train Deep Learning Model for PM2.5 Prediction
# =============================================================================
def train_dl_model(final_db):
    """Train a Dense Neural Network to predict PM2.5 from multi-pollutant features."""
    print("=" * 60)
    print("STEP 7: Training Deep Learning model...")

    # Lazy import to avoid forcing TF installation for non-DL users
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    target = "PM2.5 (µg/m³)"
    exclude_cols = ["City", "Month", target]
    feature_cols = [c for c in final_db.columns if c not in exclude_cols
                    and final_db[c].dtype != "object"]

    X = final_db[feature_cols]
    y = final_db[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    history = model.fit(
        X_train_scaled, y_train,
        epochs=200, batch_size=16,
        validation_split=0.2, verbose=0
    )

    # Evaluate
    y_pred = model.predict(X_test_scaled).flatten()
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  R² = {r2:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    # Save model and scaler to models directory
    model.save(os.path.join(MODELS_DIR, "PM25_DeepLearning_Model.keras"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "PM25_Scaler.pkl"))
    print("  Saved: models/PM25_DeepLearning_Model.keras, models/PM25_Scaler.pkl")

    # Validation scatter plot in outputs/figures
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidth=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", linewidth=2, label="1:1 line")
    plt.xlabel("Actual PM2.5 (µg/m³)")
    plt.ylabel("Predicted PM2.5 (µg/m³)")
    plt.title(f"Actual vs Predicted PM2.5 (R²={r2:.3f}, MAE={mae:.1f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pm25_validation_scatter.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/pm25_validation_scatter.png")

    return model, scaler, X.columns.tolist()


# =============================================================================
# STEP 8: Compute AQI from Predicted PM2.5
# =============================================================================
def pm25_to_aqi(pm):
    """Convert PM2.5 concentration to AQI using CPCB breakpoints."""
    if pm <= 30:
        return (50 / 30) * pm
    elif pm <= 60:
        return 50 + (pm - 30) * (50 / 30)
    elif pm <= 90:
        return 100 + (pm - 60) * (100 / 30)
    elif pm <= 120:
        return 200 + (pm - 90) * (100 / 30)
    elif pm <= 250:
        return 300 + (pm - 120) * (100 / 130)
    else:
        return 400 + (pm - 250) * (100 / 130)


def aqi_category(aqi):
    """Classify AQI value into CPCB category."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def compute_aqi(final_db, model, scaler, feature_cols):
    """Predict PM2.5 for all records and compute AQI."""
    print("=" * 60)
    print("STEP 8: Computing AQI from predicted PM2.5...")

    X_all = final_db[feature_cols]
    X_scaled = scaler.transform(X_all)
    final_db["Pred_PM25"] = model.predict(X_scaled).flatten()
    final_db["Error"] = final_db["Pred_PM25"] - final_db["PM2.5 (µg/m³)"]
    final_db["AQI"] = final_db["Pred_PM25"].apply(pm25_to_aqi)
    final_db["AQI_Category"] = final_db["AQI"].apply(aqi_category)

    # Add coordinates
    final_db["Latitude"] = final_db["City"].map(lambda x: CITY_COORDS[x][0])
    final_db["Longitude"] = final_db["City"].map(lambda x: CITY_COORDS[x][1])

    print(f"  AQI distribution:\n{final_db['AQI_Category'].value_counts()}")

    # Error distribution plot in outputs/figures
    plt.figure(figsize=(8, 5))
    plt.hist(final_db["Error"], bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Prediction Error (µg/m³)")
    plt.ylabel("Frequency")
    plt.title("PM2.5 Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pm25_error_distribution.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/pm25_error_distribution.png")

    # Save final dataset in data/processed
    output_path = os.path.join(PROCESSED_DIR, "Final_Surface_AQI_Dataset_2024.csv")
    final_db.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return final_db


# =============================================================================
# STEP 9: Generate Interactive AQI Map
# =============================================================================
def generate_aqi_map(final_db, month=12):
    """Generate an interactive Folium AQI map for a given month in outputs/maps."""
    print("=" * 60)
    print(f"STEP 9: Generating AQI map for month {month}...")

    def get_color(aqi):
        if aqi <= 50:
            return "green"
        elif aqi <= 100:
            return "yellow"
        elif aqi <= 200:
            return "orange"
        elif aqi <= 300:
            return "red"
        elif aqi <= 400:
            return "purple"
        else:
            return "darkred"

    map_df = final_db[final_db["Month"] == month]

    m = folium.Map(location=[22.5, 80], zoom_start=5)

    for _, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=10,
            color=get_color(row["AQI"]),
            fill=True,
            fill_color=get_color(row["AQI"]),
            fill_opacity=0.8,
            popup=(
                f"City: {row['City']}<br>"
                f"AQI: {row['AQI']:.1f}<br>"
                f"Category: {row['AQI_Category']}<br>"
                f"PM2.5: {row['Pred_PM25']:.1f}"
            ),
        ).add_to(m)

    output_path = os.path.join(MAPS_DIR, f"India_Surface_AQI_Month{month}_2024.html")
    m.save(output_path)
    print(f"  Saved: {output_path}")

    return m


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OBJECTIVE 1: Surface AQI Development Using Satellite Data")
    print("=" * 60)

    # Step 1-2: Load and clean CPCB data
    daily_df = load_cpcb_data()
    daily_df = clean_cpcb_data(daily_df)

    # Step 3: Merge ERA5 meteorological data
    merged_df = merge_era5_data(daily_df)

    # Step 4: Merge fire count data
    merged_df = merge_fire_data(merged_df)

    # Step 5: Extract satellite data
    satellite_db = extract_satellite_data()

    # Step 6: Build monthly multi-pollutant database
    final_db = build_monthly_database(merged_df, satellite_db)

    # Step 7: Train DL model
    model, scaler, feature_cols = train_dl_model(final_db)

    # Step 8: Compute AQI
    final_db = compute_aqi(final_db, model, scaler, feature_cols)

    # Step 9: Generate AQI maps for key months
    for month in [1, 6, 10, 12]:
        generate_aqi_map(final_db, month=month)

    print("\n" + "=" * 60)
    print("OBJECTIVE 1 COMPLETE")
    print("=" * 60)
