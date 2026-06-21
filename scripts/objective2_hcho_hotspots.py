"""
=============================================================================
Objective 2: Identification of HCHO Hotspots Over India Using Satellite Data
=============================================================================

This script implements the complete pipeline for:
1. Loading and processing TROPOMI HCHO monthly GeoTIFFs
2. Computing annual mean HCHO map over India
3. Computing biomass burning season HCHO composite
4. Identifying HCHO hotspots using statistical thresholds (Z-score)
5. Extracting city-level HCHO time series
6. Integrating VIIRS/MODIS fire count data
7. Analyzing correlation between fire activity and HCHO levels
8. Identifying major source regions (Indo-Gangetic Plain, forest fire zones)
9. Generating spatial visualizations and interactive maps

Datasets Used:
- Sentinel-5P TROPOMI HCHO monthly GeoTIFFs (12 months, 2024)
- VIIRS JPSS-1 fire detection data (2024, India)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import folium

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================
# Resolve paths relative to the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Raw data directories
HCHO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "tropomi", "HCHO")
FIRMS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "firms")

# Processed directory
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Outputs directories
MAPS_DIR = os.path.join(PROJECT_ROOT, "outputs", "maps")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# City coordinates (same 28 CPCB cities)
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

# Burning season months
BURNING_SEASON_MONTHS = [10, 11, 12, 1]  # Oct-Jan (stubble + winter)
FOREST_FIRE_MONTHS = [3, 4, 5]           # Mar-May (forest fires)


# =============================================================================
# STEP 1: Load HCHO Monthly GeoTIFFs
# =============================================================================
def load_hcho_monthly_rasters():
    """Load all 12 monthly HCHO GeoTIFFs into a stack."""
    print("=" * 60)
    print("STEP 1: Loading HCHO monthly GeoTIFFs...")

    hcho_files = []
    for month in range(1, 13):
        f = os.path.join(HCHO_DIR, f"HCHO_2024_Month_{month}.tif")
        if os.path.exists(f):
            hcho_files.append(f)

    print(f"  Found {len(hcho_files)} monthly HCHO files")

    hcho_stack = []
    transform = None
    crs = None

    for f in sorted(hcho_files):
        with rasterio.open(f) as src:
            data = src.read(1)
            data = np.where(data < 0, np.nan, data)  # Remove invalid values
            hcho_stack.append(data)
            if transform is None:
                transform = src.transform
                crs = src.crs

    print(f"  Raster shape: {hcho_stack[0].shape}")
    return hcho_stack, transform, crs


# =============================================================================
# STEP 2: Compute Annual Mean HCHO Map
# =============================================================================
def compute_annual_hcho(hcho_stack):
    """Compute annual mean HCHO from the monthly stack."""
    print("=" * 60)
    print("STEP 2: Computing annual mean HCHO map...")

    annual_hcho = np.nanmean(np.stack(hcho_stack), axis=0)
    print(f"  Annual HCHO shape: {annual_hcho.shape}")
    print(f"  Value range: [{np.nanmin(annual_hcho):.6f}, {np.nanmax(annual_hcho):.6f}]")

    plt.figure(figsize=(12, 8))
    plt.imshow(annual_hcho, cmap="YlOrRd")
    plt.colorbar(label="HCHO Column Density (mol/m²)", shrink=0.7)
    plt.title("Annual Mean HCHO Over India (2024)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "India_HCHO_Annual_Mean_2024.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/India_HCHO_Annual_Mean_2024.png")

    return annual_hcho


# =============================================================================
# STEP 3: Compute Burning Season HCHO Composite
# =============================================================================
def compute_burning_season_hcho(hcho_stack):
    """Compute HCHO composite for biomass burning season (Oct-Jan)."""
    print("=" * 60)
    print("STEP 3: Computing burning season HCHO composite...")

    # Month indices (0-based): Oct=9, Nov=10, Dec=11, Jan=0
    burn_indices = [m - 1 for m in BURNING_SEASON_MONTHS]
    burn_arrays = [hcho_stack[i] for i in burn_indices if i < len(hcho_stack)]

    burn_hcho = np.nanmean(np.stack(burn_arrays), axis=0)

    plt.figure(figsize=(12, 8))
    plt.imshow(burn_hcho, cmap="YlOrRd")
    plt.colorbar(label="HCHO Column Density (mol/m²)", shrink=0.7)
    plt.title("Biomass Burning Season HCHO Hotspots (Oct-Jan 2024)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "India_HCHO_Burning_Season_2024.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/India_HCHO_Burning_Season_2024.png")

    # Also compute forest fire season (Mar-May)
    forest_indices = [m - 1 for m in FOREST_FIRE_MONTHS]
    forest_arrays = [hcho_stack[i] for i in forest_indices if i < len(hcho_stack)]
    forest_hcho = np.nanmean(np.stack(forest_arrays), axis=0)

    plt.figure(figsize=(12, 8))
    plt.imshow(forest_hcho, cmap="YlOrRd")
    plt.colorbar(label="HCHO Column Density (mol/m²)", shrink=0.7)
    plt.title("Forest Fire Season HCHO (Mar-May 2024)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "India_HCHO_Forest_Fire_Season_2024.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/India_HCHO_Forest_Fire_Season_2024.png")

    return burn_hcho, forest_hcho


# =============================================================================
# STEP 4: Hotspot Detection Using Z-Score
# =============================================================================
def detect_hotspots(hcho_stack, annual_hcho):
    """Identify HCHO hotspots using Z-score > 2 threshold."""
    print("=" * 60)
    print("STEP 4: Detecting HCHO hotspots using Z-score method...")

    stacked = np.stack(hcho_stack)
    hcho_std = np.nanstd(stacked, axis=0)

    # Avoid division by zero
    hcho_std[hcho_std == 0] = np.nan

    # Z-score for each month relative to annual baseline
    hotspot_months = []
    for i, month_data in enumerate(hcho_stack):
        z_score = (month_data - annual_hcho) / hcho_std
        hotspot_mask = z_score > 2.0
        n_hotspot_pixels = np.nansum(hotspot_mask)
        hotspot_months.append({
            "month": i + 1,
            "hotspot_pixels": int(n_hotspot_pixels),
            "max_zscore": float(np.nanmax(z_score)),
        })
        print(f"  Month {i+1:2d}: {n_hotspot_pixels:6d} hotspot pixels (Z > 2)")

    # Compute hotspot frequency map (how many months each pixel is a hotspot)
    hotspot_freq = np.zeros_like(annual_hcho)
    for month_data in hcho_stack:
        z = (month_data - annual_hcho) / hcho_std
        hotspot_freq += (z > 2.0).astype(float)

    plt.figure(figsize=(12, 8))
    plt.imshow(hotspot_freq, cmap="hot_r")
    plt.colorbar(label="Number of months with Z > 2", shrink=0.7)
    plt.title("HCHO Hotspot Frequency Map (2024)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "India_HCHO_Hotspot_Frequency_2024.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/India_HCHO_Hotspot_Frequency_2024.png")

    return pd.DataFrame(hotspot_months), hotspot_freq


# =============================================================================
# STEP 5: Extract City-Level HCHO Time Series
# =============================================================================
def extract_city_hcho():
    """Extract monthly HCHO values at each city from GeoTIFFs."""
    print("=" * 60)
    print("STEP 5: Extracting city-level HCHO time series...")

    results = []
    for month in range(1, 13):
        tif_file = os.path.join(HCHO_DIR, f"HCHO_2024_Month_{month}.tif")

        with rasterio.open(tif_file) as src:
            for city, (lat, lon) in CITY_COORDS.items():
                row, col = src.index(lon, lat)
                value = src.read(1)[row, col]
                results.append({"City": city, "Month": month, "HCHO": float(value)})

    city_hcho = pd.DataFrame(results)

    # Rank cities by annual mean HCHO
    city_ranking = (
        city_hcho.groupby("City")["HCHO"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n  Top 10 cities by mean HCHO:")
    print(city_ranking.head(10).to_string())

    city_hcho.to_csv(os.path.join(PROCESSED_DIR, "city_hcho_2024.csv"), index=False)
    print("\n  Saved: data/processed/city_hcho_2024.csv")

    return city_hcho, city_ranking


# =============================================================================
# STEP 6: Integrate VIIRS Fire Data
# =============================================================================
def load_fire_data():
    """Load VIIRS fire detection data and compute monthly fire counts."""
    print("=" * 60)
    print("STEP 6: Loading VIIRS fire data...")

    fire_df = pd.read_csv(os.path.join(FIRMS_DIR, "viirs-jpss1_2024_India.csv"))
    fire_df["acq_date"] = pd.to_datetime(fire_df["acq_date"])
    fire_df["Month"] = fire_df["acq_date"].dt.month

    monthly_fire = (
        fire_df.groupby("Month")
        .size()
        .reset_index(name="Fire_Count")
    )

    print(f"  Total fire detections: {len(fire_df)}")
    print(f"  Monthly fire counts:\n{monthly_fire.to_string()}")

    return fire_df, monthly_fire


# =============================================================================
# STEP 7: Fire-HCHO Correlation Analysis
# =============================================================================
def analyze_fire_hcho_correlation(city_hcho, monthly_fire):
    """Analyze correlation between monthly fire activity and HCHO levels."""
    print("=" * 60)
    print("STEP 7: Analyzing fire-HCHO correlation...")

    # Merge HCHO with fire data
    merged = city_hcho.merge(monthly_fire, on="Month", how="left")

    # Compute correlation
    corr = merged["HCHO"].corr(merged["Fire_Count"])
    print(f"  Pearson correlation (HCHO vs Fire_Count): {corr:.4f}")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(merged["Fire_Count"], merged["HCHO"], alpha=0.5, edgecolors="k", linewidth=0.3)
    plt.xlabel("Monthly Fire Count (VIIRS)", fontsize=12)
    plt.ylabel("HCHO Column Density (mol/m²)", fontsize=12)
    plt.title(f"Fire Activity vs HCHO Concentration (r = {corr:.3f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fire_HCHO_Correlation.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/Fire_HCHO_Correlation.png")

    return corr


# =============================================================================
# STEP 8: Fire Density Maps (Major Source Regions)
# =============================================================================
def plot_fire_source_regions(fire_df):
    """Generate fire density hexbin maps to identify major source regions."""
    print("=" * 60)
    print("STEP 8: Identifying major fire source regions...")

    # Full year fire density
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    hb1 = axes[0].hexbin(
        fire_df["longitude"], fire_df["latitude"],
        gridsize=60, C=fire_df["frp"],
        reduce_C_function=sum, cmap="hot", mincnt=1
    )
    plt.colorbar(hb1, ax=axes[0], label="Total FRP (MW)")
    axes[0].set_title("Annual Fire Radiative Power (2024)", fontsize=13)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Burning season (Oct-Nov stubble burning)
    burning_fire = fire_df[fire_df["Month"].isin([10, 11])]
    hb2 = axes[1].hexbin(
        burning_fire["longitude"], burning_fire["latitude"],
        gridsize=60, C=burning_fire["frp"],
        reduce_C_function=sum, cmap="hot", mincnt=1
    )
    plt.colorbar(hb2, ax=axes[1], label="Total FRP (MW)")
    axes[1].set_title("Stubble Burning Season (Oct-Nov 2024)", fontsize=13)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fire_Source_Regions_2024.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/Fire_Source_Regions_2024.png")

    # Forest fire season (Mar-May)
    plt.figure(figsize=(10, 8))
    forest_fire = fire_df[fire_df["Month"].isin(FOREST_FIRE_MONTHS)]
    hb = plt.hexbin(
        forest_fire["longitude"], forest_fire["latitude"],
        gridsize=60, C=forest_fire["frp"],
        reduce_C_function=sum, cmap="hot", mincnt=1
    )
    plt.colorbar(hb, label="Fire Radiative Power (FRP)")
    plt.title("Forest Fire Season Hotspots (Mar-May 2024)", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Forest_Fire_Hotspots_2024.png"), dpi=300)
    plt.close()
    print("  Saved: outputs/figures/Forest_Fire_Hotspots_2024.png")


# =============================================================================
# STEP 9: Interactive HCHO Hotspot Map
# =============================================================================
def generate_hcho_map(city_hcho):
    """Generate an interactive Folium map of HCHO hotspots in outputs/maps."""
    print("=" * 60)
    print("STEP 9: Generating interactive HCHO hotspot map...")

    hcho_hotspot = city_hcho.groupby("City")["HCHO"].mean().reset_index()
    hcho_hotspot["lat"] = hcho_hotspot["City"].map(
        {k: v[0] for k, v in CITY_COORDS.items()}
    )
    hcho_hotspot["lon"] = hcho_hotspot["City"].map(
        {k: v[1] for k, v in CITY_COORDS.items()}
    )

    m = folium.Map(location=[22, 79], zoom_start=5)

    for _, row in hcho_hotspot.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["HCHO"] * 100000,
            popup=f"{row['City']}<br>Mean HCHO={row['HCHO']:.6f}",
            color="red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    output_path = os.path.join(MAPS_DIR, "India_HCHO_Hotspot_Map_2024.html")
    m.save(output_path)
    print(f"  Saved: {output_path}")

    return m


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OBJECTIVE 2: HCHO Hotspot Identification Over India")
    print("=" * 60)

    # Step 1: Load HCHO rasters
    hcho_stack, transform, crs = load_hcho_monthly_rasters()

    # Step 2: Annual mean HCHO
    annual_hcho = compute_annual_hcho(hcho_stack)

    # Step 3: Burning season composite
    burn_hcho, forest_hcho = compute_burning_season_hcho(hcho_stack)

    # Step 4: Hotspot detection
    hotspot_stats, hotspot_freq = detect_hotspots(hcho_stack, annual_hcho)

    # Step 5: City-level HCHO time series
    city_hcho, city_ranking = extract_city_hcho()

    # Step 6: Load fire data
    fire_df, monthly_fire = load_fire_data()

    # Step 7: Fire-HCHO correlation
    corr = analyze_fire_hcho_correlation(city_hcho, monthly_fire)

    # Step 8: Fire source regions
    plot_fire_source_regions(fire_df)

    # Step 9: Interactive HCHO map
    generate_hcho_map(city_hcho)

    print("\n" + "=" * 60)
    print("OBJECTIVE 2 COMPLETE")
    print("=" * 60)
