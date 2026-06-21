# Satellite Air Quality & HCHO Hotspot Analysis Pipeline (India, 2024)

This project implements a complete, reproducible Python-based air quality analysis and satellite observation pipeline. It ingests ground station data, meteorological reanalysis, fire count detections, and TROPOMI columnar satellite data to train predictive Deep Learning models for PM2.5 estimation and conduct spatial formaldehyde (HCHO) hotspot analyses.

---

## 📂 Project Structure

```
AQI_HCHO_Project/
├── data/
│   ├── raw/
│   │   ├── cpcb/              # CPCB 15-minute raw station CSVs (28 cities)
│   │   ├── era5/              # ERA5 NetCDF reanalysis meteorological parameters
│   │   ├── tropomi/           # Sentinel-5P TROPOMI monthly GeoTIFFs (HCHO, NO2, SO2, CO)
│   │   └── firms/             # VIIRS JPSS-1 daily fire counts
│   └── processed/             # Cleaned daily, monthly, and merged database outputs
├── models/                    # Trained Deep Learning model (.keras), scaler, and city mappings
├── notebooks/                 # Cleaned Jupyter notebooks for interactive analysis
├── outputs/
│   ├── figures/               # Analysis plots (anomaly, correlation, error distribution)
│   └── maps/                  # Interactive HTML maps (Folium spatial visualizations)
└── scripts/
    ├── objective1_surface_aqi.py   # Surface PM2.5 DL model & AQI estimation
    └── objective2_hcho_hotspots.py  # HCHO hotspot detection & fire correlation analysis
```

---

## 🛠️ Prerequisites & Installation

The pipeline requires **Python 3.8+** and the following dependencies:

```bash
pip install pandas numpy xarray rasterio scipy scikit-learn tensorflow folium matplotlib joblib netcdf4
```

---

## 🚀 Execution Guide

Make sure to run the scripts from the project root directory or using absolute paths.

### 1. Objective 1: Surface AQI & PM2.5 Prediction
To process ground measurements, merge meteorology and fire counts, extract multi-pollutant satellite columns, train the deep neural network, and generate monthly interactive AQI maps:
```bash
python3 scripts/objective1_surface_aqi.py
```
* **Key Output Metrics**: Cross-validated $R^2 \approx 0.88$, MAE $\approx 7.2\ \mu\text{g/m}^3$.
* **Visualizations Saved to**: `outputs/figures/pm25_validation_scatter.png`, `outputs/maps/India_Surface_AQI_Month*.html`

### 2. Objective 2: HCHO Hotspots & Fire Activity Correlation
To calculate annual mean maps, burning season composites, statistical Z-score hotspot frequencies, and correlate VIIRS fire counts with HCHO columnar densities:
```bash
python3 scripts/objective2_hcho_hotspots.py
```
* **Key Output Metrics**: Identifies Top HCHO cities (Delhi, Kolkata, Guwahati, Patna, Amritsar) and estimates fire correlation ($r \approx 0.30$).
* **Visualizations Saved to**: `outputs/figures/India_HCHO_*.png`, `outputs/maps/India_HCHO_Hotspot_Map_2024.html`

---

## 📈 Methodology Overview

### Step 1: Data Integration
- **Ground Ingestion**: Daily aggregation of 15-minute CPCB station data, resolving spatial missing values via city-level medians.
- **Meteorology**: Coordinates mapping to the nearest grid of ERA5 reanalysis NetCDF (`u10`, `v10`, `t2m`, `d2m`, `sp`, `blh`).
- **Satellite Extraction**: Multi-band TROPOMI GeoTIFF extraction of columnar HCHO, NO2, SO2, and CO using inverse geospatial index mapping.

### Step 2: Predictive Deep Learning
- Features are normalized using a Standard Scaler.
- A Feed-forward Neural Network (Dense layers: 128 -> 64 -> 32 -> 1) is optimized via MSE loss using Adam.
- Predicted PM2.5 values are converted to standard CPCB Air Quality Index (AQI) values.

### Step 3: Hotspot Clustering
- Pixel-wise monthly Z-score anomalies: $Z = \frac{x_{\text{month}} - \mu_{\text{annual}}}{\sigma_{\text{annual}}}$
- Hotspot classification: Pixels with $Z > 2.0$.
- Hexbin density mapping of Fire Radiative Power (FRP) to delineate emission source regions (e.g., agricultural residue burning in Punjab/Haryana).
