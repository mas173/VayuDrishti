# Satellite Air Quality Index (AQI) Analysis and Hotspot Identification Pipeline

This repository contains a professional-grade, reproducible Python pipeline for surface PM2.5 estimation, CPCB-compliant Air Quality Index (AQI) calculation, and satellite-derived Formaldehyde (HCHO) hotspot identification over India for the year 2024. 

The pipeline integrates multi-source datasets including ground-based monitoring stations, atmospheric model reanalysis, satellite instruments, and thermal anomaly sensors.

---

## Project Objectives

1. **Surface AQI Development**: Combine ground-level measurements with meteorological variables, regional fire activity counts, and Sentinel-5P TROPOMI columnar pollutants (HCHO, NO2, SO2, CO) to build a multi-pollutant dataset. Train a Deep Learning model (Dense Neural Network) to predict monthly ground-level PM2.5 concentrations, calculate AQI using the Central Pollution Control Board (CPCB) guidelines, and visualize the spatial distribution.
2. **HCHO Hotspot Identification**: Analyze monthly TROPOMI HCHO column densities to compute annual averages and biomass/forest fire season composites. Identify statistically significant anomalies using a pixel-wise Z-score method, extract city-level time series, and correlate chemical observations with VIIRS fire detections to map major emission source regions.

---

## Directory Structure

The project follows a standard, modular layout:

```
AQI_HCHO_Project/
├── README.md                  # Comprehensive project documentation
├── data/
│   ├── raw/
│   │   ├── cpcb/              # CPCB 15-minute raw station CSVs (28 cities)
│   │   ├── era5/              # ERA5 NetCDF reanalysis meteorological parameters
│   │   ├── tropomi/           # Sentinel-5P TROPOMI monthly GeoTIFFs (HCHO, NO2, SO2, CO)
│   │   └── firms/             # VIIRS JPSS-1 daily fire counts
│   └── processed/             # Cleaned daily, monthly, and merged database outputs
├── models/                    # Trained Deep Learning model (.keras), scaler, and city mappings
├── notebooks/                 # Jupyter notebooks for exploratory data analysis
├── outputs/
│   ├── figures/               # Analysis plots (anomaly, correlation, error distribution)
│   └── maps/                  # Interactive HTML maps (Folium spatial visualizations)
└── scripts/
    ├── objective1_surface_aqi.py   # Surface PM2.5 DL model & AQI estimation
    └── objective2_hcho_hotspots.py  # HCHO hotspot detection & fire correlation analysis
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Installation Steps
Install all required libraries using the package manager:

```bash
pip install pandas numpy xarray rasterio scipy scikit-learn tensorflow folium matplotlib joblib netcdf4
```

---

## Technical Specifications

### Objective 1: Surface AQI Development (`objective1_surface_aqi.py`)

#### 1. Data Ingestion & Cleaning
* **Ground station data**: 15-minute measurements from 28 cities are aggregated to daily averages. Missing values are imputed using city-level medians.
* **Feature Pruning**: Sparse columns with high missing rates are dropped: `O Xylene`, `Xylene`, `Toluene`, `Eth-Benzene`, `MP-Xylene`, and `VWS`.
* **Spatial Matching**: City centroids are matched to the nearest grid cells in the ERA5 NetCDF file to extract meteorological variables:
  * `u10` & `v10` (10m Wind Components)
  * `t2m` (2m Temperature)
  * `d2m` (2m Dewpoint Temperature)
  * `sp` (Surface Pressure)
  * `blh` (Boundary Layer Height)
* **Fire Count Merging**: VIIRS fire count detections across India are aggregated daily and merged with CPCB data. A 1-day temporal lag feature for PM2.5 (`pm25_lag1`) is created to capture temporal dependency.
* **Satellite Column Extraction**: Columnar measurements of HCHO, NO2, SO2, and CO are extracted for each city coordinate from monthly TROPOMI GeoTIFFs.

#### 2. Database Consolidation
The daily ground-meteorological-fire database is aggregated to monthly averages and merged via an inner join with the monthly TROPOMI satellite database based on `["City", "Month"]`. Remaining missing values are resolved using global column medians.

#### 3. Deep Learning Architecture
A multi-layer perceptron (feed-forward neural network) is constructed in TensorFlow/Keras:
* **Input Layer**: Dimension matches the number of features, normalized using a trained `StandardScaler`.
* **Hidden Layer 1**: 128 units, ReLU activation.
* **Hidden Layer 2**: 64 units, ReLU activation.
* **Hidden Layer 3**: 32 units, ReLU activation.
* **Output Layer**: 1 unit, Linear activation (predicts PM2.5 concentration in $\mu\text{g/m}^3$).
* **Optimizer**: Adam
* **Loss Function**: Mean Squared Error (MSE)
* **Training Hyperparameters**: Epochs = 200, Batch Size = 16, Validation Split = 20%.

#### 4. AQI and Categorization
Predicted PM2.5 concentrations are mapped to the Indian CPCB AQI scale using segment-wise linear interpolation:

$$\text{AQI} = \frac{I_{\text{high}} - I_{\text{low}}}{C_{\text{high}} - C_{\text{low}}} \times (C - C_{\text{low}}) + I_{\text{low}}$$

Where:
* $C$: Concentration of PM2.5
* $[C_{\text{low}}, C_{\text{high}}]$: Concentration breakpoints
* $[I_{\text{low}}, I_{\text{high}}]$: AQI index breakpoints

| PM2.5 Range ($\mu\text{g/m}^3$) | AQI Range | Class Category |
| :--- | :--- | :--- |
| 0 – 30 | 0 – 50 | Good |
| 30 – 60 | 50 – 100 | Satisfactory |
| 60 – 90 | 100 – 200 | Moderate |
| 90 – 120 | 200 – 300 | Poor |
| 120 – 250 | 300 – 400 | Very Poor |
| > 250 | 400 – 500 | Severe |

---

### Objective 2: HCHO Hotspot Identification (`objective2_hcho_hotspots.py`)

#### 1. Temporal Compositing
* **Annual Mean**: Pixel-wise average HCHO density across all 12 monthly rasters.
* **Biomass Burning Season Composite**: Average density across October, November, December, and January.
* **Forest Fire Season Composite**: Average density across March, April, and May.

#### 2. Anomaly Detection (Z-Score Method)
Monthly variations are evaluated against the annual baseline using pixel-wise Z-scores:

$$Z(x, y, m) = \frac{X(x, y, m) - \mu_{\text{annual}}(x, y)}{\sigma_{\text{annual}}(x, y)}$$

Where:
* $X(x, y, m)$: HCHO concentration at pixel $(x, y)$ for month $m$
* $\mu_{\text{annual}}(x, y)$: Pixel annual mean
* $\sigma_{\text{annual}}(x, y)$: Pixel standard deviation across 12 months

A pixel is identified as a hotspot if $Z > 2.0$. The hotspot frequency map displays the number of months a pixel is classified as a hotspot.

#### 3. Fire-HCHO Correlation
City-level monthly HCHO time series are extracted and merged with national monthly VIIRS fire counts. The correlation is computed using Pearson's Correlation Coefficient ($r$):

$$r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$$

Hexbin density maps weighted by Fire Radiative Power (FRP) are generated to identify major regional emission sources (e.g., agricultural burning in the Indo-Gangetic Plain and forest fire zones in Central India).

---

## Pipeline Execution

Run the scripts from the directory `/home/vicky-raj/Desktop/test/AQI_HCHO_Project`:

### Run Objective 1 Script
This script performs data cleaning, meteorology and satellite extraction, trains the Deep Learning model, computes the AQI, and outputs validation figures and interactive maps:
```bash
python3 scripts/objective1_surface_aqi.py
```
* **Performance Metrics**: Generates R², MAE, and RMSE values.
* **Outputs Generated**:
  * Scaled data: `data/processed/india_pm25_era5_2024.csv`
  * Satellite summary: `data/processed/satellite_multi_pollutant_database.csv`
  * Model artifacts: `models/PM25_DeepLearning_Model.keras`, `models/PM25_Scaler.pkl`, `models/city_mapping.pkl`
  * Validation plots: `outputs/figures/pm25_validation_scatter.png`, `outputs/figures/pm25_error_distribution.png`
  * Interactive Maps: `outputs/maps/India_Surface_AQI_Month*.html` (for months 1, 6, 10, 12)

### Run Objective 2 Script
This script calculates composites, runs the Z-score anomaly detection, evaluates the correlation between fire counts and HCHO, and maps hotspots:
```bash
python3 scripts/objective2_hcho_hotspots.py
```
* **Outputs Generated**:
  * Time Series: `data/processed/city_hcho_2024.csv`
  * Spatial Plots: `outputs/figures/India_HCHO_Annual_Mean_2024.png`, `outputs/figures/India_HCHO_Burning_Season_2024.png`, `outputs/figures/India_HCHO_Forest_Fire_Season_2024.png`, `outputs/figures/India_HCHO_Hotspot_Frequency_2024.png`
  * Fire Analyses: `outputs/figures/Fire_HCHO_Correlation.png`, `outputs/figures/Fire_Source_Regions_2024.png`, `outputs/figures/Forest_Fire_Hotspots_2024.png`
  * Interactive Map: `outputs/maps/India_HCHO_Hotspot_Map_2024.html`
