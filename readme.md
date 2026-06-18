# VayuDrishti

**AI-Based Surface AQI Prediction & HCHO Hotspot Detection using Satellite Data**

## Overview

VayuDrishti is an AI-powered geospatial analytics platform developed for the **Bharatiya Antariksh Hackathon 2026**. The project aims to estimate surface Air Quality Index (AQI) across India using satellite observations and identify Formaldehyde (HCHO) hotspots associated with biomass burning events.

The platform integrates multi-source satellite datasets, meteorological information, ground-based air quality measurements, and machine learning techniques to generate spatially continuous air quality intelligence for regions lacking monitoring infrastructure.

---

## Problem Statement

**Challenge 3: Development of Surface AQI & Identification of HCHO Hotspots over India using Satellite Data**

Air quality monitoring stations are sparse and unevenly distributed across India. Large regions lack access to real-time pollution information, making it difficult to assess environmental and public health risks.

This project addresses the challenge by:

* Predicting surface AQI using satellite-derived atmospheric observations.
* Detecting and mapping HCHO hotspots across India.
* Analyzing the relationship between biomass burning activities and HCHO concentrations.
* Visualizing pollution patterns through interactive geospatial dashboards.

---

## Objectives

### Objective 1: Surface AQI Estimation

* Collect satellite-derived atmospheric parameters.
* Integrate weather and ground station observations.
* Train machine learning models to predict AQI.
* Generate spatial AQI maps over India.

### Objective 2: HCHO Hotspot Detection

* Analyze TROPOMI HCHO observations.
* Identify abnormal HCHO concentration regions.
* Correlate HCHO levels with fire activity.
* Investigate transport patterns using wind data.

---

## Datasets

### Satellite Data

#### Sentinel-5P (TROPOMI)

* NO₂
* SO₂
* CO
* O₃
* HCHO

Source:
https://developers.google.com/earth-engine/datasets

#### INSAT-3D (Optional)

* Aerosol Optical Depth (AOD)

Source:
https://www.mosdac.gov.in

---

### Ground-Based Data

#### CPCB Air Quality Monitoring Stations

Parameters:

* AQI
* PM2.5
* PM10
* NO₂
* SO₂
* CO
* O₃

Source:
https://airquality.cpcb.gov.in

---

### Fire Data

#### NASA FIRMS (VIIRS)

Parameters:

* Latitude
* Longitude
* Fire Radiative Power (FRP)
* Detection Confidence

Source:
https://firms.modaps.eosdis.nasa.gov

---

### Meteorological Data

#### ERA5 Reanalysis

Parameters:

* Temperature
* Humidity
* Wind Components (U, V)

Source:
https://cds.climate.copernicus.eu

---

## Methodology

### AQI Prediction Pipeline

```text
Satellite Data
(NO₂, SO₂, CO, O₃, AOD)

        +
Weather Data

        +
CPCB AQI Data

        ↓

Data Processing

        ↓

Feature Engineering

        ↓

XGBoost Model

        ↓

Surface AQI Prediction
```

### HCHO Hotspot Detection Pipeline

```text
TROPOMI HCHO

        +
VIIRS Fire Data

        +
ERA5 Wind Data

        ↓

Hotspot Detection

        ↓

Spatial Analysis

        ↓

Source Region Identification
```

---

## Technology Stack

### Data Processing

* Python
* Pandas
* NumPy

### Geospatial Analysis

* Google Earth Engine
* GeoPandas
* Rasterio

### Machine Learning

* Scikit-Learn
* XGBoost

### Visualization

* React
* Leaflet
* Plotly

### Version Control

* Git
* GitHub

---

## Project Structure

```text
VayuDrishti/

├── data/
│   ├── cpcb/
│   ├── fire/
│   ├── era5/
│   └── exports/
│
├── gee/
│
├── notebooks/
│
├── models/
│
├── frontend/
│
├── docs/
│
└── README.md
```

---

## Expected Outcomes

* Surface AQI prediction maps over India.
* HCHO concentration and hotspot maps.
* Fire-HCHO correlation analysis.
* Wind-based transport assessment.
* Interactive geospatial dashboard.

---

## Team

Developed for **Bharatiya Antariksh Hackathon 2026**.

Project Name: **VayuDrishti**

Tagline:

*"Seeing Air Quality from Space."*
