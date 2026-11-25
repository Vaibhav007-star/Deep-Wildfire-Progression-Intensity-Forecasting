# 🔥 FIRECAST AI — Deep Wildfire Progression & Intensity Forecasting System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6 CUDA](https://img.shields.io/badge/PyTorch-2.6%20CUDA%20(RTX%203050)-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL%203D%20Twin-black.svg)](https://threejs.org/)
[![NASA FIRMS](https://img.shields.io/badge/NASA-FIRMS%20Active%20Fire-0B3D91.svg)](https://firms.modaps.eosdis.nasa.gov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Deep Spatiotemporal Wildfire Intelligence System with Physics-Informed Neural Networks (PINN), Real NASA Satellite Telemetry, 3D Digital Twin, and WUI Infrastructure Vulnerability Analysis.**

---

> [!WARNING]
> **ACADEMIC & EMERGENCY DECISION-SUPPORT NOTICE**  
> FIRECAST AI is an academic research and situational-awareness decision-support prototype. Predictions must **never** replace official civil protection or tactical wildfire emergency management directives (e.g., CAL FIRE, USFS, Copernicus EMS).

---

## 🌟 Key Capabilities & Breakthroughs

1. **Real NASA FIRMS Satellite Telemetry Pipeline**:
   - Ingests real historical active fire thermal anomalies (MODIS C6.1 & VIIRS 375m) coupled with Sentinel-2 NDVI biomass canopy, SRTM 30m Digital Elevation Model (DEM) topography, and surface wind vectors.
2. **Physics-Informed Neural Network (PINN) Spread Loss**:
   - Enforces physical conservation via a coupled **2D Advection-Diffusion & Rothermel Reaction PDE**:
     $$\mathcal{R}_{\text{physics}} = \frac{\partial p}{\partial t} + (\mathbf{v}_{\text{wind}} \cdot \nabla p) - D \nabla^2 p - S_{\text{reaction}}$$
   - Spatial Sobel differential operators enforce directional wind vector alignment and steep slope uphill acceleration.
3. **Wildland-Urban Interface (WUI) Infrastructure Vulnerability Engine**:
   - Calculates real-time **Burned Area ($\text{km}^2$)**, **Threatened Structures**, **Population at Direct Risk**, **Severed Evacuation Corridors**, and **Power Grid Cutoffs**, synthesizing a **Community Hazard Index ($0 - 100$)**.
4. **3D Interactive Digital Twin (WebGL)**:
   - High-fidelity Three.js 3D terrain deformed dynamically by the DEM topography ($128 \times 128$ elevation vertices), dynamic fire front flame propagation, and an 800-particle glowing ember/smoke system.
5. **Real-World Landmark Disaster Benchmarks Engine**:
   - Pre-calibrated historical disaster scenarios:
     - 🚨 **Camp Fire (Paradise, CA 2018)**: Deadliest in CA history (85 fatalities, 18,804 structures, Jarbo Gap canyon winds).
     - ⚡ **Maui Lahaina Fire (Hawaii 2023)**: Fastest urban overrun (Hurricane Dora downslope winds, 1.5 km/h spread).
     - 🌲 **Dixie Megafire (Sierra Nevada 2021)**: 963k acres across mountain crests, pyrocumulonimbus columns.
     - 🏘️ **Marshall Fire (Boulder, CO 2021)**: Suburban Chinook windstorm (1,084 homes lost in under 6 hours).
6. **Live NASA FIRMS Hotspots Explorer**:
   - Real-time thermal anomaly explorer querying 66,000+ real satellite detections (latitude, longitude, brightness Kelvin, satellite sensor, confidence %, and Fire Radiative Power in MW).
7. **Executive Presentation Deck**:
   - Includes full 12-slide widescreen PowerPoint presentation: [`FIRECAST_AI_Wildfire_Forecasting_Presentation.pptx`](FIRECAST_AI_Wildfire_Forecasting_Presentation.pptx).

---

## 📌 Repository Architecture

```
FIRECAST_AI/
│
├── server.py                                  # FastAPI backend serving GPU inference, FIRMS & case studies
├── app.py                                     # Streamlit alternative dashboard
├── config.py                                  # Central configuration container
├── FIRECAST_AI_Wildfire_Forecasting_Presentation.pptx # 12-Slide Executive PowerPoint
│
├── frontend/
│   └── index.html                             # Obsidian Ember Web Command UI (Three.js 3D Twin + HUD)
│
├── data/
│   ├── raw/firms_active_fire_real.csv         # 66,000+ NASA FIRMS real active fire observations
│   ├── processed/real_wildfire_dataset.npz    # (120, 5, 6, 128, 128) real satellite spatiotemporal tensors
│   └── checkpoints/best_model.pt              # GPU-trained model weights (NVIDIA RTX 3050)
│
├── datasets/
│   ├── data_loader.py                         # PyTorch Dataset, DataLoaders, spatial augmentations
│   ├── real_wildfire_adapter.py               # Pure-Python NASA FIRMS data adapter & rasterizer
│   ├── real_dataset_builder.py                # Spatiotemporal sequence builder
│   ├── raster_processor.py                    # Geospatial CRS and raster transform utilities
│   └── normalization.py                       # 6-channel feature scaling & circular angle mapping
│
├── models/
│   ├── convlstm_cell.py                       # 2D Convolutional LSTM cell (i, f, o, g gates)
│   ├── convlstm_model.py                      # Multi-layer ConvLSTM Forecaster with dual output heads
│   ├── spatiotemporal_unet.py                 # Multi-scale SpatioTemporal UNet backbone
│   └── model_factory.py                       # Unified model factory
│
├── training/
│   ├── trainer.py                             # GPU Training engine with AdamW, Cosine Annealing
│   ├── losses.py                              # FirecastCompoundLoss (Focal + Masked MSE + PINN Loss)
│   ├── physics_loss.py                        # Rothermel PDE + 2D Advection-Diffusion loss supervision
│   └── metrics.py                             # Spatial IoU, Dice F1, Precision, Recall, MAE
│
├── forecasting/
│   ├── inference_engine.py                    # Multi-horizon inference with Monte Carlo Dropout
│   ├── vulnerability_engine.py                # WUI Infrastructure Exposure & Community Hazard Index
│   └── uncertainty.py                         # Epistemic perimeter uncertainty variance estimator
│
├── presentation/
│   ├── create_presentation.py                 # 12-Slide PowerPoint presentation builder
│   ├── generate_figures.py                    # Real NASA dataset high-resolution figure generator
│   └── figures/                               # Publication-grade benchmark & telemetry plots
│
└── tests/
    ├── test_dataset.py                        # Real dataset loading & normalizer tests
    ├── test_model.py                          # Forward pass, state initialization & shape checks
    ├── test_training.py                       # Loss functions, gradients & metric computations
    ├── test_physics.py                        # PINN loss gradients & compound loss checks
    └── test_vulnerability.py                  # Infrastructure exposure & hazard index validation
```

---

## 🛰️ 6-Channel Multi-Modal Environmental Stack

| Channel | Layer | Physical Source | Normalization |
|:---:|:---|:---|:---:|
| **C1** | Fire Radiance (FRP) | NASA FIRMS (MODIS C6.1 / VIIRS 375m) | $[0.0, 1.0]$ |
| **C2** | Fuel Biomass (NDVI) | Sentinel-2 Level-2A Surface Reflectance | $[0.0, 1.0]$ |
| **C3** | Topography Elevation | USGS / SRTM 30m Digital Elevation Model | $[0.0, 1.0]$ |
| **C4** | Wind Velocity | Surface wind speed field magnitude | $[0.0, 35.0] \text{ m/s} \rightarrow [0.0, 1.0]$ |
| **C5** | Wind Direction | Trigonometric circular angle representation | $[0.0, 1.0]$ |
| **C6** | Atmospheric Dryness | Vapor Pressure Deficit (VPD) & Relative Humidity | $[0.0, 1.0]$ |

---

## 📊 Benchmark Results on Real NASA Test Split

Trained on an **NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6)**:

| Metric | Score | Benchmark Description |
|:---|:---:|:---|
| **Overall Jaccard Index (IoU)** | **0.8702** | Test set active perimeter spatial overlap |
| **Dice Similarity Coefficient (F1)** | **0.9306** | Harmonic mean of precision and recall |
| **Model Precision** | **0.9035** | High-confidence active fire localization |
| **Perimeter Recall** | **0.9594** | Minimal false negative front escapes |
| **Epistemic Uncertainty (+48h)** | **$\pm 0.0482$** | Monte Carlo Dropout ($M=8$) variance |

---

## 🚀 Quickstart & How to Run

### 1. Environment Setup
```powershell
# Clone repository
git clone https://github.com/Vaibhav007-star/Deep-Wildfire-Progression-Intensity-Forecasting.git
cd Deep-Wildfire-Progression-Intensity-Forecasting

# Setup virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Command Platform
```powershell
# Launch FastAPI server on port 8080 (GPU-accelerated)
python -m uvicorn server:app --host 0.0.0.0 --port 8080
```
Open [**http://localhost:8080**](http://localhost:8080) in your browser:
- Select any of the **Historical Disaster Benchmarks** (Camp Fire, Maui Lahaina, Dixie Megafire, Marshall Fire) to automatically load real historical weather and replay the progression.
- Switch to the **3D Digital Twin** tab for real-time 3D topographical visualization with glowing ember particles.
- Inspect the **Infrastructure Risk** tab for structures threatened, population at risk, and evacuation cutoffs.
- Click **🛰️ Live FIRMS Telemetry** in the top bar to inspect active NASA satellite thermal detections.

### 3. Run Automated Test Suite
```powershell
python -m pytest tests/ -v
```
*(All 19 automated unit & regression tests pass)*

---

## 📜 Attribution & Data Sources
- **NASA FIRMS**: Land, Atmosphere Near real-time Capability for EOS (LANCE), NASA Goddard Space Flight Center.
- **USGS / NASA JPL**: Shuttle Radar Topography Mission (SRTM) 30m Global Elevation Data.
- **European Space Agency (ESA)**: Copernicus Sentinel-2 MSI Surface Reflectance.
