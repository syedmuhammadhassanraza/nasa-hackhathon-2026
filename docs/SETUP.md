# 🛠️ Project Implementation & Setup Guide: NASA CDSS

This document provides complete, step-by-step instructions to set up, train, validate, and launch the NASA Astronaut Health CDSS platform.

---

## 1. System Requirements & Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+).
- **Python**: Version `3.10`, `3.11`, or `3.12` (Python 3.12 recommended).
- **RAM**: Minimum 4 GB (8 GB recommended for 2.5 GB WESAD unzipping).
- **Disk Space**: ~3.5 GB for raw datasets, processed feature tables, and model weights.
- **Web Browser**: Google Chrome, Microsoft Edge, Firefox, or Safari with WebGL enabled.
- **Camera (Optional)**: Standard laptop or USB webcam for live contactless computer vision rPPG scanning (fallback astronaut simulation mode available).

---

## 2. Step-by-Step Installation

### Step 2.1: Clone or Open Project Folder
```powershell
cd "c:\Users\HP\Documents\nasa ai hackhathon"
```

### Step 2.2: Create and Activate Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Linux / macOS
# python3 -m venv .venv
# source .venv/bin/activate
```

### Step 2.3: Install Dependencies
```powershell
pip install -r requirements.txt
pip install scikit-learn
```

Dependencies installed:
- `numpy`: Numerical linear algebra and FFT signal processing.
- `pandas`: DataFrames for tabular multi-analyte OSD-575 serum matrices.
- `scipy`: Digital Butterworth filtering and peak detection for rPPG.
- `scikit-learn`: Gradient Boosting, Random Forest, Isolation Forest, and cross-validation.
- `requests`: Automated WESAD repository downloading.
- `openpyxl`: Excel parser for Inspiration4 Alamar multiplex immune panels.

---

## 3. Dataset Preprocessing Pipeline

Run the end-to-end preprocessing pipeline to clean and tidy both datasets:

### Preprocess OSD-575 Serum Biomarkers:
```powershell
python preprocess.py osd575 --in "osd-575 dataset" --out processed
```
*Generates `processed/osd575_long.csv`, `osd575_wide.csv`, `osd575_features.csv`, and `osd575_qc.json`.*

### Download & Preprocess WESAD Wearable Data:
```powershell
python preprocess.py wesad --download --data "wesad dataset" --out processed
```
*(If already downloaded, omit `--download`). Generates `processed/wesad_features.csv` (1,377 sliding windows across 15 subjects) and `wesad_qc.json`.*

---

## 4. Model Training & Swarm Intelligence Pipeline

### Step 4.1: Train Core Machine Learning Models
```powershell
python src/models/train_models.py
```
- Trains Gradient Boosting Stress Classifier (5-Fold Group CV ROC-AUC: **0.940**).
- Trains Random Forest 4-class state model.
- Trains Unsupervised Isolation Forest Anomaly Detector.
- Serializes artifacts to `models/saved/*.joblib` and `models/saved/model_metadata.json`.

### Step 4.2: Run Swarm Intelligence (ACO Feature Selection)
```powershell
python src/models/aco_feature_selection.py
```
- Executes 12-ant colony optimization over 15 iterations.
- Compresses 22 features down to 12 optimal features (45.5% reduction) with **97.17% accuracy**.
- Saves results to `models/saved/aco_results.json`.

---

## 5. Initialize Persistent Database
```powershell
python src/database/db.py
```
- Creates SQLite database schema at `data/astronaut_health.db`.
- Seeds Inspiration4 crew profiles (`C001`–`C004`) with baseline vitals.

---

## 6. Run the Validation Test Suite
```powershell
python tests/test_system.py
```
Expected output:
```text
=================================================================
ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)
=================================================================
```

---

## 7. Launch Mission Control Dashboard & REST API

Start the background server daemon:
```powershell
python src/server.py 8080
```

Open your browser to:
👉 **[http://127.0.0.1:8080](http://127.0.0.1:8080)**

---

## 8. Dashboard Navigation Quick-Guide

1. **`MISSION CONTROL HUD`**: Real-time biometric streaming gauges, Inspiration4 4-panel organ matrix, and automated clinical countermeasure protocols.
2. **`COMPUTER VISION VITALS (rPPG)`**: Real-time face tracking. Click `LIVE WEBCAM` to scan yourself or `ASTRONAUT CABIN FEED` for simulated cockpit telemetry.
3. **`SWARM ACO OPTIMIZATION`**: Inspect ant colony pheromone concentrations and convergence curves.
4. **`GRAVITY & ENVIRONMENT SIMULATOR`**: Adjust gravity ($0.0\text{--}2.0\text{ G}$), pressure, $\text{CO}_2$, and mission duration to view live SANS, fluid shift, and bone loss changes.
5. **`NASA MARS JEZERO EXPEDITION`**: View empirical Curiosity RAD radiation & Perseverance MEDA weather metrics with interactive 3D parameter visualizers.
6. **`921-ANALYTE EXPLORER`**: Select any biomarker from Inspiration4 to graph its trajectory across all mission timepoints.
