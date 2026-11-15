# Astronaut Health Monitoring Prototype

An educational prototype for exploring public spaceflight research data, simulated telemetry, experimental camera-based pulse estimates, and mission-environment scenarios. It is not connected to spacecraft or astronaut sensors and is not validated for medical or flight use.

## What it includes

- **Telemetry demo:** replays WESAD-derived features and synthetic scenarios.
- **Biomarker explorer:** displays longitudinal Inspiration4 measurements from NASA OSDR dataset OSD-575.
- **Camera demo:** estimates pulse rate from browser camera pixels or a clearly labeled simulation. Camera video is processed in the browser; RGB summaries are sent to the local server. Results are experimental and not clinical measurements.
- **Environment scenarios:** illustrates selected gravity, pressure, and radiation model outputs. It does not predict an individual's health or survival.
- **Local API and SQLite:** runs on loopback and stores local demo records.

All model outputs are demonstrations. Do not use them to diagnose, treat, clear, or make safety decisions about anyone.

## Run locally

Python 3.10–3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install scikit-learn joblib
python tests/test_system.py
python src/server.py 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). Allow camera access only if you choose to try the camera demo.

## Data and model files

Saved model artifacts are included under `models/saved/`. Raw OSD-575 and WESAD datasets, generated tables, local database files, and the virtual environment are not included. Dataset-dependent features may need their corresponding input files; see [docs/SETUP.md](docs/SETUP.md) for the data preparation instructions.

## Project layout

```text
src/                 API server, models, database, and dashboard
tests/               Validation suite
models/saved/        Saved model artifacts
docs/                Setup, design, architecture, and QA notes
preprocess.py        Dataset preprocessing
make_synthetic.py    Synthetic test-data generator
```
