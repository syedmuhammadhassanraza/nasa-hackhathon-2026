# 🏛️ System Architecture: NASA Astronaut Health CDSS

## 1. High-Level Architecture Diagram

```mermaid
graph TD
    subgraph ClientLayer["Frontend Client Layer (Browser)"]
        UI_HUD["Mission Control HUD\n(Chart.js Telemetry Charts)"]
        UI_CV["Contactless rPPG Scanner\n(HTML5 Canvas & Video)"]
        UI_SWARM["ACO Swarm Dashboard\n(Pheromone Matrix & Convergence)"]
        UI_GRAV["Planetary Environment Simulator\n(Interactive Parameter Sliders)"]
        UI_MARS["Mars 3D Expedition Center\n(Curiosity/Perseverance Data)"]
        UI_BIO["921-Analyte Explorer\n(Longitudinal Trajectory Plots)"]
    end

    subgraph APILayer["REST API & Server Layer (src/server.py)"]
        ROUTER["HTTP Threading Server & Router\n(Port 8080)"]
        EP_STATUS["/api/status"]
        EP_TELEM["/api/telemetry/next & /scenario"]
        EP_CV["/api/cv/frame"]
        EP_ACO["/api/aco/results"]
        EP_ENV["/api/environment/simulate"]
        EP_MARS["/api/mars/data"]
        EP_BIO["/api/biomarker/timeline"]
    end

    subgraph ServiceLayer["Core Analytics & Intelligence Engines"]
        FUSION_ENG["Multimodal Fusion Engine\n(src/models/fusion_engine.py)\nComputes AHRI & Triage Badge"]
        ML_MODELS["Machine Learning Models\n• Gradient Boosting (Stress CV: 0.940)\n• Isolation Forest (Anomaly Detector)\n• Feature Scaler (StandardScaler)"]
        RPPG_ENG["rPPG Vital Estimator\n(src/models/rppg_engine.py)\nCHROM Dermal Capillary Analyzer"]
        ACO_ENG["Swarm ACO Selector\n(src/models/aco_feature_selection.py)\nPheromone Graph Search (97.17% Acc)"]
        ORGAN_ENG["Organ Biomarker Engine\n(src/models/organ_risk_engine.py)\nRenal, Hepatic, Immune, Cardio"]
        ENV_ENG["Space Environment Simulator\n(src/models/environment_engine.py)\nGravity, SANS, Bone Loss, Radiation"]
    end

    subgraph DataStorageLayer["Data & Persistence Layer"]
        SQLITE[("SQLite Database\ndata/astronaut_health.db")]
        PROCESSED_DATA["Cleaned Dataset Files\n• processed/osd575_features.csv\n• processed/wesad_features.csv\n• models/saved/*.joblib"]
    end

    ClientLayer <-->|HTTP / REST JSON| APILayer
    APILayer <--> ServiceLayer
    ServiceLayer <--> DataStorageLayer
    FUSION_ENG --> ML_MODELS & ORGAN_ENG
```

---

## 2. Database Entity-Relationship (ER) Schema

The persistence layer is implemented in `src/database/db.py` using SQLite (`data/astronaut_health.db`):

```mermaid
erDiagram
    ASTRONAUTS ||--o{ TELEMETRY_STREAM : generates
    ASTRONAUTS ||--o{ CV_VITAL_SCANS : performs
    ASTRONAUTS ||--o{ CLINICAL_ALERTS : triggers

    ASTRONAUTS {
        string id PK "C001, C002, etc."
        string name "Astronaut Full Name"
        string role "Commander, Pilot, Medical Officer"
        string mission "Inspiration4, Artemis"
        float baseline_hr "Resting Baseline HR (bpm)"
        float baseline_rmssd "Resting Baseline HRV (ms)"
        float baseline_resp "Resting Respiration (br/min)"
        timestamp created_at
    }

    TELEMETRY_STREAM {
        integer id PK
        string astronaut_id FK
        timestamp timestamp
        float heart_rate
        float rmssd
        float respiration_rate
        float eda
        float temperature
        float acceleration
        string top_state
        float stress_prob
        float anomaly_score
        float ahri_score
        string triage_level
    }

    CV_VITAL_SCANS {
        integer id PK
        string astronaut_id FK
        timestamp timestamp
        float heart_rate_bpm
        float hrv_rmssd_ms
        float respiration_rate
        float fatigue_index
        float signal_snr_db
        string clinical_status
        string notes
    }

    CLINICAL_ALERTS {
        integer id PK
        string astronaut_id FK
        timestamp timestamp
        string protocol_code
        string title
        string severity
        string action
        string rationale
        integer acknowledged
    }

    MARS_SIMULATIONS {
        integer id PK
        timestamp timestamp
        float gravity_g
        float pressure_kpa
        float radiation_rate
        float shielding_cm
        integer mission_days
        float cumulative_dose_msv
        float bmd_loss_pct
        float sans_risk_pct
        string summary_json
    }
```

---

## 3. REST API Interface Specification

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/` or `/index.html` | `GET` | None | Serves the Mission Control Web Dashboard. |
| `/api/status` | `GET` | None | Returns server health, model metadata, and crew member IDs. |
| `/api/crew` | `GET` | `timepoint` (e.g. `R+1`) | Returns multi-organ risk profiles for all 4 Inspiration4 astronauts. |
| `/api/telemetry/next` | `GET` | `astronaut`, `timepoint` | Advances simulator and executes multimodal fusion evaluation. |
| `/api/telemetry/scenario` | `GET` | `name` (`nominal`, `eva`, `stress`, `anomaly`), `astronaut`, `timepoint` | Injects controlled aerospace mission scenario. |
| `/api/telemetry/set_subject`| `GET` | `subject` (`S2`–`S17`) | Sets the active WESAD wearable streaming subject. |
| `/api/biomarker/timeline` | `GET` | `subject`, `analyte` | Queries longitudinal trajectory across `L-92` to `R+194`. |
| `/api/aco/results` | `GET` | None | Retrieves Ant Colony Optimization feature selection results, pheromones, and convergence history. |
| `/api/environment/simulate`| `GET` | `preset`, `gravity`, `pressure`, `o2`, `co2`, `radiation`, `days` | Computes gravitational and atmospheric physiological responses. |
| `/api/mars/data` | `GET` | None | Returns empirical Curiosity RAD & Perseverance MEDA Mars data. |
| `/api/cv/frame` | `POST` | `{ r, g, b, timestamp }` | Ingests facial dermal capillary color samples and returns rPPG estimates. |
| `/api/predict` | `POST` | Custom JSON telemetry packet | Performs real-time inference on arbitrary biometric data. |

---

## 4. Security & Isolation Boundaries

- **Zero External Egress**: The inference engines, database, and computer vision algorithms execute entirely in-process without transmitting data over the public internet.
- **Microgravity Edge Deployment**: Designed to run directly on the spacecraft's local computing server (e.g. Astrobee / ISS onboard network) with no dependency on ground connectivity.
