# 📋 Project Requirements: NASA Astronaut Health CDSS

## 1. Executive Summary & Purpose
The **NASA Astronaut Health Clinical Decision Support System (CDSS)** is an autonomous, in-flight aerospace medical intelligence platform designed for deep space exploration missions (Artemis lunar surface operations, long-duration Mars transit, and Commercial Low Earth Orbit missions).

Due to deep space communication latency (up to 22 minutes round-trip to Mars), astronauts cannot rely on real-time telemedicine support from Houston Mission Control during acute emergencies. The platform must continuously fuse **high-frequency wearable telemetry** (WESAD wearable sensors) with **molecular serum multi-omics biomarkers** (NASA OSDR OSD-575 Inspiration4 longitudinal dataset) to autonomously detect autonomic strain, assess organ deconditioning, and trigger life-saving countermeasures.

---

## 2. Functional Requirements (FR)

### FR-1: High-Frequency Wearable Telemetry Ingestion & Real-Time Inference
- **FR-1.1**: The system shall ingest multi-channel physiological vitals: Heart Rate (BPM), Heart Rate Variability (RMSSD & SDNN in ms), Respiratory Rate (breaths/min), Galvanic Skin Response / Electrodermal Activity (EDA mean, std, slope in $\mu\text{S}$), Skin Temperature ($^\circ\text{C}$), and 3-axis Accelerometry ($g$).
- **FR-1.2**: The system shall execute a trained Gradient Boosting classifier to predict instantaneous autonomic state across 4 classes: `baseline`, `meditation`, `stress`, and `amusement`.
- **FR-1.3**: The system shall compute a continuous probability of acute stress $P(\text{stress}) \in [0.0, 1.0]$.
- **FR-1.4**: The system shall execute an unsupervised Isolation Forest anomaly detector trained on resting baseline vitals to flag physiological anomalies (e.g., vagal collapse, cardiac arrhythmias, hyperventilation).

### FR-2: Contactless Computer Vision Vital Scanner (rPPG & Face Mesh)
- **FR-2.1**: The system shall provide contactless camera-based vital extraction using Remote Photoplethysmography (rPPG).
- **FR-2.2**: The computer vision model shall detect the human face, extract a Region of Interest (ROI) covering the forehead and cheek dermal capillary beds, and analyze the green spectral channel ($\sim 520\text{--}580\text{ nm}$).
- **FR-2.3**: The rPPG engine shall filter out high-frequency noise and motion artifacts via a 0.75 Hz to 3.0 Hz (45 to 180 BPM) Butterworth bandpass filter and estimate Heart Rate, HRV (RMSSD), and Respiration Rate via FFT Power Spectral Density.
- **FR-2.4**: The system shall calculate a Signal-to-Noise Ratio (SNR) in dB and assess eye-blink / fatigue indices.
- **FR-2.5**: The system shall support both **Live Webcam capture** (for the person in front of the screen) and **Astronaut Cabin Video Feed Simulation**.

### FR-3: Swarm Intelligence: Ant Colony Optimization (ACO) Feature Selection
- **FR-3.1**: The system shall employ Ant Colony Optimization (ACO) to discover the minimal, most discriminative subset of multi-modal features.
- **FR-3.2**: Artificial ants shall stochastically traverse feature graphs using pheromone trails ($\tau$) and heuristic Fisher discriminant information ($\eta$).
- **FR-3.3**: The optimization shall penalize subset cardinality to minimize onboard computational complexity and telemetry transmission bandwidth.
- **FR-3.4**: The system shall log and visualize the convergence curves and pheromone distribution across iterations.

### FR-4: Longitudinal Biomarker & Multi-Organ Spaceflight Risk Engine (OSD-575)
- **FR-4.1**: The system shall ingest all 921 serum analytes from the Inspiration4 mission across 4 astronauts (`C001`–`C004`) and 7 timepoints (`L-92`, `L-44`, `L-3`, `R+1`, `R+45`, `R+82`, `R+194`).
- **FR-4.2**: The system shall classify biomarkers into 4 distinct clinical spaceflight panels:
  1. **Renal & Fluid Dynamics** (Creatinine, BUN, Sodium, Potassium, Calcium).
  2. **Hepatic & Metabolic Function** (ALT, AST, Alk Phos, Bilirubin, Albumin, Glucose).
  3. **Inflammatory & Cytokine Regulation** (CRP, IL-6, TNFα, IFNγ, IL-10, MCP-1).
  4. **Cardiovascular & Endothelial Stress** (Fibrinogen, SAP, Haptoglobin, PF4, L-Selectin).
- **FR-4.3**: The system shall calculate baseline fold change ($FC = V / V_{\text{baseline}}$), log2 fold change, and out-of-range clinical boundaries.

### FR-5: Multimodal Fusion & Astronaut Health Readiness Index (AHRI)
- **FR-5.1**: The system shall fuse acute wearable telemetry with longitudinal organ vulnerability into a single **Astronaut Health Readiness Index (AHRI)** on a 0 to 100 scale.
- **FR-5.2**: The system shall trigger autonomous triage states:
  - 🟢 **NOMINAL**: $\text{AHRI} \ge 80.0$ and no anomaly detected.
  - 🟡 **CAUTION / ELEVATED RISK**: $55.0 \le \text{AHRI} < 80.0$.
  - 🔴 **CRITICAL EMERGENCY**: $\text{AHRI} < 55.0$ or severe physiological anomaly detected.

### FR-6: Planetary Gravitational & Spaceflight Environment Simulator
- **FR-6.1**: The system shall model physiological adaptation across 4 environments: Earth (1.0G), Spaceship Freefall (0.0G), Moon (0.166G), and Mars (0.376G).
- **FR-6.2**: The system shall provide interactive sliders for Gravity ($0.0\text{--}2.0\text{ G}$), Cabin Pressure ($40\text{--}120\text{ kPa}$), Ambient Oxygen ($15\text{--}40\%$), $\text{CO}_2$ ($0.04\text{--}1.5\%$), Cosmic Radiation ($0.3\text{--}100\text{ }\mu\text{Sv/hr}$), and Mission Days ($1\text{--}900\text{ d}$).
- **FR-6.3**: The system shall calculate Cephalad Fluid Shift (%), SANS Neuro-ocular Syndrome Risk (%), Optic Nerve Sheath Diameter (ONSD in mm), Bone Mineral Density Loss (%), Hypercalciuria Stone Risk (%), Alveolar $\text{pO}_2$ (mmHg), and Cumulative Radiation Dose (mSv) against NASA Permissible Exposure Limits ($600\text{ mSv}$).

### FR-7: Real NASA Mars Environment & Countermeasure Engine
- **FR-7.1**: The system shall integrate empirical data from NASA Curiosity (RAD radiation detector: $0.66\text{ mSv/day}$) and Perseverance (MEDA weather station: $750\text{ Pa}$ pressure, $95.3\%\text{ CO}_2$).
- **FR-7.2**: The system shall provide interactive 3D visualizations depicting Martian gravity well compression, atmospheric density, and cosmic radiation shielding penetration.
- **FR-7.3**: The system shall recommend targeted countermeasures (subsurface lava tube habitats, Potassium Iodide competitive perchlorate blockers, and Lower Body Negative Pressure suits).

---

## 3. Non-Functional Requirements (NFR)

- **NFR-1 (Latency)**: Real-time telemetry inference and anomaly detection must execute within $<100\text{ ms}$ per 60-second window.
- **NFR-2 (Reliability & Offline Independence)**: The system must run completely offline without relying on external cloud APIs or external inference servers.
- **NFR-3 (Data Integrity & Persistence)**: All telemetry streams, computer vision vital scans, and clinical alerts must be persistently recorded in an onboard SQLite database.
- **NFR-4 (Explainability)**: Every alert and triage transition must provide clear clinical rationales and feature attribution.
- **NFR-5 (Cross-Platform Compatibility)**: The web dashboard must operate across modern standards-compliant web browsers (Chrome, Edge, Firefox, Safari) using vanilla Web APIs, WebGL, and HTML5 Canvas.
