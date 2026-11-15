# 📐 System Design & Mathematical Formulations: NASA CDSS

## 1. Architectural Design Overview
The NASA Astronaut Health CDSS follows a modular, decoupled architecture comprising:
1. **Data Ingestion & Feature Engineering Layer** (WESAD 700 Hz sensors + NASA OSD-575 serum multiplex arrays).
2. **Machine Learning & Swarm Intelligence Layer** (ACO Feature Selection + Gradient Boosting + Random Forest + Isolation Forest).
3. **Contactless Computer Vision Layer** (Remote Photoplethysmography rPPG signal processing).
4. **Physiological & Spaceflight Environmental Simulation Core** (Gravitational fluid mechanics, SANS intracranial dynamics, radiation transport).
5. **Data Persistence Layer** (Onboard SQLite database `astronaut_health.db`).
6. **Mission Control Web Dashboard** (Responsive HTML5, TailwindCSS, Chart.js, Three.js 3D parameter visualizers).

---

## 2. Mathematical Formulations

### 2.1 Multimodal Fusion: Astronaut Health Readiness Index (AHRI)
The composite Astronaut Health Readiness Index ($\text{AHRI} \in [0, 100]$) quantifies whole-body homeostatic resilience by fusing acute autonomic arousal with chronic organ resilience:

$$\text{AHRI} = 100 - \left( w_1 \cdot S_{\text{stress}} + w_2 \cdot A_{\text{anomaly}} + w_3 \cdot O_{\text{organ}} \right)$$

Where:
- $S_{\text{stress}} = P(\text{stress}) \times 100$: Calibrated posterior probability of acute sympathetic crisis from the Gradient Boosting classifier.
- $A_{\text{anomaly}} = \min(100, \max(0, (-\text{score}_{\text{IF}} - 0.45) \times 200))$: Normalized anomaly severity score from the Isolation Forest estimator.
- $O_{\text{organ}} = \frac{1}{M} \sum_{k=1}^M \text{Risk}_k$: Mean risk score across the 4 spaceflight organ panels (Renal, Hepatic, Immune, Cardiovascular).
- Weights: $w_1 = 0.45$, $w_2 = 0.25$, $w_3 = 0.30$ (sum to 1.00).

---

### 2.2 Contactless Computer Vision Vital Sensing (rPPG CHROM Algorithm)
Remote photoplethysmography measures diffuse light reflectance from human skin. Arterial blood volume pulsations modulate the optical absorption spectrum of oxygenated hemoglobin ($\text{HbO}_2$), with maximum sensitivity in the green channel ($\sim 520\text{--}580\text{ nm}$).

To remove motion and ambient illumination artifacts, we employ the **Chrominance-based rPPG (CHROM)** method (de Haan & Jeanne):

1. **Color Normalization**:
   $$R_n(t) = \frac{R(t)}{\mu_R}, \quad G_n(t) = \frac{G(t)}{\mu_G}, \quad B_n(t) = \frac{B(t)}{\mu_B}$$

2. **Orthogonal Chrominance Signals**:
   $$X(t) = 3 R_n(t) - 2 G_n(t)$$
   $$Y(t) = 1.5 R_n(t) + G_n(t) - 1.5 B_n(t)$$

3. **Motion-Resistant Pulse Waveform**:
   $$S(t) = X(t) - \alpha Y(t), \quad \text{where } \alpha = \frac{\sigma(X)}{\sigma(Y)}$$

4. **Spectral Decomposition & Heart Rate**:
   A Butterworth bandpass filter ($0.75 \le f \le 3.0\text{ Hz}$) conditions $S(t)$. The Fast Fourier Transform (FFT) identifies the dominant spectral peak:
   $$\text{HR}_{\text{bpm}} = 60 \cdot \arg\max_{f \in [0.75, 3.0]} | \mathcal{F}\{ S_{\text{filt}}(t) \} |$$

---

### 2.3 Swarm Intelligence: Ant Colony Optimization (ACO)
Given a set of $N$ candidate biometric features $F = \{f_1, f_2, \dots, f_N\}$, an artificial ant colony searches for the optimal subset $S^* \subset F$ that maximizes classification accuracy while minimizing subset cardinality.

1. **State Transition Rule**:
   The probability of ant $k$ choosing unselected feature $i$ is:
   $$P_i^k = \frac{[\tau_i]^\alpha \cdot [\eta_i]^\beta}{\sum_{j \in \text{Allowed}^k} [\tau_j]^\alpha \cdot [\eta_j]^\beta}$$
   - $\tau_i$: Pheromone concentration on feature $i$.
   - $\eta_i$: Fisher discriminant ratio (heuristic relevance):
     $$\eta_i = \frac{\sum_{c} N_c (\mu_{c,i} - \mu_i)^2}{\sum_{c} \sum_{x \in c} (x_{i} - \mu_{c,i})^2 + \epsilon}$$
   - $\alpha = 1.0, \beta = 1.2$: Relative influence parameters.

2. **Fitness Function with Parsimony Penalty**:
   $$\text{Fitness}(S) = \text{Accuracy}_{\text{CV}}(S) - \lambda \cdot \frac{|S|}{N}$$
   where $\lambda = 0.05$ enforces feature compression.

3. **Pheromone Evaporation & Reinforcement**:
   $$\tau_i(t+1) = (1 - \rho) \cdot \tau_i(t) + \sum_{k=1}^K \Delta \tau_i^k + \Delta \tau_i^{\text{elite}}$$
   - $\rho = 0.18$: Evaporation rate.
   - $\Delta \tau_i^k = \text{Fitness}(S^k)$ if $f_i \in S^k$, else 0.
   - $\Delta \tau_i^{\text{elite}} = 0.5 \cdot \text{Fitness}(S^*)$ if $f_i \in S^*$, providing elitist swarm convergence.

---

### 2.4 Spaceflight Gravitational & Environmental Dynamics

#### A. Cephalad Fluid Shift Model
In microgravity ($G=0$), hydrostatic blood columns disappear, shifting $\approx 1.5\text{--}2.0\text{ L}$ of fluid toward the head and neck:
$$\text{ShiftIndex}(G, t) = 100 \cdot \left(1.0 - G^{0.65}\right) \cdot \min\left(1.0, \frac{t_{\text{days}}}{14}\right)$$

#### B. Spaceflight-Associated Neuro-ocular Syndrome (SANS)
Intracranial pressure and cerebrospinal fluid (CSF) expansion cause optic disc edema (papilledema). The optic nerve sheath diameter (ONSD) expands from nominal $5.0\text{ mm}$:
$$\text{ONSD}_{\text{est}} = 5.0 + 1.8 \cdot \left( \frac{\text{SANS}_{\text{risk}}}{100} \right) \quad (\text{mm})$$
$$\text{SANS}_{\text{risk}} = \min\left(100, 0.6 \cdot \text{ShiftIndex} + 15 \cdot \text{CO}_2\% + 25 \cdot \frac{t_{\text{days}}}{360}\right)$$

#### C. Skeletal Bone Resorption & Hypercalciuria
Absence of weight-bearing loading induces osteoclastic bone turnover at $\approx 1.25\%\text{ per month}$ in zero gravity:
$$\Delta \text{BMD}_{\text{loss}}\% = 1.25 \cdot \left(1.0 - G^{0.55}\right) \cdot \left( \frac{t_{\text{days}}}{30} \right)$$
Excess resorbed calcium filters through the glomerulus, spiking urinary calcium concentration and elevating kidney stone crystallization risk ($\text{StoneRisk} = \min(100, 3.5 \cdot \Delta \text{BMD} + 20(1 - G))$).

#### D. Ionizing Cosmic Radiation Transport
Total cumulative radiological exposure from Galactic Cosmic Rays (GCR) and Solar Particle Events (SPE):
$$\text{Dose}_{\text{cumulative}} = \frac{t_{\text{days}} \times 24 \times R_{\mu\text{Sv/h}}}{1000} \quad (\text{mSv})$$
NASA Career Permissible Exposure Limit (PEL) threshold: $600\text{ mSv}$ (corresponding to a 3% Risk of Exposure-Induced Death from fatal cancer).
