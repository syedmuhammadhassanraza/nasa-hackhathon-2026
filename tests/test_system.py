"""
Automated Test & Validation Suite for NASA Astronaut Health CDSS
Verifies:
1. Multimodal Machine Learning & Fusion Pipeline
2. Organ Biomarker Risk Engine (NASA OSD-575)
3. Live Telemetry Simulator & Scenario Injector
4. Demo Model Flags and Review Guidance
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.models.fusion_engine import MultimodalFusionEngine
from src.models.organ_risk_engine import OrganRiskEngine
from src.simulator import TelemetrySimulator
from src.database.db import (
    init_database,
    log_telemetry_packet,
    log_cv_scan,
    get_recent_telemetry,
    get_recent_cv_scans,
    log_demo_pulse_scan,
    get_recent_demo_pulse_scans,
    log_mars_simulation,
    get_crew_list,
)
from src.models.environment_engine import SpaceEnvironmentSimulator, NASA_MARS_ENVIRONMENT
from src.models.rppg_engine import RPPGVitalEstimator
import json


def run_all_tests():
    print("=" * 65)
    print("RUNNING NASA ASTRONAUT HEALTH CDSS VALIDATION SUITE")
    print("=" * 65)

    # 1. Test Organ Risk Engine
    print("\n[TEST 1] Testing Organ Biomarker Risk Engine (OSD-575)...")
    organ_engine = OrganRiskEngine()
    crew_overview = organ_engine.get_crew_overview("R+1")
    assert len(crew_overview) == 4, f"Expected 4 astronauts, got {len(crew_overview)}"
    for astro in crew_overview:
        sid = astro["subject"]
        score = astro["biological_readiness"]
        panels = astro["panels"]
        print(f"  Astronaut {sid}: Biological Readiness = {score}/100, Panels = {list(panels.keys())}")
        assert "renal" in panels and "hepatic" in panels and "immune" in panels and "cardio" in panels
    print("  --> PASS: All 4 astronauts & 4 organ systems verified.")

    # 2. Test Analyte Timeline
    print("\n[TEST 2] Testing Longitudinal Analyte Timeline (L-92 to R+194)...")
    crp_timeline = organ_engine.get_analyte_data("C001", "CRP")
    assert "timeline" in crp_timeline and len(crp_timeline["timeline"]) > 0
    print(f"  Astronaut C001 CRP Baseline: {crp_timeline['timeline'][0]['baseline']} {crp_timeline['unit']}")
    print(f"  Timepoints verified: {[t['timepoint'] for t in crp_timeline['timeline']]}")
    print("  --> PASS: Longitudinal analyte tracking verified.")

    # 3. Test Telemetry Simulator
    print("\n[TEST 3] Testing Telemetry Simulator & Scenarios...")
    simulator = TelemetrySimulator()
    pkt = simulator.get_next_packet()
    assert "hr_mean" in pkt and "rmssd" in pkt and "eda_mean" in pkt
    print(f"  Sequential Packet: HR={pkt['hr_mean']} bpm, RMSSD={pkt['rmssd']} ms, Subject={pkt.get('subject')}")

    for scen in ["nominal", "eva", "stress", "anomaly"]:
        scen_pkt = simulator.generate_synthetic_scenario(scen)
        assert scen_pkt["scenario"] == scen
        print(f"  Scenario '{scen}': HR={scen_pkt['hr_mean']} bpm, EDA={scen_pkt['eda_mean']} uS")
    print("  --> PASS: Telemetry Simulator & 4 Scenarios verified.")

    # 4. Test Multimodal Fusion & Triage Engine
    print("\n[TEST 4] Testing Multimodal Fusion & Triage Engine...")
    fusion = MultimodalFusionEngine()

    # Case A: Nominal baseline packet
    norm_pkt = simulator.generate_synthetic_scenario("nominal")
    norm_eval = fusion.evaluate_telemetry_packet(norm_pkt, "C001", "R+1")
    print(f"  Nominal Packet -> AHRI: {norm_eval['ahri_score']} | Triage: {norm_eval['triage_badge']} | Stress Prob: {norm_eval['wearable_metrics']['stress_probability']}")
    assert norm_eval["ahri_score"] >= 75.0, f"Expected nominal AHRI >= 75, got {norm_eval['ahri_score']}"

    # Case B: Stress scenario packet
    stress_pkt = simulator.generate_synthetic_scenario("stress")
    stress_eval = fusion.evaluate_telemetry_packet(stress_pkt, "C001", "R+1")
    print(f"  Stress Packet  -> AHRI: {stress_eval['ahri_score']} | Triage: {stress_eval['triage_badge']} | Stress Prob: {stress_eval['wearable_metrics']['stress_probability']}")
    assert stress_eval["wearable_metrics"]["stress_probability"] > 0.5, "Expected stress probability > 0.5"
    assert len(stress_eval["countermeasures"]) > 0, "Expected model review flags"
    assert all(
        "does not prescribe" in item["action"].lower()
        or "does not recommend" in item["action"].lower()
        or "model flag only" in item["action"].lower()
        for item in stress_eval["countermeasures"]
    ), "Model outputs must not prescribe interventions"
    print(f"  Model review flags: {[c['code'] for c in stress_eval['countermeasures']]}")

    # Case C: Anomaly scenario packet
    anomaly_pkt = simulator.generate_synthetic_scenario("anomaly")
    anomaly_eval = fusion.evaluate_telemetry_packet(anomaly_pkt, "C001", "R+1")
    print(f"  Anomaly Packet -> AHRI: {anomaly_eval['ahri_score']} | Triage: {anomaly_eval['triage_badge']} | Anomaly Score: {anomaly_eval['wearable_metrics']['anomaly_score']}%")
    assert anomaly_eval["triage_level"] in ("CAUTION", "CRITICAL")
    print("  --> PASS: Multimodal Fusion, Triage, and Protocols verified.")

    # 5. Test SQLite Database Layer
    print("\n[TEST 5] Testing SQLite Database Persistence & Audit Records...")
    init_database()
    crew = get_crew_list()
    assert len(crew) >= 4, f"Expected at least 4 crew members, got {len(crew)}"
    print(f"  Crew seeded in DB: {[c['id'] for c in crew]}")

    # Log a telemetry packet and check retrieval
    telemetry_row = log_telemetry_packet("C001", norm_pkt, norm_eval)
    assert telemetry_row > 0
    recent_telem = get_recent_telemetry(limit=5)
    assert len(recent_telem) > 0
    print(f"  Logged Telemetry Packet #{telemetry_row}, retrieved {len(recent_telem)} records from DB.")

    # Log a CV vital scan
    scan_row = log_cv_scan("C001", {
        "heart_rate_bpm": 72.4,
        "hrv_rmssd_ms": 48.2,
        "respiration_rate": 14.8,
        "signal_quality_snr": 18.5,
        "clinical_status": "NOMINAL",
        "notes": "Verified Contactless rPPG facial dermal absorption scan"
    })
    assert scan_row > 0
    cv_records = get_recent_cv_scans(limit=5)
    assert len(cv_records) > 0
    demo_scan_row = log_demo_pulse_scan({
        "heart_rate_bpm": 72.4,
        "signal_quality_snr": 4.2,
        "confidence_score": 0.78,
        "source_mode": "camera",
        "notes": "Local-user experimental estimate"
    })
    demo_records = get_recent_demo_pulse_scans(limit=5)
    assert demo_scan_row > 0
    assert any(record["id"] == demo_scan_row for record in demo_records)
    assert "astronaut_id" not in demo_records[0]
    print(f"  Logged legacy CV scan #{scan_row} and separate demo scan #{demo_scan_row}.")
    print("  --> PASS: SQLite Database persistence and audit logging verified.")

    # 6. Test Spaceflight Environment Engine
    print("\n[TEST 6] Testing Spaceflight Environment Engine (Mars, Moon, Spaceship)...")
    env_sim = SpaceEnvironmentSimulator()
    mars_eval = env_sim.simulate(gravity_g=0.376, cabin_pressure_kpa=56.0, oxygen_pct=32.0, co2_pct=0.35, radiation_usv_hr=27.5, mission_days=500)
    assert mars_eval["parameters"]["gravity_g"] == 0.376
    assert mars_eval["physiological_impacts"]["total_bmd_loss_pct"] > 0
    print(f"  Mars 500-Day Simulation: Bone Loss = -{mars_eval['physiological_impacts']['total_bmd_loss_pct']}%, Cephalad Shift = {mars_eval['physiological_impacts']['fluid_shift_score']}/100, SANS Risk = {mars_eval['physiological_impacts']['sans_risk_score']}/100")
    
    # Moon Simulation
    moon_eval = env_sim.simulate(gravity_g=0.166, cabin_pressure_kpa=56.5, oxygen_pct=34.0, co2_pct=0.30, radiation_usv_hr=43.0, mission_days=180)
    assert moon_eval["parameters"]["gravity_g"] == 0.166
    print(f"  Moon 180-Day Simulation: Bone Loss = -{moon_eval['physiological_impacts']['total_bmd_loss_pct']}%, Cephalad Shift = {moon_eval['physiological_impacts']['fluid_shift_score']}/100")
    print("  --> PASS: Planetary gravitational & radiation physics models verified.")

    # 7. Test Swarm ACO Optimization
    print("\n[TEST 7] Testing Swarm Ant Colony Optimization (ACO) Artifacts...")
    aco_file = root_dir / "models" / "saved" / "aco_results.json"
    assert aco_file.exists(), "ACO results file missing!"
    with open(aco_file, "r") as f:
        aco_data = json.load(f)
    assert len(aco_data["best_features"]) == 12
    assert aco_data["best_accuracy"] > 0.95
    print(f"  ACO Selected: {len(aco_data['best_features'])} features (Accuracy: {aco_data['best_accuracy']*100:.2f}%, F1: {aco_data['best_f1']:.3f})")
    print(f"  Pheromone trail top features: {sorted(aco_data['final_pheromones'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    print("  --> PASS: Swarm ACO feature selection artifacts verified.")

    # 8. Test Contactless CV rPPG Engine
    print("\n[TEST 8] Testing Contactless Face CV rPPG Vital Engine...")
    rppg = RPPGVitalEstimator(fps=30.0, buffer_seconds=6.0)
    # Simulate a 1.25 Hz (75 BPM) pulse wave in green channel
    import math
    for i in range(180):
        t = i / 30.0
        g_val = 128.0 + 4.0 * math.sin(2 * math.pi * 1.25 * t)
        rppg.add_frame_sample(100.0, g_val, 90.0, t)
    vitals = rppg.estimate_vitals()
    print(f"  rPPG Synthetic Signal -> Status: {vitals['status']}, HR: {vitals['heart_rate_bpm']} BPM, RMSSD: {vitals['hrv_rmssd_ms']} ms, Respiration: {vitals['respiration_rate']} RPM")
    assert vitals["status"] == "LOCKED"
    assert vitals["heart_rate_bpm"] is not None and abs(vitals["heart_rate_bpm"] - 75.0) <= 2.5
    rppg.reset()
    assert rppg.estimate_vitals()["status"] == "INITIALIZING"
    for i in range(180):
        rppg.add_frame_sample(100.0, 120.0, 90.0, i / 30.0)
    low_signal = rppg.estimate_vitals()
    assert low_signal["status"] == "LOW_SIGNAL"
    assert low_signal["heart_rate_bpm"] is None
    print("  --> PASS: Computer vision rPPG signal processing verified.")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
