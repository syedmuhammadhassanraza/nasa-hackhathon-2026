"""
Astronaut Health Monitoring System - Multimodal Fusion & Triage Engine
Fuses:
1. High-frequency wearable telemetry (acute autonomic stress, HRV, breathing rate, EDA, anomaly score).
2. Longitudinal serum biomarkers (NASA OSD-575 organ system status: renal, hepatic, immune, cardio).
Produces:
- Astronaut Health Readiness Index (AHRI, 0 - 100).
- Operational Triage Level (NOMINAL, CAUTION, CRITICAL).
- Actionable Aerospace Medicine Countermeasure Protocols.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path if needed
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import numpy as np
import pandas as pd
from src.models.organ_risk_engine import OrganRiskEngine


class MultimodalFusionEngine:
    def __init__(
        self,
        models_dir: Path | str = "models/saved",
        features_csv: Path | str = "processed/osd575_features.csv",
    ):
        self.models_dir = Path(models_dir)
        self.scaler = joblib.load(self.models_dir / "feature_scaler.joblib")
        self.binary_model = joblib.load(self.models_dir / "binary_stress_model.joblib")
        self.multi_model = joblib.load(self.models_dir / "multi_state_model.joblib")
        self.anomaly_detector = joblib.load(self.models_dir / "anomaly_detector.joblib")
        
        with open(self.models_dir / "model_metadata.json") as f:
            self.metadata = json.load(f)
            self.feature_names = self.metadata["features"]
            self.classes = self.metadata["classes"]

        self.organ_engine = OrganRiskEngine(features_csv)
        print("[FusionEngine] Models, scaler, and organ risk engine successfully loaded.")

    def evaluate_telemetry_packet(self, telemetry: dict, astronaut_id: str = "C001", timepoint: str = "R+1") -> dict:
        """
        Takes one biometric telemetry packet (from wearable device or simulator),
        infers acute autonomic stress state & anomaly score, retrieves astronaut's
        organ risk profile, and computes the composite Astronaut Health Readiness Index (AHRI).
        """
        # Ensure all 22 features exist (defaulting missing z-scores if only raw provided)
        feature_vector = []
        for feat in self.feature_names:
            if feat in telemetry:
                val = telemetry[feat]
            elif feat.endswith("_z") and feat[:-2] in telemetry:
                # If z-score not explicitly provided, estimate from default normal baseline
                raw_name = feat[:-2]
                val = telemetry.get(feat, 0.0)
            else:
                val = 0.0
            feature_vector.append(float(val))

        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)

        # 1. Wearable Inference
        stress_prob = float(self.binary_model.predict_proba(X_scaled)[0, 1])
        multi_probs = self.multi_model.predict_proba(X_scaled)[0]
        multi_state_preds = {cls: round(float(prob), 4) for cls, prob in zip(self.classes, multi_probs)}
        top_state = max(multi_state_preds, key=multi_state_preds.get)

        # Anomaly Detection: score_samples returns opposite of anomaly score; lower = more abnormal
        raw_anomaly = float(self.anomaly_detector.score_samples(X_scaled)[0])
        # Normalize anomaly score to 0 (nominal) - 100 (severe anomaly)
        # Typically normal baseline scores around -0.4 to -0.6, anomalies < -0.7
        anomaly_severity = round(min(100.0, max(0.0, (-raw_anomaly - 0.45) * 200.0)), 1)
        is_anomaly = bool(self.anomaly_detector.predict(X_scaled)[0] == -1)

        # 2. Longitudinal Organ Biomarker Evaluation
        organ_summary = self.organ_engine.get_astronaut_summary(astronaut_id, timepoint)
        composite_organ_risk = organ_summary["composite_organ_risk"]  # 0 to 100

        # 3. Multimodal Fusion Calculation: Astronaut Health Readiness Index (AHRI)
        # AHRI is 0 to 100 (100 = Peak Homeostasis, <50 = Critical)
        acute_strain_penalty = stress_prob * 100.0 * 0.45
        anomaly_penalty = anomaly_severity * 0.25
        organ_penalty = composite_organ_risk * 0.30

        total_penalty = acute_strain_penalty + anomaly_penalty + organ_penalty
        ahri_score = round(max(0.0, min(100.0, 100.0 - total_penalty)), 1)

        # 4. Triage State Determination
        if ahri_score >= 80.0 and not is_anomaly:
            triage_level = "NOMINAL"
            triage_badge = "🟢 GREEN"
        elif ahri_score >= 55.0:
            triage_level = "CAUTION"
            triage_badge = "🟡 YELLOW"
        else:
            triage_level = "CRITICAL"
            triage_badge = "🔴 RED"

        # 5. Generate model review flags, not operational instructions.
        countermeasures = self._generate_countermeasures(
            triage_level=triage_level,
            stress_prob=stress_prob,
            is_anomaly=is_anomaly,
            anomaly_severity=anomaly_severity,
            organ_summary=organ_summary,
            telemetry=telemetry,
        )

        return {
            "astronaut_id": astronaut_id,
            "timepoint": timepoint,
            "ahri_score": ahri_score,
            "triage_level": triage_level,
            "triage_badge": triage_badge,
            "wearable_metrics": {
                "top_state": top_state,
                "stress_probability": round(stress_prob, 3),
                "state_probabilities": multi_state_preds,
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_severity,
                "heart_rate_bpm": round(float(telemetry.get("hr_mean", 72.0)), 1),
                "hrv_rmssd_ms": round(float(telemetry.get("rmssd", 45.0)), 1),
                "respiration_rate_bpm": round(float(telemetry.get("resp_rate", 16.0)), 1),
                "eda_microsiemens": round(float(telemetry.get("eda_mean", 1.5)), 3),
                "skin_temp_c": round(float(telemetry.get("temp_mean", 33.5)), 2),
            },
            "organ_health": organ_summary,
            "countermeasures": countermeasures,
        }

    def _generate_countermeasures(
        self,
        triage_level: str,
        stress_prob: float,
        is_anomaly: bool,
        anomaly_severity: float,
        organ_summary: dict,
        telemetry: dict,
    ) -> list[dict]:
        protocols = []

        if triage_level == "CRITICAL" or is_anomaly:
            protocols.append({
                "code": "PROTOCOL-OMEGA",
                "title": "High-priority operational and medical review",
                "severity": "CRITICAL",
                "action": "Model flag only: verify sensor quality and assess the situation using approved mission procedures and qualified human oversight.",
                "rationale": f"Demo model detected a high-strain pattern or anomaly (anomaly score: {anomaly_severity}%)."
            })

        if stress_prob >= 0.65:
            protocols.append({
                "code": "PROTOCOL-ALPHA",
                "title": "Autonomic stress-pattern review",
                "severity": "HIGH",
                "action": "Review the stress-pattern flag against validated measurements and mission context; this prototype does not prescribe a countermeasure.",
                "rationale": f"Demo model estimated a stress-pattern probability of {stress_prob*100:.1f}%."
            })

        # Check organ specific alerts
        panels = organ_summary.get("panels", {})
        renal = panels.get("renal", {})
        if renal.get("risk_score", 0) >= 25.0:
            protocols.append({
                "code": "PROTOCOL-BRAVO",
                "title": "Renal and fluid-marker review",
                "severity": "MEDIUM",
                "action": "Model flag only: review the historical renal-panel signal with validated laboratory measurements and qualified medical oversight; no fluid or electrolyte treatment is recommended here.",
                "rationale": "The demo renal-panel score crossed its configured review threshold."
            })

        immune = panels.get("immune", {})
        if immune.get("risk_score", 0) >= 30.0:
            protocols.append({
                "code": "PROTOCOL-DELTA",
                "title": "Immune and inflammatory-marker review",
                "severity": "MEDIUM",
                "action": "Review historical inflammatory-marker changes with qualified medical oversight; this prototype does not recommend sleep, diet, or treatment changes.",
                "rationale": "The demo immune-panel score crossed its configured review threshold."
            })

        if not protocols:
            protocols.append({
                "code": "PROTOCOL-NOMINAL",
                "title": "No elevated demo review flag",
                "severity": "NOMINAL",
                "action": "No configured model threshold was crossed for this demo input. This is not a health clearance or assessment.",
                "rationale": "The model output reflects simulated telemetry and historical biomarker data, not a live crew assessment."
            })

        return protocols


if __name__ == "__main__":
    engine = MultimodalFusionEngine()
    
    # Test sample normal packet
    sample_normal = {
        "hr_mean": 72.0, "sdnn": 50.0, "rmssd": 45.0, "resp_rate": 15.0,
        "eda_mean": 1.2, "eda_std": 0.05, "eda_slope": 0.0001,
        "temp_mean": 33.8, "temp_slope": 0.0005, "acc_mean": 0.92, "acc_std": 0.005,
        "hr_mean_z": 0.1, "sdnn_z": 0.2, "rmssd_z": 0.2, "resp_rate_z": -0.1,
        "eda_mean_z": 0.1, "eda_std_z": 0.0, "eda_slope_z": 0.0,
        "temp_mean_z": 0.2, "temp_slope_z": 0.1, "acc_mean_z": 0.0, "acc_std_z": 0.0
    }
    res_normal = engine.evaluate_telemetry_packet(sample_normal, "C001", "R+1")
    print("Normal Packet Evaluation:")
    print(f"  AHRI: {res_normal['ahri_score']} | Triage: {res_normal['triage_badge']}")
    print(f"  Stress Prob: {res_normal['wearable_metrics']['stress_probability']}")

    # Test sample acute stress / anomaly packet
    sample_stress = {
        "hr_mean": 115.0, "sdnn": 18.0, "rmssd": 12.0, "resp_rate": 28.0,
        "eda_mean": 7.8, "eda_std": 0.8, "eda_slope": 0.05,
        "temp_mean": 35.2, "temp_slope": 0.02, "acc_mean": 1.25, "acc_std": 0.35,
        "hr_mean_z": 3.8, "sdnn_z": -2.4, "rmssd_z": -2.8, "resp_rate_z": 2.9,
        "eda_mean_z": 3.5, "eda_std_z": 2.8, "eda_slope_z": 3.2,
        "temp_mean_z": 2.1, "temp_slope_z": 2.5, "acc_mean_z": 1.5, "acc_std_z": 2.8
    }
    res_stress = engine.evaluate_telemetry_packet(sample_stress, "C001", "R+1")
    print("\nStress Packet Evaluation:")
    print(f"  AHRI: {res_stress['ahri_score']} | Triage: {res_stress['triage_badge']}")
    print(f"  Stress Prob: {res_stress['wearable_metrics']['stress_probability']}")
    print("  Countermeasures:", [c["code"] for c in res_stress["countermeasures"]])
