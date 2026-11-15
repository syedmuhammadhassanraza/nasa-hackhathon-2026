"""
Astronaut Health Monitoring System - Spaceflight Organ & Biomarker Risk Engine
Analyzes NASA OSD-575 multi-panel serum biomarkers (CMP, Cytokines, Cardiovascular, Alamar/Eve):
- Computes organ system risk scores (Renal, Hepatic, Immune, Cardiovascular).
- Tracks baseline-relative drift across pre-flight (L-minus) and post-flight (R-plus) mission phases.
- Detects clinical reference range violations and aerospace medical vulnerabilities.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

PANEL_DEFINITIONS = {
    "renal": {
        "name": "Renal & Fluid Dynamics",
        "primary_analytes": ["CREATININE", "UREA NITROGEN (BUN)", "BUN/CREATININE RATIO", "SODIUM", "POTASSIUM", "CHLORIDE", "CALCIUM"],
        "critical_thresholds": {
            "CREATININE": {"high_pct_change": 30.0, "weight": 1.5},
            "UREA NITROGEN (BUN)": {"high_pct_change": 40.0, "weight": 1.2},
            "CALCIUM": {"high_pct_change": 15.0, "weight": 1.4},  # Bone demineralization marker
            "POTASSIUM": {"high_pct_change": 20.0, "weight": 1.3},
        },
        "description": "Monitors fluid shifts, microgravity renal deconditioning, and space nephrolithiasis risk."
    },
    "hepatic": {
        "name": "Hepatic & Metabolic Function",
        "primary_analytes": ["ALT", "AST", "ALKALINE PHOSPHATASE", "BILIRUBIN; TOTAL", "ALBUMIN", "GLUCOSE"],
        "critical_thresholds": {
            "ALT": {"high_pct_change": 50.0, "weight": 1.4},
            "AST": {"high_pct_change": 50.0, "weight": 1.4},
            "GLUCOSE": {"high_pct_change": 25.0, "weight": 1.0},
        },
        "description": "Tracks liver enzyme stress, metabolic regulation, and G-force cellular adaptation."
    },
    "immune": {
        "name": "Immune & Cytokine Regulation",
        "primary_analytes": ["CRP", "IL-6", "TNFα", "IFNγ", "IL-10", "MCP-1", "IL-1β", "EGF"],
        "critical_thresholds": {
            "CRP": {"high_pct_change": 100.0, "weight": 1.8},     # Acute re-entry inflammation
            "IL-6": {"high_pct_change": 150.0, "weight": 1.6},    # Cytokine flare
            "TNFα": {"high_pct_change": 80.0, "weight": 1.4},
            "IFNγ": {"high_pct_change": 100.0, "weight": 1.4},
        },
        "description": "Quantifies spaceflight immune dysregulation, cytokine release syndrome risk, and post-flight flares."
    },
    "cardio": {
        "name": "Cardiovascular & Endothelial Stress",
        "primary_analytes": ["CRP", "Fibrinogen", "Haptoglobin", "SAP", "PF4", "L-Selectin", "a-2 Macroglobulin"],
        "critical_thresholds": {
            "Fibrinogen": {"high_pct_change": 40.0, "weight": 1.4},
            "SAP": {"high_pct_change": 50.0, "weight": 1.2},
            "PF4": {"high_pct_change": 60.0, "weight": 1.1},
        },
        "description": "Assesses vascular shear stress, endothelial activation, and orthostatic deconditioning."
    }
}


class OrganRiskEngine:
    def __init__(self, features_csv: Path | str = "processed/osd575_features.csv"):
        self.path = Path(features_csv)
        self.df = pd.read_csv(self.path)
        self.subjects = sorted(self.df["subject"].unique().tolist())
        self.timepoints = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82", "R+194"]
        print(f"[OrganRiskEngine] Initialized with {len(self.df)} records for crew {self.subjects}")

    def get_analyte_data(self, subject: str, analyte: str) -> dict:
        """Fetch trajectory and reference ranges for a specific analyte and astronaut."""
        sub_df = self.df[(self.df["subject"] == subject) & (self.df["analyte"] == analyte)]
        if sub_df.empty:
            # Try case-insensitive matching
            sub_df = self.df[(self.df["subject"] == subject) & (self.df["analyte"].str.upper() == analyte.upper())]
        if sub_df.empty:
            return {"error": f"Analyte {analyte} not found for subject {subject}"}

        timeline = []
        for tp in self.timepoints:
            row = sub_df[sub_df["timepoint"] == tp]
            if not row.empty:
                r = row.iloc[0]
                timeline.append({
                    "timepoint": tp,
                    "day_rel": float(r["day_rel"]) if pd.notna(r["day_rel"]) else None,
                    "phase": str(r["phase"]),
                    "value": float(r["value"]) if pd.notna(r["value"]) else None,
                    "baseline": float(r["baseline"]) if pd.notna(r["baseline"]) else None,
                    "delta": float(r["delta"]) if pd.notna(r["delta"]) else None,
                    "fold_change": float(r["fold_change"]) if pd.notna(r["fold_change"]) else None,
                    "log2_fc": float(r["log2_fc"]) if pd.notna(r["log2_fc"]) else None,
                    "out_of_range": bool(r["out_of_range"]) if pd.notna(r["out_of_range"]) else False,
                    "range_min": float(r["range_min"]) if pd.notna(r["range_min"]) else None,
                    "range_max": float(r["range_max"]) if pd.notna(r["range_max"]) else None,
                    "unit": str(r["unit"]) if pd.notna(r["unit"]) else "",
                })
        return {
            "subject": subject,
            "analyte": analyte,
            "unit": timeline[0]["unit"] if timeline else "",
            "timeline": timeline,
        }

    def compute_panel_risk(self, subject: str, panel_key: str, timepoint: str = "R+1") -> dict:
        """
        Calculates a risk score (0-100, where 0 is pristine/nominal and 100 is critical risk)
        for a physiological system at a specific mission timepoint.
        """
        panel_cfg = PANEL_DEFINITIONS.get(panel_key)
        if not panel_cfg:
            raise ValueError(f"Unknown panel {panel_key}")

        analytes = panel_cfg["primary_analytes"]
        sub_df = self.df[(self.df["subject"] == subject) & (self.df["timepoint"] == timepoint)]

        biomarker_results = []
        weighted_penalties = []
        total_weights = []

        for an in analytes:
            match = sub_df[sub_df["analyte"].str.upper() == an.upper()]
            if match.empty:
                # Try partial match (e.g. for TNFα vs TNF)
                match = sub_df[sub_df["analyte"].str.startswith(an, na=False)]
            if match.empty:
                continue

            r = match.iloc[0]
            val = float(r["value"]) if pd.notna(r["value"]) else None
            base = float(r["baseline"]) if pd.notna(r["baseline"]) else None
            fc = float(r["fold_change"]) if pd.notna(r["fold_change"]) else 1.0
            oor = bool(r["out_of_range"]) if pd.notna(r["out_of_range"]) else False
            rmin = float(r["range_min"]) if pd.notna(r["range_min"]) else None
            rmax = float(r["range_max"]) if pd.notna(r["range_max"]) else None

            # Calculate individual biomarker strain percentage
            pct_change = abs(fc - 1.0) * 100.0
            crit_cfg = panel_cfg["critical_thresholds"].get(an, {"high_pct_change": 50.0, "weight": 1.0})
            crit_pct = crit_cfg["high_pct_change"]
            weight = crit_cfg["weight"]

            # Biomarker penalty: 0 to 100
            severity = min(100.0, (pct_change / crit_pct) * 50.0)
            if oor:
                severity = max(severity, 75.0)  # Reference range violation immediately escalates severity

            weighted_penalties.append(severity * weight)
            total_weights.append(weight)

            biomarker_results.append({
                "analyte": an,
                "value": val,
                "baseline": base,
                "fold_change": round(fc, 2),
                "pct_change": round(pct_change, 1),
                "out_of_range": oor,
                "range_min": rmin,
                "range_max": rmax,
                "severity_score": round(severity, 1),
            })

        panel_risk_score = (
            sum(weighted_penalties) / sum(total_weights) if total_weights else 10.0
        )
        panel_risk_score = round(min(100.0, max(0.0, panel_risk_score)), 1)

        status = "NOMINAL"
        if panel_risk_score >= 65:
            status = "CRITICAL"
        elif panel_risk_score >= 35:
            status = "ELEVATED"

        return {
            "panel_key": panel_key,
            "panel_name": panel_cfg["name"],
            "risk_score": panel_risk_score,
            "status": status,
            "description": panel_cfg["description"],
            "biomarkers": biomarker_results,
        }

    def get_astronaut_summary(self, subject: str, timepoint: str = "R+1") -> dict:
        """Aggregates all organ systems into a composite astronaut biological profile."""
        panels = {}
        for p_key in PANEL_DEFINITIONS:
            panels[p_key] = self.compute_panel_risk(subject, p_key, timepoint)

        # Composite Biological Readiness Index (0 - 100, where 100 is optimal health)
        avg_risk = np.mean([p["risk_score"] for p in panels.values()])
        biological_readiness = round(max(0.0, 100.0 - avg_risk), 1)

        # Clinical alerts
        alerts = []
        for p in panels.values():
            if p["status"] == "CRITICAL":
                alerts.append(f"Critical stress in {p['panel_name']} (Risk Score: {p['risk_score']})")
            elif p["status"] == "ELEVATED":
                alerts.append(f"Elevated biomarker shift in {p['panel_name']} (Risk Score: {p['risk_score']})")

        return {
            "subject": subject,
            "timepoint": timepoint,
            "biological_readiness": biological_readiness,
            "composite_organ_risk": round(avg_risk, 1),
            "panels": panels,
            "alerts": alerts,
        }

    def get_crew_overview(self, timepoint: str = "R+1") -> list[dict]:
        """Returns risk profiles for all 4 crew members."""
        return [self.get_astronaut_summary(s, timepoint) for s in self.subjects]


if __name__ == "__main__":
    engine = OrganRiskEngine()
    overview = engine.get_crew_overview("R+1")
    print(json.dumps(overview, indent=2))
