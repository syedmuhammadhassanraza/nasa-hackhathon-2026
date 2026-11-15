"""
NASA Spaceflight Environmental & Gravitational Physiology Simulator
Models the physiological impact of gravitational shifts (Earth, Spaceship Microgravity, Moon, Mars),
cabin pressure, oxygen fractions, and cosmic radiation based on NASA Human Research Program (HRP) evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

# Real NASA Mars Environmental Data (Curiosity REMS, RAD, and Perseverance MEDA)
NASA_MARS_ENVIRONMENT = {
    "mission_site": "Jezero Crater & Gale Crater, Mars",
    "surface_gravity_g": 0.376,                  # 3.72 m/s^2 (~38% of Earth)
    "atmospheric_pressure_kpa": 0.75,            # 750 Pa (< 1% of Earth 101.3 kPa)
    "atmosphere_composition": {
        "CO2": 95.32,
        "N2": 2.60,
        "Ar": 1.90,
        "O2": 0.13,
        "CO": 0.08
    },
    "diurnal_temp_range_c": {"min": -88.0, "max": -12.0, "mean": -63.0},
    "surface_radiation_rate_usv_hr": 27.5,       # ~0.66 mSv/day from Curiosity RAD
    "transit_radiation_rate_usv_hr": 75.0,       # ~1.8 mSv/day in deep space transit
    "regolith_perchlorate_wt_pct": 0.65,        # ClO4- thyroid/metabolic inhibitor
    "dust_storm_opacity_tau": 0.85,              # Atmospheric optical depth
    "solar_irradiance_w_m2": 590,                # ~43% of Earth (1361 W/m2)
}

PRESET_ENVIRONMENTS = {
    "earth": {
        "name": "Earth Surface (Nominal Baseline)",
        "gravity_g": 1.00,
        "cabin_pressure_kpa": 101.3,
        "oxygen_pct": 20.95,
        "co2_pct": 0.04,
        "radiation_usv_hr": 0.34,
        "description": "Standard sea-level terrestrial gravity and atmospheric composition. Complete magnetic shield protection."
    },
    "spaceship": {
        "name": "Spaceship Transit / LEO (Microgravity)",
        "gravity_g": 0.00,
        "cabin_pressure_kpa": 101.3,
        "oxygen_pct": 21.0,
        "co2_pct": 0.45,  # Real ISS cabin levels (10x Earth)
        "radiation_usv_hr": 25.0,
        "description": "Zero-G freefall. Cephalad fluid shift, spaceflight neuro-ocular syndrome (SANS) risk, continuous cosmic radiation."
    },
    "moon": {
        "name": "Lunar Surface (Artemis Base Camp)",
        "gravity_g": 0.166,  # 1/6th G
        "cabin_pressure_kpa": 56.5,  # 8.2 psi Exploration Atmosphere
        "oxygen_pct": 34.0,  # Hyperoxic to maintain sea-level equivalent alveolar pO2
        "co2_pct": 0.30,
        "radiation_usv_hr": 43.0,
        "description": "One-sixth Earth gravity. Vacuum exterior, abrasive electrostatic regolith dust, severe solar particle event (SPE) exposure."
    },
    "mars": {
        "name": "Mars Surface Habitat (Jezero Crater)",
        "gravity_g": 0.376,  # 3/8th G
        "cabin_pressure_kpa": 56.0,
        "oxygen_pct": 32.0,
        "co2_pct": 0.35,
        "radiation_usv_hr": 27.5,
        "description": "Three-eighths Earth gravity. Extreme hypobaric exterior (0.75 kPa CO2), toxic perchlorates, prolonged galactic cosmic rays (GCR)."
    }
}


class SpaceEnvironmentSimulator:
    def __init__(self):
        self.presets = PRESET_ENVIRONMENTS
        self.mars_data = NASA_MARS_ENVIRONMENT

    def simulate(
        self,
        gravity_g: float = 0.0,
        cabin_pressure_kpa: float = 101.3,
        oxygen_pct: float = 21.0,
        co2_pct: float = 0.4,
        radiation_usv_hr: float = 25.0,
        mission_days: int = 180,
    ) -> dict:
        """
        Calculates dynamic physiological adaptation and organ strain metrics
        given gravitational, atmospheric, and radiological space parameters.
        """
        # Clamp inputs
        g = max(0.0, min(2.5, float(gravity_g)))
        p = max(35.0, min(140.0, float(cabin_pressure_kpa)))
        o2 = max(10.0, min(50.0, float(oxygen_pct)))
        co2 = max(0.01, min(3.0, float(co2_pct)))
        rad_rate = max(0.1, min(300.0, float(radiation_usv_hr)))
        days = max(1, min(1200, int(mission_days)))

        # 1. Alveolar Oxygen Partial Pressure (pO2 in mmHg)
        # Wet air pO2 = (P_cabin - 47 mmHg water vapor) * (O2% / 100)
        # 1 kPa = 7.50062 mmHg
        p_cabin_mmhg = p * 7.50062
        alveolar_po2 = max(0.0, (p_cabin_mmhg - 47.0) * (o2 / 100.0))
        # Normal sea level alveolar pO2 is ~100 mmHg. <60 mmHg is hypoxemia.
        hypoxia_index = max(0.0, min(100.0, (100.0 - alveolar_po2) * 1.8 if alveolar_po2 < 95.0 else 0.0))

        # 2. Fluid Redistribution & Cephalad Shift (0 - 100)
        # Microgravity causes 1.5 - 2L venous shift towards head; partial gravity reduces it
        # Model: Fluid Shift = 100 * (1.0 - G^0.75) with time saturation
        time_factor = min(1.0, days / 14.0)  # Reaches peak within 14 days
        fluid_shift_score = max(0.0, min(100.0, 100.0 * (1.0 - (g ** 0.65)) * time_factor))

        # 3. Spaceflight-Associated Neuro-ocular Syndrome (SANS) Risk
        # Driven by cephalad fluid shift + microgravity + elevated cabin CO2 + mission duration
        co2_aggravation = (co2 / 0.04) ** 0.35  # CO2 causes cerebral vasodilation
        sans_risk = min(100.0, (fluid_shift_score * 0.6 + (co2 * 15.0) + (days / 360.0 * 25.0)))
        optic_nerve_dilation_mm = 5.0 + (sans_risk / 100.0) * 1.8  # Normal ~5.0 mm, expands to 6.8 mm

        # 4. Bone Demineralization & Hypercalciuria Rate
        # Skeletal unloading: ~1.2% per month in 0-G; reduced by partial gravity G^0.5
        monthly_bmd_loss_pct = 1.25 * (1.0 - (g ** 0.55))
        total_bmd_loss_pct = min(40.0, monthly_bmd_loss_pct * (days / 30.0))
        # Urinary calcium concentration index (kidney stone risk)
        hypercalciuria_risk = min(100.0, total_bmd_loss_pct * 3.5 + (1.0 - g) * 20.0)

        # 5. Cardiovascular Deconditioning & Orthostatic Intolerance
        # Cardiac stroke volume drops and baroreflex down-regulates
        orthostatic_intolerance_risk = min(100.0, (1.0 - g) * 55.0 + min(45.0, days * 0.15))

        # 6. Cumulative Cosmic Radiation Exposure & Cancer Risk
        # Total dose in mSv = hours * uSv/hr / 1000
        total_hours = days * 24.0
        cumulative_dose_msv = round((total_hours * rad_rate) / 1000.0, 1)
        # NASA career limit is ~600 mSv (REID 3% risk of exposure-induced death)
        radiation_limit_pct = min(200.0, (cumulative_dose_msv / 600.0) * 100.0)

        # 7. Overall Space Environmental Stress Index (0 - 100, where 0 is Earth-like and 100 is lethal)
        env_stress_index = (
            fluid_shift_score * 0.25 +
            sans_risk * 0.20 +
            hypercalciuria_risk * 0.20 +
            orthostatic_intolerance_risk * 0.15 +
            (radiation_limit_pct * 0.5) * 0.15 +
            hypoxia_index * 0.05
        )
        env_stress_index = round(min(100.0, max(0.0, env_stress_index)), 1)

        # 8. Recommended Aerospace Countermeasures
        countermeasures = self._generate_environmental_countermeasures(
            g, p, o2, co2, rad_rate, days, fluid_shift_score, sans_risk, total_bmd_loss_pct, cumulative_dose_msv
        )

        return {
            "parameters": {
                "gravity_g": round(g, 3),
                "cabin_pressure_kpa": round(p, 1),
                "oxygen_pct": round(o2, 1),
                "co2_pct": round(co2, 2),
                "radiation_usv_hr": round(rad_rate, 1),
                "mission_days": days,
            },
            "atmospheric_status": {
                "alveolar_po2_mmhg": round(alveolar_po2, 1),
                "hypoxia_risk_pct": round(hypoxia_index, 1),
                "status": "NORMAL" if alveolar_po2 >= 90 else "MILD HYPOXIA" if alveolar_po2 >= 65 else "SEVERE HYPOXIA",
            },
            "physiological_impacts": {
                "env_stress_index": env_stress_index,
                "fluid_shift_score": round(fluid_shift_score, 1),
                "sans_risk_score": round(sans_risk, 1),
                "estimated_onsd_mm": round(optic_nerve_dilation_mm, 2),
                "total_bmd_loss_pct": round(total_bmd_loss_pct, 2),
                "hypercalciuria_risk": round(hypercalciuria_risk, 1),
                "orthostatic_intolerance_risk": round(orthostatic_intolerance_risk, 1),
                "cumulative_dose_msv": cumulative_dose_msv,
                "radiation_limit_consumed_pct": round(radiation_limit_pct, 1),
            },
            "countermeasures": countermeasures,
        }

    def _generate_environmental_countermeasures(
        self, g, p, o2, co2, rad_rate, days, fluid_shift, sans_risk, bmd_loss, dose_msv
    ) -> list[dict]:
        cm = []

        if g < 0.2:
            cm.append({
                "system": "GRAVITY & MUSCULOSKELETAL",
                "countermeasure": "Artificial Gravity Centrifugation & ARED Resistive Load",
                "prescription": "Candidate mission-design options only. Exercise and artificial-gravity plans require mission-specific evidence and flight-surgeon approval.",
                "target": "Designed to study loading and fluid-shift mitigation; this model does not predict an individual response."
            })

        if sans_risk >= 35.0:
            cm.append({
                "system": "NEURO-OCULAR (SANS)",
                "countermeasure": "LBNP (research countermeasure; clinician oversight required)",
                "prescription": "Research countermeasure only. Do not apply a pressure or duration from this simulation; use approved mission protocols and clinician oversight.",
                "target": "Fluid-shift effects and countermeasure efficacy require individual clinical assessment."
            })

        if bmd_loss >= 3.0:
            cm.append({
                "system": "RENAL & SKELETAL",
                "countermeasure": "Bone and renal health review",
                "prescription": "Use validated bone, renal, and electrolyte measurements. Medication or supplement decisions belong to a licensed clinician.",
                "target": "This environmental model cannot diagnose bone loss, kidney-stone risk, or select treatment."
            })

        if dose_msv >= 100.0:
            cm.append({
                "system": "RADIATION MITIGATION",
                "countermeasure": "Shielding and solar-particle-event shelter planning",
                "prescription": "Prioritize engineering shielding and solar-particle-event planning. Obtain dose estimates from validated mission radiation models; this model is illustrative only.",
                "target": "Actual dose depends on mission trajectory, solar conditions, habitat layout, and material composition."
            })

        if co2 >= 0.35:
            cm.append({
                "system": "LIFE SUPPORT (ECLSS)",
                "countermeasure": "Life-support sensor and scrubber review",
                "prescription": "Review cabin sensor readings and life-support procedures with the responsible engineering and medical teams.",
                "target": "The scenario input is not a real-time cabin measurement or equipment command."
            })

        if not cm:
            cm.append({
                "system": "GENERAL HEALTH",
                "countermeasure": "Standard Terrestrial Homeostasis Maintenance",
                "prescription": "No action is recommended by this prototype. Follow mission-specific medical guidance.",
                "target": "A model output is not an individual health assessment."
            })

        return cm


if __name__ == "__main__":
    sim = SpaceEnvironmentSimulator()
    print("=== EARTH ===")
    print(json.dumps(sim.simulate(1.0, 101.3, 21.0, 0.04, 0.34, 30), indent=2))
    print("\n=== MARS ===")
    print(json.dumps(sim.simulate(0.376, 56.0, 32.0, 0.35, 27.5, 500), indent=2))
