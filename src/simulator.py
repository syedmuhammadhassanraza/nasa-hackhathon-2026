"""
Astronaut Health Monitoring System - Telemetry Simulator
Streams real multi-modal biometric windows from WESAD subjects or generates
synthetic high-stress / EVA / microgravity anomaly scenarios.
"""
from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import pandas as pd


class TelemetrySimulator:
    def __init__(self, data_path: Path | str = "processed/wesad_features.csv"):
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.subjects = sorted(self.df["subject"].unique().tolist())
        self.current_subject = "S10"
        self.current_idx = 0
        self._filter_subject_data()

    def _filter_subject_data(self):
        self.sub_df = self.df[self.df["subject"] == self.current_subject].sort_values("t_start_s").reset_index(drop=True)
        self.current_idx = 0

    def set_subject(self, subject: str):
        if subject in self.subjects:
            self.current_subject = subject
            self._filter_subject_data()

    def get_next_packet(self) -> dict:
        """Returns the next sequential telemetry window for the active subject."""
        if self.sub_df.empty:
            return self.generate_synthetic_scenario("nominal")

        if self.current_idx >= len(self.sub_df):
            self.current_idx = 0  # Loop around

        row = self.sub_df.iloc[self.current_idx].to_dict()
        self.current_idx += 1
        return row

    def generate_synthetic_scenario(self, scenario_name: str) -> dict:
        """
        Generates controlled real-time aerospace scenarios:
        - 'nominal': Resting orbit baseline
        - 'eva': Extravehicular Activity (high exertion, moderate HR, high movement)
        - 'stress': Acute cognitive crisis / system emergency (tachycardia, low HRV, EDA surge)
        - 'anomaly': Autonomic collapse / hypoxia / heat stress
        """
        scenario = scenario_name.lower()

        if scenario == "nominal":
            hr = random.uniform(62.0, 75.0)
            rmssd = random.uniform(40.0, 65.0)
            resp = random.uniform(12.0, 16.0)
            eda = random.uniform(0.6, 1.4)
            temp = random.uniform(33.5, 34.2)
            acc = random.uniform(0.90, 0.94)
            acc_std = random.uniform(0.003, 0.008)
            hr_z = random.uniform(-0.4, 0.3)
            rmssd_z = random.uniform(0.2, 0.8)
            resp_z = random.uniform(-0.3, 0.2)
            eda_z = random.uniform(-0.2, 0.3)
            temp_z = random.uniform(-0.1, 0.3)
            acc_std_z = random.uniform(-0.3, 0.1)
            eda_slope = random.uniform(-0.0001, 0.0001)

        elif scenario == "eva":
            hr = random.uniform(92.0, 110.0)
            rmssd = random.uniform(28.0, 42.0)
            resp = random.uniform(20.0, 26.0)
            eda = random.uniform(2.5, 4.8)
            temp = random.uniform(34.4, 35.1)
            acc = random.uniform(1.10, 1.35)
            acc_std = random.uniform(0.08, 0.18)
            hr_z = random.uniform(1.5, 2.2)
            rmssd_z = random.uniform(-0.8, -0.2)
            resp_z = random.uniform(1.2, 1.9)
            eda_z = random.uniform(1.4, 2.2)
            temp_z = random.uniform(1.1, 1.8)
            acc_std_z = random.uniform(1.6, 2.5)
            eda_slope = random.uniform(0.001, 0.005)

        elif scenario == "stress":
            hr = random.uniform(112.0, 130.0)
            rmssd = random.uniform(10.0, 19.0)  # Severe vagal withdrawal
            resp = random.uniform(25.0, 34.0)  # Hyperventilation
            eda = random.uniform(6.5, 12.0)    # Massive sympathetic sweat surge
            temp = random.uniform(34.8, 35.6)
            acc = random.uniform(0.92, 1.05)   # Tremor / postural tension
            acc_std = random.uniform(0.015, 0.04)
            hr_z = random.uniform(2.8, 4.1)
            rmssd_z = random.uniform(-2.8, -2.1)
            resp_z = random.uniform(2.4, 3.8)
            eda_z = random.uniform(2.8, 4.5)
            temp_z = random.uniform(1.8, 2.9)
            acc_std_z = random.uniform(0.5, 1.4)
            eda_slope = random.uniform(0.02, 0.08)

        elif scenario == "anomaly":
            # Hypoxia / acute vital dissonance
            hr = random.uniform(135.0, 155.0)
            rmssd = random.uniform(6.0, 12.0)
            resp = random.uniform(8.0, 11.0)   # Bradypnea with extreme tachycardia (pathologic mismatch)
            eda = random.uniform(14.0, 22.0)
            temp = random.uniform(36.2, 37.4)  # Thermal distress / heat accumulation
            acc = random.uniform(0.88, 0.95)
            acc_std = random.uniform(0.001, 0.005)
            hr_z = random.uniform(4.5, 6.2)
            rmssd_z = random.uniform(-3.5, -2.6)
            resp_z = random.uniform(-2.5, -1.8)
            eda_z = random.uniform(4.5, 7.0)
            temp_z = random.uniform(3.5, 5.0)
            acc_std_z = random.uniform(-1.0, -0.5)
            eda_slope = random.uniform(0.05, 0.15)

        else:
            return self.generate_synthetic_scenario("nominal")

        packet = {
            "subject": f"SIM_{scenario.upper()}",
            "scenario": scenario,
            "t_start_s": 0.0,
            "t_end_s": 60.0,
            "hr_mean": round(hr, 2),
            "sdnn": round(rmssd * 1.3, 2),
            "rmssd": round(rmssd, 2),
            "resp_rate": round(resp, 2),
            "eda_mean": round(eda, 3),
            "eda_std": round(eda * 0.15, 3),
            "eda_slope": round(eda_slope, 5),
            "temp_mean": round(temp, 2),
            "temp_slope": round(random.uniform(-0.001, 0.002), 4),
            "acc_mean": round(acc, 3),
            "acc_std": round(acc_std, 4),
            "hr_mean_z": round(hr_z, 2),
            "sdnn_z": round(rmssd_z * 0.9, 2),
            "rmssd_z": round(rmssd_z, 2),
            "resp_rate_z": round(resp_z, 2),
            "eda_mean_z": round(eda_z, 2),
            "eda_std_z": round(eda_z * 0.8, 2),
            "eda_slope_z": round(eda_z * 0.9, 2),
            "temp_mean_z": round(temp_z, 2),
            "temp_slope_z": round(temp_z * 0.5, 2),
            "acc_mean_z": round(random.uniform(-0.2, 0.5), 2),
            "acc_std_z": round(acc_std_z, 2),
        }
        return packet
