"""
NASA Astronaut Health CDSS - Database Layer
SQLite persistent data store for crew profiles, telemetry streaming logs,
contactless computer vision scans, clinical alerts, and planetary simulations.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "astronaut_health.db"


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initializes schema and seeds baseline Inspiration4 crew profiles."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Astronauts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS astronauts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                mission TEXT NOT NULL,
                baseline_hr REAL DEFAULT 70.0,
                baseline_rmssd REAL DEFAULT 45.0,
                baseline_resp REAL DEFAULT 15.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Telemetry streaming logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_stream (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                astronaut_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                heart_rate REAL,
                rmssd REAL,
                respiration_rate REAL,
                eda REAL,
                temperature REAL,
                acceleration REAL,
                top_state TEXT,
                stress_prob REAL,
                anomaly_score REAL,
                ahri_score REAL,
                triage_level TEXT,
                FOREIGN KEY (astronaut_id) REFERENCES astronauts (id)
            )
        """)

        # 3. Computer Vision Contactless Face Vital Scans
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cv_vital_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                astronaut_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                heart_rate_bpm REAL,
                hrv_rmssd_ms REAL,
                respiration_rate REAL,
                fatigue_index REAL,
                signal_snr_db REAL,
                clinical_status TEXT,
                notes TEXT,
                FOREIGN KEY (astronaut_id) REFERENCES astronauts (id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS demo_pulse_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pulse_rate_bpm REAL NOT NULL,
                signal_snr_db REAL,
                confidence_score REAL,
                source_mode TEXT NOT NULL,
                notes TEXT
            )
        """)

        # 4. Clinical Alerts & Aerospace Protocols
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                astronaut_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                protocol_code TEXT NOT NULL,
                title TEXT NOT NULL,
                severity TEXT NOT NULL,
                action TEXT NOT NULL,
                rationale TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0,
                FOREIGN KEY (astronaut_id) REFERENCES astronauts (id)
            )
        """)

        # 5. Planetary Mars & Gravity Simulation Logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mars_simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                gravity_g REAL,
                pressure_kpa REAL,
                radiation_rate REAL,
                shielding_cm REAL,
                mission_days INTEGER,
                cumulative_dose_msv REAL,
                bmd_loss_pct REAL,
                sans_risk_pct REAL,
                summary_json TEXT
            )
        """)

        # Seed initial crew if empty
        cursor.execute("SELECT COUNT(*) FROM astronauts")
        if cursor.fetchone()[0] == 0:
            crew = [
                ("C001", "Jared Isaacman", "Commander", "Inspiration4", 71.0, 48.0, 14.5),
                ("C002", "Sian Proctor", "Mission Pilot", "Inspiration4", 68.5, 52.0, 15.0),
                ("C003", "Hayley Arceneaux", "Medical Officer", "Inspiration4", 74.0, 42.0, 16.0),
                ("C004", "Christopher Sembroski", "Mission Specialist", "Inspiration4", 69.5, 46.5, 15.2),
            ]
            cursor.executemany("""
                INSERT INTO astronauts (id, name, role, mission, baseline_hr, baseline_rmssd, baseline_resp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, crew)
            print("[Database] Seeded Inspiration4 crew profiles into SQLite.")

        conn.commit()
        print(f"[Database] SQLite schema verified at {DB_PATH}")


def log_telemetry_packet(astronaut_id: str, telemetry_data: dict, evaluation: dict) -> int:
    """Logs a live telemetry packet and its model evaluation into database."""
    wm = evaluation.get("wearable_metrics", {})
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO telemetry_stream (
                astronaut_id, heart_rate, rmssd, respiration_rate, eda, temperature,
                acceleration, top_state, stress_prob, anomaly_score, ahri_score, triage_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            astronaut_id,
            wm.get("heart_rate_bpm"),
            wm.get("hrv_rmssd_ms"),
            wm.get("respiration_rate_bpm"),
            wm.get("eda_microsiemens"),
            wm.get("skin_temp_c"),
            telemetry_data.get("acc_mean", 0.92),
            wm.get("top_state"),
            wm.get("stress_probability"),
            wm.get("anomaly_score"),
            evaluation.get("ahri_score"),
            evaluation.get("triage_level"),
        ))
        conn.commit()
        return cursor.lastrowid


def log_cv_scan(astronaut_id: str, scan_data: dict) -> int:
    """Records a contactless computer vision facial scan into database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO cv_vital_scans (
                astronaut_id, heart_rate_bpm, hrv_rmssd_ms, respiration_rate,
                fatigue_index, signal_snr_db, clinical_status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            astronaut_id,
            scan_data.get("heart_rate_bpm"),
            scan_data.get("hrv_rmssd_ms"),
            scan_data.get("respiration_rate"),
            scan_data.get("fatigue_index", 15.0),
            scan_data.get("signal_quality_snr"),
            scan_data.get("clinical_status", "NOMINAL"),
            scan_data.get("notes", "Contactless rPPG facial dermal absorption scan"),
        ))
        conn.commit()
        return cursor.lastrowid


def log_demo_pulse_scan(scan_data: dict) -> int:
    """Stores a local camera/demo estimate separately from astronaut medical records."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO demo_pulse_scans (
                pulse_rate_bpm, signal_snr_db, confidence_score, source_mode, notes
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            scan_data["heart_rate_bpm"],
            scan_data.get("signal_quality_snr"),
            scan_data.get("confidence_score"),
            scan_data["source_mode"],
            scan_data.get("notes"),
        ))
        conn.commit()
        return cursor.lastrowid


def get_recent_demo_pulse_scans(limit: int = 20) -> list[dict]:
    """Retrieves local camera/demo estimates without astronaut identity."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM demo_pulse_scans ORDER BY id DESC LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_recent_telemetry(limit: int = 50) -> list[dict]:
    """Retrieves recent telemetry history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM telemetry_stream ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_recent_cv_scans(limit: int = 20) -> list[dict]:
    """Retrieves recent CV facial vital scans."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM cv_vital_scans ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def log_mars_simulation(sim_data: dict) -> int:
    """Logs a planetary Mars or environmental simulation experiment."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO mars_simulations (
                gravity_g, pressure_kpa, radiation_rate, shielding_cm,
                mission_days, cumulative_dose_msv, bmd_loss_pct, sans_risk_pct, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sim_data.get("gravity_g", 0.38),
            sim_data.get("pressure_kpa", 0.6),
            sim_data.get("radiation_usv_day", 610.0),
            sim_data.get("shielding_cm", 0.0),
            sim_data.get("mission_duration_days", 500),
            sim_data.get("radiation_risk", {}).get("cumulative_dose_msv", 0.0),
            sim_data.get("bone_density_loss_pct", 0.0),
            sim_data.get("sans_risk", {}).get("risk_probability_pct", 0.0),
            json.dumps(sim_data)
        ))
        conn.commit()
        return cursor.lastrowid


def get_recent_simulations(limit: int = 20) -> list[dict]:
    """Retrieves recent planetary simulation records."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM mars_simulations ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_crew_list() -> list[dict]:
    """Retrieves all astronaut profiles."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM astronauts ORDER BY id ASC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_database()
    print("Crew list:", get_crew_list())
