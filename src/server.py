"""
NASA Astronaut Health CDSS - Complete REST API & Dashboard Server
Integrates SQLite persistence, Multimodal Fusion, Swarm ACO,
3D Mars planetary simulations, and Contactless Computer Vision rPPG.
"""
from __future__ import annotations

import json
import math
import mimetypes
import sys
from threading import Lock
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.database.db import (
    init_database,
    log_telemetry_packet,
    get_recent_telemetry,
    get_recent_cv_scans,
    get_crew_list,
    log_demo_pulse_scan,
    get_recent_demo_pulse_scans,
)
from src.models.fusion_engine import MultimodalFusionEngine
from src.simulator import TelemetrySimulator
from src.models.environment_engine import SpaceEnvironmentSimulator, NASA_MARS_ENVIRONMENT, PRESET_ENVIRONMENTS
from src.models.rppg_engine import RPPGVitalEstimator

STATIC_DIR = root_dir / "src" / "static"
FUSION_ENGINE = None
SIMULATOR = None
ENV_SIMULATOR = None
RPPG_ESTIMATOR = None
RPPG_SESSION_ID = None
RPPG_LOCK = Lock()


def get_engine():
    global FUSION_ENGINE
    if FUSION_ENGINE is None:
        FUSION_ENGINE = MultimodalFusionEngine(
            models_dir=root_dir / "models" / "saved",
            features_csv=root_dir / "processed" / "osd575_features.csv",
        )
    return FUSION_ENGINE


def get_simulator():
    global SIMULATOR
    if SIMULATOR is None:
        SIMULATOR = TelemetrySimulator(
            data_path=root_dir / "processed" / "wesad_features.csv"
        )
    return SIMULATOR


def get_env_simulator():
    global ENV_SIMULATOR
    if ENV_SIMULATOR is None:
        ENV_SIMULATOR = SpaceEnvironmentSimulator()
    return ENV_SIMULATOR


def get_rppg_estimator():
    global RPPG_ESTIMATOR
    if RPPG_ESTIMATOR is None:
        RPPG_ESTIMATOR = RPPGVitalEstimator()
    return RPPG_ESTIMATOR


class HealthMonitoringHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict | list, status=HTTPStatus.OK):
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, filepath: Path):
        if not filepath.exists() or not filepath.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        mime_type, _ = mimetypes.guess_type(str(filepath))
        content = filepath.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        engine = get_engine()
        simulator = get_simulator()
        env_sim = get_env_simulator()

        # 1. Static Web Dashboard
        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html")
            return

        # 2. System Status & Metadata
        if path == "/api/status":
            self._send_json({
                "status": "ONLINE",
                "system": "NASA Astronaut Health CDSS",
                "model_metadata": engine.metadata,
                "crew_members": get_crew_list(),
                "timepoints": engine.organ_engine.timepoints,
                "database_active": True,
            })
            return

        # 3. Crew Overview
        if path == "/api/crew":
            tp = query.get("timepoint", ["R+1"])[0]
            crew_data = engine.organ_engine.get_crew_overview(tp)
            self._send_json(crew_data)
            return

        # 4. Next Telemetry Stream Packet (with SQLite logging)
        if path in ("/api/telemetry/next", "/api/telemetry"):
            astro = query.get("astronaut", ["C001"])[0]
            tp = query.get("timepoint", ["R+1"])[0]
            packet = simulator.get_next_packet()
            evaluation = engine.evaluate_telemetry_packet(packet, astronaut_id=astro, timepoint=tp)
            
            # Persist to database
            try:
                log_telemetry_packet(astro, packet, evaluation)
            except Exception as e:
                pass

            self._send_json(evaluation)
            return

        # 5. Flight Scenario Injection
        if path == "/api/telemetry/scenario":
            scenario = query.get("name", ["nominal"])[0]
            astro = query.get("astronaut", ["C001"])[0]
            tp = query.get("timepoint", ["R+1"])[0]
            packet = simulator.generate_synthetic_scenario(scenario)
            evaluation = engine.evaluate_telemetry_packet(packet, astronaut_id=astro, timepoint=tp)
            
            try:
                log_telemetry_packet(astro, packet, evaluation)
            except Exception:
                pass

            self._send_json(evaluation)
            return

        # 6. Telemetry History from Database
        if path == "/api/telemetry/history":
            limit = int(query.get("limit", [30])[0])
            rows = get_recent_telemetry(limit)
            self._send_json(rows)
            return

        # 7. CV Scan History
        if path == "/api/cv/history":
            limit = int(query.get("limit", [20])[0])
            rows = get_recent_cv_scans(limit)
            self._send_json(rows)
            return

        if path == "/api/cv/demo_history":
            limit = int(query.get("limit", [20])[0])
            self._send_json(get_recent_demo_pulse_scans(limit))
            return

        # 8. Set Simulation Subject
        if path == "/api/telemetry/set_subject":
            subj = query.get("subject", ["S10"])[0]
            simulator.set_subject(subj)
            self._send_json({"status": "OK", "current_subject": simulator.current_subject})
            return

        # 9. Biomarker Trajectory Timeline
        if path == "/api/biomarker/timeline":
            subj = query.get("subject", ["C001"])[0]
            analyte = query.get("analyte", ["CRP"])[0]
            data = engine.organ_engine.get_analyte_data(subj, analyte)
            self._send_json(data)
            return

        # 10. Swarm ACO Results
        if path == "/api/aco/results":
            aco_file = root_dir / "models" / "saved" / "aco_results.json"
            if aco_file.exists():
                with open(aco_file) as f:
                    aco_data = json.load(f)
                self._send_json(aco_data)
            else:
                self._send_json({"error": "ACO results not yet computed", "status": "PENDING"})
            return

        # 11. Planetary Environment Simulation
        if path == "/api/environment/simulate":
            preset = query.get("preset", [None])[0]
            days = int(query.get("days", [180])[0])
            if preset and preset in PRESET_ENVIRONMENTS:
                cfg = PRESET_ENVIRONMENTS[preset]
                g = cfg["gravity_g"]
                p = cfg["cabin_pressure_kpa"]
                o2 = cfg["oxygen_pct"]
                co2 = cfg["co2_pct"]
                rad = cfg["radiation_usv_hr"]
            else:
                g = float(query.get("gravity", [0.0])[0])
                p = float(query.get("pressure", [101.3])[0])
                o2 = float(query.get("o2", [21.0])[0])
                co2 = float(query.get("co2", [0.4])[0])
                rad = float(query.get("radiation", [25.0])[0])

            sim_result = env_sim.simulate(g, p, o2, co2, rad, days)
            sim_result["presets"] = PRESET_ENVIRONMENTS
            self._send_json(sim_result)
            return

        # 12. Real NASA Mars Data & Enhanced Interactive Shielding Model
        if path == "/api/mars/data":
            # Optional query parameters for customizable Mars habitat simulation
            g = float(query.get("gravity", [0.376])[0])
            p_kpa = float(query.get("pressure", [56.0])[0])
            shield_cm = float(query.get("shielding_cm", [50.0])[0])  # cm of regolith/water
            days = int(query.get("days", [500])[0])
            
            # Shielding attenuation factor: 50 cm regolith reduces GCR dose by ~50%, 150 cm reduces by ~80%
            # Half-value layer (HVL) of regolith is ~35 cm
            shield_factor = 0.5 ** (shield_cm / 35.0)
            base_rad_usv = NASA_MARS_ENVIRONMENT["surface_radiation_rate_usv_hr"]
            attenuated_rad_usv = round(base_rad_usv * shield_factor, 2)
            
            sim_result = env_sim.simulate(
                gravity_g=g,
                cabin_pressure_kpa=p_kpa,
                oxygen_pct=32.0,
                co2_pct=0.35,
                radiation_usv_hr=attenuated_rad_usv,
                mission_days=days
            )
            
            self._send_json({
                "nasa_mars_environment": NASA_MARS_ENVIRONMENT,
                "custom_simulation": {
                    "gravity_g": g,
                    "habitat_pressure_kpa": p_kpa,
                    "shielding_thickness_cm": shield_cm,
                    "mission_days": days,
                    "effective_radiation_usv_hr": attenuated_rad_usv,
                    "shielding_attenuation_pct": round((1.0 - shield_factor) * 100, 1),
                    "atmospheric_status": sim_result["atmospheric_status"],
                    "physiological_impacts": sim_result["physiological_impacts"],
                    "countermeasures": sim_result["countermeasures"]
                }
            })
            return

        # 13. Mars Simulation History
        if path == "/api/mars/history":
            from src.database.db import get_recent_simulations
            limit = int(query.get("limit", [20])[0])
            rows = get_recent_simulations(limit)
            self._send_json(rows)
            return

        # Fallback to static assets
        static_file = STATIC_DIR / path.lstrip("/")
        if static_file.exists() and static_file.is_file():
            self._serve_file(static_file)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, f"Endpoint {path} not found")

    def do_POST(self):
        global RPPG_SESSION_ID
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        engine = get_engine()
        rppg = get_rppg_estimator()

        # 1. Computer Vision Frame Ingestion
        if path == "/api/cv/frame":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": "Invalid JSON", "details": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return

            samples = data.get("samples")
            if samples is None:
                samples = [data]
            if not isinstance(samples, list) or not samples or len(samples) > 90:
                self._send_json({"error": "samples must contain between 1 and 90 RGB samples"}, status=HTTPStatus.BAD_REQUEST)
                return

            try:
                parsed_samples = []
                for sample in samples:
                    if not isinstance(sample, dict):
                        raise ValueError("each sample must be an object")
                    rgb = [float(sample[key]) for key in ("r", "g", "b")]
                    if not all(math.isfinite(value) and 0 <= value <= 255 for value in rgb):
                        raise ValueError("RGB channels must be finite values from 0 to 255")
                    timestamp = sample.get("timestamp")
                    if timestamp is not None:
                        timestamp = float(timestamp)
                        if not math.isfinite(timestamp):
                            raise ValueError("timestamp must be finite")
                    parsed_samples.append((*rgb, timestamp))
            except (KeyError, TypeError, ValueError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return

            session_id = data.get("session_id")
            with RPPG_LOCK:
                if session_id and session_id != RPPG_SESSION_ID:
                    self._send_json({"error": "rPPG session has ended; start a new scan"}, status=HTTPStatus.CONFLICT)
                    return
                for sample in parsed_samples:
                    rppg.add_frame_sample(*sample)
                estimates = rppg.estimate_vitals()
            estimates["clinical_status"] = "NOT_ASSESSED"
            estimates["clinical_flags"] = []
            self._send_json(estimates)
            return

        if path == "/api/cv/reset":
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                session_id = data["session_id"]
                if not isinstance(session_id, str) or not session_id or len(session_id) > 100:
                    raise ValueError("session_id must be a non-empty string of at most 100 characters")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json({"error": f"Invalid session: {error}"}, status=HTTPStatus.BAD_REQUEST)
                return
            with RPPG_LOCK:
                rppg.reset()
                RPPG_SESSION_ID = session_id
            self._send_json({"status": "RESET", "session_id": session_id})
            return

        # 2. Log Contactless CV Scan to Database
        if path == "/api/cv/save_scan":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": "Invalid JSON", "details": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return

            try:
                pulse_rate = float(data["heart_rate_bpm"])
                source_mode = data["source_mode"]
                if not math.isfinite(pulse_rate) or not 30 <= pulse_rate <= 240:
                    raise ValueError("pulse rate must be a finite value between 30 and 240 bpm")
                if not isinstance(source_mode, str) or source_mode not in ("camera", "simulation"):
                    raise ValueError("source_mode must be camera or simulation")
                for field in ("signal_quality_snr", "confidence_score"):
                    value = data.get(field)
                    if value is not None:
                        numeric_value = float(value)
                        if not math.isfinite(numeric_value):
                            raise ValueError(f"{field} must be finite")
                        if field == "signal_quality_snr" and numeric_value < 0:
                            raise ValueError("signal_quality_snr must be non-negative")
                        if field == "confidence_score" and not 0 <= numeric_value <= 1:
                            raise ValueError("confidence_score must be between 0 and 1")
                        data[field] = numeric_value
            except (KeyError, TypeError, ValueError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return

            row_id = log_demo_pulse_scan(data)
            self._send_json({
                "status": "SUCCESS",
                "record_id": row_id,
                "message": "Experimental demo estimate saved separately; it is not an astronaut medical record."
            })
            return

        # 3. Arbitrary Predict Endpoint
        if path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json({"error": "Invalid JSON", "details": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return

            telemetry = data.get("telemetry", data)
            astro = data.get("astronaut", query.get("astronaut", ["C001"])[0])
            tp = data.get("timepoint", query.get("timepoint", ["R+1"])[0])

            evaluation = engine.evaluate_telemetry_packet(telemetry, astronaut_id=astro, timepoint=tp)
            self._send_json(evaluation)
            return

        self.send_error(HTTPStatus.NOT_FOUND, f"POST endpoint {path} not found")

    def log_message(self, format, *args):
        pass


def run_server(port: int = 8080):
    init_database()
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, HealthMonitoringHandler)
    print(f"\n=======================================================")
    print(f" NASA Astronaut Health Monitoring CDSS is RUNNING")
    print(f" Database:      SQLite (data/astronaut_health.db)")
    print(f" Dashboard URL: http://127.0.0.1:{port}")
    print(f" REST API:      http://127.0.0.1:{port}/api/status")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
