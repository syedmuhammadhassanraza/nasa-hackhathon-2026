"""
Computer Vision Remote Photoplethysmography (rPPG) & Facial Vital Estimator
Processes facial pixel stream or frame packets for an experimental pulse-rate estimate.
This prototype does not estimate HRV, respiration, oxygen saturation, or fatigue.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


class RPPGVitalEstimator:
    def __init__(self, fps: float = 30.0, buffer_seconds: float = 12.0):
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.minimum_samples = min(int(fps * 8.0), self.buffer_size)
        self.green_buffer = []
        self.red_buffer = []
        self.blue_buffer = []
        self.timestamps = []

    def reset(self):
        self.green_buffer.clear()
        self.red_buffer.clear()
        self.blue_buffer.clear()
        self.timestamps.clear()

    def add_frame_sample(self, r_mean: float, g_mean: float, b_mean: float, timestamp: float | None = None):
        """Adds average RGB values extracted from the facial Region of Interest (ROI)."""
        self.green_buffer.append(float(g_mean))
        self.red_buffer.append(float(r_mean))
        self.blue_buffer.append(float(b_mean))
        self.timestamps.append(timestamp if timestamp is not None else len(self.green_buffer) / self.fps)

        if len(self.green_buffer) > self.buffer_size:
            self.green_buffer.pop(0)
            self.red_buffer.pop(0)
            self.blue_buffer.pop(0)
            self.timestamps.pop(0)

    def estimate_vitals(self) -> dict:
        """
        Applies chrominance / green channel filtering to estimate pulse rate.
        """
        n = len(self.green_buffer)
        if n < self.minimum_samples:
            # Not enough frames yet, return baseline initialization
            return {
                "status": "INITIALIZING",
                "progress_pct": round((n / self.minimum_samples) * 100, 1),
                "heart_rate_bpm": None,
                "hrv_rmssd_ms": None,
                "respiration_rate": None,
                "signal_quality_snr": None,
                "confidence_score": 0.0,
            }

        g = np.array(self.green_buffer)
        r = np.array(self.red_buffer)
        b = np.array(self.blue_buffer)
        sample_times = np.asarray(self.timestamps, dtype=float)
        intervals = np.diff(sample_times)
        if np.all(np.isfinite(sample_times)) and np.all(intervals > 0):
            uniform_times = np.arange(
                sample_times[0],
                sample_times[-1] + 0.5 / self.fps,
                1.0 / self.fps,
            )
            if len(uniform_times) >= self.minimum_samples:
                r = np.interp(uniform_times, sample_times, r)
                g = np.interp(uniform_times, sample_times, g)
                b = np.interp(uniform_times, sample_times, b)

        # CHROM algorithm (de Haan & Jeanne): combination of normalized color channels
        # X = 3R - 2G, Y = 1.5R + G - 1.5B
        r_norm = r / (np.mean(r) + 1e-6)
        g_norm = g / (np.mean(g) + 1e-6)
        b_norm = b / (np.mean(b) + 1e-6)
        
        # Green channel chrominance signal
        raw_pulse = 3.0 * r_norm - 2.0 * g_norm - (1.5 * r_norm + g_norm - 1.5 * b_norm)
        # Detrend
        raw_pulse = raw_pulse - np.mean(raw_pulse)

        # Bandpass filter between 0.7 Hz (42 bpm) and 3.0 Hz (180 bpm)
        nyq = 0.5 * self.fps
        low = 0.75 / nyq
        high = 3.0 / nyq
        b_filt, a_filt = butter(2, [low, high], btype="band")
        
        try:
            pulse_clean = filtfilt(b_filt, a_filt, raw_pulse)
        except Exception:
            pulse_clean = raw_pulse

        # FFT Power Spectral Density
        fft_size = max(1024, len(pulse_clean) * 8)
        fft_vals = np.abs(np.fft.rfft(pulse_clean, n=fft_size))
        freqs = np.fft.rfftfreq(fft_size, 1.0 / self.fps)

        # Mask within physiologic heart rate band [0.75 Hz, 3.0 Hz]
        band_mask = (freqs >= 0.75) & (freqs <= 3.0)
        valid_freqs = freqs[band_mask]
        valid_psd = fft_vals[band_mask]

        amplitude = float(np.std(raw_pulse))
        if len(valid_psd) == 0 or amplitude < 0.0005:
            return {
                "status": "LOW_SIGNAL",
                "progress_pct": 100.0,
                "heart_rate_bpm": None,
                "hrv_rmssd_ms": None,
                "respiration_rate": None,
                "signal_quality_snr": None,
                "confidence_score": 0.0,
            }

        peak_idx = np.argmax(valid_psd)
        best_freq = valid_freqs[peak_idx]
        hr_bpm = float(best_freq * 60.0)

        peak_power = valid_psd[peak_idx]
        bg_power = np.mean(valid_psd) + 1e-6
        snr = float(peak_power / bg_power)
        if snr < 1.5:
            return {
                "status": "LOW_SIGNAL",
                "progress_pct": 100.0,
                "heart_rate_bpm": None,
                "hrv_rmssd_ms": None,
                "respiration_rate": None,
                "signal_quality_snr": round(snr, 2),
                "confidence_score": 0.0,
            }

        confidence = min(0.98, snr / (snr + 4.0))

        return {
            "status": "LOCKED",
            "heart_rate_bpm": round(hr_bpm, 1),
            "hrv_rmssd_ms": None,
            "respiration_rate": None,
            "signal_quality_snr": round(snr, 2),
            "confidence_score": round(confidence, 2),
            "pulse_waveform": [round(float(v), 4) for v in pulse_clean[-40:]],
        }


if __name__ == "__main__":
    estimator = RPPGVitalEstimator()
    # Simulate 120 frames with synthetic 75 bpm (1.25 Hz) cardiac pulse in green channel
    t = np.arange(120) / 30.0
    for i in range(120):
        val = 120.0 + 3.0 * np.sin(2 * np.pi * 1.25 * t[i]) + np.random.normal(0, 0.2)
        estimator.add_frame_sample(100.0, val, 90.0, t[i])
    res = estimator.estimate_vitals()
    print("Synthetic rPPG Test Result:", res)
