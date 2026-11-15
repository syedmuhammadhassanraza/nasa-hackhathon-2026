"""Synthetic stand-ins for testing the pipeline offline (NOT real data)."""
import pickle, numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(0)
FS = 700

def make_wesad(root, sid, hr_base, hr_stress):
    d = Path(root)/f"S{sid}"; d.mkdir(parents=True, exist_ok=True)
    segs = [(1, 240, hr_base), (2, 180, hr_stress), (1, 120, hr_base), (3, 120, hr_base+5), (0, 30, hr_base)]
    ecg, lab, resp, eda, temp = [], [], [], [], []
    for L, dur, hr in segs:
        n = dur*FS; t = np.arange(n)/FS
        e = rng.normal(0, 0.02, n)
        beat_t = np.cumsum(rng.normal(60/hr, 0.03, int(dur*hr/60)+5)); beat_t = beat_t[beat_t < dur-0.1]
        for bt in beat_t:
            i = int(bt*FS); e[i:i+8] += np.hanning(8)*1.2      # sharp R-like spike
        ecg.append(e); lab.append(np.full(n, L))
        resp.append(np.sin(2*np.pi*0.25*t) + rng.normal(0, 0.05, n))
        eda.append(2 + 0.01*t + rng.normal(0, 0.01, n)); temp.append(33 + 0.001*t + rng.normal(0, 0.01, n))
    cat = lambda x: np.concatenate(x)
    N = len(cat(ecg))
    sig = {"chest": {"ACC": rng.normal(0, 0.05, (N, 3)) + [0, 0, 1], "ECG": cat(ecg)[:, None],
                     "EMG": rng.normal(0, 0.01, (N, 1)), "EDA": cat(eda)[:, None],
                     "Temp": cat(temp)[:, None], "Resp": cat(resp)[:, None]},
           "wrist": {"ACC": np.zeros((10, 3)), "BVP": np.zeros((10, 1)), "EDA": np.zeros((10, 1)), "TEMP": np.zeros((10, 1))}}
    with open(d/f"S{sid}.pkl", "wb") as f:
        pickle.dump({"signal": sig, "label": cat(lab), "subject": f"S{sid}"}, f)

def make_osd(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    tps = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82"]
    rows = []
    for c in ["C001", "C002", "C003", "C004"]:
        for tp in tps:
            for an, lo, hi, base in [("Glucose", 70, 99, 90), ("Creatinine", 0.6, 1.3, 0.9), ("ALT", 7, 56, 25), ("CRP", 0, 3, 1.2)]:
                v = base * (1.4 if (tp == "R+1" and an == "CRP") else 1) * rng.normal(1, 0.05)
                val = f"{v:.2f}"
                if an == "CRP" and tp == "L-44" and c == "C002": val = "<0.5"
                rows.append({"SAMPLE_NAME": f"{c}_{tp}", "ANALYTE": an, "VALUE": val, "UNITS": "mg/dL",
                             "RANGE_MIN": lo, "RANGE_MAX": hi})
    pd.DataFrame(rows).to_csv(root/"LSDS-8_CMP_SUBMITTED.csv", index=False)
    wide = []
    for c in ["C001", "C002", "C003", "C004"]:
        for tp in tps:
            r = {"Sample": f"{c}_{tp}"}
            for an in ["IL-6", "TNF-a", "IL-10", "IL-1b", "MCP-1", "IL-8"]:
                r[an] = f"{abs(rng.normal(5, 1)):.2f}"
            if c == "C003" and tp == "R+1": r["IL-6"] = "OOR >"; r["IL-10"] = "OOR <"
            wide.append(r)
    pd.DataFrame(wide).to_csv(root/"multiplex_cytokine_array.csv", index=False)

if __name__ == "__main__":
    make_wesad("test/WESAD", 2, 68, 95)
    make_wesad("test/WESAD", 3, 75, 100)
    make_osd("test/osd575")
    print("synthetic data written")
