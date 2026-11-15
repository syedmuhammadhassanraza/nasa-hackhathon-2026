#!/usr/bin/env python3
"""
Astronaut health monitoring: dataset preprocessing pipeline (cleaning only, no modeling).

Datasets
--------
1. NASA OSDR OSD-575  (Inspiration4 blood serum: CMP + multiplex cytokine/cardiac arrays)
   Longitudinal, 4 crew, pre-flight (L-92, L-44, L-3) and post-flight (R+1, R+45, R+82).
2. WESAD (wearable stress/affect, 15 subjects): ECG, respiration, EDA, temperature, acceleration.
   Used as the live-vitals stand-in (heart rate, HRV, breathing rate, skin temperature, activity).

Usage
-----
    pip install numpy pandas scipy requests openpyxl

    # OSD-575: point at the folder (or zip) you downloaded from OSDR
    python preprocess.py osd575 --in data/osd575 --out processed

    # WESAD: download (2.5 GB) + process
    python preprocess.py wesad --download --data data/WESAD --out processed

    # both
    python preprocess.py all --in data/osd575 --download --data data/WESAD --out processed

Outputs (in --out)
------------------
    osd575_long.csv        tidy: subject, timepoint, day_rel, phase, analyte, value, flag, unit, range_min, range_max
    osd575_wide.csv        one row per subject x timepoint, one column per analyte
    osd575_features.csv    wide + per-subject baseline-relative values (fold change, log2 FC, out-of-range flag)
    osd575_qc.json         what was found / mapped / dropped (paste this back if anything looks off)
    wesad_features.csv     one row per 60 s window: HR, HRV, breathing rate, EDA, temp, ACC, label, per-subject z-scores
    wesad_qc.json          per-subject sanity checks
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Shared helpers
# =============================================================================

GREEK_CHAR_MAP = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e",
    "Α": "a", "Β": "b", "Γ": "g", "Δ": "d", "Ε": "e",
}

def norm_name(s) -> str:
    """lower_snake_case column names."""
    t = str(s).strip()
    for grk, rep in GREEK_CHAR_MAP.items():
        t = t.replace(grk, rep)
    return re.sub(r"[^0-9a-z]+", "_", t.lower()).strip("_")


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str))


# =============================================================================
# 1) OSD-575  (schema-adaptive: handles long or wide tables, CSV/TSV/XLSX)
# =============================================================================

MISSION_DAYS = 3  # Inspiration4 flew ~3 days; used only to put R+N on the same day axis as L-N

SUBJECT_COLS = ["subject", "subject_id", "crew", "crew_id", "astronaut", "participant",
                "participant_id", "patient", "donor"]
TIME_COLS = ["timepoint", "time_point", "collection_time", "collection", "visit", "time",
             "sample_time", "mission_phase", "day"]
SAMPLE_COLS = ["sample", "sample_id", "sample_name", "source_name", "specimen", "id"]
ANALYTE_COLS = ["analyte", "test", "test_name", "biomarker", "marker", "parameter", "assay",
                "compound", "component", "name"]
VALUE_COLS = ["value", "result", "result_value", "concentration", "conc", "measurement", "level"]
UNIT_COLS = ["unit", "units"]
RMIN_COLS = ["range_min", "ref_low", "reference_low", "ref_min", "lower_limit", "low"]
RMAX_COLS = ["range_max", "ref_high", "reference_high", "ref_max", "upper_limit", "high"]

# lookarounds (not \b) so "C001_L-92" works: underscore counts as a word character
SUBJECT_RE = re.compile(r"(?<![A-Za-z0-9])(C\s?0?\d{2,3})(?![0-9])", re.I)     # C001 ... C004
TIME_RE = re.compile(r"(?<![A-Za-z0-9])([LR])\s*([+-])\s*(\d+)(?![0-9])", re.I)  # L-92, R+1
NA_STRINGS = {"", "na", "n/a", "nan", "nd", "n.d.", "none", "null", "-", "--", "*", "ns", "nq"}


def find_col(cols, candidates, exclude=("range", "unit", "flag", "sample", "subject")):
    """Exact match first (in candidate priority order), then substring match."""
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for c in cols:
            if cand in c and not any(x in c for x in exclude):
                return c
    return None


def numeric_fraction(series: pd.Series) -> float:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    ok = s.str.replace(",", "", regex=False).str.match(r"^\s*[<>]?\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$")
    return float(ok.mean())


def read_any_table(path: Path):
    """Yield (label, DataFrame) for csv/tsv/txt/xlsx/xls."""
    suf = path.suffix.lower()
    try:
        if suf in (".csv", ".tsv", ".txt"):
            df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding_errors="replace")
            yield path.name, df
        elif suf in (".xlsx", ".xls"):
            for sheet, df in pd.read_excel(path, sheet_name=None, dtype=str).items():
                yield f"{path.name}[{sheet}]", df
    except Exception as e:  # keep going; report in QC
        yield f"{path.name}", e


def parse_timepoint(text):
    """'L-92' -> ('L-92', -92, 'pre'); 'R+45' -> ('R+45', 45+MISSION_DAYS, 'post')."""
    m = TIME_RE.search(str(text))
    if not m:
        return None, np.nan, None
    side, sign, n = m.group(1).upper(), m.group(2), int(m.group(3))
    label = f"{side}{sign}{n}"
    if side == "L":
        return label, -n, "pre"
    return label, n + MISSION_DAYS, "post"


def parse_censored(val):
    """
    Return (number, flag). Handles '<0.5', '>1000', 'OOR <', 'OOR >', '1,234.5', blanks.
    flag in {ok, below_range, above_range, missing}
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan, "missing"
    s = str(val).strip()
    if s.lower() in NA_STRINGS:
        return np.nan, "missing"
    low = s.lower()
    flag = "ok"
    if "<" in low or "below" in low or low.startswith("oor <") or "oorl" in low:
        flag = "below_range"
    elif ">" in low or "above" in low or low.startswith("oor >") or "oorh" in low:
        flag = "above_range"
    num = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace(",", ""))
    if num is None:
        return np.nan, ("below_range" if flag == "below_range" else "above_range" if flag == "above_range" else "missing")
    return float(num.group()), flag


def apply_censor_policy(value, flag, policy):
    if flag == "below_range":
        return {"half_lod": value / 2 if pd.notna(value) else np.nan, "limit": value, "nan": np.nan}[policy]
    if flag == "above_range":
        return value if policy in ("half_lod", "limit") else np.nan
    return value


def collect_osd_files(in_path: Path) -> list[Path]:
    """Accept a folder or a zip; unzip zips next to themselves."""
    if in_path.is_file() and in_path.suffix.lower() == ".zip":
        out = in_path.with_suffix("")
        if not out.exists():
            with zipfile.ZipFile(in_path) as z:
                z.extractall(out)
        in_path = out
    if in_path.is_file():
        return [in_path]
    for z in in_path.rglob("*.zip"):
        out = z.with_suffix("")
        if not out.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    return sorted(p for p in in_path.rglob("*") if p.suffix.lower() in exts and not p.name.startswith("."))


def table_to_long(label, df, overrides, qc):
    """Convert one raw table into the tidy long schema. Returns DataFrame or None."""
    df = df.copy()
    df.columns = [norm_name(c) for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    cols = list(df.columns)
    info = {"table": label, "rows": len(df), "columns": cols[:40]}

    subj = overrides.get("subject") or find_col(cols, SUBJECT_COLS)
    tcol = overrides.get("time") or find_col(cols, TIME_COLS)
    scol = find_col(cols, SAMPLE_COLS, exclude=("range", "unit", "flag"))
    acol = overrides.get("analyte") or find_col(cols, ANALYTE_COLS)
    vcol = overrides.get("value") or find_col(cols, VALUE_COLS)
    ucol = find_col(cols, UNIT_COLS, exclude=())
    rmin = find_col(cols, RMIN_COLS, exclude=())
    rmax = find_col(cols, RMAX_COLS, exclude=())

    # ---- decide layout
    if acol and vcol and acol != vcol:
        layout = "long"
    else:
        meta = {c for c in (subj, tcol, scol, ucol, rmin, rmax) if c}
        numeric_cols = [c for c in cols if c not in meta and numeric_fraction(df[c]) > 0.6]
        layout = "wide" if len(numeric_cols) >= 5 else "unknown"
    info["layout"] = layout
    info["mapped"] = dict(subject=subj, time=tcol, sample=scol, analyte=acol, value=vcol,
                          unit=ucol, range_min=rmin, range_max=rmax)

    if layout == "unknown":
        info["skipped_reason"] = "no analyte/value columns and fewer than 5 numeric columns"
        qc["tables"].append(info)
        return None

    if layout == "wide":
        id_vars = [c for c in (subj, tcol, scol) if c]
        value_vars = numeric_cols
        long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="analyte", value_name="raw_value")
        long["unit"] = np.nan
        long["range_min"] = np.nan
        long["range_max"] = np.nan
    else:
        long = pd.DataFrame({
            "analyte": df[acol].astype(str).str.strip(),
            "raw_value": df[vcol],
        })
        for src, dst in ((subj, subj), (tcol, tcol), (scol, scol)):
            if src:
                long[src] = df[src].values
        long["unit"] = df[ucol].values if ucol else np.nan
        long["range_min"] = pd.to_numeric(df[rmin].astype(str).str.replace(",", ""), errors="coerce").values if rmin else np.nan
        long["range_max"] = pd.to_numeric(df[rmax].astype(str).str.replace(",", ""), errors="coerce").values if rmax else np.nan

    # ---- subject + timepoint: explicit columns, else parse from sample id / any text in row
    text_cols = [c for c in (subj, tcol, scol) if c and c in long.columns]
    if not text_cols:
        text_cols = [c for c in long.columns if c in df.columns and long[c].dtype == object]
    row_text = long[text_cols].astype(str).agg(" ".join, axis=1) if text_cols else pd.Series([""] * len(long))

    if subj and subj in long.columns:
        long["subject"] = long[subj].astype(str).str.strip()
    else:
        long["subject"] = row_text.str.extract(SUBJECT_RE, expand=False).str.replace(" ", "", regex=False).str.upper()
    parsed = (long[tcol].astype(str) if tcol and tcol in long.columns else row_text).map(parse_timepoint)
    long["timepoint"] = [p[0] for p in parsed]
    long["day_rel"] = [p[1] for p in parsed]
    long["phase"] = [p[2] for p in parsed]

    # fallback: timepoint column exists but isn't L-/R- style (e.g., 'pre-flight', 'R+1' elsewhere)
    miss = long["timepoint"].isna()
    if miss.any():
        alt = row_text[miss].map(parse_timepoint)
        long.loc[miss, "timepoint"] = [p[0] for p in alt]
        long.loc[miss, "day_rel"] = [p[1] for p in alt]
        long.loc[miss, "phase"] = [p[2] for p in alt]

    keep = ["subject", "timepoint", "day_rel", "phase", "analyte", "raw_value", "unit", "range_min", "range_max"]
    long = long[keep].copy()
    long["source_table"] = label
    info["rows_long"] = len(long)
    info["subjects_found"] = sorted(long["subject"].dropna().unique().tolist())[:20]
    info["timepoints_found"] = sorted(long["timepoint"].dropna().unique().tolist())
    qc["tables"].append(info)
    return long


def process_osd575(in_path: Path, out_dir: Path, overrides: dict, censor_policy: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    qc = {"input": str(in_path), "tables": [], "errors": []}
    files = collect_osd_files(in_path)
    if not files:
        sys.exit(f"[osd575] no csv/tsv/xlsx files found under {in_path}")
    print(f"[osd575] {len(files)} candidate file(s)")

    frames = []
    for f in files:
        for label, df in read_any_table(f):
            if isinstance(df, Exception):
                qc["errors"].append({"file": label, "error": str(df)})
                continue
            long = table_to_long(label, df, overrides, qc)
            if long is not None:
                frames.append(long)
    if not frames:
        save_json(qc, out_dir / "osd575_qc.json")
        sys.exit("[osd575] no usable tables. Open osd575_qc.json, check 'columns', then re-run with "
                 "--subject-col / --time-col / --analyte-col / --value-col.")

    long = pd.concat(frames, ignore_index=True)

    # ---- clean values
    parsed = long["raw_value"].map(parse_censored)
    long["value"] = [p[0] for p in parsed]
    long["flag"] = [p[1] for p in parsed]
    long["value"] = [apply_censor_policy(v, f, censor_policy) for v, f in zip(long["value"], long["flag"])]
    long["analyte"] = long["analyte"].astype(str).str.strip()
    long = long.drop(columns=["raw_value"])

    before = len(long)
    long = long.dropna(subset=["subject", "timepoint"])
    qc["rows_dropped_no_subject_or_timepoint"] = before - len(long)
    no_num = long["value"].isna()
    qc["rows_dropped_no_numeric_value"] = long.loc[no_num, "flag"].value_counts().to_dict()  # e.g. 'OOR <' with no limit
    long = long.loc[~no_num]
    long = long.drop_duplicates(subset=["subject", "timepoint", "analyte", "source_table"])
    long = long.sort_values(["subject", "day_rel", "analyte"]).reset_index(drop=True)

    cols = ["subject", "timepoint", "day_rel", "phase", "analyte", "value", "flag", "unit",
            "range_min", "range_max", "source_table"]
    long = long[cols]
    long.to_csv(out_dir / "osd575_long.csv", index=False)

    # ---- wide (duplicate analyte names across tables get a suffix so nothing is silently merged)
    dup = long.groupby(["subject", "timepoint", "analyte"])["source_table"].nunique()
    clash = set(dup[dup > 1].index.get_level_values("analyte"))
    w = long.copy()
    if clash:
        w.loc[w["analyte"].isin(clash), "analyte"] = w["analyte"] + "__" + w["source_table"].str.replace(r"\W+", "_", regex=True)
    wide = w.pivot_table(index=["subject", "timepoint", "day_rel", "phase"], columns="analyte",
                         values="value", aggfunc="mean").reset_index().sort_values(["subject", "day_rel"])
    wide.columns.name = None
    wide.to_csv(out_dir / "osd575_wide.csv", index=False)

    # ---- baseline-relative features (pre-flight mean per subject x analyte)
    pre = long[long["phase"] == "pre"].groupby(["subject", "analyte"])["value"].mean().rename("baseline")
    feat = long.merge(pre.reset_index(), on=["subject", "analyte"], how="left")
    eps = 1e-9
    feat["delta"] = feat["value"] - feat["baseline"]
    feat["fold_change"] = feat["value"] / feat["baseline"].where(feat["baseline"].abs() > eps)
    feat["log2_fc"] = np.log2(feat["fold_change"].where(feat["fold_change"] > 0))
    feat["out_of_range"] = np.where(
        feat["range_min"].notna() & feat["range_max"].notna(),
        (feat["value"] < feat["range_min"]) | (feat["value"] > feat["range_max"]), np.nan)
    feat.to_csv(out_dir / "osd575_features.csv", index=False)

    # ---- QC
    qc["summary"] = {
        "rows_long": int(len(long)),
        "subjects": sorted(long["subject"].unique().tolist()),
        "timepoints": sorted(long["timepoint"].unique().tolist(), key=lambda t: parse_timepoint(t)[1]),
        "n_analytes": int(long["analyte"].nunique()),
        "censored_values": long["flag"].value_counts().to_dict(),
        "analytes_without_baseline": sorted(set(long["analyte"]) - set(pre.reset_index()["analyte"]))[:30],
        "censor_policy": censor_policy,
        "mission_days_assumed": MISSION_DAYS,
    }
    save_json(qc, out_dir / "osd575_qc.json")
    s = qc["summary"]
    print(f"[osd575] long rows={s['rows_long']}  subjects={s['subjects']}  timepoints={s['timepoints']}  analytes={s['n_analytes']}")
    print(f"[osd575] censored: {s['censored_values']}")
    print(f"[osd575] wrote {out_dir}/osd575_long.csv, _wide.csv, _features.csv, _qc.json")


# =============================================================================
# 2) WESAD  (chest ECG/Resp/EDA/Temp/ACC at 700 Hz -> 60 s window features)
# =============================================================================

WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"   # public share, ~2.5 GB zip
FS = 700
LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}   # 0,5,6,7 = undefined/ignore
FEATURES = ["hr_mean", "sdnn", "rmssd", "resp_rate", "eda_mean", "eda_std", "eda_slope",
            "temp_mean", "temp_slope", "acc_mean", "acc_std"]


def download_wesad(dest: Path):
    import requests
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("S*.pkl")):
        print(f"[wesad] already present in {dest}, skipping download")
        return
    zpath = dest / "WESAD.zip"
    print(f"[wesad] downloading {WESAD_URL} -> {zpath}")
    with requests.get(WESAD_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(zpath, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="")
    print("\n[wesad] unzipping")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)


def _butter(x, fs, lo=None, hi=None, order=3):
    """Zero-phase Butterworth. SOS form: the b/a form is numerically unstable for low cutoffs at 700 Hz."""
    from scipy.signal import butter, sosfiltfilt
    if lo and hi:
        sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    elif hi:
        sos = butter(order, hi, btype="low", fs=fs, output="sos")
    else:
        sos = butter(order, lo, btype="high", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def detect_r_peaks(ecg, fs=FS):
    """Light Pan-Tompkins style detector: band-pass 5-15 Hz -> derivative^2 -> moving average -> peaks."""
    from scipy.signal import find_peaks
    x = _butter(ecg - np.nanmean(ecg), fs, 5, 15)
    x = np.square(np.diff(x, prepend=x[0]))
    win = max(1, int(0.12 * fs))
    x = np.convolve(x, np.ones(win) / win, mode="same")
    thr = 0.35 * np.percentile(x, 99)
    peaks, _ = find_peaks(x, height=thr, distance=int(0.30 * fs))
    return peaks


def detect_breaths(resp, fs=FS):
    from scipy.signal import find_peaks
    x = _butter(resp - np.nanmean(resp), fs, 0.1, 0.5)
    peaks, _ = find_peaks(x, distance=int(1.5 * fs), prominence=0.3 * np.std(x))
    return peaks


def _slope(y, fs):
    if len(y) < 3:
        return np.nan
    t = np.arange(len(y)) / fs
    return float(np.polyfit(t, y, 1)[0])


def wesad_subject_windows(pkl_path: Path, win_s=60, step_s=30, min_purity=0.9):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest = d["signal"]["chest"]
    labels = np.asarray(d["label"]).astype(int).ravel()
    ecg = np.asarray(chest["ECG"]).ravel().astype(float)
    resp = np.asarray(chest["Resp"]).ravel().astype(float)
    eda = _butter(np.asarray(chest["EDA"]).ravel().astype(float), FS, hi=5.0)
    temp = np.asarray(chest["Temp"]).ravel().astype(float)
    acc = np.linalg.norm(np.asarray(chest["ACC"]).astype(float), axis=1)

    n = min(len(ecg), len(labels), len(resp), len(eda), len(temp), len(acc))
    ecg, resp, eda, temp, acc, labels = ecg[:n], resp[:n], eda[:n], temp[:n], acc[:n], labels[:n]

    r_peaks = detect_r_peaks(ecg)
    breaths = detect_breaths(resp)

    W, S = int(win_s * FS), int(step_s * FS)
    rows, dropped = [], {"undefined_label": 0, "low_purity": 0}
    for start in range(0, n - W + 1, S):
        end = start + W
        lab = labels[start:end]
        vals, counts = np.unique(lab, return_counts=True)
        top = int(vals[np.argmax(counts)])
        purity = counts.max() / len(lab)
        if top not in LABELS:
            dropped["undefined_label"] += 1
            continue
        if purity < min_purity:
            dropped["low_purity"] += 1
            continue

        pk = r_peaks[(r_peaks >= start) & (r_peaks < end)]
        hr = sdnn = rmssd = np.nan
        if len(pk) >= 4:
            ibi = np.diff(pk) / FS * 1000.0
            ibi = ibi[(ibi > 300) & (ibi < 2000)]
            if len(ibi) >= 3:
                hr, sdnn = 60000.0 / ibi.mean(), ibi.std(ddof=1)
                rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
        nb = int(((breaths >= start) & (breaths < end)).sum())

        rows.append({
            "subject": pkl_path.stem,
            "t_start_s": start / FS, "t_end_s": end / FS,
            "label": top, "state": LABELS[top], "label_purity": round(float(purity), 3),
            "hr_mean": hr, "sdnn": sdnn, "rmssd": rmssd,
            "resp_rate": nb * 60.0 / win_s,
            "eda_mean": eda[start:end].mean(), "eda_std": eda[start:end].std(),
            "eda_slope": _slope(eda[start:end:10], FS / 10),
            "temp_mean": temp[start:end].mean(), "temp_slope": _slope(temp[start:end:10], FS / 10),
            "acc_mean": acc[start:end].mean(), "acc_std": acc[start:end].std(),
        })
    return pd.DataFrame(rows), dropped, n / FS


def process_wesad(data_dir: Path, out_dir: Path, subjects=None, win_s=60, step_s=30, min_purity=0.9):
    out_dir.mkdir(parents=True, exist_ok=True)
    pkls = sorted(data_dir.rglob("S*.pkl"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if subjects:
        want = {s.upper() for s in subjects}
        pkls = [p for p in pkls if p.stem.upper() in want]
    if not pkls:
        sys.exit(f"[wesad] no S*.pkl files under {data_dir} (run with --download, or check --data)")
    print(f"[wesad] {len(pkls)} subject file(s)")

    frames, qc = [], {"subjects": {}}
    for p in pkls:
        df, dropped, dur = wesad_subject_windows(p, win_s, step_s, min_purity)
        if df.empty:
            qc["subjects"][p.stem] = {"windows": 0, "note": "no usable windows"}
            continue
        df["subject"] = p.stem  # file stem is the source of truth (S2..S17)
        frames.append(df)
        qc["subjects"][p.stem] = {
            "duration_min": round(dur / 60, 1), "windows": int(len(df)), "dropped": dropped,
            "state_counts": df["state"].value_counts().to_dict(),
            "hr_median_bpm": round(float(df["hr_mean"].median()), 1),
            "hr_nan_pct": round(float(df["hr_mean"].isna().mean() * 100), 1),
        }
        print(f"  {p.stem:>4}: {dur / 60:5.1f} min  windows={len(df):3d}  HR median={df['hr_mean'].median():5.1f}  "
              f"states={df['state'].value_counts().to_dict()}")

    feats = pd.concat(frames, ignore_index=True)

    # ---- clean: physiologic plausibility, then fill small gaps within subject
    feats.loc[(feats["hr_mean"] < 35) | (feats["hr_mean"] > 200), ["hr_mean", "sdnn", "rmssd"]] = np.nan
    feats.loc[(feats["resp_rate"] < 4) | (feats["resp_rate"] > 40), "resp_rate"] = np.nan
    feats = feats.sort_values(["subject", "t_start_s"]).reset_index(drop=True)
    feats[FEATURES] = feats.groupby("subject")[FEATURES].transform(lambda s: s.interpolate(limit=2, limit_direction="both"))

    # ---- per-subject normalisation against that subject's own baseline windows (label 1)
    for col in FEATURES:
        base = feats[feats["label"] == 1].groupby("subject")[col].agg(["mean", "std"])
        m = feats["subject"].map(base["mean"])
        sd = feats["subject"].map(base["std"]).replace(0, np.nan)
        feats[f"{col}_z"] = (feats[col] - m) / sd

    feats.to_csv(out_dir / "wesad_features.csv", index=False)
    qc["summary"] = {"windows": int(len(feats)), "subjects": int(feats["subject"].nunique()),
                     "window_s": win_s, "step_s": step_s, "min_label_purity": min_purity,
                     "state_counts": feats["state"].value_counts().to_dict(),
                     "feature_nan_pct": (feats[FEATURES].isna().mean() * 100).round(1).to_dict()}
    save_json(qc, out_dir / "wesad_qc.json")
    print(f"[wesad] wrote {out_dir}/wesad_features.csv ({len(feats)} windows), wesad_qc.json")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_osd(p):
        p.add_argument("--in", dest="osd_in", required=True, help="folder or zip with the OSD-575 files")
        p.add_argument("--subject-col"); p.add_argument("--time-col")
        p.add_argument("--analyte-col"); p.add_argument("--value-col")
        p.add_argument("--censor-policy", choices=["half_lod", "limit", "nan"], default="half_lod",
                       help="how to treat '<x' / '>x' values (default: half the limit for '<', the limit for '>')")

    def add_wesad(p):
        p.add_argument("--data", default="data/WESAD", help="WESAD folder (contains S2/, S3/, ...)")
        p.add_argument("--download", action="store_true", help="download WESAD (2.5 GB) first")
        p.add_argument("--subjects", nargs="*", help="e.g. S2 S3 (default: all)")
        p.add_argument("--window", type=int, default=60); p.add_argument("--step", type=int, default=30)
        p.add_argument("--min-purity", type=float, default=0.9)

    for name in ("osd575", "wesad", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="processed")
        if name in ("osd575", "all"):
            add_osd(p)
        if name in ("wesad", "all"):
            add_wesad(p)

    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd in ("osd575", "all"):
        ov = {k: norm_name(v) for k, v in (("subject", a.subject_col), ("time", a.time_col),
                                           ("analyte", a.analyte_col), ("value", a.value_col)) if v}
        process_osd575(Path(a.osd_in), out, ov, a.censor_policy)
    if a.cmd in ("wesad", "all"):
        if a.download:
            download_wesad(Path(a.data))
        process_wesad(Path(a.data), out, a.subjects, a.window, a.step, a.min_purity)


if __name__ == "__main__":
    main()
