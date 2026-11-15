"""
Astronaut Health Monitoring System - Model Training Pipeline
Trains:
1. Multimodal State Classifier (baseline, stress, meditation, amusement)
2. Binary Acute Stress Detector (stress vs non-stress)
3. Unsupervised Physiological Anomaly Detector (Isolation Forest)
Evaluated with GroupKFold cross-validation across subjects.
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FEATURES_RAW = [
    "hr_mean", "sdnn", "rmssd", "resp_rate",
    "eda_mean", "eda_std", "eda_slope",
    "temp_mean", "temp_slope", "acc_mean", "acc_std"
]

FEATURES_Z = [f"{col}_z" for col in FEATURES_RAW]
ALL_FEATURES = FEATURES_RAW + FEATURES_Z


def train_and_evaluate(data_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"[Training] Loaded {len(df)} windows across {df['subject'].nunique()} subjects.")

    X = df[ALL_FEATURES].values
    groups = df["subject"].values
    
    # 1. Multi-class target
    y_multi = df["state"].values
    classes = sorted(np.unique(y_multi).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_multi_idx = np.array([class_to_idx[c] for c in y_multi])

    # 2. Binary target: Stress (1) vs Non-stress (0)
    y_binary = (df["state"] == "stress").astype(int).values

    print("\n--- Running 5-Fold Group Cross-Validation (Binary Stress) ---")
    gkf = GroupKFold(n_splits=5)
    binary_accs, binary_f1s, binary_aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_binary, groups=groups), 1):
        X_tr, y_tr = X[train_idx], y_binary[train_idx]
        X_va, y_va = X[val_idx], y_binary[val_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
        clf.fit(X_tr_sc, y_tr)
        preds = clf.predict(X_va_sc)
        probs = clf.predict_proba(X_va_sc)[:, 1]

        acc = accuracy_score(y_va, preds)
        f1 = f1_score(y_va, preds, zero_division=0)
        auc = roc_auc_score(y_va, probs)

        binary_accs.append(acc)
        binary_f1s.append(f1)
        binary_aucs.append(auc)
        print(f"  Fold {fold} - Acc: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {auc:.3f}")

    print(f"Binary Mean CV - Accuracy: {np.mean(binary_accs):.3f} +/- {np.std(binary_accs):.3f}")
    print(f"Binary Mean CV - F1 Score: {np.mean(binary_f1s):.3f} +/- {np.std(binary_f1s):.3f}")
    print(f"Binary Mean CV - ROC-AUC:  {np.mean(binary_aucs):.3f} +/- {np.std(binary_aucs):.3f}")

    # Full Dataset Training & Scaler Fit
    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X)

    # Train Final Binary Stress Classifier
    final_binary_model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
    )
    final_binary_model.fit(X_scaled, y_binary)

    # Train Final Multiclass State Classifier (Random Forest for robust multi-class probability calibration)
    final_multi_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    final_multi_model.fit(X_scaled, y_multi_idx)

    # Train Unsupervised Anomaly Detector (fit on baseline and meditation data)
    normal_mask = df["state"].isin(["baseline", "meditation"]).values
    anomaly_detector = IsolationForest(
        n_estimators=100, contamination=0.05, random_state=42
    )
    anomaly_detector.fit(X_scaled[normal_mask])

    # Feature Importance
    importances = dict(zip(ALL_FEATURES, final_binary_model.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    metrics = {
        "dataset_windows": len(df),
        "subjects": df["subject"].nunique(),
        "states_distribution": df["state"].value_counts().to_dict(),
        "cv_binary_accuracy_mean": float(np.mean(binary_accs)),
        "cv_binary_accuracy_std": float(np.std(binary_accs)),
        "cv_binary_f1_mean": float(np.mean(binary_f1s)),
        "cv_binary_f1_std": float(np.std(binary_f1s)),
        "cv_binary_roc_auc_mean": float(np.mean(binary_aucs)),
        "cv_binary_roc_auc_std": float(np.std(binary_aucs)),
        "classes": classes,
        "feature_importances": top_features,
    }

    # Save artifacts
    joblib.dump(final_binary_model, output_dir / "binary_stress_model.joblib")
    joblib.dump(final_multi_model, output_dir / "multi_state_model.joblib")
    joblib.dump(anomaly_detector, output_dir / "anomaly_detector.joblib")
    joblib.dump(full_scaler, output_dir / "feature_scaler.joblib")

    meta = {
        "features": ALL_FEATURES,
        "features_raw": FEATURES_RAW,
        "features_z": FEATURES_Z,
        "classes": classes,
        "metrics": metrics,
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Training] Completed successfully. Artifacts saved in {output_dir}")
    print("\nTop 5 Most Predictive Biometric Features for Acute Stress:")
    for feat, imp in top_features[:5]:
        print(f"  - {feat}: {imp*100:.2f}%")


if __name__ == "__main__":
    train_and_evaluate(
        data_path=Path("processed/wesad_features.csv"),
        output_dir=Path("models/saved"),
    )
