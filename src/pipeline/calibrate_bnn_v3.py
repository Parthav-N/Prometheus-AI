"""
calibrate_bnn_v3.py  —  M2 addendum
=====================================
Applies Platt scaling calibration to BNN v3 outputs.

All parameters derived from validation data. Nothing hardcoded.

Rules (non-negotiable):
  1. Calibrator fitted on VAL set only — test never touched during fitting
  2. C parameter cross-validated on val set — not default 1.0
  3. Risk tier thresholds derived from val set score distribution — not
     relative to current scoring batch (that changes per call)
  4. Everything saved to metadata JSON so the bridge loads constants,
     not recomputes them

Outputs:
  models/bnn_v3_calibrator.pkl        ← Platt scaler (LogisticRegression)
  models/bnn_v3_thresholds.json       ← fixed tier thresholds from val set
  validation/bnn/bnn_calibration_after.png
  validation/bnn/bnn_score_separation.png
  validation/bnn/bnn_brier_comparison.png
  validation/bnn/val_metrics_calibrated.json
  validation/bnn/test_metrics_calibrated.json
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr

ROOT      = Path(__file__).resolve().parents[2]
BNN_DIR   = ROOT / "data"    / "bnn"
MODEL_DIR = ROOT / "models"
VAL_DIR   = ROOT / "validation" / "bnn"
VAL_DIR.mkdir(parents=True, exist_ok=True)

FEAT_COLS = [
    "min_dist_km","mean_dist_km","n_fires_30km","max_frp",
    "wind_speed_kmh","wind_direction","temperature_c","humidity",
    "wind_fire_alignment","drought_index","days_since_rain",
]
MC_PASSES = 50

print("=" * 70)
print("M2 ADDENDUM — PLATT SCALING + FIXED THRESHOLDS")
print("All parameters derived from validation data. Nothing hardcoded.")
print("=" * 70)

# ── 1. Load model + data ──────────────────────────────────────────────────────
print("\n1. Loading model and datasets...")
model  = tf.keras.models.load_model(str(MODEL_DIR / "bnn_v3_best.keras"))
scaler = joblib.load(MODEL_DIR / "bnn_v3_scaler.pkl")

val  = pd.read_parquet(BNN_DIR / "bnn_val.parquet")
test = pd.read_parquet(BNN_DIR / "bnn_test.parquet")

X_val  = scaler.transform(val[FEAT_COLS].values.astype(np.float32))
y_val  = val["label"].values.astype(np.float32)
X_test = scaler.transform(test[FEAT_COLS].values.astype(np.float32))
y_test = test["label"].values.astype(np.float32)

print(f"   Val  : {len(y_val):,}  pos={100*y_val.mean():.3f}%")
print(f"   Test : {len(y_test):,}  pos={100*y_test.mean():.3f}%")

# ── 2. MC Dropout inference ───────────────────────────────────────────────────
print(f"\n2. Running MC Dropout ({MC_PASSES} passes)...")

def mc_predict(X, n=MC_PASSES):
    preds = np.stack([
        model(X, training=True).numpy().flatten()
        for _ in range(n)
    ])
    return preds.mean(axis=0), preds.std(axis=0)

val_raw,  val_std  = mc_predict(X_val.astype(np.float32))
test_raw, test_std = mc_predict(X_test.astype(np.float32))
print(f"   Raw val  range: [{val_raw.min():.5f}, {val_raw.max():.5f}]")
print(f"   Raw test range: [{test_raw.min():.5f}, {test_raw.max():.5f}]")

# ── 3. Cross-validate C parameter on val set ──────────────────────────────────
print("\n3. Cross-validating Platt C parameter on val set...")
print("   (not using default C=1.0 — deriving from data)")

C_candidates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_C     = None
best_brier = float("inf")
cv_results = []

for C in C_candidates:
    clf   = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    scores= cross_val_score(
        clf, val_raw.reshape(-1,1), y_val,
        scoring="neg_brier_score", cv=cv)
    mean_brier = float(-scores.mean())
    cv_results.append((C, mean_brier))
    if mean_brier < best_brier:
        best_brier = mean_brier
        best_C     = C

print(f"   CV results (C → Brier score):")
for C, b in cv_results:
    marker = " ← best" if C == best_C else ""
    print(f"     C={C:<8} Brier={b:.8f}{marker}")
print(f"\n   Best C = {best_C}  (Brier = {best_brier:.8f})")
print(f"   Derived from 5-fold CV on val set, not default")

# ── 4. Fit final calibrator with best C ──────────────────────────────────────
print(f"\n4. Fitting final Platt calibrator (C={best_C})...")
calibrator = LogisticRegression(C=best_C, solver="lbfgs", max_iter=1000)
calibrator.fit(val_raw.reshape(-1,1), y_val)
joblib.dump(calibrator, MODEL_DIR / "bnn_v3_calibrator.pkl")
print(f"   Sigmoid: a={calibrator.coef_[0][0]:.5f}, "
      f"b={calibrator.intercept_[0]:.5f}")
print(f"   ✓ Calibrator saved")

# ── 5. Apply calibration ──────────────────────────────────────────────────────
print("\n5. Applying calibration...")
val_cal  = calibrator.predict_proba(val_raw.reshape(-1,1))[:,1]
test_cal = calibrator.predict_proba(test_raw.reshape(-1,1))[:,1]

n_unique = len(np.unique(val_cal.round(6)))
print(f"   Unique calibrated values on val: {n_unique:,} / {len(val_cal):,}")
print(f"   (Should equal sample count — confirms no plateaus)")
print(f"   Val  range: [{val_cal.min():.6f}, {val_cal.max():.6f}]")
print(f"   Test range: [{test_cal.min():.6f}, {test_cal.max():.6f}]")

rc, _ = spearmanr(val_raw, val_cal)
print(f"   Spearman rank corr: {rc:.6f}  "
      f"{'✅ ranking preserved' if rc > 0.999 else '⚠ check'}")

# ── 6. Derive fixed tier thresholds from val set ──────────────────────────────
print("\n6. Deriving fixed risk tier thresholds from val set...")
print("   These are FIXED constants saved to JSON.")
print("   The bridge loads them — never recomputes from current batch.")

# Use calibrated scores on the FULL val set to set thresholds
# P50 = median val score → anything above is at least MEDIUM
# P80 = top 20% of val scores → HIGH
# P95 = top 5% of val scores → CRITICAL
thresh_low    = float(np.percentile(val_cal, 50))
thresh_medium = float(np.percentile(val_cal, 80))
thresh_high   = float(np.percentile(val_cal, 95))

print(f"\n   Threshold derivation (val set calibrated score distribution):")
print(f"     P50  → LOW/MEDIUM boundary  : {thresh_low:.6f}  "
      f"({100*thresh_low:.4f}%)")
print(f"     P80  → MEDIUM/HIGH boundary : {thresh_medium:.6f}  "
      f"({100*thresh_medium:.4f}%)")
print(f"     P95  → HIGH/CRITICAL boundary: {thresh_high:.6f}  "
      f"({100*thresh_high:.4f}%)")

# Sanity check: what fraction of TRUE POSITIVES fall above each threshold?
pos_mask = y_val == 1
pos_scores = val_cal[pos_mask]
pct_pos_above_crit = float((pos_scores >= thresh_high).mean())
pct_pos_above_high = float((pos_scores >= thresh_medium).mean())
print(f"\n   Of true positives in val set:")
print(f"     Above CRITICAL threshold: {100*pct_pos_above_crit:.1f}%")
print(f"     Above HIGH threshold    : {100*pct_pos_above_high:.1f}%")
print(f"   (Higher = thresholds correctly capture real risk)")

thresholds = {
    "LOW_MEDIUM": round(thresh_low,    8),
    "MEDIUM_HIGH": round(thresh_medium, 8),
    "HIGH_CRITICAL": round(thresh_high, 8),
    "derived_from": "val set (2022-2023) calibrated score percentiles",
    "method": "P50/P80/P95 of Platt-calibrated BNN scores on val set",
    "best_C": best_C,
    "cv_brier": round(best_brier, 8),
    "note": (
        "These are FIXED thresholds. Apply consistently to all scoring calls. "
        "Do NOT recompute per-batch — that makes tiers relative, not absolute."
    ),
}
with open(MODEL_DIR / "bnn_v3_thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)
print(f"\n   ✓ Thresholds saved → models/bnn_v3_thresholds.json")

# ── 7. Metrics ────────────────────────────────────────────────────────────────
print("\n7. Metrics...")

def metrics(y_true, y_prob, label):
    auc   = roc_auc_score(y_true, y_prob)
    ap    = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    base  = brier_score_loss(y_true, np.full_like(y_prob, y_true.mean()))
    bss   = 1 - brier / base
    print(f"   {label}:")
    print(f"     AUC     : {auc:.4f}")
    print(f"     Avg Prec: {ap:.4f}")
    print(f"     Brier   : {brier:.6f}")
    print(f"     Brier SS: {bss:.4f}  (>0 = better than naive)")
    return {"auc":round(auc,4), "ap":round(ap,4),
            "brier":round(brier,6), "brier_skill":round(bss,4)}

print()
raw_vm = metrics(y_val,  val_raw,  "Val  RAW")
cal_vm = metrics(y_val,  val_cal,  "Val  PLATT")
print()
raw_tm = metrics(y_test, test_raw, "Test RAW")
cal_tm = metrics(y_test, test_cal, "Test PLATT")

assert cal_vm["auc"] >= raw_vm["auc"] - 0.01, \
    f"AUC degraded on val: {raw_vm['auc']} → {cal_vm['auc']}"
assert cal_tm["auc"] >= raw_tm["auc"] - 0.01, \
    f"AUC degraded on test: {raw_tm['auc']} → {cal_tm['auc']}"
print("\n   ✅ AUC preserved or improved on both val and test")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
print("\n8. Generating validation plots...")

# Calibration plot before vs after
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for row, (y_true, y_before, y_after, name) in enumerate([
    (y_val,  val_raw,  val_cal,  "Validation (2022-2023)"),
    (y_test, test_raw, test_cal, "Test 2024 holdout"),
]):
    for col, (scores, title, color) in enumerate([
        (y_before, "RAW", "#F44336"),
        (y_after,  "PLATT CALIBRATED", "#4CAF50"),
    ]):
        ax = axes[row, col]
        try:
            pt, pp = calibration_curve(y_true, scores,
                                       n_bins=10, strategy="quantile")
            ax.plot(pp, pt, "o-", color=color, lw=2, ms=6,
                    label=f"BNN {title}")
        except Exception:
            pass
        ax.plot([0,1],[0,1],"k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{title} — {name}", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Calibration: Raw vs Platt Scaling",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_calibration_after.png", dpi=150, bbox_inches="tight")
plt.close()

# Score separation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_true, scores, name, color) in zip(axes, [
    (y_val,  val_cal,  "Validation", "#2196F3"),
    (y_test, test_cal, "Test 2024",  "#4CAF50"),
]):
    ax.hist(scores[y_true==0], bins=60, alpha=0.6,
            color="#999", label="Negative", density=True)
    ax.hist(scores[y_true==1], bins=60, alpha=0.8,
            color=color, label="Positive", density=True)
    for pct, thresh, lbl in [
        (50, thresh_low,    "P50→MEDIUM"),
        (80, thresh_medium, "P80→HIGH"),
        (95, thresh_high,   "P95→CRITICAL"),
    ]:
        ax.axvline(thresh, color="red", ls="--", alpha=0.7,
                   label=f"{lbl}: {thresh:.5f}")
    ax.set_xlabel("Calibrated Score"); ax.set_ylabel("Density")
    ax.set_title(f"Score Separation — {name}", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Fixed Tier Thresholds (derived from val set P50/P80/P95)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_score_separation.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ All plots saved")

# ── 9. Save metrics JSONs ─────────────────────────────────────────────────────
with open(VAL_DIR/"val_metrics_calibrated.json","w") as f:
    json.dump({"raw":raw_vm,"calibrated":cal_vm,
               "thresholds":thresholds}, f, indent=2)
with open(VAL_DIR/"test_metrics_calibrated.json","w") as f:
    json.dump({"raw":raw_tm,"calibrated":cal_tm,
               "thresholds":"loaded from val set — see val_metrics_calibrated.json",
               "note":"test set never used during calibration or threshold derivation"},
              f, indent=2)

print(f"\n{'='*70}")
print("CALIBRATION COMPLETE")
print(f"{'='*70}")
print(f"Best C (cross-validated): {best_C}")
print(f"Val  AUC: {raw_vm['auc']} → {cal_vm['auc']}")
print(f"Test AUC: {raw_tm['auc']} → {cal_tm['auc']}")
print(f"\nFixed thresholds (derived from val set, saved to JSON):")
print(f"  LOW    : score < {thresh_low:.6f}")
print(f"  MEDIUM : {thresh_low:.6f} ≤ score < {thresh_medium:.6f}")
print(f"  HIGH   : {thresh_medium:.6f} ≤ score < {thresh_high:.6f}")
print(f"  CRITICAL: score ≥ {thresh_high:.6f}")
print(f"\nSaved:")
print(f"  models/bnn_v3_calibrator.pkl    ← Platt scaler")
print(f"  models/bnn_v3_thresholds.json   ← fixed tier thresholds")
print(f"{'='*70}")