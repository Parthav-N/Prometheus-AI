"""
train_bnn_v3.py  —  M2
=======================
Trains BNN v3 on real FIRMS proximity labels from M1.

Architecture: 4-layer dense network with MC Dropout
  256 → 128 → 64 → 32 → 1 (sigmoid)

Key design decisions:
  - Binary classification (fire reached asset or not)
  - Class weights handle 0.3% positive rate imbalance
  - MC Dropout: 50 passes at inference → probability + uncertainty
  - Early stopping on val AUC (not loss — loss misleads with imbalance)
  - Strict use of train/val during training, test only at the end

Validation outputs (all saved to validation/bnn/):
  - ROC curve + AUC
  - Precision-recall curve + AP
  - Calibration plot (reliability diagram)
  - Confusion matrix at optimal threshold
  - Loss + AUC curves per epoch
  - Feature permutation importance
  - val_metrics.json + test_metrics.json

Run from project root:
  python src/pipeline/train_bnn_v3.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve,
                              confusion_matrix, f1_score)
from sklearn.calibration import calibration_curve

ROOT     = Path(__file__).resolve().parents[2]
BNN_DIR  = ROOT / "data" / "bnn"
MODEL_DIR= ROOT / "models"
VAL_DIR  = ROOT / "validation" / "bnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

FEAT_COLS = [
    "min_dist_km","mean_dist_km","n_fires_30km","max_frp",
    "wind_speed_kmh","wind_direction","temperature_c","humidity",
    "wind_fire_alignment","drought_index","days_since_rain",
]
LABEL_COL = "label"

MC_PASSES  = 50
BATCH_SIZE = 2048
MAX_EPOCHS = 100
LR         = 1e-3
DROPOUT    = 0.3

print("=" * 70)
print("M2 — TRAIN BNN v3 ON REAL FIRMS LABELS")
print("=" * 70)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n1. Loading datasets...")
train = pd.read_parquet(BNN_DIR / "bnn_train.parquet")
val   = pd.read_parquet(BNN_DIR / "bnn_val.parquet")
test  = pd.read_parquet(BNN_DIR / "bnn_test.parquet")

X_train = train[FEAT_COLS].values.astype(np.float32)
y_train = train[LABEL_COL].values.astype(np.float32)
X_val   = val[FEAT_COLS].values.astype(np.float32)
y_val   = val[LABEL_COL].values.astype(np.float32)
X_test  = test[FEAT_COLS].values.astype(np.float32)
y_test  = test[LABEL_COL].values.astype(np.float32)

print(f"   Train: {len(X_train):,}  pos={100*y_train.mean():.2f}%")
print(f"   Val  : {len(X_val):,}  pos={100*y_val.mean():.2f}%")
print(f"   Test : {len(X_test):,}  pos={100*y_test.mean():.2f}%")

# ── 2. Scale features ─────────────────────────────────────────────────────────
print("\n2. Scaling features (fit on train only)...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, MODEL_DIR / "bnn_v3_scaler.pkl")
print("   ✓ Scaler fitted and saved")

# ── 3. Class weights ──────────────────────────────────────────────────────────
n_neg  = int((y_train == 0).sum())
n_pos  = int((y_train == 1).sum())
pos_weight = n_neg / n_pos
print(f"\n3. Class imbalance: {n_pos:,} pos / {n_neg:,} neg")
print(f"   Positive class weight: {pos_weight:.1f}x")
print(f"   (Without weighting, predicting all-zeros = "
      f"{100*(1-y_train.mean()):.1f}% accuracy — meaningless)")

sample_weights = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)

# ── 4. Build model ────────────────────────────────────────────────────────────
print("\n4. Building BNN (MC Dropout architecture)...")

def build_bnn(n_features, dropout_rate=DROPOUT):
    inputs = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, output)

model = build_bnn(len(FEAT_COLS))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.AUC(name="auprc", curve="PR")],
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\n5. Training...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=8,
        mode="max", restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", factor=0.5, patience=4,
        mode="max", min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR/"bnn_v3_best.keras"),
        monitor="val_auc", save_best_only=True, mode="max"),
]

history = model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

best_epoch = int(np.argmax(history.history["val_auc"])) + 1
best_auc   = float(max(history.history["val_auc"]))
print(f"\n   Best epoch: {best_epoch}  val_auc: {best_auc:.4f}")

# ── 6. MC Dropout inference ───────────────────────────────────────────────────

def mc_predict(X, n_passes=MC_PASSES):
    """Run MC Dropout: n_passes forward passes with dropout active."""
    preds = np.stack([
        model(X, training=True).numpy().flatten()
        for _ in range(n_passes)
    ])  # (n_passes, N)
    return preds.mean(axis=0), preds.std(axis=0)

print(f"\n6. MC Dropout inference ({MC_PASSES} passes)...")
val_mean,  val_std  = mc_predict(X_val.astype(np.float32))
test_mean, test_std = mc_predict(X_test.astype(np.float32))
print("   ✓ Done")

# ── 7. Find optimal threshold on VAL (never on test) ─────────────────────────
print("\n7. Finding optimal threshold on val set...")
fpr, tpr, thresholds = roc_curve(y_val, val_mean)
j_scores    = tpr - fpr
best_thresh = float(thresholds[np.argmax(j_scores)])
print(f"   Optimal threshold (Youden J): {best_thresh:.4f}")

# ── 8. Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold, name):
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    ap     = average_precision_score(y_true, y_prob)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"\n   {name}:")
    print(f"     ROC-AUC  : {auc:.4f}")
    print(f"     Avg Prec : {ap:.4f}")
    print(f"     Precision: {precision:.4f}")
    print(f"     Recall   : {recall:.4f}")
    print(f"     F1       : {f1:.4f}")
    print(f"     TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    return {"auc":round(auc,4),"ap":round(ap,4),
            "precision":round(precision,4),"recall":round(recall,4),
            "f1":round(f1,4),"tp":int(tp),"fp":int(fp),
            "tn":int(tn),"fn":int(fn),"threshold":round(threshold,4)}

print("\n8. Computing metrics...")
val_metrics  = compute_metrics(y_val,  val_mean,  best_thresh, "Validation")
test_metrics = compute_metrics(y_test, test_mean, best_thresh, "Test (2024 holdout)")

with open(VAL_DIR/"val_metrics.json","w")  as f: json.dump(val_metrics,  f, indent=2)
with open(VAL_DIR/"test_metrics.json","w") as f: json.dump(test_metrics, f, indent=2)
print("   Saved metrics JSONs")

# ── 9. Validation plots ───────────────────────────────────────────────────────
print("\n9. Generating validation plots...")

# ── Plot 1: ROC curve ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_true, y_prob, name, color) in zip(axes, [
    (y_val,  val_mean,  "Validation (2022-2023)", "#2196F3"),
    (y_test, test_mean, "Test 2024 holdout",      "#4CAF50"),
]):
    fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)
    auc_c = roc_auc_score(y_true, y_prob)
    ax.plot(fpr_c, tpr_c, color=color, lw=2,
            label=f"AUC = {auc_c:.4f}")
    ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {name}", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("BNN v3 ROC Curves — Real FIRMS Labels", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ bnn_roc_curve.png")

# ── Plot 2: Precision-recall curve ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_true, y_prob, name, color) in zip(axes, [
    (y_val,  val_mean,  "Validation", "#2196F3"),
    (y_test, test_mean, "Test 2024",  "#4CAF50"),
]):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, color=color, lw=2, label=f"AP = {ap:.4f}")
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", ls="--", alpha=0.5,
               label=f"Baseline (random) = {baseline:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {name}", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("BNN v3 Precision-Recall Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ bnn_precision_recall.png")

# ── Plot 3: Calibration plot (reliability diagram) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_true, y_prob, name, color) in zip(axes, [
    (y_val,  val_mean,  "Validation", "#2196F3"),
    (y_test, test_mean, "Test 2024",  "#4CAF50"),
]):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile")
    ax.plot(prob_pred, prob_true, "o-", color=color, lw=2,
            label="BNN v3")
    ax.plot([0,1],[0,1],"k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Plot — {name}\n"
                 f"(If predicted prob=0.7, ~70% of those assets should actually get fire)",
                 fontweight="bold", fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("BNN v3 Calibration — Do Risk Scores Mean What They Say?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ bnn_calibration.png")

# ── Plot 4: Confusion matrix ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (y_true, y_prob, name) in zip(axes, [
    (y_val,  val_mean,  "Validation"),
    (y_test, test_mean, "Test 2024"),
]):
    y_pred = (y_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred: No fire","Pred: Fire"])
    ax.set_yticklabels(["True: No fire","True: Fire"])
    ax.set_title(f"Confusion Matrix — {name}\n"
                 f"Threshold: {best_thresh:.3f}", fontweight="bold")
fig.suptitle("BNN v3 Confusion Matrices", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ bnn_confusion_matrix.png")

# ── Plot 5: Training curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["loss"],     label="Train loss", color="#F44336")
axes[0].plot(history.history["val_loss"], label="Val loss",   color="#2196F3")
axes[0].axvline(best_epoch-1, color="green", ls="--", label=f"Best epoch {best_epoch}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].set_title("Training Loss", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history["auc"],     label="Train AUC", color="#F44336")
axes[1].plot(history.history["val_auc"], label="Val AUC",   color="#2196F3")
axes[1].axvline(best_epoch-1, color="green", ls="--", label=f"Best epoch {best_epoch}")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("ROC-AUC During Training", fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)

fig.suptitle("BNN v3 Training Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ bnn_loss_curves.png")

# ── Plot 6: Uncertainty distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_true, y_std, name) in zip(axes, [
    (y_val,  val_std,  "Validation"),
    (y_test, test_std, "Test 2024"),
]):
    ax.hist(y_std[y_true==0], bins=50, alpha=0.6,
            color="#F44336", label="True Negative", density=True)
    ax.hist(y_std[y_true==1], bins=50, alpha=0.6,
            color="#2196F3", label="True Positive", density=True)
    ax.set_xlabel("MC Dropout Uncertainty (std)")
    ax.set_ylabel("Density")
    ax.set_title(f"Uncertainty Distribution — {name}\n"
                 f"Higher uncertainty = model less confident",
                 fontweight="bold", fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("BNN v3 Uncertainty — MC Dropout Std Across 50 Passes",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_uncertainty.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ bnn_uncertainty.png")

# ── Plot 7: Feature permutation importance ────────────────────────────────────
print("\n   Computing feature permutation importance (val set)...")
baseline_auc = roc_auc_score(y_val, val_mean)
importances  = []

for fi, feat in enumerate(FEAT_COLS):
    X_perm         = X_val.copy()
    X_perm[:, fi]  = np.random.permutation(X_perm[:, fi])
    perm_mean, _   = mc_predict(X_perm.astype(np.float32), n_passes=10)
    perm_auc       = roc_auc_score(y_val, perm_mean)
    importances.append(baseline_auc - perm_auc)

fig, ax = plt.subplots(figsize=(10, 6))
order = np.argsort(importances)[::-1]
colors = ["#F44336" if imp > 0 else "#9E9E9E" for imp in np.array(importances)[order]]
ax.barh([FEAT_COLS[o] for o in order],
        [importances[o] for o in order], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("AUC Drop When Feature Permuted\n(higher = more important)")
ax.set_title("Feature Permutation Importance\n"
             "Key check: min_dist_km must be most important",
             fontweight="bold")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(VAL_DIR/"bnn_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ bnn_feature_importance.png")

# Print importance ranking
print("   Feature importances (AUC drop):")
for o in order:
    print(f"     {FEAT_COLS[o]:<25} {importances[o]:+.5f}")

# ── 10. Save model and metadata ───────────────────────────────────────────────
print("\n10. Saving model and metadata...")
model.save(str(MODEL_DIR / "bnn_v3.keras"))

metadata = {
    "model":           "BNN v3",
    "label_source":    "NASA FIRMS — real satellite fire detections",
    "label_definition":"fire within 5km of asset within next 24h",
    "no_synthetic_labels": True,
    "architecture":    {"layers":[256,128,64,32,1],
                        "dropout":DROPOUT,"activation":"relu"},
    "mc_passes":       MC_PASSES,
    "features":        FEAT_COLS,
    "n_features":      len(FEAT_COLS),
    "optimal_threshold": best_thresh,
    "class_pos_weight":  round(pos_weight, 2),
    "train_period":    "2017-2021",
    "val_period":      "2022-2023",
    "test_period":     "2024",
    "val_metrics":     val_metrics,
    "test_metrics":    test_metrics,
    "best_epoch":      best_epoch,
    "best_val_auc":    round(best_auc, 4),
}
with open(MODEL_DIR/"bnn_v3_metadata.json","w") as f:
    json.dump(metadata, f, indent=2)
print("   ✓ bnn_v3.keras + bnn_v3_metadata.json saved")

print(f"\n{'='*70}")
print("M2 COMPLETE")
print(f"{'='*70}")
print(f"Val  AUC : {val_metrics['auc']}")
print(f"Val  AP  : {val_metrics['ap']}")
print(f"Test AUC : {test_metrics['auc']}  ← 2024 holdout, never seen during training")
print(f"Test AP  : {test_metrics['ap']}")
print(f"\nKey validation outputs → {VAL_DIR}/")
print(f"  bnn_roc_curve.png")
print(f"  bnn_calibration.png       ← most important: do scores mean what they say?")
print(f"  bnn_confusion_matrix.png")
print(f"  bnn_feature_importance.png ← min_dist_km must be top feature")
print(f"  bnn_loss_curves.png")
print(f"  bnn_uncertainty.png")
print(f"  val_metrics.json + test_metrics.json")
print(f"\nNext: M3 — python src/pipeline/fetch_nifc_perimeters.py")
print(f"{'='*70}")