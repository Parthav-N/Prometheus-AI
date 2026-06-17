"""
train_tft_v2.py  —  M5
=======================
Trains TFT v2: multi-output quantile regressor for radial fire spread.

Architecture: shared encoder + 8 directional heads
  Each head outputs [P10, P50, P90] for expansion in that direction.
  24 total outputs (8 directions × 3 quantiles).

With 1,898 training samples, a small network is correct.
Larger = overfitting. We use:
  Input → BatchNorm → Dense(64) → Dense(32) → Dense(16) → 8×Dense(3)

Loss: pinball (quantile) loss — correct loss for quantile regression.
  P10 quantile: underestimates penalized at 0.9×, overestimates at 0.1×
  P50 quantile: symmetric (= MAE)
  P90 quantile: overestimates penalized at 0.1×, underestimates at 0.9×

Calibration: derived from val median ratio (actual P50 / predicted P50).
  This replaces the 20× magic number with a defensible derived value.

Validation outputs (all saved to validation/tft/):
  tft_predicted_vs_actual.png
  tft_quantile_coverage.png
  tft_loss_curves.png
  tft_calibration_factor.png
  tft_val_metrics.json
  tft_test_metrics.json
  tft_calibration_factor.json
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
from sklearn.metrics import mean_absolute_error

ROOT     = Path(__file__).resolve().parents[2]
TFT_DIR  = ROOT / "data"    / "tft_v2"
MODEL_DIR= ROOT / "models"
VAL_DIR  = ROOT / "validation" / "tft"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

N_DIRS   = 8
DIRS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
DIR_NAMES= [f"{d:03d}" for d in DIRS_DEG]
DIR_LABELS= ["N","NE","E","SE","S","SW","W","NW"]
LABEL_COLS  = [f"radial_delta_{n}" for n in DIR_NAMES]
PROFILE_COLS= [f"r_{n}" for n in DIR_NAMES]
FEAT_COLS   = [
    "area_km2","n_points","max_frp","mean_frp",
    "wind_speed_kmh","wind_direction","temperature_c",
    "humidity","drought_index","days_since_rain",
] + PROFILE_COLS

QUANTILES  = [0.1, 0.5, 0.9]
BATCH_SIZE = 64     # small — 1898 training samples
MAX_EPOCHS = 200
LR         = 1e-3
DROPOUT    = 0.3
CLIP_PCT   = 0.02   # clip labels at 2nd/98th percentile

print("=" * 70)
print("M5 — TRAIN TFT v2 (multi-output quantile regressor)")
print(f"8 directional heads × 3 quantiles = 24 total outputs")
print("=" * 70)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n1. Loading datasets...")
train = pd.read_parquet(TFT_DIR / "tft_train.parquet")
val   = pd.read_parquet(TFT_DIR / "tft_val.parquet")
test  = pd.read_parquet(TFT_DIR / "tft_test.parquet")

print(f"   Train: {len(train):,}")
print(f"   Val  : {len(val):,}")
print(f"   Test : {len(test):,}")

# ── 2. Clip extreme labels ────────────────────────────────────────────────────
print("\n2. Clipping extreme labels at 2nd/98th percentile (train only)...")
clip_bounds = {}
for col in LABEL_COLS:
    lo = float(train[col].quantile(CLIP_PCT))
    hi = float(train[col].quantile(1-CLIP_PCT))
    clip_bounds[col] = (lo, hi)
    train[col] = train[col].clip(lo, hi)
print(f"   Clip example (N expansion): "
      f"[{clip_bounds[LABEL_COLS[0]][0]:.2f}, "
      f"{clip_bounds[LABEL_COLS[0]][1]:.2f}] km")

# ── 3. Prepare features ───────────────────────────────────────────────────────
print("\n3. Preparing features...")
X_train = train[FEAT_COLS].values.astype(np.float32)
X_val   = val[FEAT_COLS].values.astype(np.float32)
X_test  = test[FEAT_COLS].values.astype(np.float32)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, MODEL_DIR / "tft_v2_scaler.pkl")

Y_train = train[LABEL_COLS].values.astype(np.float32)  # (N, 8)
Y_val   = val[LABEL_COLS].values.astype(np.float32)
Y_test  = test[LABEL_COLS].values.astype(np.float32)
print(f"   Features: {X_train.shape[1]}, Labels: {Y_train.shape[1]}")

# ── 4. Build model ────────────────────────────────────────────────────────────
print("\n4. Building model...")

def pinball_loss(q):
    """Quantile (pinball) loss for quantile q."""
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))
    loss.__name__ = f"pinball_{int(q*100)}"
    return loss

def build_model(n_features, n_dirs=N_DIRS, n_quantiles=3, dropout=DROPOUT):
    """
    Shared encoder → N_DIRS × N_QUANTILES output heads.
    Small network appropriate for ~1900 training samples.
    """
    inp = tf.keras.Input(shape=(n_features,))

    # Shared encoder
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)

    # 8 directional heads, each outputting [P10, P50, P90]
    outputs = []
    for i in range(n_dirs):
        head = tf.keras.layers.Dense(8, activation="relu",
                                     name=f"head_{i}")(x)
        out  = tf.keras.layers.Dense(n_quantiles,
                                     name=f"out_{i}")(head)
        outputs.append(out)

    # Concatenate: shape (batch, n_dirs * n_quantiles) = (batch, 24)
    output = tf.keras.layers.Concatenate(name="output")(outputs)
    return tf.keras.Model(inp, output)

model = build_model(X_train.shape[1])
model.summary()

# Build Y with shape (N, 24): each direction has 3 quantile targets
# Y layout: [dir0_P10, dir0_P50, dir0_P90, dir1_P10, ...]
Y_train_full = np.repeat(Y_train, 3, axis=1)  # (N, 24) — same target for all quantiles
Y_val_full   = np.repeat(Y_val,   3, axis=1)

# Combined quantile loss: mean of 3 pinball losses across all outputs
def combined_quantile_loss(y_true, y_pred):
    """
    y_true, y_pred: (batch, 24)
    For each direction i, outputs at indices [3i, 3i+1, 3i+2] are P10, P50, P90.
    """
    total = 0.0
    for i in range(N_DIRS):
        for j, q in enumerate(QUANTILES):
            idx = i*3 + j
            total += pinball_loss(q)(y_true[:,idx], y_pred[:,idx])
    return total / (N_DIRS * len(QUANTILES))

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=combined_quantile_loss,
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\n5. Training...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20,
        restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10,
        min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR/"tft_v2_best.keras"),
        monitor="val_loss", save_best_only=True),
]

history = model.fit(
    X_train, Y_train_full,
    validation_data=(X_val, Y_val_full),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

best_epoch = int(np.argmin(history.history["val_loss"])) + 1
best_loss  = float(min(history.history["val_loss"]))
print(f"\n   Best epoch: {best_epoch}  val_loss: {best_loss:.6f}")

# ── 6. Extract predictions ────────────────────────────────────────────────────
print("\n6. Extracting predictions...")

def predict_quantiles(X):
    """Returns dict: p10, p50, p90 each shape (N, 8)."""
    raw = model.predict(X, batch_size=256, verbose=0)  # (N, 24)
    p10 = raw[:, 0::3]   # every 3rd starting at 0
    p50 = raw[:, 1::3]   # every 3rd starting at 1
    p90 = raw[:, 2::3]   # every 3rd starting at 2
    return {"p10":p10, "p50":p50, "p90":p90}

val_pred  = predict_quantiles(X_val)
test_pred = predict_quantiles(X_test)

# ── 7. Compute metrics ────────────────────────────────────────────────────────
print("\n7. Computing metrics...")

def compute_metrics(y_true, preds, name):
    """y_true: (N,8), preds: dict p10/p50/p90 each (N,8)"""
    p50 = preds["p50"]
    p10 = preds["p10"]
    p90 = preds["p90"]

    mae_per_dir = [mean_absolute_error(y_true[:,i], p50[:,i])
                   for i in range(N_DIRS)]
    mae_overall = float(np.mean(mae_per_dir))

    # Quantile coverage: what % of actuals fall within [P10, P90]?
    covered = ((y_true >= p10) & (y_true <= p90)).mean()

    print(f"\n   {name}:")
    print(f"     MAE (P50) overall: {mae_overall:.4f} km")
    print(f"     MAE per direction:")
    for i, (dl, mae) in enumerate(zip(DIR_LABELS, mae_per_dir)):
        print(f"       {dl:2s}: {mae:.4f} km")
    print(f"     P10-P90 coverage: {100*covered:.1f}%  "
          f"(target: ~80%)")

    return {
        "mae_overall":    round(mae_overall,4),
        "mae_per_dir":    {DIR_LABELS[i]:round(mae_per_dir[i],4)
                           for i in range(N_DIRS)},
        "p10_p90_coverage": round(float(covered),4),
    }

val_metrics  = compute_metrics(Y_val,  val_pred,  "Validation")
test_metrics = compute_metrics(Y_test, test_pred, "Test 2024 holdout")

# ── 8. Calibration factor ─────────────────────────────────────────────────────
print("\n8. Deriving calibration factor from val set...")
# For each direction: median(actual / predicted_P50) where predicted != 0
cal_factors = []
for i in range(N_DIRS):
    actual = Y_val[:, i]
    pred   = val_pred["p50"][:, i]
    # Only where prediction is non-trivial
    mask   = np.abs(pred) > 0.01
    if mask.sum() > 5:
        ratios = actual[mask] / pred[mask]
        # Winsorize ratios to remove extreme outliers
        ratios = np.clip(ratios,
                         np.percentile(ratios, 5),
                         np.percentile(ratios, 95))
        cal_factors.append(float(np.median(ratios)))
    else:
        cal_factors.append(1.0)

overall_cal = float(np.median(cal_factors))
print(f"   Calibration factor per direction:")
for dl, cf in zip(DIR_LABELS, cal_factors):
    print(f"     {dl:2s}: {cf:.4f}")
print(f"   Overall median: {overall_cal:.4f}")
print(f"   (1.0 = perfectly calibrated, >1 = underpredicting, <1 = overpredicting)")

cal_data = {
    "overall":          round(overall_cal, 4),
    "per_direction":    {DIR_LABELS[i]: round(cal_factors[i], 4)
                         for i in range(N_DIRS)},
    "derived_from":     "val set median(actual/predicted_P50)",
    "val_samples":      len(Y_val),
    "interpretation":   (
        "Multiply P50 prediction by this factor before displaying on dashboard. "
        "Derived from 2022-2023 validation fires, not manually chosen."
    ),
}
with open(MODEL_DIR/"tft_v2_calibration_factor.json","w") as f:
    json.dump(cal_data, f, indent=2)
with open(VAL_DIR/"tft_calibration_factor.json","w") as f:
    json.dump(cal_data, f, indent=2)

# ── 9. Validation plots ───────────────────────────────────────────────────────
print("\n9. Generating validation plots...")

# Plot 1: Predicted vs actual per direction (val)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, (dl, dname) in enumerate(zip(DIR_LABELS, DIR_NAMES)):
    ax = axes[i]
    y_true = Y_val[:, i]
    y_pred = val_pred["p50"][:, i]
    # Apply calibration
    y_cal  = y_pred * cal_factors[i]

    lim = max(abs(y_true).max(), abs(y_cal).max()) * 1.1
    ax.scatter(y_true, y_cal, alpha=0.5, s=15, color="#FF7043")
    ax.plot([-lim,lim],[-lim,lim],"k--",alpha=0.5,label="Perfect")
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_xlabel("Actual expansion (km)")
    ax.set_ylabel("Predicted P50 (km)")
    mae = mean_absolute_error(y_true, y_cal)
    ax.set_title(f"{dl} ({dname}°)  MAE={mae:.2f}km", fontweight="bold")
    ax.grid(alpha=0.3)
fig.suptitle("TFT v2 — Predicted vs Actual Radial Expansion\n"
             "Val set (2022-2023). Cal factor applied.",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(VAL_DIR/"tft_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ tft_predicted_vs_actual.png")

# Plot 2: Loss curves
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history["loss"],     color="#F44336", lw=2, label="Train loss")
ax.plot(history.history["val_loss"], color="#2196F3", lw=2, label="Val loss")
ax.axvline(best_epoch-1, color="green", lw=2, ls="--",
           label=f"Best epoch {best_epoch}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Combined Quantile Loss")
ax.set_title("TFT v2 Training Curves", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VAL_DIR/"tft_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ tft_loss_curves.png")

# Plot 3: Quantile coverage per direction
fig, ax = plt.subplots(figsize=(12, 5))
for split, (y_true, preds, color, name) in {
    "val":  (Y_val,  val_pred,  "#2196F3", "Validation"),
    "test": (Y_test, test_pred, "#4CAF50", "Test 2024"),
}.items():
    coverages = []
    for i in range(N_DIRS):
        cov = float(((y_true[:,i] >= preds["p10"][:,i]) &
                     (y_true[:,i] <= preds["p90"][:,i])).mean())
        coverages.append(cov*100)
    x = np.arange(N_DIRS)
    offset = -0.2 if split=="val" else 0.2
    ax.bar(x + offset, coverages, 0.35,
           label=name, color=color, alpha=0.8, edgecolor="white")

ax.axhline(80, color="red", lw=2, ls="--", label="Target 80%")
ax.set_xticks(np.arange(N_DIRS)); ax.set_xticklabels(DIR_LABELS)
ax.set_ylabel("P10-P90 Coverage (%)")
ax.set_title("Quantile Coverage per Direction\n"
             "Target ~80%. >80% = too conservative. <80% = overconfident.",
             fontweight="bold")
ax.legend(); ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(VAL_DIR/"tft_quantile_coverage.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ tft_quantile_coverage.png")

# Plot 4: Calibration factor per direction
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#F44336" if cf > 1 else "#2196F3" for cf in cal_factors]
ax.bar(DIR_LABELS, cal_factors, color=colors, alpha=0.8, edgecolor="white")
ax.axhline(1.0, color="black", lw=2, ls="--", label="Perfect (1.0)")
ax.axhline(overall_cal, color="orange", lw=2, ls="-.",
           label=f"Overall median: {overall_cal:.3f}")
for i,(dl,cf) in enumerate(zip(DIR_LABELS,cal_factors)):
    ax.text(i, cf+0.02, f"{cf:.2f}", ha="center", fontsize=10)
ax.set_ylabel("Calibration Factor (actual/predicted_P50)")
ax.set_title("TFT v2 Calibration Factors per Direction\n"
             "Derived from 2022-2023 val set — NOT manually chosen",
             fontweight="bold")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(VAL_DIR/"tft_calibration_factor.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ tft_calibration_factor.png")

# ── 10. Save model and metrics ────────────────────────────────────────────────
print("\n10. Saving model and metadata...")
model.save(str(MODEL_DIR/"tft_v2.keras"))

with open(VAL_DIR/"tft_val_metrics.json","w")  as f:
    json.dump(val_metrics,  f, indent=2)
with open(VAL_DIR/"tft_test_metrics.json","w") as f:
    json.dump(test_metrics, f, indent=2)

metadata = {
    "model":              "TFT v2 (multi-output quantile regressor)",
    "label_approach":     "radial_profile",
    "n_directions":       N_DIRS,
    "directions_deg":     DIRS_DEG,
    "n_quantiles":        3,
    "quantiles":          QUANTILES,
    "architecture":       {
        "shared_encoder": [64,32,16],
        "heads":          N_DIRS,
        "outputs_per_head": 3,
        "total_outputs":  N_DIRS * 3,
        "dropout":        DROPOUT,
    },
    "loss":               "pinball (quantile) loss",
    "label_source":       "FIRMS VIIRS daily fire clusters — convex hull radial profile",
    "no_synthetic_labels":True,
    "train_period":       "2017-2021",
    "val_period":         "2022-2023",
    "test_period":        "2024",
    "best_epoch":         best_epoch,
    "calibration_factor": overall_cal,
    "val_metrics":        val_metrics,
    "test_metrics":       test_metrics,
}
with open(MODEL_DIR/"tft_v2_metadata.json","w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*70}")
print("M5 COMPLETE")
print(f"{'='*70}")
print(f"Val  MAE  : {val_metrics['mae_overall']:.4f} km")
print(f"Test MAE  : {test_metrics['mae_overall']:.4f} km  ← 2024 holdout")
print(f"Val  coverage : {100*val_metrics['p10_p90_coverage']:.1f}%")
print(f"Test coverage : {100*test_metrics['p10_p90_coverage']:.1f}%")
print(f"Calibration factor: {overall_cal:.4f}  (derived from val set)")
print(f"\nOutputs → {VAL_DIR}/")
print(f"  tft_predicted_vs_actual.png")
print(f"  tft_quantile_coverage.png")
print(f"  tft_loss_curves.png")
print(f"  tft_calibration_factor.png")
print(f"  tft_val_metrics.json + tft_test_metrics.json")
print(f"\nNext: M6 — python src/pipeline/inference_bridge.py")
print(f"{'='*70}")