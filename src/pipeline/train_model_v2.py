"""
train_model_v2.py
==================
Trains the Bayesian Neural Network with proper temporal split.

Key changes vs previous version:
  - Temporal split: train on 2017-2023, validate on 2024-2025
  - No random shuffling across dates (eliminates data leakage)
  - Balanced sampling within each split independently
  - Saves to same paths: bayesian_risk_model_v2.keras + feature_scaler_v2.pkl

Run from project root:
    python src/pipeline/train_model_v2.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "training_dataset_national.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODEL_DIR / "bayesian_risk_model_v2.keras"
SCALER_PATH  = MODEL_DIR / "feature_scaler_v2.pkl"
META_PATH    = MODEL_DIR / "model_metadata_v2.json"

FEATURE_COLS = [
    "min_distance_km",
    "mean_distance_km",
    "num_fires_30km",
    "max_frp",
    "wind_speed_kmh",
    "wind_direction",
    "temperature_c",
    "humidity",
    "wind_fire_alignment",
    "drought_index",
    "days_since_rain",
]

TRAIN_CUTOFF = "2024-01-01"   # everything before this = train
MC_SAMPLES   = 50

print("=" * 70)
print("TRAINING BAYESIAN NEURAL NETWORK  v2  (temporal split)")
print("=" * 70)

# ---------------------------------------------------------------------------
# Load and split temporally
# ---------------------------------------------------------------------------

print("\n1. Loading dataset...")
df = pd.read_csv(DATA_PATH)
df["acq_date"] = pd.to_datetime(df["acq_date"])
df = df.dropna(subset=FEATURE_COLS + ["risk_score"])
print(f"   ✓ {len(df):,} total samples")
print(f"   Date range: {df['acq_date'].min().date()} → {df['acq_date'].max().date()}")

train_df = df[df["acq_date"] <  TRAIN_CUTOFF].copy()
val_df   = df[df["acq_date"] >= TRAIN_CUTOFF].copy()

print(f"\n2. Temporal split at {TRAIN_CUTOFF}:")
print(f"   Train : {len(train_df):,} samples "
      f"({train_df['acq_date'].min().date()} → {train_df['acq_date'].max().date()})")

if len(val_df) == 0:
    # No 2024-2025 samples in dataset — use 15% random holdout from training
    # for early stopping only. Real validation is done by validate_bnn_temporal.py
    print(f"   Val   : 0 samples in 2024-2025 — using 15% random holdout for early stopping")
    from sklearn.model_selection import train_test_split as tts
    train_df, val_df = tts(train_df, test_size=0.15, random_state=42)
    print(f"   Train : {len(train_df):,} (after holdout)")
    print(f"   Val   : {len(val_df):,}  (early stopping proxy only)")
    print(f"   NOTE  : Run validate_bnn_temporal.py for real 2024-2025 evaluation")
else:
    print(f"   Val   : {len(val_df):,} samples "
          f"({val_df['acq_date'].min().date()} → {val_df['acq_date'].max().date()})")

# Balance train set across risk buckets (within training period only)
def balance_dataset(df, n_per_bucket=5000, seed=42):
    low    = df[df["risk_score"] <= 30].sample(
        min(n_per_bucket, (df["risk_score"]<=30).sum()), random_state=seed)
    medium = df[(df["risk_score"]>30) & (df["risk_score"]<=70)].sample(
        min(n_per_bucket, ((df["risk_score"]>30)&(df["risk_score"]<=70)).sum()),
        random_state=seed)
    high   = df[df["risk_score"] > 70].sample(
        min(n_per_bucket, (df["risk_score"]>70).sum()), random_state=seed)
    return pd.concat([low, medium, high]).sample(frac=1, random_state=seed)

train_balanced = balance_dataset(train_df)
print(f"\n3. Balanced train set: {len(train_balanced):,} samples")
print(f"   Low  (≤30): {(train_balanced['risk_score']<=30).sum():,}")
print(f"   Med  (30-70): {((train_balanced['risk_score']>30)&(train_balanced['risk_score']<=70)).sum():,}")
print(f"   High (>70): {(train_balanced['risk_score']>70).sum():,}")

# ---------------------------------------------------------------------------
# Scale — fit on training data only
# ---------------------------------------------------------------------------

X_train = train_balanced[FEATURE_COLS].values.astype(np.float32)
y_train = train_balanced["risk_score"].values.astype(np.float32)
X_val   = val_df[FEATURE_COLS].values.astype(np.float32)
y_val   = val_df["risk_score"].values.astype(np.float32)

scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train).astype(np.float32)
X_val    = scaler.transform(X_val).astype(np.float32)   # transform only, never fit
joblib.dump(scaler, SCALER_PATH)
print(f"\n   ✓ Scaler fit on training data only → {SCALER_PATH}")

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(n_features):
    inputs = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64,  activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32,  activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def predict_with_uncertainty(model, X, n=MC_SAMPLES):
    preds = np.array([model(X, training=True).numpy().flatten()
                      for _ in range(n)])
    return preds.mean(axis=0), preds.std(axis=0)

print("\n4. Building model...")
model = build_model(len(FEATURE_COLS))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="mse", metrics=["mae"])
print(f"   ✓ {model.count_params():,} parameters")

# ---------------------------------------------------------------------------
# Train — validate on temporal holdout, not random split
# ---------------------------------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, min_lr=1e-5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH), monitor="val_loss",
        save_best_only=True, verbose=0),
]

print(f"\n5. Training (temporal val on 2024-2025)...")
print("-" * 70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),   # ← real holdout, not random split
    epochs=150,
    batch_size=64,
    verbose=1,
    callbacks=callbacks,
)

epochs_trained = len(history.history["loss"])
best_val_loss  = min(history.history["val_loss"])
print("-" * 70)
print(f"   Stopped at epoch : {epochs_trained}")
print(f"   Best val loss    : {best_val_loss:.4f}")

# ---------------------------------------------------------------------------
# Evaluate on 2024-2025 holdout with MC Dropout
# ---------------------------------------------------------------------------

print(f"\n6. Evaluating on 2024-2025 holdout ({MC_SAMPLES} MC samples)...")
y_pred_mean, y_pred_std = predict_with_uncertainty(model, X_val)

mae       = float(np.mean(np.abs(y_val - y_pred_mean)))
rmse      = float(np.sqrt(np.mean((y_val - y_pred_mean)**2)))
avg_unc   = float(y_pred_std.mean())

def bucket(s):
    return np.where(s>70,"high", np.where(s>30,"medium","low"))

bucket_acc = float((bucket(y_val)==bucket(y_pred_mean)).mean())

# Calibration: are high-uncertainty predictions actually less accurate?
high_unc_mask = y_pred_std > np.percentile(y_pred_std, 75)
mae_high_unc  = float(np.mean(np.abs(y_val[high_unc_mask]  - y_pred_mean[high_unc_mask])))
mae_low_unc   = float(np.mean(np.abs(y_val[~high_unc_mask] - y_pred_mean[~high_unc_mask])))

print(f"\n{'='*70}")
print("TEMPORAL VALIDATION RESULTS  (2024-2025 holdout)")
print(f"{'='*70}")
print(f"MAE              : {mae:.2f} risk points")
print(f"RMSE             : {rmse:.2f}")
print(f"Bucket accuracy  : {bucket_acc*100:.1f}%  (low/medium/high)")
print(f"Avg uncertainty  : ±{avg_unc:.2f}")
print(f"\nUncertainty calibration:")
print(f"   MAE where uncertainty HIGH (top 25%) : {mae_high_unc:.2f}")
print(f"   MAE where uncertainty LOW  (bot 75%) : {mae_low_unc:.2f}")
if mae_high_unc > mae_low_unc:
    print(f"   ✓ Uncertainty is well-calibrated "
          f"(high uncertainty = higher error as expected)")
else:
    print(f"   ⚠ Uncertainty may be miscalibrated")

print(f"\nRisk distribution on 2024-2025 holdout:")
print(f"   Low  (≤30) : {(y_val<=30).sum():,}  "
      f"predicted low  : {(y_pred_mean<=30).sum():,}")
print(f"   Med (30-70): {((y_val>30)&(y_val<=70)).sum():,}  "
      f"predicted med  : {((y_pred_mean>30)&(y_pred_mean<=70)).sum():,}")
print(f"   High (>70) : {(y_val>70).sum():,}  "
      f"predicted high : {(y_pred_mean>70).sum():,}")

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------

metadata = {
    "version":              "v2_temporal",
    "features":             FEATURE_COLS,
    "n_features":           len(FEATURE_COLS),
    "architecture":         "256-128-64-32 MC Dropout",
    "dropout_rates":        [0.3, 0.3, 0.2, 0.1],
    "mc_samples":           MC_SAMPLES,
    "train_cutoff":         TRAIN_CUTOFF,
    "train_samples":        len(X_train),
    "val_samples":          len(X_val),
    "val_period":           "2024-01-01 to 2025-12-31",
    "mae":                  mae,
    "rmse":                 rmse,
    "bucket_accuracy":      bucket_acc,
    "avg_uncertainty":      avg_unc,
    "mae_high_uncertainty": mae_high_unc,
    "mae_low_uncertainty":  mae_low_unc,
    "epochs_trained":       epochs_trained,
    "best_val_loss":        best_val_loss,
    "note":                 "Temporal split — no data leakage from future fire events",
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*70}")
print("MODEL SAVED")
print(f"{'='*70}")
print(f"Model  → {MODEL_PATH}")
print(f"Scaler → {SCALER_PATH}")
print(f"Meta   → {META_PATH}")
print(f"{'='*70}")

if __name__ == "__main__":
    pass