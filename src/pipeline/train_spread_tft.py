"""
train_spread_tft.py
====================
Trains a Temporal Fusion Transformer to predict fire spread growth rate
at T+6h, T+12h, T+24h horizons.

Key design decisions:
  - Log-transform growth rate targets (heavily right-skewed distribution)
  - Temporal split: train 2018-2023, validate 2024-2025
  - Quantile outputs: P10/P50/P90 for uncertainty-aware spread polygons
  - Features: weather + fire state + terrain + fuel

Output: models/spread_tft_best.pt + spread_tft_metadata.json

Run from project root:
    python src/pipeline/train_spread_tft.py
"""

import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data"    / "training_spread_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "spread_tft_best.pt"
META_PATH  = MODEL_DIR / "spread_tft_metadata.json"
SCALER_PATH= MODEL_DIR / "spread_scaler.pkl"

TRAIN_CUTOFF = "2024-01-01"
GROWTH_CLIP  = 5.0    # km2/h — clip extreme outliers before log transform

WEATHER_FEATURES = [
    "wind_speed_kmh", "wind_direction", "temperature_c", "humidity",
    "max_wind_kmh", "wind_speed_trend", "temp_trend", "humidity_trend",
    "drought_index",
]
STATIC_FEATURES = [
    "current_area_km2", "fuel_multiplier", "slope_proxy",
    "frp_mean", "frp_max", "n_detections", "fire_density", "frp_trend",
    "slope_wind_interaction", "wind_x_fuel", "heat_dryness",
]
TARGET_COLS = ["growth_rate_6h", "growth_rate_12h", "growth_rate_24h"]
N_QUANTILES = 3   # P10, P50, P90

print("=" * 70)
print("TRAINING FIRE SPREAD TFT")
print("=" * 70)

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

print("\n1. Loading spread dataset...")
df = pd.read_csv(DATA_PATH)
df["ref_date"] = pd.to_datetime(df["ref_date"])
print(f"   ✓ {len(df):,} samples")
print(f"   Date range: {df['ref_date'].min().date()} → {df['ref_date'].max().date()}")

# Clip growth rate outliers
for col in TARGET_COLS:
    df[col] = df[col].clip(upper=GROWTH_CLIP)

# Log transform targets — log1p handles zeros cleanly
for col in TARGET_COLS:
    df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

LOG_TARGET_COLS = [f"log_{c}" for c in TARGET_COLS]

# Fill missing features with median, then clip to finite range
for col in WEATHER_FEATURES + STATIC_FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].replace([np.inf, -np.inf], df[col].median())

# Sanity check
nan_cols = [c for c in WEATHER_FEATURES + STATIC_FEATURES
            if df[c].isna().any() or np.isinf(df[c]).any()]
if nan_cols:
    print(f"   ⚠ NaN/inf found in: {nan_cols} — filling with 0")
    for c in nan_cols:
        df[c] = df[c].fillna(0).replace([np.inf, -np.inf], 0)

nan_target = [c for c in LOG_TARGET_COLS
              if df[c].isna().any() or np.isinf(df[c]).any()]
if nan_target:
    print(f"   ⚠ NaN/inf in targets: {nan_target} — filling with 0")
    for c in nan_target:
        df[c] = df[c].fillna(0).replace([np.inf, -np.inf], 0)

print(f"\n   Growth rate distribution after clip (km2/h):")
for col in TARGET_COLS:
    print(f"   {col:<25} mean={df[col].mean():.4f}  "
          f"p50={df[col].median():.4f}  p90={df[col].quantile(0.9):.4f}")

# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

print(f"\n2. Temporal split at {TRAIN_CUTOFF}...")
train_df = df[df["ref_date"] <  TRAIN_CUTOFF].copy()
val_df   = df[df["ref_date"] >= TRAIN_CUTOFF].copy()

if len(val_df) == 0:
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"   No 2024+ data — using 15% random holdout")

print(f"   Train: {len(train_df):,}  Val: {len(val_df):,}")

# ---------------------------------------------------------------------------
# Scale features
# ---------------------------------------------------------------------------

all_features = WEATHER_FEATURES + STATIC_FEATURES
X_train = train_df[all_features].values.astype(np.float32)
X_val   = val_df[all_features].values.astype(np.float32)
y_train = train_df[LOG_TARGET_COLS].values.astype(np.float32)
y_val   = val_df[LOG_TARGET_COLS].values.astype(np.float32)

scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train).astype(np.float32)
X_val    = scaler.transform(X_val).astype(np.float32)
joblib.dump(scaler, SCALER_PATH)

# Final NaN/inf check after scaling
X_train = np.nan_to_num(X_train, nan=0.0, posinf=3.0, neginf=-3.0)
X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=3.0, neginf=-3.0)
y_train = np.nan_to_num(y_train, nan=0.0)
y_val   = np.nan_to_num(y_val,   nan=0.0)

print(f"   X_train NaN after clean: {np.isnan(X_train).sum()}")
print(f"   y_train NaN after clean: {np.isnan(y_train).sum()}")

print(f"   Features: {len(all_features)}")
print(f"   Targets:  {len(LOG_TARGET_COLS)} horizons × {N_QUANTILES} quantiles")

# ---------------------------------------------------------------------------
# TFT Model
# ---------------------------------------------------------------------------

class SpreadTFT(nn.Module):
    """
    Simplified TFT for fire spread prediction.
    Input: flat feature vector (weather + fire state + terrain)
    Output: P10/P50/P90 growth rate at 3 horizons = 9 values total

    Architecture:
      VSN (Variable Selection) → LSTM encoder → 
      Multi-head attention → Quantile output heads
    """

    def __init__(self, n_features, d_model=64, n_heads=4,
                 n_lstm_layers=2, dropout=0.1,
                 n_horizons=3, n_quantiles=3):
        super().__init__()
        self.n_horizons  = n_horizons
        self.n_quantiles = n_quantiles

        # Variable Selection Network
        self.vsn = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ELU(),
        )
        self.vsn_gate = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.Sigmoid(),
        )

        # LSTM encoder — treats feature vector as single-step sequence
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=n_lstm_layers, batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0
        )

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Separate output head per horizon × quantile
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_quantiles)
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x):
        vsn_out  = self.vsn(x) * self.vsn_gate(x)
        lstm_in  = vsn_out.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)
        h = attn_out.squeeze(1)

        outputs = [head(h) for head in self.output_heads]
        stacked = torch.stack(outputs, dim=1)  # (B, n_horizons, n_quantiles)

        # Enforce quantile monotonicity: P10 <= P50 <= P90
        # Use cumsum on sorted deltas to guarantee ordering
        q0    = stacked[:, :, 0:1]                        # anchor (P10)
        deltas = torch.relu(stacked[:, :, 1:] -
                            stacked[:, :, :-1])            # positive gaps
        stacked = torch.cat([q0,
                             q0 + torch.cumsum(deltas, dim=2)], dim=2)
        return stacked


def quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    """Standard pinball loss for quantile regression."""
    losses = []
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        losses.append(torch.max(q * err, (q - 1) * err))
    return torch.stack(losses, dim=1).mean()


def combined_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    """
    pred:   (B, n_horizons, n_quantiles)
    target: (B, n_horizons)
    """
    total = 0.0
    for h in range(pred.shape[1]):
        total += quantile_loss(pred[:, h, :], target[:, h], quantiles)
    return total / pred.shape[1]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":

 print("\n3. Building model...")
 model = SpreadTFT(
    n_features=len(all_features),
    d_model=64, n_heads=4, n_lstm_layers=2, dropout=0.20
)
n_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {n_params:,}")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5, min_lr=1e-5, verbose=True)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).to(device)

BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE   = 20
QUANTILES  = [0.1, 0.5, 0.9]

n_train  = len(X_train_t)
n_batches = math.ceil(n_train / BATCH_SIZE)

best_val_loss = float("inf")
patience_ctr  = 0
best_state    = {k: v.clone() for k, v in model.state_dict().items()}  # fallback

print(f"\n4. Training (max {MAX_EPOCHS} epochs, patience={PATIENCE})...")
print("-" * 70)

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    # Shuffle
    perm = torch.randperm(n_train)
    X_train_t = X_train_t[perm]
    y_train_t = y_train_t[perm]

    train_loss = 0.0
    for b in range(n_batches):
        xb = X_train_t[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        yb = y_train_t[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        optimizer.zero_grad()
        pred = model(xb)
        loss = combined_loss(pred, yb, QUANTILES)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= n_batches

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = combined_loss(val_pred, y_val_t, QUANTILES).item()

    scheduler.step(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"   Epoch {epoch:>3}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr  = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n   Early stopping at epoch {epoch}")
            break

print("-" * 70)
print(f"   Best val loss: {best_val_loss:.4f}")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

model.load_state_dict(best_state)
torch.save(best_state, MODEL_PATH)

model.eval()
with torch.no_grad():
    val_pred = model(X_val_t).cpu().numpy()   # (N, 3, 3)
    y_val_np = y_val_t.cpu().numpy()           # (N, 3)

HORIZON_NAMES = ["6h", "12h", "24h"]
print(f"\n5. Evaluation on validation set:")
print(f"{'Horizon':<8} {'MAE (log)':<12} {'MAE (km2/h)':<14} "
      f"{'P10-P90 coverage':<18}")
print("-" * 55)

coverages = []
for h, hname in enumerate(HORIZON_NAMES):
    p10 = np.expm1(val_pred[:, h, 0])
    p50 = np.expm1(val_pred[:, h, 1])
    p90 = np.expm1(val_pred[:, h, 2])
    true_rate = np.expm1(y_val_np[:, h])

    mae_log  = float(np.mean(np.abs(y_val_np[:, h] - val_pred[:, h, 1])))
    mae_rate = float(np.mean(np.abs(true_rate - p50)))
    coverage = float(np.mean((true_rate >= p10) & (true_rate <= p90)))
    coverages.append(coverage)

    print(f"   T+{hname:<5} {mae_log:<12.4f} {mae_rate:<14.4f} "
          f"{coverage*100:.1f}%")

print(f"\n   Mean P10-P90 coverage: {np.mean(coverages)*100:.1f}% "
      f"(target: 80%)")

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------

metadata = {
    "version":           "v1",
    "weather_features":  WEATHER_FEATURES,
    "static_features":   STATIC_FEATURES,
    "all_features":      all_features,
    "n_features":        len(all_features),
    "target_cols":       TARGET_COLS,
    "horizons_h":        [6, 12, 24],
    "quantiles":         QUANTILES,
    "growth_clip_km2h":  GROWTH_CLIP,
    "train_cutoff":      TRAIN_CUTOFF,
    "train_samples":     len(X_train),
    "val_samples":       len(X_val),
    "best_val_loss":     best_val_loss,
    "p10_p90_coverage":  float(np.mean(coverages)),
    "architecture": {
        "d_model": 64, "n_heads": 4,
        "n_lstm_layers": 2, "dropout": 0.15,
    },
    "note": "Log-transform targets. Use np.expm1() to convert predictions back to km2/h.",
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