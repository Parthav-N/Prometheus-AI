"""
validate_bnn_temporal.py
=========================
Evaluates the retrained BNN on known 2024-2025 wildfire events.

For each major fire, checks whether the model scored nearby infrastructure
HIGH in the 24-48h window before the fire reached it.

This is the validation Gabe asked for — not Camp Fire (seen during training)
but genuinely unseen 2024-2025 events.

Run from project root:
    python src/pipeline/validate_bnn_temporal.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "bayesian_risk_model_v2.keras"
SCALER_PATH= ROOT / "models" / "feature_scaler_v2.pkl"
FIRES_PATH = ROOT / "data"   / "fires" / "national_fires_2017_2025.csv"
INFRA_PATH = ROOT / "data"   / "infrastructure" / "national_infrastructure.csv"
WEATHER_PATH=ROOT / "data"   / "weather" / "national_weather_grid.csv"
OUT_PATH   = ROOT / "data"   / "validation_results_2024_2025.csv"

MC_SAMPLES = 100
LOOKAHEAD_H = 48   # hours before fire arrival to check model score

# Known major 2024-2025 fire events — name, ignition date, lat, lon, affected radius km
VALIDATION_FIRES = [
    {"name": "Park Fire",        "date": "2024-07-24", "lat": 39.90, "lon": -121.43, "radius_km": 50, "state": "CA"},
    {"name": "Boquet Fire",      "date": "2024-09-10", "lat": 34.55, "lon": -118.60, "radius_km": 30, "state": "CA"},
    {"name": "Palisades Fire",   "date": "2025-01-07", "lat": 34.07, "lon": -118.53, "radius_km": 25, "state": "CA"},
    {"name": "Eaton Fire",       "date": "2025-01-07", "lat": 34.18, "lon": -118.05, "radius_km": 25, "state": "CA"},
    {"name": "Upstream Fire",    "date": "2024-06-15", "lat": 38.50, "lon": -120.80, "radius_km": 20, "state": "CA"},
    {"name": "Oregon Ridge Fire","date": "2024-08-20", "lat": 44.20, "lon": -122.10, "radius_km": 20, "state": "OR"},
]

FEATURE_COLS = [
    "min_distance_km","mean_distance_km","num_fires_30km","max_frp",
    "wind_speed_kmh","wind_direction","temperature_c","humidity",
    "wind_fire_alignment","drought_index","days_since_rain",
]

print("=" * 70)
print("TEMPORAL VALIDATION — 2024-2025 WILDFIRE EVENTS")
print("=" * 70)

# ---------------------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------------------

print("\nLoading model...")
model  = tf.keras.models.load_model(str(MODEL_PATH))
scaler = joblib.load(SCALER_PATH)

print("Loading fires and infrastructure...")
fires = pd.read_csv(FIRES_PATH, low_memory=False)
fires["datetime"] = pd.to_datetime(fires["acq_date"], errors="coerce")
fires = fires.dropna(subset=["datetime"])

infra = pd.read_csv(INFRA_PATH, low_memory=False)
infra.columns = [c.strip().lower() for c in infra.columns]
infra["lat"] = pd.to_numeric(infra["lat"], errors="coerce")
infra["lon"] = pd.to_numeric(infra["lon"], errors="coerce")
infra = infra.dropna(subset=["lat","lon"])

wx = pd.read_csv(WEATHER_PATH, low_memory=False)
wx["datetime"] = pd.to_datetime(wx["datetime"], errors="coerce")
wx = wx.dropna(subset=["datetime"])

from scipy.spatial import cKDTree
wx_coords = wx[["grid_lat","grid_lon"]].drop_duplicates().values
wx_tree   = cKDTree(wx_coords)

def haversine(lat1, lon1, lats2, lons2):
    R = 6371
    la1,lo1 = np.radians(lat1),  np.radians(lon1)
    la2,lo2 = np.radians(lats2), np.radians(lons2)
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
    la1,lo1 = np.radians(fire_lat),  np.radians(fire_lon)
    la2,lo2 = np.radians(asset_lat), np.radians(asset_lon)
    dlon = lo2 - lo1
    x = np.sin(dlon)*np.cos(la2)
    y = np.cos(la1)*np.sin(la2) - np.sin(la1)*np.cos(la2)*np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x,y)) + 360) % 360
    wind_toward = (wind_dir + 180) % 360
    diff = abs(wind_toward - bearing)
    if diff > 180: diff = 360 - diff
    return float(np.cos(np.radians(diff)))

def predict_mc(X_scaled):
    preds = np.array([model(X_scaled, training=True).numpy().flatten()
                      for _ in range(MC_SAMPLES)])
    return preds.mean(axis=0), preds.std(axis=0)

# ---------------------------------------------------------------------------
# Evaluate each fire event
# ---------------------------------------------------------------------------

all_results = []

for fire_event in VALIDATION_FIRES:
    name      = fire_event["name"]
    date      = pd.Timestamp(fire_event["date"])
    lookback  = date - pd.Timedelta(hours=LOOKAHEAD_H)
    fire_lat  = fire_event["lat"]
    fire_lon  = fire_event["lon"]
    radius_km = fire_event["radius_km"]

    print(f"\n{'─'*70}")
    print(f"  {name}  ({fire_event['date']})  "
          f"[{fire_lat:.2f}°N, {abs(fire_lon):.2f}°W]")
    print(f"{'─'*70}")

    # Fires known to the model at lookback time (48h before ignition)
    known_fires = fires[fires["datetime"] <= lookback]
    if len(known_fires) < 5:
        print(f"  ⚠ Insufficient fire data at lookback time")
        continue

    # Infrastructure within affected radius
    local_infra = infra.copy()
    dists_to_event = haversine(fire_lat, fire_lon,
                               local_infra["lat"].values,
                               local_infra["lon"].values)
    local_infra = local_infra[dists_to_event <= radius_km].copy()

    if len(local_infra) == 0:
        print(f"  ⚠ No infrastructure within {radius_km}km")
        continue

    print(f"  Assets within {radius_km}km: {len(local_infra)}")

    # Drought proxy at lookback
    recent_30 = fires[(fires["datetime"] <= lookback) &
                      (fires["datetime"] >= lookback - pd.Timedelta(days=30))]
    drought_proxy = min(len(recent_30) / 500, 4.0)
    recent_7 = fires[(fires["datetime"] <= lookback) &
                     (fires["datetime"] >= lookback - pd.Timedelta(days=7))]
    days_dry_proxy = float(len(recent_7) / 50)

    rows = []
    for _, asset in local_infra.iterrows():
        dists = haversine(asset["lat"], asset["lon"],
                          known_fires["latitude"].values,
                          known_fires["longitude"].values)

        _, wx_idx = wx_tree.query([asset["lat"], asset["lon"]])
        nlat, nlon = wx_coords[wx_idx]
        w_rows = wx[(wx["grid_lat"]==nlat) & (wx["grid_lon"]==nlon)].copy()
        w_rows["td"] = (w_rows["datetime"] - lookback).abs().dt.total_seconds()
        w_row = w_rows.nsmallest(1, "td")
        if w_row.empty or w_row.iloc[0]["td"] > 21600:
            continue
        w = w_row.iloc[0]

        cf_idx = dists.argmin()
        cf     = known_fires.iloc[cf_idx]
        wa     = wind_alignment(cf["latitude"], cf["longitude"],
                                asset["lat"], asset["lon"],
                                float(w.get("wind_direction",180)))

        rows.append([
            float(dists.min()), float(dists.mean()),
            int((dists<30).sum()), float(known_fires["frp"].max()),
            float(w.get("wind_speed_kmh",0)), float(w.get("wind_direction",180)),
            float(w.get("temp_c",20)), float(w.get("humidity",50)),
            wa, drought_proxy, days_dry_proxy
        ])

    if not rows:
        print("  ⚠ No weather data available for assets")
        continue

    X = np.array(rows, dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)
    means, stds = predict_mc(X_scaled)

    high_risk = (means > 70).sum()
    med_risk  = ((means > 30) & (means <= 70)).sum()
    low_risk  = (means <= 30).sum()
    avg_score = means.mean()
    avg_unc   = stds.mean()

    print(f"  Avg risk score   : {avg_score:.1f}%  ±{avg_unc:.1f}")
    print(f"  High risk (>70%) : {high_risk}/{len(means)} assets  "
          f"{'✓ FLAGGED' if high_risk > 0 else '✗ MISSED'}")
    print(f"  Medium (30-70%)  : {med_risk}/{len(means)} assets")
    print(f"  Low    (≤30%)    : {low_risk}/{len(means)} assets")

    # Top 5 highest-risk assets
    top_idx = np.argsort(means)[::-1][:5]
    print(f"\n  Top flagged assets:")
    asset_list = local_infra.reset_index(drop=True)
    for idx in top_idx:
        if idx < len(asset_list):
            a = asset_list.iloc[idx]
            print(f"    {a.get('name','Unknown')[:35]:<35} "
                  f"{a.get('type',''):<20} "
                  f"{means[idx]:.1f}% ±{stds[idx]:.1f}")

    for i, (m, s) in enumerate(zip(means, stds)):
        if i >= len(local_infra): break
        a = local_infra.reset_index(drop=True).iloc[i]
        all_results.append({
            "fire_event":    name,
            "fire_date":     fire_event["date"],
            "asset_name":    str(a.get("name","")),
            "asset_type":    str(a.get("type","")),
            "state":         str(a.get("state","")),
            "risk_score":    round(float(m),1),
            "uncertainty":   round(float(s),1),
            "risk_bucket":   "high" if m>70 else "medium" if m>30 else "low",
            "hours_before":  LOOKAHEAD_H,
        })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("VALIDATION SUMMARY")
print(f"{'='*70}")

if all_results:
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(OUT_PATH, index=False)

    total    = len(df_res)
    flagged  = (df_res["risk_bucket"]=="high").sum()
    per_fire = df_res.groupby("fire_event")["risk_score"].mean().round(1)

    print(f"Total assets evaluated : {total:,}")
    print(f"Flagged HIGH risk      : {flagged:,} ({100*flagged/total:.1f}%)")
    print(f"\nAvg risk score per fire event:")
    print(per_fire.to_string())
    print(f"\nResults saved → {OUT_PATH}")
else:
    print("No results generated — check data paths.")

print(f"{'='*70}")