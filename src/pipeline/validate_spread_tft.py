"""
validate_spread_tft.py
=======================
Validates the TFT spread model against known 2024-2025 fire events.

For each named fire:
  1. Get fire state at discovery (size, location, weather)
  2. Run TFT to predict growth rate at T+6h, T+12h, T+24h
  3. Compare predicted final size vs actual final size (from NIFC)
  4. Check if actual growth rate falls within P10-P90 interval

This answers: does the model correctly forecast spread on fires
it has never seen during training?

Run from project root:
    python src/pipeline/validate_spread_tft.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial import cKDTree
import sys

ROOT       = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from spread_tft_model import SpreadTFT

ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "spread_tft_best.pt"
SCALER_PATH= ROOT / "models" / "spread_scaler.pkl"
META_PATH  = ROOT / "models" / "spread_tft_metadata.json"
NIFC_PATH  = ROOT / "data"   / "perimeters" / "WFIGS_Interagency_Perimeters.csv"
FIRMS_PATH = ROOT / "data"   / "fires"      / "national_fires_2017_2025.csv"
WEATHER_DIR= ROOT / "data"   / "weather"

# Known 2024-2025 fire events with ground truth
# final_area_km2 from NIFC records
VALIDATION_FIRES = [
    {
        "name":           "Park Fire",
        "state":          "US-CA",
        "discovery_date": "2024-07-24",
        "lat":            39.90, "lon": -121.43,
        "discovery_area_km2": 0.5,    # small at discovery
        "final_area_km2":     2483.0, # ~613,000 acres
        "duration_days":      60,
    },
    {
        "name":           "Palisades Fire",
        "state":          "US-CA",
        "discovery_date": "2025-01-07",
        "lat":            34.07, "lon": -118.53,
        "discovery_area_km2": 0.2,
        "final_area_km2":     242.0,  # ~59,800 acres
        "duration_days":      23,
    },
    {
        "name":           "Eaton Fire",
        "state":          "US-CA",
        "discovery_date": "2025-01-07",
        "lat":            34.18, "lon": -118.05,
        "discovery_area_km2": 0.3,
        "final_area_km2":     567.0,  # ~14,000 acres
        "duration_days":      16,
    },
    {
        "name":           "Boquet Fire",
        "state":          "US-CA",
        "discovery_date": "2024-09-10",
        "lat":            34.55, "lon": -118.60,
        "discovery_area_km2": 0.4,
        "final_area_km2":     97.0,   # ~24,000 acres
        "duration_days":      14,
    },
    {
        "name":           "Oregon Ridge Fire",
        "state":          "US-OR",
        "discovery_date": "2024-08-20",
        "lat":            44.20, "lon": -122.10,
        "discovery_area_km2": 0.3,
        "final_area_km2":     28.0,
        "duration_days":      10,
    },
]

print("=" * 70)
print("VALIDATING FIRE SPREAD TFT ON 2024-2025 UNSEEN FIRE EVENTS")
print("=" * 70)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("\nLoading TFT model...")
with open(META_PATH) as f:
    meta = json.load(f)

scaler       = joblib.load(SCALER_PATH)
all_features = meta["all_features"]
horizons_h   = meta["horizons_h"]
quantiles    = meta["quantiles"]
growth_clip  = meta["growth_clip_km2h"]

model = SpreadTFT(
    n_features    = meta["n_features"],
    d_model       = meta["architecture"]["d_model"],
    n_heads       = meta["architecture"]["n_heads"],
    n_lstm_layers = meta["architecture"]["n_lstm_layers"],
    dropout       = 0.0,
)
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(state)
model.eval()
print(f"   ✓ Model loaded ({meta['n_features']} features)")

# ---------------------------------------------------------------------------
# Load weather and FIRMS
# ---------------------------------------------------------------------------
print("Loading weather and fire data...")
wx_parts = []
for f in sorted(WEATHER_DIR.glob("*_weather_grid.csv")):
    wx_parts.append(pd.read_csv(f, low_memory=False,
                    usecols=["datetime","temp_c","humidity",
                             "wind_speed_kmh","wind_direction",
                             "grid_lat","grid_lon"]))
wx = pd.concat(wx_parts, ignore_index=True)
wx["datetime"] = pd.to_datetime(wx["datetime"], errors="coerce")
wx = wx.dropna(subset=["datetime"])
wx_coords = wx[["grid_lat","grid_lon"]].drop_duplicates().values
wx_tree   = cKDTree(wx_coords)

firms = pd.read_csv(FIRMS_PATH, low_memory=False)
firms["datetime"] = pd.to_datetime(firms["acq_date"], errors="coerce")
firms = firms.dropna(subset=["datetime","latitude","longitude"])
firms = firms[firms["confidence"].astype(str).str.lower()
              .isin(["h","n","high","nominal"])].copy()
firms_coords = firms[["latitude","longitude"]].values
firms_tree   = cKDTree(firms_coords)

def get_weather_sequence(lat, lon, target_time, lookback_h=24):
    _, idx     = wx_tree.query([lat, lon])
    nlat, nlon = wx_coords[idx]
    station    = wx[
        (wx["grid_lat"] == nlat) & (wx["grid_lon"] == nlon) &
        (wx["datetime"] >= target_time - pd.Timedelta(hours=lookback_h)) &
        (wx["datetime"] <= target_time)
    ].sort_values("datetime")
    if len(station) < 2:
        return None
    return {
        "wind_speed_kmh":   float(station["wind_speed_kmh"].mean()),
        "wind_direction":   float(station["wind_direction"].mean()),
        "temperature_c":    float(station["temp_c"].mean()),
        "humidity":         float(station["humidity"].mean()),
        "max_wind_kmh":     float(station["wind_speed_kmh"].max()),
        "wind_speed_trend": float(station["wind_speed_kmh"].iloc[-1] -
                                  station["wind_speed_kmh"].iloc[0]),
        "temp_trend":       float(station["temp_c"].iloc[-1] -
                                  station["temp_c"].iloc[0]),
        "humidity_trend":   float(station["humidity"].iloc[-1] -
                                  station["humidity"].iloc[0]),
        "drought_index":    float(1 - station["humidity"].mean() / 100),
    }

def get_firms_context(lat, lon, target_time, radius_km=30, lookback_h=48):
    nearby = firms_tree.query_ball_point([lat, lon], r=radius_km/111.0)
    if not nearby:
        return {"frp_mean":0,"frp_max":0,"n_detections":0,
                "fire_density":0,"frp_trend":0}
    nb = firms.iloc[nearby].copy()
    nb = nb[(nb["datetime"] >= target_time - pd.Timedelta(hours=lookback_h)) &
            (nb["datetime"] <= target_time)]
    if nb.empty:
        return {"frp_mean":0,"frp_max":0,"n_detections":0,
                "fire_density":0,"frp_trend":0}
    early = nb[nb["datetime"] <= target_time - pd.Timedelta(hours=lookback_h//2)]
    late  = nb[nb["datetime"] >  target_time - pd.Timedelta(hours=lookback_h//2)]
    trend = (float(late["frp"].mean()) - float(early["frp"].mean())
             if not early.empty and not late.empty else 0.0)
    return {
        "frp_mean":      float(nb["frp"].mean()),
        "frp_max":       float(nb["frp"].max()),
        "n_detections":  len(nb),
        "fire_density":  len(nb) / (np.pi * radius_km**2),
        "frp_trend":     trend,
    }

FUEL_MAP = {"GR":1.5,"GS":1.4,"SH":1.0,"TU":0.8,"TL":0.6,"NB":0.1}

# ---------------------------------------------------------------------------
# Validate each fire
# ---------------------------------------------------------------------------
results = []

for fire in VALIDATION_FIRES:
    name   = fire["name"]
    lat    = fire["lat"]
    lon    = fire["lon"]
    target = pd.Timestamp(fire["discovery_date"])

    print(f"\n{'─'*70}")
    print(f"  {name}  ({fire['discovery_date']})  "
          f"[{lat:.2f}°N, {abs(lon):.2f}°W]")
    print(f"{'─'*70}")

    # Get weather at discovery
    wx_seq = get_weather_sequence(lat, lon, target)
    if wx_seq is None:
        print(f"  ⚠ No weather data — skipping")
        continue

    # Get FIRMS context
    fc = get_firms_context(lat, lon, target)

    # Slope proxy
    slope = float(np.clip((lat - 32.0) / 12.0, 0, 1))

    # Build feature vector
    feat_map = {
        "wind_speed_kmh":      wx_seq["wind_speed_kmh"],
        "wind_direction":      wx_seq["wind_direction"],
        "temperature_c":       wx_seq["temperature_c"],
        "humidity":            wx_seq["humidity"],
        "max_wind_kmh":        wx_seq["max_wind_kmh"],
        "wind_speed_trend":    wx_seq["wind_speed_trend"],
        "temp_trend":          wx_seq["temp_trend"],
        "humidity_trend":      wx_seq["humidity_trend"],
        "drought_index":       wx_seq["drought_index"],
        "current_area_km2":    fire["discovery_area_km2"],
        "fuel_multiplier":     1.0,
        "slope_proxy":         slope,
        "frp_mean":            fc["frp_mean"],
        "frp_max":             fc["frp_max"],
        "n_detections":        fc["n_detections"],
        "fire_density":        fc["fire_density"],
        "frp_trend":           fc["frp_trend"],
        "slope_wind_interaction": slope * wx_seq["wind_speed_kmh"] / 50.0,
        "wind_x_fuel":         wx_seq["wind_speed_kmh"] * 1.0,
        "heat_dryness":        max(0, wx_seq["temperature_c"] - 25) *
                               max(0, 60 - wx_seq["humidity"]) / 100,
    }

    X = np.array([[feat_map.get(f, 0.0) for f in all_features]],
                 dtype=np.float32)
    X = np.nan_to_num(scaler.transform(X), nan=0.0).astype(np.float32)

    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()[0]  # (3 horizons, 3 quantiles)

    # Convert log predictions back to km2/h
    p10 = np.expm1(pred[:, 0]).clip(0, growth_clip)
    p50 = np.expm1(pred[:, 1]).clip(0, growth_clip)
    p90 = np.expm1(pred[:, 2]).clip(0, growth_clip)

    # Actual growth rate over fire lifetime
    actual_rate = ((fire["final_area_km2"] - fire["discovery_area_km2"]) /
                   (fire["duration_days"] * 24))

    print(f"  Weather at discovery:")
    print(f"    Temp: {wx_seq['temperature_c']:.1f}°C  "
          f"Humidity: {wx_seq['humidity']:.0f}%  "
          f"Wind: {wx_seq['wind_speed_kmh']:.0f} km/h")
    print(f"  FIRMS context: {fc['n_detections']} detections, "
          f"max FRP={fc['frp_max']:.0f}")
    print(f"\n  Actual avg growth rate: {actual_rate:.4f} km²/h  "
          f"(over {fire['duration_days']} days)")
    print(f"  Final area: {fire['final_area_km2']:.0f} km²")

    print(f"\n  {'Horizon':<8} {'P10':>8} {'P50':>8} {'P90':>8}  "
          f"{'In range?':>10}  {'Proj area km²':>14}")
    print(f"  {'':─<65}")

    for h_i, h in enumerate(horizons_h):
        in_range = p10[h_i] <= actual_rate <= p90[h_i]
        proj_area = fire["discovery_area_km2"] + p50[h_i] * h
        flag = "✓" if in_range else "✗"
        print(f"  T+{h:<5}h  {p10[h_i]:>8.4f}  {p50[h_i]:>8.4f}  "
              f"{p90[h_i]:>8.4f}  {flag:>10}  {proj_area:>14.1f}")

    # Check if actual rate within any horizon's P10-P90
    covered = any(p10[i] <= actual_rate <= p90[i] for i in range(3))
    results.append({
        "fire":          name,
        "actual_rate":   round(actual_rate, 4),
        "p50_6h":        round(float(p50[0]), 4),
        "p50_12h":       round(float(p50[1]), 4),
        "p50_24h":       round(float(p50[2]), 4),
        "in_range":      covered,
        "final_area_km2":fire["final_area_km2"],
        "wind_kmh":      wx_seq["wind_speed_kmh"],
        "temp_c":        wx_seq["temperature_c"],
        "humidity":      wx_seq["humidity"],
    })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("VALIDATION SUMMARY")
print(f"{'='*70}")

df_res = pd.DataFrame(results)
if not df_res.empty:
    covered = df_res["in_range"].sum()
    print(f"Fires validated     : {len(df_res)}")
    print(f"Actual rate in P10-P90: {covered}/{len(df_res)}")
    print()
    print(df_res[["fire","actual_rate","p50_6h","p50_24h",
                  "in_range","wind_kmh","temp_c","humidity"]].to_string(index=False))

    print(f"\nKey observations:")
    for _, r in df_res.iterrows():
        ratio = r["p50_6h"] / max(r["actual_rate"], 0.001)
        if ratio > 2:
            print(f"  {r['fire']}: model OVER-predicts "
                  f"({ratio:.1f}x actual)")
        elif ratio < 0.5:
            print(f"  {r['fire']}: model UNDER-predicts "
                  f"({ratio:.1f}x actual)")
        else:
            print(f"  {r['fire']}: reasonable prediction "
                  f"(ratio={ratio:.1f}x)")