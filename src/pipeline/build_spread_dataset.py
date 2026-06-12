"""
build_spread_dataset.py
========================
Builds training dataset for the TFT fire spread model.

Goal: predict fire GROWTH RATE (acres/hour) at T+6h, T+12h, T+24h
given current fire state + weather sequence.

Pipeline:
  1. Load NIFC perimeters — one record per fire, gives us size + location + fuel
  2. Load FIRMS detections — gives us fire intensity (FRP) over time
  3. Load weather grid — gives us wind/temp/humidity at fire location
  4. For each NIFC fire, find FIRMS detections in 30km radius over fire lifetime
  5. Compute growth rate = (final_acres - discovery_acres) / duration_hours
  6. Build weather sequences (24h lookback) at fire location
  7. Compute slope proxy from lat (elevation API — 80 calls)
  8. Output: training_spread_dataset.csv

Run from project root:
    python src/pipeline/build_spread_dataset.py
"""

import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

NIFC_PATH    = DATA_DIR / "perimeters" / "WFIGS_Interagency_Perimeters.csv"
FIRMS_PATH   = DATA_DIR / "fires"      / "national_fires_2017_2025.csv"
OUT_PATH     = DATA_DIR / "training_spread_dataset.csv"

TARGET_STATES = [
    "US-CA","US-OR","US-WA","US-MT","US-CO",
    "US-ID","US-WY","US-UT","US-NV","US-AZ"
]

# Rothermel fuel model rate-of-spread multipliers (relative to baseline)
# Based on standard 13 Anderson fuel models collapsed to NIFC categories
FUEL_ROS_MULTIPLIER = {
    "GR1": 1.2,  "GR2": 1.5,  "GR3": 1.8,  "GR4": 2.0,
    "GS1": 1.3,  "GS2": 1.6,  "GS3": 1.9,
    "SH1": 0.8,  "SH2": 1.0,  "SH3": 1.1,  "SH5": 1.2,  "SH7": 1.4,
    "TU1": 0.7,  "TU2": 0.8,  "TU3": 0.9,  "TU4": 1.0,  "TU5": 1.1,
    "TL1": 0.5,  "TL2": 0.6,  "TL3": 0.7,  "TL8": 0.9,
    "NB1": 0.1,  "NB2": 0.1,  "NB3": 0.1,
}
DEFAULT_FUEL_MULT = 1.0

print("=" * 70)
print("BUILDING FIRE SPREAD TRAINING DATASET")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Load and filter NIFC perimeters
# ---------------------------------------------------------------------------

print("\n1. Loading NIFC perimeters...")
nifc = pd.read_csv(NIFC_PATH, low_memory=False)
nifc = nifc[nifc["attr_POOState"].isin(TARGET_STATES)].copy()

nifc["discovery_dt"] = pd.to_datetime(
    nifc["attr_FireDiscoveryDateTime"], errors="coerce", dayfirst=False)
nifc["polygon_dt"]   = pd.to_datetime(
    nifc["poly_PolygonDateTime"], errors="coerce", dayfirst=False)

# Use polygon datetime as primary, fall back to discovery
nifc["ref_dt"] = nifc["polygon_dt"].fillna(nifc["discovery_dt"])
nifc = nifc[nifc["ref_dt"].dt.year.between(2017, 2025)].copy()

# Fill missing lat/lon from county centroids where possible
nifc = nifc.dropna(subset=["attr_InitialLatitude","attr_InitialLongitude"])

# Compute fire size in km2 from Shape__Area (square degrees at ~42°N)
# 1 degree lat ~ 111km, 1 degree lon ~ 82km at 42°N → avg 96km → 1 sq deg ~ 9216 km2
nifc["area_km2"] = nifc["Shape__Area"] * 9216
nifc["area_km2"] = nifc["area_km2"].clip(lower=0.01)

# Duration from discovery to polygon
nifc["duration_h"] = (
    (nifc["polygon_dt"] - nifc["discovery_dt"])
    .dt.total_seconds() / 3600
).clip(lower=1)

# Growth rate: km2 per hour
nifc["growth_rate_km2h"] = nifc["area_km2"] / nifc["duration_h"]
nifc["growth_rate_km2h"] = nifc["growth_rate_km2h"].clip(upper=500)

# ── Filter to genuinely spreading fires ──────────────────────────────
# Removes one-and-done containments and static/admin corrections
# Only fires that actually spread over multiple days matter for our model
pre_filter = len(nifc)
nifc = nifc[
    (nifc["duration_h"]       > 12) &    # lasted more than 12 hours
    (nifc["growth_rate_km2h"] > 0.005)   # grew at least 1.2 acres/hour
].copy()
print(f"   Filtered {pre_filter:,} → {len(nifc):,} actively spreading fires "
      f"({100*len(nifc)/pre_filter:.1f}% retained)")

print(f"   ✓ {len(nifc):,} fires  "
      f"({nifc['ref_dt'].min().date()} → {nifc['ref_dt'].max().date()})")
print(f"   Size range: {nifc['area_km2'].min():.2f} – {nifc['area_km2'].max():.1f} km2")
print(f"   Growth rate range: {nifc['growth_rate_km2h'].min():.4f} – "
      f"{nifc['growth_rate_km2h'].max():.1f} km2/h")

# ---------------------------------------------------------------------------
# 2. Load FIRMS fire detections
# ---------------------------------------------------------------------------

print("\n2. Loading FIRMS detections...")
firms = pd.read_csv(FIRMS_PATH, low_memory=False)
firms["datetime"] = pd.to_datetime(firms["acq_date"], errors="coerce")
firms = firms.dropna(subset=["datetime","latitude","longitude"])
firms = firms[
    firms["confidence"].astype(str).str.lower()
    .isin(["h","n","high","nominal"])
].copy()
print(f"   ✓ {len(firms):,} high-confidence detections")

firms_coords = firms[["latitude","longitude"]].values
firms_tree   = cKDTree(firms_coords)

# ---------------------------------------------------------------------------
# 3. Load weather grid
# ---------------------------------------------------------------------------

print("\n3. Loading weather grid (10 state files)...")
wx_parts = []
for state_file in sorted((DATA_DIR / "weather").glob("*_weather_grid.csv")):
    wx_parts.append(pd.read_csv(state_file, low_memory=False))
wx = pd.concat(wx_parts, ignore_index=True)
wx["datetime"] = pd.to_datetime(wx["datetime"], errors="coerce")
wx = wx.dropna(subset=["datetime"])
if wx["datetime"].dt.tz is not None:
    wx["datetime"] = wx["datetime"].dt.tz_localize(None)

wx["grid_lat"] = wx["grid_lat"].round(4)
wx["grid_lon"] = wx["grid_lon"].round(4)

wx_coords = wx[["grid_lat","grid_lon"]].drop_duplicates().values
wx_tree   = cKDTree(wx_coords)
print(f"   ✓ {len(wx):,} records, {len(wx_coords)} grid points")

# ---------------------------------------------------------------------------
# 4. USGS Elevation API — slope proxy for each unique grid point
# ---------------------------------------------------------------------------

print("\n4. Fetching elevation for weather grid points (USGS API)...")
elevation_cache = {}

def get_elevation(lat, lon):
    key = (round(lat,3), round(lon,3))
    if key in elevation_cache:
        return elevation_cache[key]
    try:
        url = (f"https://epqs.nationalmap.gov/v1/json"
               f"?x={lon}&y={lat}&units=Meters&includeDate=false")
        r = requests.get(url, timeout=8)
        elev = float(r.json()["value"])
        elevation_cache[key] = elev
        return elev
    except Exception:
        elevation_cache[key] = 500.0  # default elevation
        return 500.0

# Sample elevations at a subset of grid points
sample_coords = wx_coords[::4]  # every 4th point
print(f"   Fetching {len(sample_coords)} elevation points...")
elevations = {}
for i, (lat, lon) in enumerate(sample_coords):
    elev = get_elevation(lat, lon)
    elevations[(round(lat,3), round(lon,3))] = elev
    if i % 10 == 0:
        print(f"   [{i}/{len(sample_coords)}] {lat:.2f},{lon:.2f} → {elev:.0f}m")
    time.sleep(0.1)  # be polite

def get_slope_proxy(lat, lon):
    """Estimate slope from elevation gradient between nearby grid points."""
    key = (round(lat,3), round(lon,3))
    elev = elevations.get(key, 500.0)
    # Simple proxy: higher absolute elevation = more likely mountainous = steeper
    # Real slope would need DEM raster; this is our lightweight approximation
    return float(np.clip(elev / 3000.0, 0, 1))  # 0=flat, 1=high mountain

# ---------------------------------------------------------------------------
# 5. Helper functions
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lats2, lons2):
    R = 6371
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2 = np.radians(np.asarray(lats2, dtype=float))
    lo2 = np.radians(np.asarray(lons2, dtype=float))
    a = (np.sin((la2-la1)/2)**2 +
         np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def get_weather_sequence(lat, lon, target_time, lookback_h=24):
    """Get weather sequence for last lookback_h hours at nearest grid point."""
    _, idx    = wx_tree.query([lat, lon])
    nlat, nlon = wx_coords[idx]
    station   = wx[
        (wx["grid_lat"] == nlat) &
        (wx["grid_lon"] == nlon) &
        (wx["datetime"] >= target_time - pd.Timedelta(hours=lookback_h)) &
        (wx["datetime"] <= target_time)
    ].sort_values("datetime")

    if len(station) < 2:
        return None

    # Aggregate to mean conditions over lookback window
    return {
        "wind_speed_kmh":   float(station["wind_speed_kmh"].mean()),
        "wind_direction":   float(station["wind_direction"].mean()),
        "temperature_c":    float(station["temp_c"].mean()),
        "humidity":         float(station["humidity"].mean()),
        "max_wind_kmh":     float(station["wind_speed_kmh"].max()),
        "wind_speed_trend": float(
            station["wind_speed_kmh"].iloc[-1] -
            station["wind_speed_kmh"].iloc[0]),
        "temp_trend":       float(
            station["temp_c"].iloc[-1] -
            station["temp_c"].iloc[0]),
        "humidity_trend":   float(
            station["humidity"].iloc[-1] -
            station["humidity"].iloc[0]),
        "drought_index":    float(station["precipitation"].clip(upper=10).mean()
                            / 10.0),  # proxy from precip
        "n_weather_obs":    len(station),
    }

def get_firms_context(lat, lon, target_time, radius_km=30, lookback_h=48):
    """Get FIRMS fire context around location."""
    nearby_idx = firms_tree.query_ball_point([lat, lon],
                                             r=radius_km/111.0)
    if not nearby_idx:
        return {"frp_mean":0, "frp_max":0, "n_detections":0,
                "fire_density":0, "frp_trend":0}

    nearby = firms.iloc[nearby_idx].copy()
    nearby = nearby[
        (nearby["datetime"] >= target_time - pd.Timedelta(hours=lookback_h)) &
        (nearby["datetime"] <= target_time)
    ]
    if nearby.empty:
        return {"frp_mean":0, "frp_max":0, "n_detections":0,
                "fire_density":0, "frp_trend":0}

    early = nearby[nearby["datetime"] <= target_time - pd.Timedelta(hours=lookback_h//2)]
    late  = nearby[nearby["datetime"] >  target_time - pd.Timedelta(hours=lookback_h//2)]
    frp_trend = (float(late["frp"].mean()) - float(early["frp"].mean())
                 if not early.empty and not late.empty else 0.0)

    return {
        "frp_mean":      float(nearby["frp"].mean()),
        "frp_max":       float(nearby["frp"].max()),
        "n_detections":  len(nearby),
        "fire_density":  len(nearby) / (np.pi * radius_km**2),
        "frp_trend":     frp_trend,
    }

# ---------------------------------------------------------------------------
# 6. Build training samples
# ---------------------------------------------------------------------------

print("\n5. Building spread training samples...")

FUEL_MAP = {
    "GR": 1.5, "GS": 1.4, "SH": 1.0, "TU": 0.8,
    "TL": 0.6, "NB": 0.1,
}

def get_fuel_multiplier(fuel_model):
    if pd.isna(fuel_model):
        return DEFAULT_FUEL_MULT
    fm = str(fuel_model).strip().upper()
    # Exact match
    if fm in FUEL_ROS_MULTIPLIER:
        return FUEL_ROS_MULTIPLIER[fm]
    # Prefix match
    for prefix, mult in FUEL_MAP.items():
        if fm.startswith(prefix):
            return mult
    return DEFAULT_FUEL_MULT

rows = []
skipped = 0

for i, (_, fire) in enumerate(nifc.iterrows()):
    if i % 500 == 0:
        print(f"   [{i:>5}/{len(nifc)}]  samples={len(rows):,}  skipped={skipped}")

    lat = float(fire["attr_InitialLatitude"])
    lon = float(fire["attr_InitialLongitude"])
    ref_time = fire["ref_dt"]

    # Get weather sequence
    wx_seq = get_weather_sequence(lat, lon, ref_time, lookback_h=24)
    if wx_seq is None:
        skipped += 1
        continue

    # Get FIRMS context
    firms_ctx = get_firms_context(lat, lon, ref_time)

    # Slope proxy
    slope = get_slope_proxy(lat, lon)

    # Fuel multiplier
    fuel_mult = get_fuel_multiplier(fire.get("attr_PredominantFuelModel"))

    # Wind-slope interaction (Rothermel: upslope + upwind = max spread)
    wind_dir_rad = np.radians(wx_seq["wind_direction"])
    # Simple upslope proxy: higher elevation = more slope effect
    slope_wind_interaction = slope * wx_seq["wind_speed_kmh"] / 50.0

    # Target: growth rate at different horizons
    # We have total growth rate for the fire lifetime
    # Approximate horizon-specific rates with decay (fires slow as they grow)
    base_rate = float(fire["growth_rate_km2h"])
    duration  = float(fire["duration_h"])

    # Exponential decay model: rate is highest early, slows as fire matures
    rate_6h  = base_rate * np.exp(-6  / max(duration, 24))
    rate_12h = base_rate * np.exp(-12 / max(duration, 24))
    rate_24h = base_rate * np.exp(-24 / max(duration, 24))

    rows.append({
        # Identity
        "fire_id":          str(fire["attr_UniqueFireIdentifier"]),
        "fire_name":        str(fire["attr_IncidentName"]),
        "state":            str(fire["attr_POOState"]),
        "ref_date":         str(ref_time.date()),
        "lat":              lat,
        "lon":              lon,

        # Fire state
        "current_area_km2": float(fire["area_km2"]),
        "duration_h":       float(fire["duration_h"]),
        "fuel_multiplier":  fuel_mult,
        "slope_proxy":      slope,

        # Weather (24h mean)
        "wind_speed_kmh":   wx_seq["wind_speed_kmh"],
        "wind_direction":   wx_seq["wind_direction"],
        "temperature_c":    wx_seq["temperature_c"],
        "humidity":         wx_seq["humidity"],
        "max_wind_kmh":     wx_seq["max_wind_kmh"],
        "wind_speed_trend": wx_seq["wind_speed_trend"],
        "temp_trend":       wx_seq["temp_trend"],
        "humidity_trend":   wx_seq["humidity_trend"],
        "drought_index":    wx_seq["drought_index"],

        # FIRMS context
        "frp_mean":         firms_ctx["frp_mean"],
        "frp_max":          firms_ctx["frp_max"],
        "n_detections":     firms_ctx["n_detections"],
        "fire_density":     firms_ctx["fire_density"],
        "frp_trend":        firms_ctx["frp_trend"],

        # Interaction features
        "slope_wind_interaction": slope_wind_interaction,
        "wind_x_fuel":      wx_seq["wind_speed_kmh"] * fuel_mult,
        "heat_dryness":     max(0, wx_seq["temperature_c"] - 25) *
                            max(0, 60 - wx_seq["humidity"]) / 100,

        # Targets — growth rate at each horizon
        "growth_rate_6h":   round(rate_6h,  4),
        "growth_rate_12h":  round(rate_12h, 4),
        "growth_rate_24h":  round(rate_24h, 4),
        "growth_rate_total":round(base_rate, 4),
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_PATH, index=False)

print(f"\n{'='*70}")
print("SPREAD DATASET COMPLETE")
print(f"{'='*70}")
print(f"Total samples      : {len(df_out):,}")
print(f"Skipped (no wx)    : {skipped:,}")
print(f"\nGrowth rate distribution (km2/h):")
for col in ["growth_rate_6h","growth_rate_12h","growth_rate_24h"]:
    print(f"  {col:<25} mean={df_out[col].mean():.3f}  "
          f"max={df_out[col].max():.2f}  "
          f"p90={df_out[col].quantile(0.9):.3f}")
print(f"\nBy state:")
print(df_out["state"].value_counts().to_string())
print(f"\nDate range: {df_out['ref_date'].min()} → {df_out['ref_date'].max()}")
print(f"\nSaved → {OUT_PATH}")
print(f"{'='*70}")

if __name__ == "__main__":
    pass