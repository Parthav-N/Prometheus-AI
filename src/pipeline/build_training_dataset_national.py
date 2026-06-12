"""
build_training_dataset_national.py  (v3 + acq_date fix)
=========================================================
Identical to v3 except:
  - acq_date preserved on every observed sample (required for temporal split)
  - acq_date assigned to synthetic samples from 2017-2023 only
    (ensures synthetic data never leaks into 2024-2025 validation set)

Output: data/training_dataset_national.csv
Run:    python src/pipeline/build_training_dataset.py
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

FIRES_PATH   = DATA_DIR / "fires"          / "national_fires_2017_2025.csv"
WEATHER_PATH = DATA_DIR / "weather"        / "national_weather_grid.csv"
INFRA_PATH   = DATA_DIR / "infrastructure" / "national_infrastructure.csv"
OUT_PATH     = DATA_DIR / "training_dataset_national.csv"

SAMPLE_FREQ_DAYS  = 5
ASSETS_PER_WINDOW = 30
MAX_FIRE_DIST_KM  = 300
NEARBY_RADIUS_KM  = 30

TARGET_HIGH = 7350
TARGET_MED  = 6300
TARGET_LOW  = 7350

# ---------------------------------------------------------------------------

def haversine_vec(lat1, lon1, lats2, lons2):
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats2 = np.radians(np.asarray(lats2, dtype=float))
    lons2 = np.radians(np.asarray(lons2, dtype=float))
    dlat, dlon = lats2 - lat1, lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lats2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
    lat1, lon1 = np.radians(fire_lat), np.radians(fire_lon)
    lat2, lon2 = np.radians(asset_lat), np.radians(asset_lon)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
    wind_toward = (wind_dir + 180) % 360
    angle_diff = abs(wind_toward - bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return float(np.cos(np.radians(angle_diff)))


def risk_score(min_dist, num_fires, wind_speed, max_frp,
               wind_align, temp, humidity, drought_idx):
    if   min_dist < 1:   base = 95
    elif min_dist < 3:   base = 85
    elif min_dist < 7:   base = 70
    elif min_dist < 15:  base = 50
    elif min_dist < 30:  base = 30
    elif min_dist < 50:  base = 15
    elif min_dist < 100: base = 8
    elif min_dist < 200: base = 4
    else:                base = 2

    wind_factor = 1.0 + max(0, (wind_speed - 20) / 40)
    if wind_align > 0.5:    wind_factor *= (1.0 + wind_align * 0.3)
    elif wind_align < -0.5: wind_factor *= 0.8

    cluster_factor   = 1.0 + min(num_fires / 20, 0.4)
    intensity_factor = 1.0 + min(max_frp / 150, 0.3)

    if   temp > 30 and humidity < 30: weather_factor = 1.2
    elif temp > 25 and humidity < 40: weather_factor = 1.1
    elif temp < 15 or  humidity > 70: weather_factor = 0.9
    else:                             weather_factor = 1.0

    drought_factor = 1.0 + drought_idx * 0.25

    return float(min(
        base * wind_factor * cluster_factor * intensity_factor
             * weather_factor * drought_factor,
        100.0
    ))


STATES = ["CA","OR","WA","MT","CO","ID","WY","UT","NV","AZ"]
ASSET_TYPES = [
    "Power Substation","Wind Farm","Solar Farm","Gas Plant","Coal Plant",
    "Hydro Plant","Hospital","Fire Station","School","Residential Area",
    "Cell Tower","Water Treatment","Airport","Medical Clinic","University",
]

# Pre-build date pool for synthetic samples — training period only (pre-2024)
# This guarantees synthetic samples never appear in the 2024-2025 val set.
_SYNTHETIC_DATES = pd.date_range("2017-01-01", "2023-12-31", freq="D")

def generate_synthetic(n: int, risk_range: tuple,
                       rng: np.random.Generator) -> list:
    lo, hi   = risk_range
    samples  = []
    attempts = 0

    while len(samples) < n and attempts < n * 20:
        attempts += 1

        if hi <= 40:
            min_dist   = rng.uniform(80, 300)
            num_fires  = int(rng.uniform(0, 20))
            wind_speed = rng.uniform(0, 25)
            max_frp    = rng.uniform(1, 50)
            wind_align = rng.uniform(-1, 0.3)
            temp       = rng.uniform(-10, 20)
            humidity   = rng.uniform(50, 100)
            drought    = rng.uniform(0, 0.3)
            dsr        = rng.uniform(0, 5)
        elif hi <= 75:
            min_dist   = rng.uniform(15, 80)
            num_fires  = int(rng.uniform(5, 100))
            wind_speed = rng.uniform(15, 45)
            max_frp    = rng.uniform(10, 100)
            wind_align = rng.uniform(-0.5, 0.7)
            temp       = rng.uniform(10, 30)
            humidity   = rng.uniform(25, 65)
            drought    = rng.uniform(0.2, 0.7)
            dsr        = rng.uniform(3, 20)
        else:
            min_dist   = rng.uniform(0.1, 15)
            num_fires  = int(rng.uniform(10, 500))
            wind_speed = rng.uniform(30, 80)
            max_frp    = rng.uniform(50, 300)
            wind_align = rng.uniform(0.3, 1.0)
            temp       = rng.uniform(25, 45)
            humidity   = rng.uniform(5, 35)
            drought    = rng.uniform(0.5, 1.0)
            dsr        = rng.uniform(10, 30)

        mean_dist = min_dist * rng.uniform(1.5, 5.0)
        wind_dir  = rng.uniform(0, 360)
        score     = risk_score(min_dist, num_fires, wind_speed, max_frp,
                               wind_align, temp, humidity, drought)

        if lo <= score < hi:
            # Assign a random date from the training period only
            synthetic_date = str(
                _SYNTHETIC_DATES[rng.integers(0, len(_SYNTHETIC_DATES))].date()
            )
            samples.append({
                "acq_date":            synthetic_date,          # ← training-period only
                "min_distance_km":     round(min_dist, 3),
                "mean_distance_km":    round(mean_dist, 3),
                "num_fires_30km":      num_fires,
                "max_frp":             round(max_frp, 2),
                "wind_speed_kmh":      round(wind_speed, 2),
                "wind_direction":      round(wind_dir, 1),
                "temperature_c":       round(temp, 2),
                "humidity":            round(humidity, 1),
                "wind_fire_alignment": round(wind_align, 4),
                "drought_index":       round(drought, 4),
                "days_since_rain":     round(dsr, 1),
                "asset_type":          rng.choice(ASSET_TYPES),
                "state":               rng.choice(STATES),
                "risk_score":          round(score, 2),
                "source":              "synthetic",
            })

    return samples


def main():
    print("=" * 70)
    print("BUILDING NATIONAL TRAINING DATASET  (v3 + acq_date)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    print("\n1. Loading fire data...")
    fires_df = pd.read_csv(FIRES_PATH, low_memory=False)
    fires_df["datetime"] = pd.to_datetime(
        fires_df["acq_date"] + " " +
        fires_df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M", errors="coerce"
    )
    fires_df = fires_df.dropna(subset=["datetime"])
    fires_df = fires_df[
        fires_df["confidence"].astype(str).str.lower()
        .isin(["h", "n", "high", "nominal"])
    ].copy()
    print(f"   ✓ {len(fires_df):,} detections")

    print("\n2. Loading weather...")
    weather_df = pd.read_csv(WEATHER_PATH, low_memory=False)
    weather_df["datetime"] = pd.to_datetime(
        weather_df["datetime"], utc=False, errors="coerce"
    )
    if weather_df["datetime"].dt.tz is not None:
        weather_df["datetime"] = weather_df["datetime"].dt.tz_localize(None)
    weather_df = weather_df.dropna(subset=["datetime"])
    weather_df["grid_lat"] = weather_df["grid_lat"].round(4)
    weather_df["grid_lon"] = weather_df["grid_lon"].round(4)
    print(f"   ✓ {len(weather_df):,} records, "
          f"{weather_df['location'].nunique()} locations")

    print("   Computing drought index...")
    weather_df = weather_df.sort_values(["location", "datetime"])
    drought_vals, dsr_vals = [], []

    for _, grp in weather_df.groupby("location", sort=False):
        precip     = grp["precipitation"].fillna(0).values
        precip_30d = pd.Series(precip).rolling(60, min_periods=1).sum().values
        drought    = 1.0 - np.clip(precip_30d / 10.0, 0, 1)
        drought_vals.extend(drought.tolist())

        last_rain = None
        for idx_r in grp.index:
            if weather_df.at[idx_r, "precipitation"] > 0.1:
                last_rain = weather_df.at[idx_r, "datetime"]
            dsr_vals.append(
                min((weather_df.at[idx_r, "datetime"] - last_rain).days, 30.0)
                if last_rain else 30.0
            )

    weather_df["drought_index"]   = drought_vals
    weather_df["days_since_rain"] = dsr_vals

    weather_coords = weather_df[["grid_lat","grid_lon"]].drop_duplicates().values
    weather_tree   = cKDTree(weather_coords)
    print(f"   ✓ KDTree: {len(weather_coords)} unique locations")

    print("\n3. Loading infrastructure...")
    infra_df = pd.read_csv(INFRA_PATH, low_memory=False)
    infra_df = infra_df[infra_df["type"] != "Transmission Line"].copy()
    print(f"   ✓ {len(infra_df):,} assets")

    fire_start   = fires_df["datetime"].min()
    fire_end     = fires_df["datetime"].max()
    time_windows = pd.date_range(fire_start, fire_end,
                                 freq=f"{SAMPLE_FREQ_DAYS}D")
    states           = infra_df["state"].dropna().unique()
    assets_per_state = max(1, ASSETS_PER_WINDOW // len(states))

    print(f"\n4. Time windows: {len(time_windows)}")

    def get_weather(lat, lon, target_time):
        _, idx   = weather_tree.query([lat, lon])
        nearest  = weather_coords[idx]
        station  = weather_df[
            (weather_df["grid_lat"] == nearest[0]) &
            (weather_df["grid_lon"] == nearest[1])
        ].copy()
        station["tdiff"] = (
            (station["datetime"] - target_time).abs().dt.total_seconds()
        )
        best = station.nsmallest(1, "tdiff")
        if len(best) > 0 and best.iloc[0]["tdiff"] < 21600:
            return best.iloc[0]
        return None

    print("\n5. Collecting real high-risk samples...")
    high_samples = []

    for i, current_time in enumerate(time_windows[:-1]):
        if len(high_samples) >= TARGET_HIGH:
            break
        if i % 100 == 0:
            print(f"   [{i:>4}/{len(time_windows)}]  high={len(high_samples):,}")

        current_fires = fires_df[fires_df["datetime"] <= current_time]
        if len(current_fires) < 5:
            continue

        parts = []
        for state in states:
            sa = infra_df[infra_df["state"] == state]
            if len(sa) == 0: continue
            parts.append(sa.sample(n=min(assets_per_state, len(sa)),
                                   replace=False, random_state=i))
        if not parts: continue
        sampled = pd.concat(parts, ignore_index=True)

        for _, asset in sampled.iterrows():
            if len(high_samples) >= TARGET_HIGH: break

            distances = haversine_vec(
                asset["lat"], asset["lon"],
                current_fires["latitude"].values,
                current_fires["longitude"].values
            )
            if distances.min() > MAX_FIRE_DIST_KM: continue

            wx = get_weather(asset["lat"], asset["lon"], current_time)
            if wx is None: continue

            closest_fire = current_fires.iloc[distances.argmin()]
            w_align = wind_alignment(
                closest_fire["latitude"], closest_fire["longitude"],
                asset["lat"], asset["lon"],
                float(wx["wind_direction"])
            )
            drought_idx  = float(wx.get("drought_index",   0.0))
            days_since_r = float(wx.get("days_since_rain", 7.0))

            score = risk_score(
                float(distances.min()),
                int((distances < NEARBY_RADIUS_KM).sum()),
                float(wx["wind_speed_kmh"]), float(current_fires["frp"].max()),
                w_align, float(wx["temp_c"]), float(wx["humidity"]),
                drought_idx
            )

            if score >= 75:
                high_samples.append({
                    "acq_date":            str(current_time.date()),  # ← preserved
                    "min_distance_km":     float(distances.min()),
                    "mean_distance_km":    float(distances.mean()),
                    "num_fires_30km":      int((distances < NEARBY_RADIUS_KM).sum()),
                    "max_frp":             float(current_fires["frp"].max()),
                    "wind_speed_kmh":      float(wx["wind_speed_kmh"]),
                    "wind_direction":      float(wx["wind_direction"]),
                    "temperature_c":       float(wx["temp_c"]),
                    "humidity":            float(wx["humidity"]),
                    "wind_fire_alignment": w_align,
                    "drought_index":       drought_idx,
                    "days_since_rain":     days_since_r,
                    "asset_type":          asset["type"],
                    "state":               asset["state"],
                    "risk_score":          score,
                    "source":              "observed",
                })

    print(f"   ✓ {len(high_samples):,} high-risk samples")

    print("\n6. Generating synthetic low and medium risk samples...")
    low_samples = generate_synthetic(TARGET_LOW, (0,  40), rng)
    med_samples = generate_synthetic(TARGET_MED, (40, 75), rng)
    print(f"   ✓ Low : {len(low_samples):,}")
    print(f"   ✓ Med : {len(med_samples):,}")

    all_samples = high_samples + low_samples + med_samples
    df_out = pd.DataFrame(all_samples)
    df_out["acq_date"] = pd.to_datetime(df_out["acq_date"])
    df_out = df_out.sample(frac=1, random_state=42).reset_index(drop=True)
    df_out.to_csv(OUT_PATH, index=False)

    n      = len(df_out)
    n_low  = (df_out["risk_score"] <  40).sum()
    n_med  = df_out["risk_score"].between(40, 75).sum()
    n_high = (df_out["risk_score"] >  75).sum()

    print(f"\n{'='*70}")
    print("DATASET COMPLETE  (v3 + acq_date)")
    print(f"{'='*70}")
    print(f"Total   : {n:,}")
    print(f"  Low   : {n_low:,}  ({n_low/n*100:.1f}%)")
    print(f"  Med   : {n_med:,}  ({n_med/n*100:.1f}%)")
    print(f"  High  : {n_high:,}  ({n_high/n*100:.1f}%)")
    print(f"\nDate range : {df_out['acq_date'].min().date()} → "
          f"{df_out['acq_date'].max().date()}")
    print(f"\nBy year:")
    print(df_out.groupby(df_out["acq_date"].dt.year).size().to_string())
    print(f"\nBy source:")
    for s, c in df_out["source"].value_counts().items():
        print(f"  {s:<12} {c:>6,}  ({c/n*100:.1f}%)")
    print(f"\nSaved → {OUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()