"""
build_tft_v2_dataset.py  —  M4 (point-based radial profile)
=============================================================
Builds TFT v2 training data from FIRMS daily fire clusters.

Label approach: RADIAL PROFILE (not centroid)
  - For each fire cluster, measure distance from centroid to hull
    boundary in 8 fixed directions (N, NE, E, SE, S, SW, W, NW)
  - Compare T vs T+1 to get expansion per direction (km)
  - TFT learns: given current profile + weather → predict 8 expansions

This produces asymmetric projected perimeters:
  - Downwind direction: large positive expansion
  - Upwind direction:   small or near-zero
  - Flanking:           moderate
  → Perimeter shape changes realistically, not as rigid body shift

Labels (8 values per sample):
  radial_delta_000  (North expansion, km)
  radial_delta_045  (Northeast)
  radial_delta_090  (East)
  radial_delta_135  (Southeast)
  radial_delta_180  (South)
  radial_delta_225  (Southwest)
  radial_delta_270  (West)
  radial_delta_315  (Northwest)

Features (12):
  area_km2, n_points, max_frp, mean_frp,
  wind_speed_kmh, wind_direction, temperature_c,
  humidity, drought_index, days_since_rain,
  + 8 current radial profile values (r_000 … r_315)

Temporal split:
  Train : 2017-01-01 → 2021-12-31
  Val   : 2022-01-01 → 2023-12-31
  Test  : 2024-01-01 → 2024-12-31
"""

import json, datetime, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

try:
    from shapely.geometry import MultiPoint
    SHAPELY = True
except ImportError:
    SHAPELY = False

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TFT_DIR  = DATA_DIR / "tft_v2"
TFT_DIR.mkdir(parents=True, exist_ok=True)

FIRMS_PATH  = DATA_DIR / "fires"   / "national_fires_2017_2025.csv"
WEATHER_DIR = DATA_DIR / "weather"

DBSCAN_EPS_DEG  = 0.15
DBSCAN_MIN_PTS  = 3
MIN_CLUSTER_PTS = 5
MIN_AREA_KM2    = 0.5    # slightly higher — need enough points for radial profile
MAX_MATCH_KM    = 50.0
MAX_MATCH_DEG   = MAX_MATCH_KM / 111.0

# Radial profile directions
N_DIRS   = 8
DIRS_DEG = np.arange(0, 360, 360/N_DIRS)   # [0,45,90,135,180,225,270,315]
DIR_NAMES= [f"{int(d):03d}" for d in DIRS_DEG]
LABEL_COLS  = [f"radial_delta_{n}" for n in DIR_NAMES]
PROFILE_COLS= [f"r_{n}" for n in DIR_NAMES]

TRAIN_END = "2021-12-31"
VAL_END   = "2023-12-31"

print("=" * 70)
print("M4 — TFT v2 DATASET (radial profile / point-based)")
print(f"8 directions: {list(DIRS_DEG.astype(int))}")
print("=" * 70)

# ── 1. Load FIRMS ─────────────────────────────────────────────────────────────
print("\n1. Loading FIRMS detections...")
fires = pd.read_csv(FIRMS_PATH, low_memory=False)
fires["date"] = pd.to_datetime(fires["acq_date"], errors="coerce").dt.date
fires = fires.dropna(subset=["date","latitude","longitude"])
fires = fires[fires["confidence"].astype(str).str.lower().isin(["h","high"])].copy()
fires["frp"] = pd.to_numeric(fires["frp"], errors="coerce").fillna(0)
fires = fires.sort_values("date").reset_index(drop=True)
fires = fires[
    (fires["latitude"].between(31,50)) &
    (fires["longitude"].between(-125,-102))
].copy()
print(f"   ✓ {len(fires):,} detections")
by_date   = {d: g for d, g in fires.groupby("date")}
all_dates = sorted(by_date.keys())

# ── 2. Load weather ───────────────────────────────────────────────────────────
print("\n2. Loading weather...")
wx_parts = []
for f in sorted(WEATHER_DIR.glob("*_weather_grid.csv")):
    wx_parts.append(pd.read_csv(f, low_memory=False))
wx = pd.concat(wx_parts, ignore_index=True)
wx["datetime"] = pd.to_datetime(wx["datetime"], errors="coerce")
wx = wx.dropna(subset=["datetime"]).copy()
wx["date"]     = wx["datetime"].dt.date
wx["grid_lat"] = wx["grid_lat"].round(4)
wx["grid_lon"] = wx["grid_lon"].round(4)
for col in ["drought_index","days_since_rain"]:
    if col not in wx.columns:
        wx[col] = 0.5 if col=="drought_index" else 7.0
wx_pts  = wx[["grid_lat","grid_lon"]].drop_duplicates().values
wx_tree = cKDTree(wx_pts)
wx["grid_idx"] = wx_tree.query(wx[["grid_lat","grid_lon"]].values)[1]
wx_lookup = {}
for row in wx[["grid_idx","date","wind_speed_kmh","wind_direction",
               "temp_c","humidity","drought_index","days_since_rain"]
             ].itertuples(index=False):
    key = (row.grid_idx, row.date)
    if key not in wx_lookup:
        wx_lookup[key] = np.array([
            row.wind_speed_kmh, row.wind_direction,
            row.temp_c, row.humidity,
            row.drought_index, row.days_since_rain
        ], dtype=np.float32)
print(f"   ✓ {len(wx_lookup):,} records indexed")

# ── 3. Radial profile helpers ─────────────────────────────────────────────────

def radial_profile(c_lat, c_lon, pt_lats, pt_lons):
    """
    For each of N_DIRS directions, find the max distance (km) from
    centroid to any hull point in the ±(180/N_DIRS)° sector around
    that direction.
    Sector half-width = 360/N_DIRS/2 = 22.5° for 8 directions.
    """
    KM_PER_DEG_LAT = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(c_lat))

    # Relative positions in km
    dy = (pt_lats - c_lat) * KM_PER_DEG_LAT   # + = north
    dx = (pt_lons - c_lon) * km_per_deg_lon    # + = east

    # Angle of each point from centroid (0=North, clockwise)
    angles = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    dists  = np.sqrt(dx**2 + dy**2)

    sector_hw = 180.0 / N_DIRS   # half-width in degrees (22.5°)
    radii = np.zeros(N_DIRS)

    for i, d in enumerate(DIRS_DEG):
        diff = np.abs(((angles - d) + 180) % 360 - 180)
        mask = diff <= sector_hw
        if mask.sum() > 0:
            radii[i] = float(dists[mask].max())
        else:
            radii[i] = 0.0

    # Fill zeros by interpolating from nearest non-zero neighbours
    for i in range(N_DIRS):
        if radii[i] == 0:
            # Try neighbouring sectors
            for step in range(1, N_DIRS):
                l = (i - step) % N_DIRS
                r = (i + step) % N_DIRS
                vals = [v for v in [radii[l], radii[r]] if v > 0]
                if vals:
                    radii[i] = float(np.mean(vals))
                    break
            if radii[i] == 0:
                radii[i] = 0.1   # fallback minimum

    return radii   # km per direction

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1,lo1 = np.radians(lat1), np.radians(lon1)
    la2,lo2 = np.radians(lat2), np.radians(lon2)
    a = (np.sin((la2-la1)/2)**2 +
         np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))

def hull_area(lats, lons):
    if SHAPELY and len(lats) >= 3:
        try:
            hull = MultiPoint(list(zip(lons,lats))).convex_hull
            lat_c = float(np.mean(lats))
            return float(hull.area
                         * 111.0
                         * (111.0 * np.cos(np.radians(lat_c))))
        except Exception:
            pass
    dlat = (max(lats)-min(lats))*111.0
    dlon = (max(lons)-min(lons))*111.0*np.cos(np.radians(np.mean(lats)))
    return max(float(dlat*dlon), 0.01)

# ── 4. Cluster and compute radial profiles ────────────────────────────────────
print("\n3. Clustering and computing radial profiles...")
dbscan = DBSCAN(eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_PTS, n_jobs=-1)
clusters_by_date = {}

for i, date_t in enumerate(all_dates):
    if i % 300 == 0:
        print(f"   [{100*i/len(all_dates):5.1f}%] {date_t}")

    day_fires = by_date[date_t]
    coords    = day_fires[["latitude","longitude"]].values
    if len(coords) < DBSCAN_MIN_PTS:
        continue

    labels = dbscan.fit_predict(coords)
    clusters = []

    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        if mask.sum() < MIN_CLUSTER_PTS:
            continue

        c_lats = coords[mask,0]
        c_lons = coords[mask,1]
        c_frps = day_fires["frp"].values[mask]
        c_lat  = float(np.mean(c_lats))
        c_lon  = float(np.mean(c_lons))
        area   = hull_area(c_lats, c_lons)

        if area < MIN_AREA_KM2:
            continue

        # Radial profile
        profile = radial_profile(c_lat, c_lon, c_lats, c_lons)

        # Weather
        _, gi  = wx_tree.query([c_lat, c_lon])
        wx_row = wx_lookup.get((int(gi), date_t))
        if wx_row is None:
            continue

        clusters.append({
            "date":        date_t,
            "centroid_lat":round(c_lat,5),
            "centroid_lon":round(c_lon,5),
            "area_km2":    round(area,4),
            "n_points":    int(mask.sum()),
            "max_frp":     round(float(c_frps.max()),2),
            "mean_frp":    round(float(c_frps.mean()),2),
            "wind_speed_kmh":  round(float(wx_row[0]),2),
            "wind_direction":  round(float(wx_row[1]),2),
            "temperature_c":   round(float(wx_row[2]),2),
            "humidity":        round(float(wx_row[3]),2),
            "drought_index":   round(float(wx_row[4]),4),
            "days_since_rain": round(float(wx_row[5]),2),
            "profile":     profile,   # ndarray shape (N_DIRS,)
        })

    if clusters:
        clusters_by_date[date_t] = clusters

n_total = sum(len(v) for v in clusters_by_date.values())
print(f"\n   ✓ {n_total:,} fire clusters across {len(clusters_by_date):,} days")

# ── 5. Match clusters T → T+1, compute radial deltas ─────────────────────────
print("\n4. Matching clusters and computing radial deltas...")

rows = []
for date_t in sorted(clusters_by_date.keys()):
    date_t1 = date_t + datetime.timedelta(days=1)
    if date_t1 not in clusters_by_date:
        continue

    ct_list  = clusters_by_date[date_t]
    ct1_list = clusters_by_date[date_t1]

    t1_cents = np.array([[c["centroid_lat"],c["centroid_lon"]]
                          for c in ct1_list])
    t1_tree  = cKDTree(t1_cents)

    for ct in ct_list:
        dist_deg, idx = t1_tree.query(
            [ct["centroid_lat"],ct["centroid_lon"]], k=1)
        if dist_deg > MAX_MATCH_DEG:
            continue

        ct1    = ct1_list[int(idx)]
        prof_t  = ct["profile"]
        prof_t1 = ct1["profile"]
        deltas  = prof_t1 - prof_t   # expansion per direction (km)

        row = {
            "date":         str(date_t),
            "year":         date_t.year,
            # Features at T
            "area_km2":         ct["area_km2"],
            "n_points":         ct["n_points"],
            "max_frp":          ct["max_frp"],
            "mean_frp":         ct["mean_frp"],
            "wind_speed_kmh":   ct["wind_speed_kmh"],
            "wind_direction":   ct["wind_direction"],
            "temperature_c":    ct["temperature_c"],
            "humidity":         ct["humidity"],
            "drought_index":    ct["drought_index"],
            "days_since_rain":  ct["days_since_rain"],
        }
        # Current radial profile (model input)
        for j, name in enumerate(DIR_NAMES):
            row[f"r_{name}"] = round(float(prof_t[j]), 4)
        # Radial delta labels (model target)
        for j, name in enumerate(DIR_NAMES):
            row[f"radial_delta_{name}"] = round(float(deltas[j]), 4)

        rows.append(row)

print(f"   ✓ {len(rows):,} matched pairs with radial profiles")

# ── 6. Temporal split ─────────────────────────────────────────────────────────
print("\n5. Applying temporal split...")
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])

train = df[df["date"] <= TRAIN_END].copy()
val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
test  = df[df["date"] > VAL_END].copy()

assert train["date"].max() <= pd.Timestamp(TRAIN_END)
assert val["date"].max()   <= pd.Timestamp(VAL_END)
assert test["date"].min()  >  pd.Timestamp(VAL_END)
print("   ✅ No temporal leakage")
print(f"   Train: {len(train):,}")
print(f"   Val  : {len(val):,}")
print(f"   Test : {len(test):,}")

# Print label stats
print(f"\n   Radial delta stats (km/day) across all 8 directions:")
all_deltas = df[LABEL_COLS].values.flatten()
print(f"     Median : {np.median(all_deltas):.2f}")
print(f"     P90    : {np.percentile(all_deltas,90):.2f}")
print(f"     Max    : {np.max(all_deltas):.2f}")
print(f"     Min    : {np.min(all_deltas):.2f}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
train.to_parquet(TFT_DIR/"tft_train.parquet", index=False)
val.to_parquet(  TFT_DIR/"tft_val.parquet",   index=False)
test.to_parquet( TFT_DIR/"tft_test.parquet",  index=False)

# ── 8. Validation plots ───────────────────────────────────────────────────────
print("\n6. Generating validation plots...")

FEAT_COLS = ["area_km2","n_points","max_frp","mean_frp",
             "wind_speed_kmh","wind_direction","temperature_c",
             "humidity","drought_index","days_since_rain"]

# Radial delta distribution — polar + histogram
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
dir_labels = ["N(0°)","NE(45°)","E(90°)","SE(135°)",
              "S(180°)","SW(225°)","W(270°)","NW(315°)"]
for i, (col, dlabel) in enumerate(zip(LABEL_COLS, dir_labels)):
    ax = axes[i]
    vals = train[col].clip(
        train[col].quantile(0.02),
        train[col].quantile(0.98))
    ax.hist(vals, bins=40, color="#FF7043", alpha=0.8, edgecolor="white")
    ax.axvline(0,  color="black", lw=1, ls="--", alpha=0.5)
    ax.axvline(train[col].median(), color="blue", lw=1.5, ls="--",
               label=f"med={train[col].median():.1f}")
    ax.set_title(dlabel, fontweight="bold")
    ax.set_xlabel("Expansion (km)")
    ax.legend(fontsize=8)
fig.suptitle("Radial Delta Distributions per Direction — Train Set\n"
             "Positive = expansion, Negative = retreat",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(TFT_DIR/"label_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ label_distribution.png")

# Polar plot of median expansion per direction
fig, ax = plt.subplots(figsize=(8, 8),
                        subplot_kw={"projection":"polar"})
angles_rad = np.radians(DIRS_DEG)
medians    = [float(train[col].median()) for col in LABEL_COLS]
medians_closed = medians + [medians[0]]
angles_closed  = np.append(angles_rad, angles_rad[0])
# Polar: 0=East in matplotlib, but we want 0=North
# Convert: N=0° → 90° in polar; rotate by 90°
angles_polar = (np.pi/2 - angles_closed) % (2*np.pi)
ax.plot(angles_polar, medians_closed, "o-",
        color="#FF7043", lw=2, ms=8)
ax.fill(angles_polar, [max(0,m) for m in medians_closed],
        alpha=0.3, color="#FF7043")
ax.set_thetagrids(DIRS_DEG, labels=dir_labels)
ax.set_title("Median Radial Expansion per Direction\n(Train set)",
             fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(TFT_DIR/"radial_expansion_polar.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ radial_expansion_polar.png")

# Samples per year
fig, ax = plt.subplots(figsize=(12,4))
yr = df.groupby("year").size()
ax.bar(yr.index, yr.values, color="#FF7043", alpha=0.85, edgecolor="white")
ax.axvline(2021.5, color="orange", lw=2, ls="--", label="Train/Val split")
ax.axvline(2023.5, color="green",  lw=2, ls="--", label="Val/Test split")
ax.set_xlabel("Year"); ax.set_ylabel("Samples")
ax.set_title("Samples per Year", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(TFT_DIR/"samples_per_year.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ samples_per_year.png")

# Feature correlation with one label (north expansion)
fig, ax = plt.subplots(figsize=(12, 6))
corr_with = {}
for col in FEAT_COLS + PROFILE_COLS:
    if col in train.columns:
        corr_with[col] = float(train[col].corr(train["radial_delta_000"]))
corr_series = pd.Series(corr_with).sort_values()
colors = ["#F44336" if v < 0 else "#2196F3" for v in corr_series]
ax.barh(corr_series.index, corr_series.values, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Correlation with North Expansion (radial_delta_000)")
ax.set_title("Feature Correlation — North Direction Label\n"
             "Expected: wind_direction and wind_speed should correlate",
             fontweight="bold")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(TFT_DIR/"feature_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ feature_correlation.png")

# ── 9. Summary ────────────────────────────────────────────────────────────────
summary = {
    "total_samples":       len(df),
    "train":  {"n":len(train),"period":"2017-2021"},
    "val":    {"n":len(val),  "period":"2022-2023"},
    "test":   {"n":len(test), "period":"2024"},
    "label_approach":      "radial_profile",
    "n_directions":        N_DIRS,
    "directions_deg":      [int(d) for d in DIRS_DEG],
    "label_columns":       LABEL_COLS,
    "feature_columns":     FEAT_COLS,
    "profile_columns":     PROFILE_COLS,
    "label_source":        "FIRMS VIIRS daily fire clusters",
    "no_synthetic_labels": True,
    "temporal_leakage":    False,
    "limitation": (
        "Convex hull approximation — real perimeters follow terrain. "
        "Accurate for large fires, less so for small/sparse detections."
    ),
}
with open(TFT_DIR/"dataset_summary.json","w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print("M4 COMPLETE — radial profile labels")
print(f"{'='*70}")
print(f"Total  : {len(df):,}")
print(f"Train  : {len(train):,}")
print(f"Val    : {len(val):,}")
print(f"Test   : {len(test):,}")
print(f"\n8 directional labels per sample (N NE E SE S SW W NW)")
print(f"Positive = fire expanded in that direction")
print(f"Negative = fire retreated in that direction")
print(f"\nNext: M5 — python src/pipeline/train_tft_v2.py")
print(f"{'='*70}")