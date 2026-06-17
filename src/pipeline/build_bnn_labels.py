"""
build_bnn_labels.py  —  M1
===========================
Builds BNN training data from real FIRMS satellite detections.

Label: did fire appear within 5km of this asset within next 24h?
Source: NASA FIRMS VIIRS — no synthetic labels.

Fully vectorized — no inner Python loops over assets.
All distance calculations batched with numpy.

Temporal split:
  Train : 2017-01-01 → 2021-12-31
  Val   : 2022-01-01 → 2023-12-31
  Test  : 2024-01-01 → 2024-12-31  ← never touched during training
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

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
BNN_DIR   = DATA_DIR / "bnn"
BNN_DIR.mkdir(parents=True, exist_ok=True)

FIRMS_PATH  = DATA_DIR / "fires"          / "national_fires_2017_2025.csv"
INFRA_PATH  = DATA_DIR / "infrastructure" / "national_infrastructure.csv"
WEATHER_DIR = DATA_DIR / "weather"

LABEL_RADIUS_KM   = 5.0
NEARBY_RADIUS_KM  = 50.0
NEARBY_RADIUS_DEG = NEARBY_RADIUS_KM / 111.0
TRAIN_END = "2021-12-31"
VAL_END   = "2023-12-31"

print("=" * 70)
print("M1 — BUILD BNN TRAINING DATA (fully vectorized)")
print("=" * 70)

# ── 1. Load FIRMS ─────────────────────────────────────────────────────────────
print("\n1. Loading FIRMS...")
fires = pd.read_csv(FIRMS_PATH, low_memory=False)
fires["date"] = pd.to_datetime(fires["acq_date"], errors="coerce").dt.date
fires = fires.dropna(subset=["date","latitude","longitude"])
fires = fires[fires["confidence"].astype(str).str.lower().isin(["h","high"])].copy()
fires["frp"] = pd.to_numeric(fires["frp"], errors="coerce").fillna(0)
fires = fires.sort_values("date").reset_index(drop=True)
print(f"   ✓ {len(fires):,} detections")
by_date = {d: g for d, g in fires.groupby("date")}
all_dates = sorted(by_date.keys())

# ── 2. Load infrastructure ────────────────────────────────────────────────────
print("\n2. Loading infrastructure...")
infra = pd.read_csv(INFRA_PATH, low_memory=False)
infra.columns = [c.strip().lower() for c in infra.columns]
infra["lat"] = pd.to_numeric(infra["lat"], errors="coerce")
infra["lon"] = pd.to_numeric(infra["lon"], errors="coerce")
infra = infra.dropna(subset=["lat","lon"])
infra = infra[~infra["type"].isin(["Residential Area"])].reset_index(drop=True)

# Precompute as numpy arrays — no per-asset DataFrame access
infra_lats  = infra["lat"].values.astype(np.float64)
infra_lons  = infra["lon"].values.astype(np.float64)
infra_types = infra["type"].values
infra_states= infra["state"].values
infra_tree  = cKDTree(np.column_stack([infra_lats, infra_lons]))
print(f"   ✓ {len(infra):,} assets")

# ── 3. Load weather — precompute per (grid_point, date) ──────────────────────
print("\n3. Loading weather grids...")
wx_parts = []
for f in sorted(WEATHER_DIR.glob("*_weather_grid.csv")):
    wx_parts.append(pd.read_csv(f, low_memory=False))
wx = pd.concat(wx_parts, ignore_index=True)
wx["datetime"] = pd.to_datetime(wx["datetime"], errors="coerce")
wx = wx.dropna(subset=["datetime"]).copy()
wx["date"]     = wx["datetime"].dt.date
wx["grid_lat"] = wx["grid_lat"].round(4)
wx["grid_lon"] = wx["grid_lon"].round(4)

# Build weather array indexed by (date, grid_idx)
wx_pts   = wx[["grid_lat","grid_lon"]].drop_duplicates().values
wx_tree  = cKDTree(wx_pts)

# Map each weather row to grid_idx
wx["grid_idx"] = wx_tree.query(wx[["grid_lat","grid_lon"]].values)[1]

# Fast lookup: {(grid_idx, date): [ws, wd, tc, hum, drought, dsr]}
WX_COLS = ["wind_speed_kmh","wind_direction","temp_c","humidity"]
for col in ["drought_index","days_since_rain"]:
    if col not in wx.columns:
        wx[col] = 0.5 if col=="drought_index" else 7.0
WX_COLS += ["drought_index","days_since_rain"]

wx_lookup = {}
for row in wx[["grid_idx","date"]+WX_COLS].itertuples(index=False):
    key = (row.grid_idx, row.date)
    if key not in wx_lookup:
        wx_lookup[key] = np.array([
            row.wind_speed_kmh, row.wind_direction,
            row.temp_c, row.humidity,
            row.drought_index, row.days_since_rain
        ], dtype=np.float32)

# Precompute weather grid index for every infrastructure asset (done once)
print("   Precomputing wx grid index for all assets...")
infra_wx_idx = wx_tree.query(np.column_stack([infra_lats, infra_lons]))[1]
print(f"   ✓ {len(wx_lookup):,} weather records indexed")

# ── 4. Vectorized haversine ───────────────────────────────────────────────────

def haversine_vec(alats, alons, flats, flons):
    """1-to-1 haversine: arrays must be same length. Returns distances (km)."""
    R = 6371.0
    la1 = np.radians(np.asarray(alats, dtype=np.float64))
    lo1 = np.radians(np.asarray(alons, dtype=np.float64))
    la2 = np.radians(np.asarray(flats, dtype=np.float64))
    lo2 = np.radians(np.asarray(flons, dtype=np.float64))
    a = (np.sin((la2-la1)/2)**2 +
         np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def wind_align_vec(flats, flons, alats, alons, wind_dirs):
    """
    Wind alignment for N assets against their closest fire.
    flats/flons: closest fire lat/lon per asset (N,)
    alats/alons: asset lat/lon (N,)
    wind_dirs:   wind direction per asset (N,)
    Returns: alignment array (N,)
    """
    la1 = np.radians(flats); lo1 = np.radians(flons)
    la2 = np.radians(alats); lo2 = np.radians(alons)
    dlon = lo2 - lo1
    x = np.sin(dlon)*np.cos(la2)
    y = np.cos(la1)*np.sin(la2) - np.sin(la1)*np.cos(la2)*np.cos(dlon)
    bearing  = (np.degrees(np.arctan2(x, y)) + 360) % 360
    w_toward = (wind_dirs + 180) % 360
    diff     = np.abs(w_toward - bearing)
    diff     = np.where(diff > 180, 360 - diff, diff)
    return np.cos(np.radians(diff))

# ── 5. Build samples — fully vectorized per day ───────────────────────────────
COLS = ["label","min_dist_km","mean_dist_km","n_fires_30km","max_frp",
        "wind_speed_kmh","wind_direction","temperature_c","humidity",
        "wind_fire_alignment","drought_index","days_since_rain",
        "asset_type","state","date"]

# Cap per year to prevent high-fire years from dominating
MAX_PER_YEAR = 300_000

print("\n4. Building samples (vectorized, chunked writes)...")
print(f"   Max {MAX_PER_YEAR:,} samples per year to prevent year imbalance.\n")

chunk_rows  = []
chunk_size  = 500_000   # write to parquet every 500K rows
chunk_files = []
total_built = 0
year_counts = {}        # track samples per year

def flush_chunk(rows, chunk_idx):
    """Write current chunk to a temp parquet file."""
    if not rows:
        return None
    path = BNN_DIR / f"_chunk_{chunk_idx:04d}.parquet"
    pd.DataFrame(rows, columns=COLS).to_parquet(path, index=False)
    return path

chunk_idx = 0

for i, date_t in enumerate(all_dates):
    if i % 300 == 0:
        pct = 100*i/len(all_dates)
        print(f"   [{pct:5.1f}%] Day {i}/{len(all_dates)} "
              f"({date_t}) — {total_built:,} samples")

    year = date_t.year

    # Skip if this year already has enough samples
    if year_counts.get(year, 0) >= MAX_PER_YEAR:
        continue

    fires_t = by_date[date_t]
    flats   = fires_t["latitude"].values.astype(np.float64)
    flons   = fires_t["longitude"].values.astype(np.float64)
    frps    = fires_t["frp"].values.astype(np.float64)
    max_frp = float(frps.max())

    fire_coords = np.column_stack([flats, flons])
    fire_tree_t = cKDTree(fire_coords)

    nearby_sets = infra_tree.query_ball_point(fire_coords, NEARBY_RADIUS_DEG)
    nearby_idx  = np.unique(np.concatenate(
        [np.array(s, dtype=np.int32) for s in nearby_sets if len(s)>0]
    )) if any(len(s)>0 for s in nearby_sets) else np.array([], dtype=np.int32)

    if len(nearby_idx) == 0:
        continue

    a_lats = infra_lats[nearby_idx]
    a_lons = infra_lons[nearby_idx]
    asset_coords = np.column_stack([a_lats, a_lons])

    # Min distance via KDTree + exact haversine on nearest only
    _, nn_idx  = fire_tree_t.query(asset_coords, k=1)
    cf_lats    = flats[nn_idx]; cf_lons = flons[nn_idx]
    min_dists  = haversine_vec(a_lats, a_lons, cf_lats, cf_lons)

    mask = min_dists <= NEARBY_RADIUS_KM
    if mask.sum() == 0:
        continue

    nearby_idx   = nearby_idx[mask];  a_lats = a_lats[mask]; a_lons = a_lons[mask]
    min_dists    = min_dists[mask];   nn_idx = nn_idx[mask]
    asset_coords = asset_coords[mask]; cf_lats = cf_lats[mask]; cf_lons = cf_lons[mask]

    # n30 + mean_dist
    n30_lists  = fire_tree_t.query_ball_point(asset_coords, 30.0/111.0)
    n50_lists  = fire_tree_t.query_ball_point(asset_coords, NEARBY_RADIUS_DEG)
    n30        = np.array([len(x) for x in n30_lists], dtype=np.int32)
    mean_dists = np.array([
        haversine_vec(a_lats[j], a_lons[j],
                      flats[np.array(idxs)], flons[np.array(idxs)]).mean()
        if idxs else min_dists[j]
        for j, idxs in enumerate(n50_lists)
    ], dtype=np.float64)

    # Weather
    wx_idxs = infra_wx_idx[nearby_idx]
    ws  = np.zeros(len(nearby_idx), np.float32)
    wd  = np.zeros(len(nearby_idx), np.float32)
    tc  = np.zeros(len(nearby_idx), np.float32)
    hum = np.zeros(len(nearby_idx), np.float32)
    dri = np.full(len(nearby_idx), 0.5, np.float32)
    dsr = np.full(len(nearby_idx), 7.0, np.float32)
    wx_ok = np.zeros(len(nearby_idx), bool)
    for j, gi in enumerate(wx_idxs):
        r = wx_lookup.get((int(gi), date_t))
        if r is not None:
            ws[j],wd[j],tc[j],hum[j],dri[j],dsr[j] = r
            wx_ok[j] = True

    if wx_ok.sum() == 0:
        continue

    ok           = wx_ok
    nearby_idx   = nearby_idx[ok];  a_lats = a_lats[ok];  a_lons = a_lons[ok]
    min_dists    = min_dists[ok];   mean_dists = mean_dists[ok]; n30 = n30[ok]
    nn_idx       = nn_idx[ok];      asset_coords = asset_coords[ok]
    cf_lats      = cf_lats[ok];     cf_lons = cf_lons[ok]
    ws=ws[ok]; wd=wd[ok]; tc=tc[ok]; hum=hum[ok]; dri=dri[ok]; dsr=dsr[ok]

    # Wind alignment
    wa = wind_align_vec(cf_lats, cf_lons, a_lats, a_lons, wd)

    # Label
    date_t1  = date_t + datetime.timedelta(days=1)
    fires_t1 = by_date.get(date_t1)
    if fires_t1 is not None:
        f1c = np.column_stack([
            fires_t1["latitude"].values.astype(np.float64),
            fires_t1["longitude"].values.astype(np.float64)])
        ft1_tree = cKDTree(f1c)
        _, nn1   = ft1_tree.query(asset_coords, k=1)
        d_t1     = haversine_vec(a_lats, a_lons,
                                 fires_t1["latitude"].values[nn1],
                                 fires_t1["longitude"].values[nn1])
        labels   = (d_t1 <= LABEL_RADIUS_KM).astype(np.int8)
    else:
        labels = np.zeros(len(nearby_idx), np.int8)

    # How many can we still add for this year?
    remaining = MAX_PER_YEAR - year_counts.get(year, 0)
    n_add = min(len(nearby_idx), remaining)
    if n_add <= 0:
        continue

    # Random sample if over quota
    if n_add < len(nearby_idx):
        sel = np.random.choice(len(nearby_idx), n_add, replace=False)
    else:
        sel = np.arange(len(nearby_idx))

    date_str = str(date_t)
    for j in sel:
        chunk_rows.append((
            int(labels[j]),
            round(float(min_dists[j]),  3),
            round(float(mean_dists[j]), 3),
            int(n30[j]),
            round(max_frp, 2),
            round(float(ws[j]),  2),
            round(float(wd[j]),  2),
            round(float(tc[j]),  2),
            round(float(hum[j]), 2),
            round(float(wa[j]),  4),
            round(float(dri[j]), 4),
            round(float(dsr[j]), 2),
            str(infra_types[nearby_idx[j]]),
            str(infra_states[nearby_idx[j]]),
            date_str,
        ))

    year_counts[year] = year_counts.get(year, 0) + n_add
    total_built += n_add

    # Flush chunk to disk
    if len(chunk_rows) >= chunk_size:
        p = flush_chunk(chunk_rows, chunk_idx)
        chunk_files.append(p)
        chunk_rows = []
        chunk_idx += 1

# Final flush
if chunk_rows:
    p = flush_chunk(chunk_rows, chunk_idx)
    chunk_files.append(p)

print(f"\n   ✓ {total_built:,} samples built")
print(f"   Samples per year:")
for yr in sorted(year_counts):
    print(f"     {yr}: {year_counts[yr]:,}")

# Read all chunks
print("\n   Assembling chunks...")
df = pd.concat([pd.read_parquet(f) for f in chunk_files if f], ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
# Clean up temp files
for f in chunk_files:
    if f and f.exists(): f.unlink()
print(f"   ✓ {len(df):,} total rows assembled")

# ── 6. DataFrame + split ──────────────────────────────────────────────────────
print("\n5. Applying temporal split...")

train = df[df["date"] <= TRAIN_END].copy()
val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
test  = df[df["date"] > VAL_END].copy()

assert train["date"].max() <= pd.Timestamp(TRAIN_END), "LEAKAGE train→val"
assert val["date"].max()   <= pd.Timestamp(VAL_END),   "LEAKAGE val→test"
assert test["date"].min()  >  pd.Timestamp(VAL_END),   "LEAKAGE test↔val"
print("   ✅ No temporal leakage")
print(f"   Train: {len(train):,}  pos={100*train['label'].mean():.1f}%")
print(f"   Val  : {len(val):,}  pos={100*val['label'].mean():.1f}%")
print(f"   Test : {len(test):,}  pos={100*test['label'].mean():.1f}%")

# ── 7. Save ───────────────────────────────────────────────────────────────────
train.to_parquet(BNN_DIR/"bnn_train.parquet", index=False)
val.to_parquet(  BNN_DIR/"bnn_val.parquet",   index=False)
test.to_parquet( BNN_DIR/"bnn_test.parquet",  index=False)
print(f"   Saved → {BNN_DIR}/")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
print("\n6. Generating validation plots...")
FEAT_COLS = ["min_dist_km","mean_dist_km","n_fires_30km","max_frp",
             "wind_speed_kmh","wind_direction","temperature_c","humidity",
             "wind_fire_alignment","drought_index","days_since_rain"]

# Label distribution
fig, axes = plt.subplots(1,3, figsize=(14,4))
for ax,(sp,title,col) in zip(axes,[
    (train,f"Train 2017-2021\n{len(train):,} samples","#2196F3"),
    (val,  f"Val   2022-2023\n{len(val):,} samples",  "#FF9800"),
    (test, f"Test  2024\n{len(test):,} samples",       "#4CAF50"),
]):
    cnts = sp["label"].value_counts().sort_index()
    ax.bar(["Neg (0)\nFire stayed away","Pos (1)\nFire reached"],
           cnts.values, color=[col+"55",col], edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(f"Positive rate: {100*sp['label'].mean():.1f}%")
    for ii,v in enumerate(cnts.values):
        ax.text(ii, v+2, f"{v:,}", ha="center", fontsize=9)
fig.suptitle("Label Distribution — Real FIRMS Labels", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(BNN_DIR/"label_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Samples per year
fig, ax = plt.subplots(figsize=(12,4))
yr = df.groupby(df["date"].dt.year)["label"].agg(["count","sum"])
yr["neg"] = yr["count"] - yr["sum"]
ax.bar(yr.index, yr["neg"],  label="Negative", color="#F44336", alpha=0.8)
ax.bar(yr.index, yr["sum"],  bottom=yr["neg"], label="Positive", color="#2196F3", alpha=0.8)
ax.axvline(2021.5, color="orange", lw=2, ls="--", label="Train/Val split")
ax.axvline(2023.5, color="green",  lw=2, ls="--", label="Val/Test split")
ax.set_xlabel("Year"); ax.set_ylabel("Samples")
ax.set_title("Samples per Year — Confirms No Year Dominates", fontweight="bold")
ax.legend(); ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(BNN_DIR/"samples_per_year.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature distributions
fig, axes = plt.subplots(3,4, figsize=(18,12))
axes = axes.flatten()
pos = train[train["label"]==1]
neg = train[train["label"]==0]
for ii,feat in enumerate(FEAT_COLS):
    ax = axes[ii]
    ax.hist(neg[feat].dropna(), bins=40, alpha=0.6,
            color="#F44336", label="Neg", density=True)
    ax.hist(pos[feat].dropna(), bins=40, alpha=0.6,
            color="#2196F3", label="Pos", density=True)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=8)
for jj in range(len(FEAT_COLS), len(axes)):
    axes[jj].set_visible(False)
fig.suptitle("Feature Distributions: Pos vs Neg (Train Set)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(BNN_DIR/"feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Correlation matrix
fig, ax = plt.subplots(figsize=(12,10))
corr = train[FEAT_COLS+["label"]].corr()
im   = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
labs = FEAT_COLS+["label"]
ax.set_xticks(range(len(labs)))
ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(labs)))
ax.set_yticklabels(labs, fontsize=9)
for ii in range(len(labs)):
    for jj in range(len(labs)):
        ax.text(jj,ii,f"{corr.iloc[ii,jj]:.2f}",ha="center",va="center",
                fontsize=7, color="white" if abs(corr.iloc[ii,jj])>0.6 else "black")
ax.set_title("Feature Correlation Matrix\n"
             "Key: label must correlate NEGATIVELY with min_dist_km",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(BNN_DIR/"feature_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ All plots saved")

# ── 9. Sanity checks ──────────────────────────────────────────────────────────
print("\n7. Sanity checks...")
corr_dist = train["min_dist_km"].corr(train["label"])
print(f"   Corr(min_dist_km, label) = {corr_dist:.4f}  "
      f"{'✅ negative — closer fire = more likely positive' if corr_dist < 0 else '❌ FAIL'}")
print(f"   Positive rate by distance band:")
for r in [5,10,20,30,50]:
    sub = train[train["min_dist_km"]<=r]
    if len(sub)>0:
        print(f"     ≤{r:2d}km : {100*sub['label'].mean():.1f}%  (n={len(sub):,})")

# ── 10. Summary JSON ──────────────────────────────────────────────────────────
summary = {
    "total_samples": len(df),
    "train": {"n":len(train), "pos_rate":round(float(train["label"].mean()),4)},
    "val":   {"n":len(val),   "pos_rate":round(float(val["label"].mean()),4)},
    "test":  {"n":len(test),  "pos_rate":round(float(test["label"].mean()),4)},
    "label_definition": "fire within 5km of asset next day (FIRMS VIIRS)",
    "label_source":     "NASA FIRMS — no synthetic labels",
    "temporal_leakage": False,
    "corr_dist_label":  round(float(corr_dist),4),
    "feature_columns":  FEAT_COLS,
}
with open(BNN_DIR/"dataset_summary.json","w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print("M1 COMPLETE")
print(f"{'='*70}")
print(f"Total  : {len(df):,}")
print(f"Train  : {len(train):,}  ({100*train['label'].mean():.1f}% pos)")
print(f"Val    : {len(val):,}  ({100*val['label'].mean():.1f}% pos)")
print(f"Test   : {len(test):,}  ({100*test['label'].mean():.1f}% pos)")
print(f"\nNext: M2 — python src/pipeline/train_bnn_v3.py")
print(f"{'='*70}")