"""
build_xgboost_labels.py  (v4 — robust, balanced, full date range)
==================================================================
Key fixes over v3:
  1. Per-year quota for both positives and negatives (balanced 2017-2025)
  2. Extended radius for hard negatives (50-150km) — teaches distance signal
  3. Physical geo features instead of state encoder (lat/lon/elevation proxy)
  4. Ensures 2024-2025 data exists for temporal validation in training script
  5. n_fires_50km computed via bulk query_ball_tree (no Python loop)

Label definitions:
  Positive (1): fire within 50km + county outage > 2% customers
  Negative (0): fire within 150km but NO significant outage
    - Type A: fire 0-50km, no outage (survived proximity)
    - Type B: fire 50-150km, no outage (far but active fire environment)

Output: data/xgboost_training_labels.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

FIRMS_PATH  = DATA_DIR / "fires"          / "national_fires_2017_2025.csv"
INFRA_PATH  = DATA_DIR / "infrastructure" / "national_infrastructure.csv"
MCC_PATH    = DATA_DIR / "outages"        / "MCC.csv"
OUTAGE_DIR  = DATA_DIR / "outages"
WEATHER_DIR = DATA_DIR / "weather"
OUT_PATH    = DATA_DIR / "xgboost_training_labels.csv"

TARGET_STATES = {
    "California":"CA","Oregon":"OR","Washington":"WA","Montana":"MT",
    "Colorado":"CO","Idaho":"ID","Wyoming":"WY","Utah":"UT",
    "Nevada":"NV","Arizona":"AZ"
}
STATE_ABBREV   = set(TARGET_STATES.values())
OUTAGE_THRESH  = 0.02
FIRE_RADIUS_KM = 50     # positive label radius
HARD_NEG_KM    = 150    # hard negative radius (fire present but survived)
RADIUS_DEG     = FIRE_RADIUS_KM / 111.0
HARD_NEG_DEG   = HARD_NEG_KM   / 111.0

# Per-year quota — ensures balanced temporal coverage
YEARS          = list(range(2017, 2026))
POS_PER_YEAR   = 17_000   # ~153K total positives
NEG_PER_YEAR   = 17_000   # ~153K total negatives (50% type A, 50% type B)

VULN = {
    "Power Substation":65,"Wind Farm":70,"Solar Farm":60,
    "Gas Plant":55,"Coal Plant":55,"Hydro Plant":50,
    "Hospital":45,"Fire Station":40,"School":35,
    "Airport":45,"Water Treatment":40,"Cell Tower":30,
}
VULN_DEFAULT = 40

print("=" * 70)
print("BUILDING XGBOOST LABELS  (v4 — robust, balanced, full date range)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. MCC
# ---------------------------------------------------------------------------
print("\n1. MCC...")
mcc = pd.read_csv(MCC_PATH)
mcc["fips"] = mcc["County_FIPS"].astype(str).str.zfill(5)
mcc_dict = mcc.set_index("fips")["Customers"].to_dict()

# ---------------------------------------------------------------------------
# 2. EAGLE-I (fire season only)
# ---------------------------------------------------------------------------
print("\n2. EAGLE-I outages...")
parts = []
for year in YEARS:
    fpath = OUTAGE_DIR / f"eaglei_outages_{year}.csv"
    if not fpath.exists(): continue
    df = pd.read_csv(fpath, low_memory=False)
    if "sum" in df.columns and "customers_out" not in df.columns:
        df = df.rename(columns={"sum":"customers_out"})
    keep = ["fips_code","county","state","customers_out","run_start_time"]
    if "total_customers" in df.columns: keep.append("total_customers")
    df = df[keep].copy()
    df = df[df["state"].isin(TARGET_STATES.keys())]
    df["run_start_time"] = pd.to_datetime(df["run_start_time"], errors="coerce")
    df = df.dropna(subset=["run_start_time"])
    df = df[df["run_start_time"].dt.month.isin([5,6,7,8,9,10,11])]
    parts.append(df)
    print(f"   {year}: {len(df):,}")

outages = pd.concat(parts, ignore_index=True)
outages["fips"] = outages["fips_code"].astype(str).str.zfill(5)
if "total_customers" not in outages.columns:
    outages["total_customers"] = outages["fips"].map(mcc_dict).fillna(1000)
else:
    outages["total_customers"] = (outages["total_customers"]
                                  .fillna(outages["fips"].map(mcc_dict))
                                  .fillna(1000))
outages["outage_pct"] = (outages["customers_out"] /
                          outages["total_customers"]).clip(0, 1)
outages["date"]       = outages["run_start_time"].dt.date
outages["year"]       = outages["run_start_time"].dt.year

daily = (outages[outages["outage_pct"] > OUTAGE_THRESH]
         .groupby(["fips","state","county","date","year"])
         .agg(max_outage_pct=("outage_pct","max"),
              customers_out=("customers_out","max"))
         .reset_index())
daily["state_abbrev"] = daily["state"].map(TARGET_STATES)
daily = daily.dropna(subset=["state_abbrev"]).reset_index(drop=True)
print(f"\n   Daily outage events: {len(daily):,}")
print(f"   By year:")
print(daily.groupby("year").size().to_string())

# ---------------------------------------------------------------------------
# 3. FIRMS
# ---------------------------------------------------------------------------
print("\n3. FIRMS...")
fires = pd.read_csv(FIRMS_PATH, low_memory=False)
fires["date"] = pd.to_datetime(fires["acq_date"], errors="coerce").dt.date
fires["year"] = pd.to_datetime(fires["acq_date"], errors="coerce").dt.year
fires = fires.dropna(subset=["date","latitude","longitude"])
fires = fires[fires["confidence"].astype(str).str.lower()
              .isin(["h","n","high","nominal"])].copy()
fires["glat"] = (fires["latitude"]  / 0.5).round() * 0.5
fires["glon"] = (fires["longitude"] / 0.5).round() * 0.5
daily_fires = (fires.groupby(["date","year","glat","glon"])
               .agg(n_detections=("frp","count"),
                    max_frp=("frp","max"))
               .reset_index())
print(f"   Daily fire grid cells: {len(daily_fires):,}")

# ---------------------------------------------------------------------------
# 4. Infrastructure
# ---------------------------------------------------------------------------
print("\n4. Infrastructure...")
infra = pd.read_csv(INFRA_PATH, low_memory=False)
infra.columns = [c.strip().lower() for c in infra.columns]
infra["lat"] = pd.to_numeric(infra["lat"], errors="coerce")
infra["lon"] = pd.to_numeric(infra["lon"], errors="coerce")
infra = infra.dropna(subset=["lat","lon"])
infra = infra[infra["state"].isin(STATE_ABBREV)].copy()
infra = infra[~infra["type"].isin(["Residential Area","Transmission Line"])].copy()
infra = infra.reset_index(drop=True)
infra_coords = infra[["lat","lon"]].values
infra_tree   = cKDTree(infra_coords)
infra_states = infra["state"].values
infra_types  = infra["type"].fillna("Unknown").values
infra_vuln   = np.array([VULN.get(t, VULN_DEFAULT) for t in infra_types])
print(f"   {len(infra):,} assets")

# ---------------------------------------------------------------------------
# 5. Weather
# ---------------------------------------------------------------------------
print("\n5. Weather...")
wx_parts = []
for f in sorted(WEATHER_DIR.glob("*_weather_grid.csv")):
    wx_parts.append(pd.read_csv(f, low_memory=False,
                    usecols=["datetime","temp_c","humidity",
                             "wind_speed_kmh","wind_direction",
                             "grid_lat","grid_lon"]))
wx = pd.concat(wx_parts, ignore_index=True)
wx["datetime"] = pd.to_datetime(wx["datetime"], errors="coerce")
wx = wx.dropna(subset=["datetime"])
wx["date"] = wx["datetime"].dt.date
wx_daily = (wx.groupby(["date","grid_lat","grid_lon"])
            .agg(temp_c=("temp_c","mean"),
                 humidity=("humidity","mean"),
                 wind_speed_kmh=("wind_speed_kmh","mean"),
                 wind_direction=("wind_direction","mean"))
            .reset_index())
print(f"   {len(wx_daily):,} daily wx records")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wx_for_assets(day_wx_df, asset_lats, asset_lons):
    if day_wx_df.empty or len(asset_lats) == 0:
        n = len(asset_lats)
        return (np.full(n, 20.0), np.full(n, 50.0),
                np.full(n, 10.0), np.full(n, 270.0))
    pts  = np.column_stack([asset_lats, asset_lons])
    gpts = day_wx_df[["grid_lat","grid_lon"]].values
    gt   = cKDTree(gpts)
    _, idx = gt.query(pts)
    return (day_wx_df["temp_c"].values[idx],
            day_wx_df["humidity"].values[idx],
            day_wx_df["wind_speed_kmh"].values[idx],
            day_wx_df["wind_direction"].values[idx])

def compute_wind_alignment(al, aln, nf_la, nf_lo, wdir):
    brd  = np.arctan2(
        np.sin(np.radians(aln - nf_lo)) * np.cos(np.radians(al)),
        np.cos(np.radians(nf_la)) * np.sin(np.radians(al)) -
        np.sin(np.radians(nf_la)) * np.cos(np.radians(al)) *
        np.cos(np.radians(aln - nf_lo)))
    brgs = (np.degrees(brd) + 360) % 360
    wt   = (wdir + 180) % 360
    dfs  = np.abs(wt - brgs)
    dfs  = np.where(dfs > 180, 360 - dfs, dfs)
    return np.cos(np.radians(dfs))

def elevation_proxy(lats):
    """Rough elevation proxy from latitude — higher lat = more mountainous in western US."""
    return np.clip((lats - 32.0) / 12.0, 0, 1)

def build_rows(cand, dist_k, nfi_, al, aln, states,
               day_fires, day_wx, label, outage_pct_arr, dt):
    if len(cand) == 0:
        return []
    tc, hum, wspd, wdir = get_wx_for_assets(day_wx, al, aln)
    nf_la = day_fires["glat"].values[nfi_]
    nf_lo = day_fires["glon"].values[nfi_]
    wa    = compute_wind_alignment(al, aln, nf_la, nf_lo, wdir)
    max_frp = float(day_fires["max_frp"].max())

    # ── n_fires_50km: bulk query_ball_tree — no Python loop over assets ──
    fire_tree_local = cKDTree(day_fires[["glat","glon"]].values)
    asset_tree_local = cKDTree(np.column_stack([al, aln]))
    counts = fire_tree_local.query_ball_tree(asset_tree_local, r=50/111.0)
    n50 = np.array([len(c) for c in counts], dtype=int)
    # counts[i] = assets within 50km of fire i; we want fires within 50km of
    # each asset, so transpose: use asset_tree querying fire_tree instead
    counts2 = asset_tree_local.query_ball_tree(fire_tree_local, r=50/111.0)
    n50 = np.array([len(c) for c in counts2], dtype=int)

    elev  = elevation_proxy(al)
    coast = np.minimum(np.abs(aln - (-124.0)), np.abs(aln - (-117.0)))
    rows  = []
    for j in range(len(cand)):
        ai = cand[j]
        rows.append({
            "outage_label":   label,
            "min_dist_km":    round(float(dist_k[j]), 2),
            "n_fires_50km":   int(n50[j]),
            "max_frp":        max_frp,
            "wind_alignment": round(float(wa[j]), 4),
            "wind_speed_kmh": float(wspd[j]),
            "wind_direction": float(wdir[j]),
            "temperature_c":  float(tc[j]),
            "humidity":       float(hum[j]),
            "latitude":       round(float(al[j]), 3),
            "longitude":      round(float(aln[j]), 3),
            "elevation_proxy":round(float(elev[j]), 3),
            "dist_from_coast":round(float(coast[j]), 2),
            "asset_type":     infra_types[ai],
            "vulnerability":  int(infra_vuln[ai]),
            "outage_pct":     float(outage_pct_arr[j]) if label == 1 else 0.0,
            "event_date":     str(dt),
            "year":           pd.Timestamp(str(dt)).year,
        })
    return rows

# ---------------------------------------------------------------------------
# 6. Build labels — per-year quota
# ---------------------------------------------------------------------------
print("\n6. Building labels (per-year quota)...")

outage_date_state = set(zip(daily["date"], daily["state_abbrev"]))
pos_rows = []
neg_rows = []

year_pos = {y: 0 for y in YEARS}
year_neg = {y: 0 for y in YEARS}

all_dates = sorted(set(daily["date"].tolist() +
                       daily_fires["date"].tolist()))

for i, dt in enumerate(all_dates):
    yr = pd.Timestamp(str(dt)).year
    if yr not in YEARS: continue

    pos_done = year_pos[yr] >= POS_PER_YEAR
    neg_done = year_neg[yr] >= NEG_PER_YEAR
    if pos_done and neg_done: continue

    if i % 1000 == 0:
        print(f"   [{i:>5}/{len(all_dates)}]  "
              f"pos={sum(year_pos.values()):,}  "
              f"neg={sum(year_neg.values()):,}")

    day_fires_df = daily_fires[daily_fires["date"] == dt]
    if day_fires_df.empty: continue
    day_wx = wx_daily[wx_daily["date"] == dt]
    if day_wx.empty: continue

    day_fire_coords = day_fires_df[["glat","glon"]].values
    day_fire_tree   = cKDTree(day_fire_coords)

    # ── Positive labels ────────────────────────────────────────────────
    if not pos_done:
        day_out = daily[daily["date"] == dt]
        if not day_out.empty:
            outage_states = set(day_out["state_abbrev"].values)
            s_idx = np.where(np.isin(infra_states, list(outage_states)))[0]
            if len(s_idx) > 0:
                dists_deg, nfi = day_fire_tree.query(infra_coords[s_idx])
                dists_km = dists_deg * 111.0
                in_rad   = dists_km <= FIRE_RADIUS_KM
                if in_rad.any():
                    cand   = s_idx[in_rad];    dist_k = dists_km[in_rad]
                    nfi_   = nfi[in_rad]
                    al     = infra_coords[cand, 0]; aln = infra_coords[cand, 1]
                    cap = min(len(cand), POS_PER_YEAR - year_pos[yr], 200)
                    if len(cand) > cap:
                        sel = np.random.choice(len(cand), cap, replace=False)
                        cand=cand[sel]; dist_k=dist_k[sel]; nfi_=nfi_[sel]
                        al=al[sel]; aln=aln[sel]
                    state_to_opc = day_out.groupby("state_abbrev")["max_outage_pct"].max().to_dict()
                    opc = np.array([state_to_opc.get(infra_states[ai], 0.0)
                                    for ai in cand])
                    new = build_rows(cand, dist_k, nfi_, al, aln,
                                     infra_states[cand], day_fires_df, day_wx,
                                     1, opc, dt)
                    pos_rows.extend(new)
                    year_pos[yr] += len(new)

    # ── Negative labels ────────────────────────────────────────────────
    if not neg_done:
        outage_states_today = {s for (d,s) in outage_date_state if d == dt}
        no_out_mask = ~np.isin(infra_states, list(outage_states_today))
        s_idx_neg   = np.where(no_out_mask)[0]

        if len(s_idx_neg) > 0:
            dists_deg, nfi = day_fire_tree.query(infra_coords[s_idx_neg])
            dists_km = dists_deg * 111.0

            # Type A: 2-50km
            in_rad_a = (dists_km >= 2) & (dists_km <= FIRE_RADIUS_KM)
            if in_rad_a.any():
                cand   = s_idx_neg[in_rad_a]; dist_k = dists_km[in_rad_a]
                nfi_   = nfi[in_rad_a]
                al     = infra_coords[cand, 0]; aln = infra_coords[cand, 1]
                cap = min(len(cand), (NEG_PER_YEAR - year_neg[yr]) // 2 + 1, 75)
                if len(cand) > cap:
                    sel = np.random.choice(len(cand), cap, replace=False)
                    cand=cand[sel]; dist_k=dist_k[sel]; nfi_=nfi_[sel]
                    al=al[sel]; aln=aln[sel]
                new = build_rows(cand, dist_k, nfi_, al, aln,
                                 infra_states[cand], day_fires_df, day_wx,
                                 0, np.zeros(len(cand)), dt)
                neg_rows.extend(new)
                year_neg[yr] += len(new)

            # Type B: 50-150km
            if year_neg[yr] < NEG_PER_YEAR:
                in_rad_b = ((dists_km > FIRE_RADIUS_KM) &
                            (dists_km <= HARD_NEG_KM))
                if in_rad_b.any():
                    cand   = s_idx_neg[in_rad_b]; dist_k = dists_km[in_rad_b]
                    nfi_   = nfi[in_rad_b]
                    al     = infra_coords[cand, 0]; aln = infra_coords[cand, 1]
                    cap = min(len(cand), (NEG_PER_YEAR - year_neg[yr]) // 2 + 1, 50)
                    if len(cand) > cap:
                        sel = np.random.choice(len(cand), cap, replace=False)
                        cand=cand[sel]; dist_k=dist_k[sel]; nfi_=nfi_[sel]
                        al=al[sel]; aln=aln[sel]
                    new = build_rows(cand, dist_k, nfi_, al, aln,
                                     infra_states[cand], day_fires_df, day_wx,
                                     0, np.zeros(len(cand)), dt)
                    neg_rows.extend(new)
                    year_neg[yr] += len(new)

print(f"\n   Positive: {len(pos_rows):,}")
print(f"   Negative: {len(neg_rows):,}")
print(f"\n   Per year:")
for yr in YEARS:
    print(f"   {yr}  pos={year_pos[yr]:>6,}  neg={year_neg[yr]:>6,}")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
all_rows = pos_rows + neg_rows
df_out   = pd.DataFrame(all_rows)
df_out["event_date"] = pd.to_datetime(df_out["event_date"])
df_out   = df_out.sample(frac=1, random_state=42).reset_index(drop=True)
df_out.to_csv(OUT_PATH, index=False)

n     = len(df_out)
n_pos = (df_out["outage_label"]==1).sum()
n_neg = (df_out["outage_label"]==0).sum()

print(f"\n{'='*70}")
print("LABELS COMPLETE  (v4)")
print(f"{'='*70}")
print(f"Total    : {n:,}")
print(f"Positive : {n_pos:,}  ({100*n_pos/n:.1f}%)")
print(f"Negative : {n_neg:,}  ({100*n_neg/n:.1f}%)")
print(f"Date range: {df_out['event_date'].min().date()} → "
      f"{df_out['event_date'].max().date()}")
print(f"\nBy year:")
print(df_out.groupby("year")["outage_label"]
      .value_counts().unstack(fill_value=0).to_string())
print(f"\nBy asset type (top 10):")
print(df_out["asset_type"].value_counts().head(10).to_string())
print(f"\nSaved → {OUT_PATH}")
print(f"{'='*70}")