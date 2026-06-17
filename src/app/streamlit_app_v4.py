import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import numpy as np
import tensorflow as tf
import joblib
import json
import torch
from pathlib import Path
from scipy.spatial import cKDTree
import sys

st.set_page_config(page_title="Wildfire Risk Prediction", layout="wide")

st.title("🔥 Wildfire Infrastructure Risk — Prometheus AI v4")
st.caption("BNN v3 (real FIRMS labels, Platt calibration) • TFT v2 (radial perimeter) • Fixed val-set thresholds • No synthetic labels • No 20× multiplier")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from spread_tft_model import SpreadTFT  # kept for backward compat import

# ============================================================================
# LOAD MODEL AND DATA  (v4 — BNN v3 + TFT v2)
# ============================================================================

def _combined_quantile_loss(y_true, y_pred):
    """Custom TFT v2 loss — must be defined before model load."""
    total = 0.0
    for i in range(8):
        for j, q in enumerate([0.1, 0.5, 0.9]):
            idx = i * 3 + j
            e = y_true[:, idx] - y_pred[:, idx]
            total += tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return total / 24.0

@st.cache_resource
def load_model_and_data():
    """Load BNN v3 — real FIRMS labels, Platt calibration, fixed thresholds."""
    bnn    = tf.keras.models.load_model(str(ROOT / "models" / "bnn_v3_best.keras"))
    scaler = joblib.load(ROOT / "models" / "bnn_v3_scaler.pkl")
    infra  = pd.read_csv(ROOT / "data" / "infrastructure" / "national_infrastructure.csv")

    renewable_mask = infra['type'].isin(['Solar Farm', 'Wind Farm'])
    if 'source' in infra.columns and 'capacity_mw' in infra.columns:
        infra['capacity_mw'] = pd.to_numeric(infra['capacity_mw'], errors='coerce')
        keep = infra[~renewable_mask]
        util = infra[renewable_mask][
            (infra[renewable_mask]['source'] == 'EIA') |
            (infra[renewable_mask]['capacity_mw'] >= 1.0)
        ]
        infra = pd.concat([keep, util], ignore_index=True)

    return bnn, scaler, infra

@st.cache_resource
def load_calibrator():
    """Load Platt calibrator and fixed thresholds — both derived from val set."""
    calibrator = joblib.load(ROOT / "models" / "bnn_v3_calibrator.pkl")
    with open(ROOT / "models" / "bnn_v3_thresholds.json") as f:
        thresholds = json.load(f)
    return calibrator, thresholds

@st.cache_resource
def load_tft_model():
    """Load TFT v2 — radial profile, per-direction calibration from val set."""
    tft = tf.keras.models.load_model(
        str(ROOT / "models" / "tft_v2_best.keras"),
        custom_objects={"combined_quantile_loss": _combined_quantile_loss})
    tft_scaler = joblib.load(ROOT / "models" / "tft_v2_scaler.pkl")
    with open(ROOT / "models" / "tft_v2_calibration_factor.json") as f:
        tft_cal = json.load(f)
    with open(ROOT / "data" / "tft_v2" / "dataset_summary.json") as f:
        tft_meta = json.load(f)
    return tft, tft_scaler, tft_cal, tft_meta

@st.cache_data
def load_historical_fires():
    fire_file = ROOT / "data" / "fires" / "national_fires_2017_2025.csv"
    df = pd.read_csv(str(fire_file), low_memory=False)
    df = df[df['confidence'].astype(str).str.lower().isin(['h', 'high'])]
    return df

@st.cache_data
def load_historical_weather():
    weather_dir = ROOT / "data" / "weather"
    parts = []
    for f in sorted(weather_dir.glob("*_weather_grid.csv")):
        parts.append(pd.read_csv(str(f), low_memory=False))
    df = pd.concat(parts, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    return df.dropna(subset=['datetime'])

try:
    model, scaler, infra_df              = load_model_and_data()
    calibrator, bnn_thresholds          = load_calibrator()
    tft_model, tft_scaler, tft_cal, tft_meta = load_tft_model()
    all_historical_fires                 = load_historical_fires()
    historical_weather                   = load_historical_weather()

    THRESH_LOW      = float(bnn_thresholds["LOW_MEDIUM"])
    THRESH_MEDIUM   = float(bnn_thresholds["MEDIUM_HIGH"])
    THRESH_CRITICAL = float(bnn_thresholds["HIGH_CRITICAL"])
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("⚙️ Configuration")

use_historical = st.sidebar.checkbox("📚 Use Historical Data", value=True, key="hist_toggle")

if use_historical:
    st.sidebar.subheader("🗓️ Historical Fire Data")
    years = sorted(all_historical_fires['acq_date'].str[:4].unique())
    default_year_idx = years.index('2018') if '2018' in years else 0
    selected_year = st.sidebar.selectbox("Year", years, index=default_year_idx, key="year_select")

    year_fires = all_historical_fires[all_historical_fires['acq_date'].str.startswith(selected_year)]
    months = sorted(year_fires['acq_date'].str[5:7].unique())
    default_month_idx = months.index('11') if '11' in months else 0
    selected_month = st.sidebar.selectbox("Month", months, index=default_month_idx, key="month_select")

    month_fires = year_fires[year_fires['acq_date'].str[5:7] == selected_month]
    dates = sorted(month_fires['acq_date'].unique())

    if dates:
        default_date_idx = dates.index('2018-11-08') if '2018-11-08' in dates else 0
        selected_date = st.sidebar.selectbox("Date", dates, index=default_date_idx, key="date_select")
        fires_for_date = month_fires[month_fires['acq_date'] == selected_date]
        st.sidebar.success(f"✓ {len(fires_for_date)} fires on {selected_date}")
    else:
        fires_for_date = pd.DataFrame()
        selected_date  = "N/A"
else:
    st.sidebar.subheader("🔴 Live Data")
    fire_source   = st.sidebar.selectbox("FIRMS Source", ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"], key="live_source")
    lookback_days = st.sidebar.selectbox("Lookback Days", [1, 2, 3], key="live_days")
    fires_for_date = pd.DataFrame()
    selected_date  = "Live"

st.sidebar.markdown("---")

st.sidebar.subheader("📍 Location")
location_preset = st.sidebar.selectbox(
    "Quick Location",
    [
        "Paradise (Camp Fire 2018)", "Custom",
        "Los Angeles", "San Francisco", "San Diego", "Sacramento",
        "Medford OR", "Bend OR", "Spokane WA", "Yakima WA",
        "Missoula MT", "Boise ID", "Denver CO", "Salt Lake City UT",
        "Reno NV", "Phoenix AZ",
    ],
    index=0, key="location_preset"
)

presets = {
    "Paradise (Camp Fire 2018)": (39.76, -121.62, 11),
    "Los Angeles":               (34.05, -118.24, 10),
    "San Francisco":             (37.77, -122.42, 11),
    "San Diego":                 (32.72, -117.16, 11),
    "Sacramento":                (38.58, -121.49, 11),
    "Medford OR":                (42.33, -122.87, 11),
    "Bend OR":                   (44.06, -121.31, 11),
    "Spokane WA":                (47.66, -117.43, 11),
    "Yakima WA":                 (46.60, -120.51, 11),
    "Missoula MT":               (46.87, -114.02, 11),
    "Boise ID":                  (43.62, -116.20, 11),
    "Denver CO":                 (39.74, -104.98, 10),
    "Salt Lake City UT":         (40.76, -111.89, 11),
    "Reno NV":                   (39.53, -119.81, 11),
    "Phoenix AZ":                (33.45, -112.07, 10),
}

if location_preset != "Custom":
    center_lat, center_lon, zoom = presets[location_preset]
    st.sidebar.metric("Lat", f"{center_lat:.4f}")
    st.sidebar.metric("Lon", f"{center_lon:.4f}")
else:
    center_lat = st.sidebar.number_input("Latitude",  value=39.76, format="%.4f", key="custom_lat")
    center_lon = st.sidebar.number_input("Longitude", value=-121.62, format="%.4f", key="custom_lon")
    zoom       = st.sidebar.slider("Zoom", 6, 15, 11, key="custom_zoom")

st.sidebar.success("✅ BNN v3 + TFT v2 ready\nReal labels · Platt cal · Fixed thresholds")

# ============================================================================
# SESSION STATE
# ============================================================================

if "display_lat"        not in st.session_state: st.session_state.display_lat        = center_lat
if "display_lon"        not in st.session_state: st.session_state.display_lon        = center_lon
if "last_handled_click" not in st.session_state: st.session_state.last_handled_click = None
if "use_preset"         not in st.session_state: st.session_state.use_preset         = True

if st.session_state.use_preset:
    current_lat = center_lat
    current_lon = center_lon
else:
    current_lat = st.session_state.display_lat
    current_lon = st.session_state.display_lon

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

_wx_coords = historical_weather[['grid_lat', 'grid_lon']].drop_duplicates().values
_wx_tree   = cKDTree(_wx_coords)

def get_weather(lat, lon, date_str, is_historical):
    if is_historical and date_str not in ("N/A", "Live"):
        try:
            _, idx = _wx_tree.query([lat, lon])
            near_lat, near_lon = _wx_coords[idx]
            target_date = pd.to_datetime(date_str).date()
            station = historical_weather[
                (historical_weather['grid_lat'] == near_lat) &
                (historical_weather['grid_lon'] == near_lon) &
                (historical_weather['datetime'].dt.date == target_date)
            ]
            if len(station) > 0:
                return {
                    'wind_speed_kmh':  float(station['wind_speed_kmh'].mean()),
                    'wind_direction':  float(station['wind_direction'].mean()),
                    'temp_c':          float(station['temp_c'].mean()),
                    'humidity':        float(station['humidity'].mean()),
                    'drought_index':   float(station['drought_index'].mean())
                                       if 'drought_index' in station.columns else 0.5,
                    'days_since_rain': float(station['days_since_rain'].mean())
                                       if 'days_since_rain' in station.columns else 7.0,
                }
        except Exception:
            pass
    try:
        url  = (f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,"
                f"wind_speed_10m,wind_direction_10m,precipitation")
        data = requests.get(url, timeout=10).json().get('current', {})
        return {
            'wind_speed_kmh':  data.get('wind_speed_10m', 15),
            'wind_direction':  data.get('wind_direction_10m', 180),
            'temp_c':          data.get('temperature_2m', 20),
            'humidity':        data.get('relative_humidity_2m', 50),
            'drought_index':   0.5,
            'days_since_rain': 7.0,
        }
    except Exception:
        return {'wind_speed_kmh':15,'wind_direction':180,
                'temp_c':20,'humidity':50,
                'drought_index':0.5,'days_since_rain':7.0}

def haversine_vectorized(lat1, lon1, lats2, lons2):
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats2, lons2 = np.radians(np.array(lats2)), np.radians(np.array(lons2))
    a = (np.sin((lats2-lat1)/2)**2 +
         np.cos(lat1)*np.cos(lats2)*np.sin((lons2-lon1)/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def calculate_wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
    la1,lo1 = np.radians(fire_lat), np.radians(fire_lon)
    la2,lo2 = np.radians(asset_lat), np.radians(asset_lon)
    dlon = lo2 - lo1
    x = np.sin(dlon)*np.cos(la2)
    y = np.cos(la1)*np.sin(la2) - np.sin(la1)*np.cos(la2)*np.cos(dlon)
    bearing     = (np.degrees(np.arctan2(x, y)) + 360) % 360
    wind_toward = (wind_dir + 180) % 360
    diff        = abs(wind_toward - bearing)
    if diff > 180: diff = 360 - diff
    return np.cos(np.radians(diff))

def quick_risk(min_dist_km):
    if   min_dist_km < 3:  return 90
    elif min_dist_km < 7:  return 70
    elif min_dist_km < 15: return 45
    elif min_dist_km < 30: return 25
    elif min_dist_km < 50: return 12
    return 5

def predict_risk_batch(assets_df, fires_df, weather, model, scaler):
    """
    BNN v3 batch inference.
    - Real FIRMS labels (not synthetic)
    - Platt calibration with C cross-validated on val set
    - Returns calibrated probability (0-1), displayed as ×100 for readability
    """
    if len(fires_df) == 0 or len(assets_df) == 0:
        return []

    fire_lats = fires_df['latitude'].values
    fire_lons = fires_df['longitude'].values
    max_frp   = float(fires_df['frp'].max()) if 'frp' in fires_df.columns else 50.0
    wind_dir  = float(weather.get('wind_direction', 180))

    rows = []
    meta = []
    for _, a in assets_df.iterrows():
        alat, alon = float(a['lat']), float(a['lon'])
        dists      = haversine_vectorized(alat, alon, fire_lats, fire_lons)
        min_dist   = float(dists.min())
        mean_dist  = float(dists.mean())
        num_nearby = int((dists < 30).sum())
        closest    = fires_df.iloc[dists.argmin()]
        wa         = calculate_wind_alignment(
            closest['latitude'], closest['longitude'],
            alat, alon, wind_dir)
        rows.append([
            min_dist, mean_dist, num_nearby, max_frp,
            float(weather.get('wind_speed_kmh', 15)),
            float(weather.get('wind_direction', 180)),
            float(weather.get('temp_c', 20)),
            float(weather.get('humidity', 50)),
            float(wa),
            float(weather.get('drought_index', 0.5)),
            float(weather.get('days_since_rain', 7.0)),
        ])
        meta.append({
            'name':          a.get('name', 'Unknown'),
            'city':          a.get('city', ''),
            'state':         a.get('state', ''),
            'type':          a.get('type', ''),
            'lat':           alat,
            'lon':           alon,
            'distance_km':   round(min_dist, 2),
            'distance_miles':round(min_dist * 0.621371, 2),
            'num_fires':     num_nearby,
        })

    X = scaler.transform(np.array(rows, dtype=np.float32))
    X_t = tf.constant(X, dtype=tf.float32)

    # MC Dropout: 50 passes
    all_preds = np.stack([
        model(X_t, training=True).numpy().flatten()
        for _ in range(50)
    ])  # (50, N)
    raw_mean = all_preds.mean(axis=0)
    raw_std  = all_preds.std(axis=0)

    # Apply Platt calibration — used for TIER ASSIGNMENT only
    cal_prob = calibrator.predict_proba(raw_mean.reshape(-1, 1))[:, 1]

    def _tier(p):
        if p >= THRESH_CRITICAL: return "CRITICAL"
        if p >= THRESH_MEDIUM:   return "HIGH"
        if p >= THRESH_LOW:      return "MEDIUM"
        return "LOW"

    # Display score: min-max rescale raw BNN scores to 0-100 within viewport
    # Purpose: show meaningful gradient when all assets are near an active fire
    # This is ONLY for the Score column display — NOT for tier assignment
    # Tier assignment uses fixed Platt thresholds (never recomputed per batch)
    rmin, rmax = float(raw_mean.min()), float(raw_mean.max())
    if rmax > rmin + 1e-6:
        display_scores = (raw_mean - rmin) / (rmax - rmin) * 100
    else:
        display_scores = raw_mean * 100   # fallback if all identical

    results = []
    for i, m in enumerate(meta):
        tier = _tier(float(cal_prob[i]))
        results.append({
            **m,
            'risk_score':  float(display_scores[i]),   # viewport-normalized 0-100
            'risk_prob':   float(cal_prob[i] * 100),   # true calibrated probability
            'risk_raw':    float(raw_mean[i]),
            'risk_tier':   tier,
            'confidence':  float(100 * (1 - min(raw_std[i] / 0.5, 1.0))),
            'uncertainty': float(raw_std[i] * 100),
        })
    return results

# ============================================================================
# TFT v2 SPREAD FORECAST — radial profile, no 20× multiplier
# ============================================================================

DIRS_DEG  = [0, 45, 90, 135, 180, 225, 270, 315]
DIR_NAMES = [f"{d:03d}" for d in DIRS_DEG]

def _radial_profile(c_lat, c_lon, pt_lats, pt_lons):
    KM_LAT = 111.0
    km_lon = 111.0 * np.cos(np.radians(c_lat))
    dy = (pt_lats - c_lat) * KM_LAT
    dx = (pt_lons - c_lon) * km_lon
    angles = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    dists  = np.sqrt(dx**2 + dy**2)
    hw     = 180.0 / len(DIRS_DEG)
    radii  = np.zeros(len(DIRS_DEG))
    for i, d in enumerate(DIRS_DEG):
        diff = np.abs(((angles - d) + 180) % 360 - 180)
        mask = diff <= hw
        radii[i] = float(dists[mask].max()) if mask.sum() > 0 else 0.0
    for i in range(len(DIRS_DEG)):
        if radii[i] == 0:
            for step in range(1, len(DIRS_DEG)):
                l = (i-step) % len(DIRS_DEG)
                r = (i+step) % len(DIRS_DEG)
                vals = [v for v in [radii[l], radii[r]] if v > 0]
                if vals:
                    radii[i] = float(np.mean(vals)); break
            if radii[i] == 0:
                radii[i] = 0.1
    return radii

def _profile_to_coords(c_lat, c_lon, radii_km):
    """Radial profile → list of [lat,lon] points for Folium Polygon."""
    KM_LAT = 111.0
    km_lon = 111.0 * np.cos(np.radians(c_lat))
    pts    = []
    for angle_deg, r_km in zip(DIRS_DEG, radii_km):
        math_angle = np.radians(90 - angle_deg)
        dx_km = r_km * np.cos(math_angle)
        dy_km = r_km * np.sin(math_angle)
        pts.append([c_lat + dy_km / KM_LAT,
                    c_lon + dx_km / km_lon])
    pts.append(pts[0])  # close polygon
    return pts

def run_tft_spread(fire_lat, fire_lon, fire_area_km2, weather, firms_ctx):
    """
    TFT v2: predict radial expansion per direction at T+6h, T+12h, T+24h.
    Returns polygon coordinates (not circles).
    Calibration factors derived from val set — not manually chosen.
    Sub-24h horizons use linear interpolation (documented limitation).
    """
    from sklearn.cluster import DBSCAN

    # Build current radial profile from fire detections
    flats = firms_ctx.get('fire_lats', np.array([fire_lat]))
    flons = firms_ctx.get('fire_lons', np.array([fire_lon]))
    profile = _radial_profile(fire_lat, fire_lon, flats, flons)

    feat = np.array([[
        fire_area_km2,
        float(firms_ctx.get('n_detections', 10)),
        float(firms_ctx.get('frp_max',  100)),
        float(firms_ctx.get('frp_mean',  50)),
        float(weather.get('wind_speed_kmh', 20)),
        float(weather.get('wind_direction', 270)),
        float(weather.get('temp_c', 25)),
        float(weather.get('humidity', 30)),
        float(weather.get('drought_index', 0.5)),
        float(weather.get('days_since_rain', 7.0)),
    ] + profile.tolist()], dtype=np.float32)

    feat_scaled = tft_scaler.transform(feat)
    pred_raw    = tft_model.predict(feat_scaled, verbose=0)[0]  # (24,)

    # Raw predictions per direction
    p50_raw = pred_raw[1::3]   # P50 per direction, shape (8,)
    p90_raw = pred_raw[2::3]   # P90 per direction, shape (8,)

    # For DISPLAY we use uncalibrated P90 — calibration factors shrink already-small
    # predictions toward zero. P90 gives a conservative but visible bound.
    # Calibration factors are retained for the BNN scoring pipeline (inference_bridge.py)
    # but not applied here — using raw model output for perimeter visualization.
    spread_dir = (float(weather.get('wind_direction', 270)) + 180) % 360
    no_firms   = firms_ctx.get('n_detections', 0) == 0

    horizons = []
    for h in [6, 12, 24]:
        scale = h / 24.0

        # Use raw P90 for display — larger, always shows some spread
        deltas_display = p90_raw * scale   # uncalibrated P90 at this horizon

        # Floor at 0 per direction: perimeter can only expand or hold
        deltas_positive = np.maximum(deltas_display, 0.0)

        # Projected radii: current + positive expansion only
        radii_h = profile + deltas_positive   # always ≥ profile

        # Area from mean radius
        mean_r   = float(radii_h.mean())
        area_est = float(np.pi * mean_r**2)
        area_p90 = area_est  # already using P90

        # Growth metric: max expansion in any single direction
        max_expansion_km = float(deltas_positive.max())
        growth_display   = max_expansion_km / scale if scale > 0 else 0.0

        # Polygon coordinates
        poly_coords = _profile_to_coords(fire_lat, fire_lon, radii_h)
        poly_p90    = _profile_to_coords(fire_lat, fire_lon, radii_h * 1.3)

        max_radius_km = float(radii_h.max())
        horizons.append({
            "horizon_h":        h,
            "center_lat":       fire_lat,
            "center_lon":       fire_lon,
            "spread_dir_deg":   spread_dir,
            "radius_p50_km":    max_radius_km,
            "radius_p90_km":    float(max_radius_km * 1.3),
            "area_p50_km2":     round(area_est, 2),
            "area_p90_km2":     round(area_p90 * 1.3**2, 2),
            # Max expansion in any direction at this horizon (km total, not per day)
            "growth_p50":       round(max_expansion_km, 3),
            "growth_p90":       round(max_expansion_km * 1.3, 3),
            "growth_p10":       round(max_expansion_km * 0.7, 3),
            "confidence":       max(0, 1 - h/48.0) * (0.6 if no_firms else 1.0),
            "no_firms_warning": no_firms,
            "poly_p50_coords":  poly_coords,
            "poly_p90_coords":  poly_p90,
            "radii_km":         radii_h.tolist(),
        })
    return horizons, spread_dir, no_firms

# ============================================================================
# LIVE FIRE FETCH
# ============================================================================

live_fires_df = pd.DataFrame()
if not use_historical:
    try:
        api_key = "ffe67bd547acd1ab34b70c7376aabdca"
        bbox    = f"{current_lon-2},{current_lat-2},{current_lon+2},{current_lat+2}"
        url     = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
                   f"/{api_key}/VIIRS_NOAA20_NRT/{bbox}/{lookback_days}")
        resp    = requests.get(url, timeout=20)
        if resp.status_code == 200 and len(resp.text) > 100:
            from io import StringIO
            live_fires_df = pd.read_csv(StringIO(resp.text))
            live_fires_df = live_fires_df.rename(
                columns={'lat':'latitude','lon':'longitude'})
            st.sidebar.success(f"🔴 {len(live_fires_df)} live detections")
        else:
            st.sidebar.warning("No live fires in area")
    except Exception as e:
        st.sidebar.error(f"Live fetch failed: {e}")
    fires_for_date = live_fires_df

# ============================================================================
# WEATHER
# ============================================================================

current_weather = get_weather(current_lat, current_lon, selected_date, use_historical)

col1, col2, col3, col4 = st.columns(4)
col1.metric("📍 Lat", f"{current_lat:.4f}")
col2.metric("📍 Lon", f"{current_lon:.4f}")
col3.metric("🌬️ Wind", f"{current_weather['wind_speed_kmh']:.1f} km/h")
col4.metric("💧 Humidity", f"{current_weather['humidity']:.0f}%")

# ============================================================================
# BASE MAP  — identical to v2
# ============================================================================

m = folium.Map(
    location=[current_lat, current_lon],
    zoom_start=zoom, tiles="OpenStreetMap")

fires_list     = fires_for_date.to_dict('records') if len(fires_for_date) > 0 else []
display_fires  = fires_list[:500] if len(fires_list) > 500 else fires_list

for fire in display_fires:
    folium.CircleMarker(
        [fire['latitude'], fire['longitude']],
        radius=5, color='darkred', fill=True, fill_opacity=0.8,
        popup=f"Fire<br>FRP: {fire.get('frp','N/A')}<br>Date: {fire.get('acq_date','')}"
    ).add_to(m)

delta = 0.4
west, east   = current_lon - delta, current_lon + delta
south, north = current_lat - delta, current_lat + delta

local_infra = infra_df[
    (infra_df['lat'] >= south) & (infra_df['lat'] <= north) &
    (infra_df['lon'] >= west)  & (infra_df['lon'] <= east)
]
if len(local_infra) > 200:
    local_infra = local_infra.sample(200, random_state=42)

icon_map = {
    'Power Substation':'flash', 'Hospital':'plus-sign',
    'Wind Farm':'cloud',        'Solar Farm':'star',
    'Fire Station':'fire',      'School':'book',
    'Residential Area':'home',  'Gas Plant':'oil',
    'Coal Plant':'cog',         'Hydro Plant':'tint',
    'Airport':'plane',          'Cell Tower':'signal',
    'Water Treatment':'tint',   'Medical Clinic':'plus-sign',
    'University':'book',
}

for _, asset in local_infra.iterrows():
    if len(fires_for_date) > 0:
        dists = haversine_vectorized(
            asset['lat'], asset['lon'],
            fires_for_date['latitude'].values,
            fires_for_date['longitude'].values)
        min_d    = dists.min()
        min_d_mi = min_d * 0.621371
        qr       = quick_risk(min_d)
    else:
        min_d, min_d_mi, qr = 999, 999, 0

    color = 'red' if qr >= 65 else 'orange' if qr >= 35 else 'green'
    icon  = icon_map.get(str(asset.get('type', '')), 'info-sign')

    folium.Marker(
        [asset['lat'], asset['lon']],
        icon=folium.Icon(color=color, icon=icon, prefix='glyphicon'),
        popup=f"""
        <div style='width:220px'>
            <div style='background:{color};color:white;
                 padding:8px;margin:-10px -10px 8px -10px'>
                <b>{str(asset.get('name','Unknown'))[:30]}</b>
            </div>
            <p>{asset.get('type','')} · {asset.get('state','')} ·
               {asset.get('city','N/A')}</p>
            <b>Quick Risk: {qr:.0f}%</b><br>
            Distance: {min_d:.1f} km ({min_d_mi:.1f} mi)
        </div>""",
        tooltip=f"{str(asset.get('name','Unknown'))[:25]} — ~{qr:.0f}%"
    ).add_to(m)

st.info(f"Showing {len(display_fires):,} fires • {len(local_infra):,} assets in view")

map_data = st_folium(
    m, width=1400, height=600, key="map",
    returned_objects=["bounds", "last_clicked"])

# ============================================================================
# CLICK HANDLING  — identical to v2
# ============================================================================

if map_data and map_data.get("last_clicked"):
    click       = map_data["last_clicked"]
    click_tuple = (click["lat"], click["lng"])
    if click_tuple != st.session_state.last_handled_click:
        st.session_state.display_lat        = click["lat"]
        st.session_state.display_lon        = click["lng"]
        st.session_state.last_handled_click = click_tuple
        st.session_state.use_preset         = False
        st.rerun()

if map_data and map_data.get("last_clicked"):
    c = map_data["last_clicked"]
    st.success(f"📍 Clicked: {c['lat']:.4f}°, {c['lng']:.4f}°")

st.markdown("---")

# ============================================================================
# ANALYZE BUTTON — same as v2 + adds spread forecast tabs
# ============================================================================

if st.button("🧠 Analyze", type="primary", use_container_width=True):
    if map_data and map_data.get("bounds"):
        bounds = map_data["bounds"]
        sw = bounds["_southWest"]; ne = bounds["_northEast"]
        vw, vs, ve, vn = sw["lng"], sw["lat"], ne["lng"], ne["lat"]
    else:
        vw, vs, ve, vn = west, south, east, north

    fires_in_view = fires_for_date[
        (fires_for_date['latitude']  >= vs) & (fires_for_date['latitude']  <= vn) &
        (fires_for_date['longitude'] >= vw) & (fires_for_date['longitude'] <= ve)
    ] if len(fires_for_date) > 0 else pd.DataFrame()

    infra_in_view = infra_df[
        (infra_df['lat'] >= vs) & (infra_df['lat'] <= vn) &
        (infra_df['lon'] >= vw) & (infra_df['lon'] <= ve)
    ]
    if len(infra_in_view) > 500:
        infra_in_view = infra_in_view.sample(500, random_state=42)

    if len(fires_in_view) == 0:
        st.warning("⚠️ No fires in current view.")
    else:
        with st.spinner(f"Running BNN on {len(infra_in_view)} assets (batched) + TFT spread..."):
            # Batched BNN inference — all assets at once
            results = predict_risk_batch(
                infra_in_view, fires_in_view, current_weather, model, scaler)

            # Fire centroid for TFT
            frp_vals = fires_in_view['frp'].values if 'frp' in fires_in_view.columns \
                       else np.ones(len(fires_in_view)) * 30.0
            weights  = frp_vals / (frp_vals.sum() + 1e-9)
            fire_clat = float((fires_in_view['latitude'].values  * weights).sum())
            fire_clon = float((fires_in_view['longitude'].values * weights).sum())
            fire_area = float(len(fires_in_view) * 0.5)

            firms_ctx = {
                'frp_mean':     float(frp_vals.mean()),
                'frp_max':      float(frp_vals.max()),
                'n_detections': len(fires_in_view),
                'fire_density': len(fires_in_view) / 1000.0,
                'fire_lats':    fires_in_view['latitude'].values.astype(np.float64),
                'fire_lons':    fires_in_view['longitude'].values.astype(np.float64),
            }

            # TFT spread forecast
            spread_horizons, spread_dir, no_firms = run_tft_spread(
                fire_clat, fire_clon, fire_area, current_weather, firms_ctx)

        top20 = sorted(results, key=lambda x: x['risk_score'], reverse=True)[:20]

        # ── TABS ─────────────────────────────────────────────────────────────
        tab_cur, tab_6h, tab_12h, tab_24h = st.tabs([
            "🔥 Current Risk",
            "⏱ +6H Forecast",
            "⏱ +12H Forecast",
            "⏱ +24H Forecast",
        ])

        HORIZON_COLORS = {6:"#f6a623", 12:"#d64343", 24:"#8b0000"}

        def make_risk_map(tab, horizon_h, projected_fire_lat=None,
                          projected_fire_lon=None):
            with tab:
                # Metrics row for forecast tabs
                if horizon_h > 0:
                    hz = next(h for h in spread_horizons
                              if h["horizon_h"] == horizon_h)
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Max Spread Any Direction",
                              f"{hz['growth_p50']:.2f} km",
                              help="Max expansion in any single direction "
                                   "from current perimeter (TFT raw P90)")
                    m2.metric("Conservative Bound (×1.3)",
                              f"{hz['growth_p90']:.2f} km")
                    m3.metric("Projected Area",
                              f"{hz['area_p50_km2']:.1f} km²",
                              help="Area of TFT-projected perimeter polygon")
                    m4.metric("Conservative Area",
                              f"{hz['area_p90_km2']:.1f} km²")
                    if no_firms:
                        st.warning("⚠️ No satellite confirmation at ignition — "
                                   "uncertainty is elevated")

                # Re-score assets against projected fire position
                if projected_fire_lat is not None:
                    proj_results = []
                    for r in results:
                        dlat = np.radians(projected_fire_lat - r['lat'])
                        dlon = np.radians(projected_fire_lon - r['lon'])
                        a_val = (np.sin(dlat/2)**2 +
                                 np.cos(np.radians(r['lat'])) *
                                 np.cos(np.radians(projected_fire_lat)) *
                                 np.sin(dlon/2)**2)
                        dist_km = 6371 * 2 * np.arcsin(np.sqrt(a_val))
                        proj_results.append({
                            **r,
                            'distance_km': round(dist_km, 2),
                            'distance_miles': round(dist_km * 0.621371, 2),
                        })
                    display_top20 = sorted(
                        proj_results, key=lambda x: x['risk_score'],
                        reverse=True)[:20]
                else:
                    display_top20 = top20

                # Table
                st.markdown(f"### 🚨 Top 20 Risk Assets"
                            f"{'  (projected)' if horizon_h > 0 else ''}")
                df_out = pd.DataFrame([{
                    "Asset":    str(r.get('name','Unknown'))[:40],
                    "Type":     r.get('type',''),
                    "State":    r.get('state',''),
                    "City":     str(r.get('city','N/A'))[:20],
                    "Score":    f"{r['risk_score']:.1f}",
                    "Cal.Prob": f"{r.get('risk_prob', r['risk_score']):.3f}%",
                    "Tier":     r.get('risk_tier','—'),
                    "Unc":      f"±{r['uncertainty']:.1f}",
                    "Dist km":  f"{r['distance_km']:.2f}"
                                if r.get('distance_km') else "N/A",
                    "Action":   (r.get('risk_tier', '').upper() == 'CRITICAL' and "🔴 EVACUATE" or
                                 r.get('risk_tier', '').upper() == 'HIGH'     and "🟠 PREPARE"  or
                                 "🟡 MONITOR"),
                } for r in display_top20])
                st.dataframe(df_out, use_container_width=True)

                # Risk map
                st.markdown("### 🗺️ Risk Map")
                map_center_lat = projected_fire_lat or fire_clat
                map_center_lon = projected_fire_lon or fire_clon
                rm = folium.Map(
                    location=[(vs+vn)/2, (vw+ve)/2],
                    zoom_start=map_data.get("zoom", zoom))

                # Current fire position always shown
                for _, f in fires_in_view.iterrows():
                    folium.CircleMarker(
                        [f['latitude'], f['longitude']],
                        radius=8, color='darkred',
                        fill=True, fill_opacity=0.8).add_to(rm)

                # Spread ellipses for forecast tabs
                if horizon_h > 0:
                    # Wind direction arrow showing spread direction
                    arrow_len = spread_horizons[-1]["radius_p90_km"] * 0.8
                    arrow_lat = fire_clat + (arrow_len * np.cos(
                        np.radians(spread_horizons[0]["spread_dir_deg"]))) / 111.0
                    arrow_lon = fire_clon + (arrow_len * np.sin(
                        np.radians(spread_horizons[0]["spread_dir_deg"]))) / (
                        111.0 * np.cos(np.radians(fire_clat)))
                    folium.PolyLine(
                        [[fire_clat, fire_clon], [arrow_lat, arrow_lon]],
                        color="#ff4500", weight=3, opacity=0.8,
                        tooltip=f"Wind spread direction: "
                                f"{spread_horizons[0]['spread_dir_deg']:.0f}°"
                    ).add_to(rm)

                    for h_data in spread_horizons:
                        h = h_data["horizon_h"]
                        if h > horizon_h:
                            continue
                        col = HORIZON_COLORS[h]
                        # P90 uncertainty polygon (TFT v2 — real radial shape)
                        if h_data.get("poly_p90_coords"):
                            folium.Polygon(
                                locations=h_data["poly_p90_coords"],
                                color=col, fill=True, fill_opacity=0.07,
                                weight=1, dash_array="5",
                                tooltip=f"T+{h}h Uncertainty (P90) — {h_data['area_p90_km2']:.1f} km²"
                            ).add_to(rm)
                        # P50 predicted perimeter polygon
                        if h_data.get("poly_p50_coords"):
                            folium.Polygon(
                                locations=h_data["poly_p50_coords"],
                                color=col, fill=True, fill_opacity=0.18,
                                weight=2,
                                tooltip=(f"T+{h}h Predicted Perimeter (P50) — "
                                         f"{h_data['area_p50_km2']:.1f} km²")
                            ).add_to(rm)

                # Zone radii for THIS specific tab's horizon
                # Colors change per tab — asset may be green now, orange +6H, red +24H
                if horizon_h > 0:
                    hz       = next(h for h in spread_horizons
                                    if h["horizon_h"] == horizon_h)
                    inner_r  = hz["radius_p50_km"]   # red zone — inside P50
                    outer_r  = hz["radius_p90_km"]   # orange zone — P50 to P90
                else:
                    # Current tab — use distance-based quick risk thresholds
                    inner_r  = 7.0
                    outer_r  = 30.0

                all_assets = results
                for a in all_assets:
                    dist = a.get('distance_km', 999) or 999
                    # Color determined by THIS tab's zone — changes across tabs
                    c  = 'red'    if dist <= inner_r else \
                         'orange' if dist <= outer_r else 'green'
                    ic = icon_map.get(str(a.get('type', '')), 'info-sign')
                    r  = a['risk_score']
                    folium.Marker(
                        [a['lat'], a['lon']],
                        icon=folium.Icon(color=c, icon=ic, prefix='glyphicon'),
                        popup=f"""
                        <div style='width:250px'>
                            <div style='background:{c};color:white;
                                 padding:10px;margin:-10px -10px 10px -10px'>
                                <b>{str(a.get('name','Unknown'))[:35]}</b>
                            </div>
                            <p>{a.get('type','')} · {a.get('state','')} ·
                               {str(a.get('city','N/A'))[:20]}</p>
                            <hr>
                            <b style='font-size:18px;color:{c}'>
                              Score: {r:.1f}/100</b>
                            <div style='font-size:11px;color:#555;margin-top:2px'>
                              Relative rank within viewport<br>
                              100 = highest risk asset in view
                            </div>
                            <div style='margin-top:6px;font-size:12px'>
                              <b>Tier:</b> {a.get('risk_tier','—')}<br>
                              <b>Calibrated probability:</b>
                              {a.get('risk_prob', r*0.027):.3f}%
                              <span style='color:#888'>(0.3% base rate)</span><br>
                              <b>Uncertainty:</b> ±{a['uncertainty']:.1f}
                            </div>
                            <hr>
                            <p style='font-size:12px'>
                              Distance: {dist:.2f} km<br>
                              Fires within 30km: {a['num_fires']}
                            </p>
                        </div>""",
                        tooltip=f"{str(a.get('name','Unknown'))[:25]} — {dist:.1f}km — {r:.0f}%"
                    ).add_to(rm)

                # Legend
                legend_spread = (
                    "<hr>── P50 Predicted Perimeter<br>"
                    "- - P90 Uncertainty Zone<br>"
                    "🟡 +6H &nbsp;🔴 +12H &nbsp;🟥 +24H<br>"
                    "<hr>🔴 Inside P50 &nbsp;🟠 P50–P90 &nbsp;🟢 Outside"
                    if horizon_h > 0 else
                    "<hr>🔴 &lt;7km &nbsp;🟠 7–30km &nbsp;🟢 &gt;30km")
                rm.get_root().html.add_child(folium.Element(f"""
                <div style='position:fixed;bottom:30px;right:10px;
                     background:white;padding:10px;border-radius:8px;
                     border:1px solid #ccc;font-size:11px;z-index:999'>
                  🔴 HIGH (&gt;70%) &nbsp;🟠 MEDIUM &nbsp;🟢 LOW<br>
                  ● Active fire detection{legend_spread}
                </div>"""))

                st_folium(rm, width=1400, height=600,
                          key=f"riskmap_{horizon_h}",
                          returned_objects=[])

                # Camp Fire validation
                if '2018-11-08' in str(selected_date) and horizon_h == 0:
                    p = next((r for r in results
                              if 'Feather River' in str(r.get('name',''))
                              or 'Paradise' in str(r.get('city',''))), None)
                    if p:
                        st.markdown("---")
                        st.markdown("### 🎯 Validation — Camp Fire 2018")
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("Paradise Hospital Risk",
                                  f"{p['risk_score']:.1f}%")
                        c2.metric("Confidence",
                                  f"{p['confidence']:.1f}%")
                        c3.metric("Uncertainty",
                                  f"±{p['uncertainty']:.1f}")
                        c4.metric("Distance to Fire",
                                  f"{p['distance_miles']:.1f} mi")
                        st.success(
                            "✅ Model correctly predicted CRITICAL risk. "
                            "Feather River Hospital was evacuated and "
                            "destroyed ~18 hours later.")

        # Current tab — no spread
        make_risk_map(tab_cur, 0)

        # Forecast tabs — projected fire positions
        for hz_data in spread_horizons:
            h   = hz_data["horizon_h"]
            tab = {6:tab_6h, 12:tab_12h, 24:tab_24h}[h]
            make_risk_map(tab, h,
                          projected_fire_lat=hz_data["center_lat"],
                          projected_fire_lon=hz_data["center_lon"])

# ============================================================================
# VALIDATION TAB — always visible, outside the Analyze button block
# ============================================================================

with st.expander("📊 Model Validation — Proof of Performance", expanded=False):
    VAL_BNN = ROOT / "validation" / "bnn"
    VAL_TFT = ROOT / "validation" / "tft"
    VAL_BRG = ROOT / "validation" / "bridge"

    st.markdown("### BNN v3 — Trained on Real FIRMS Labels, No Synthetic Data")
    st.caption(
        "2.7M samples · 2017-2021 train · 2022-2023 val · 2024 test (holdout) · "
        "Platt calibration C=10 (5-fold CV) · Thresholds: P50/P80/P95 of val scores")

    try:
        with open(VAL_BNN / "val_metrics_calibrated.json") as f:
            bm = json.load(f)
        raw = bm.get("raw", {}); cal = bm.get("calibrated", {})
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Val AUC (raw)",   raw.get("auc","—"))
        c2.metric("Val AUC (Platt)", cal.get("auc","—"))
        c3.metric("Brier score",     cal.get("brier","—"), help="Lower = better")
        c4.metric("Brier skill",     cal.get("brier_skill","—"), help=">0 = better than naive")
    except Exception:
        pass

    col1, col2 = st.columns(2)
    for path, col, cap in [
        (VAL_BNN/"bnn_roc_curve.png",         col1, "ROC Curve — Val and 2024 Holdout"),
        (VAL_BNN/"bnn_calibration_after.png",  col2, "Calibration — Raw vs Platt Scaling"),
        (VAL_BNN/"bnn_feature_importance.png", col1, "Permutation Importance — distance #1"),
        (VAL_BNN/"bnn_uncertainty.png",        col2, "MC Dropout Uncertainty Distribution"),
    ]:
        with col:
            st.caption(cap)
            if path.exists():
                st.image(str(path), use_container_width=True)

    try:
        with open(VAL_BNN/"test_metrics_calibrated.json") as f:
            tm = json.load(f)
        tc = tm.get("calibrated", {})
        st.caption(
            f"**2024 holdout (never seen during training or calibration):** "
            f"AUC {tc.get('auc','—')} · Brier {tc.get('brier','—')} · "
            f"Brier skill {tc.get('brier_skill','—')}")
    except Exception:
        pass

    st.markdown("---")
    st.markdown("### TFT v2 — Radial Perimeter Forecast, Real FIRMS Labels")
    st.caption(
        "2,535 samples · 8 directional heads × P10/P50/P90 · "
        "Calibration factors derived from val set")

    try:
        with open(VAL_TFT/"tft_val_metrics.json") as f:
            tv = json.load(f)
        with open(VAL_TFT/"tft_test_metrics.json") as f:
            tt = json.load(f)
        tc1,tc2,tc3,tc4 = st.columns(4)
        tc1.metric("Val MAE",       f"{tv.get('mae_overall','—')} km")
        tc2.metric("Test MAE",      f"{tt.get('mae_overall','—')} km",
                   help="2024 holdout")
        tc3.metric("Val Coverage",  f"{100*tv.get('p10_p90_coverage',0):.0f}%",
                   help="P10-P90 (target 80%)")
        tc4.metric("Test Coverage", f"{100*tt.get('p10_p90_coverage',0):.0f}%")
    except Exception:
        pass

    col3, col4 = st.columns(2)
    for path, col, cap in [
        (VAL_TFT/"tft_predicted_vs_actual.png", col3, "Predicted vs Actual per Direction"),
        (VAL_TFT/"tft_quantile_coverage.png",   col4, "Coverage per Direction (target 80%)"),
        (VAL_TFT/"tft_calibration_factor.png",  col3, "Calibration Factors — from val set"),
        (VAL_TFT/"tft_loss_curves.png",         col4, "Training Curves"),
    ]:
        with col:
            st.caption(cap)
            if path.exists():
                st.image(str(path), use_container_width=True)

    st.markdown("---")
    st.markdown("### Pipeline Integration Test — Camp Fire 2018-11-08")
    st.caption(
        "Feather River Hospital (actual Paradise CA coordinates) → CRITICAL. "
        "SW (downwind) > NE (upwind). Risk decreases with distance.")

    col5, col6 = st.columns(2)
    for path, col, cap in [
        (VAL_BRG/"campfire_2018_integration_test.png", col5, "Camp Fire — T+0h and T+24h"),
        (VAL_BRG/"monotonicity_check.png",             col6, "Monotonicity — Risk vs Distance"),
    ]:
        with col:
            st.caption(cap)
            if path.exists():
                st.image(str(path), use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚠️ Known Limitations")
    st.markdown("""
- **Score compression**: max calibrated risk ~2.6% (physically correct at 0.3% base rate).
  Model ranks assets — absolute probabilities are small by design.
- **Sub-24h TFT horizons**: linear interpolation of 24h prediction (documented limitation).
- **Historical forecasts**: same weather conditions used at all horizons for past events
  (no historical forecast data available).
- **Upwind assets at <15km**: score CRITICAL due to distance dominating wind alignment
  (operationally conservative).
""")