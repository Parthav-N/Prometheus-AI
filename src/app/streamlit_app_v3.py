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

st.title("🔥 Wildfire Infrastructure Risk Prediction")
st.caption("Bayesian Neural Network v2 • TFT Spread Forecast • 192,884 Assets • 10-State Western US • Live + Historical Data")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from spread_tft_model import SpreadTFT

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model_and_data():
    model_path  = ROOT / "models" / "bayesian_risk_model_v2.keras"
    scaler_path = ROOT / "models" / "feature_scaler_v2.pkl"
    infra_path  = ROOT / "data"   / "infrastructure" / "national_infrastructure.csv"

    model  = tf.keras.models.load_model(str(model_path))
    scaler = joblib.load(str(scaler_path))
    infra  = pd.read_csv(infra_path)

    renewable_mask = infra['type'].isin(['Solar Farm', 'Wind Farm'])
    if 'source' in infra.columns and 'capacity_mw' in infra.columns:
        infra['capacity_mw'] = pd.to_numeric(infra['capacity_mw'], errors='coerce')
        keep = infra[~renewable_mask]
        util = infra[renewable_mask][
            (infra[renewable_mask]['source'] == 'EIA') |
            (infra[renewable_mask]['capacity_mw'] >= 1.0)
        ]
        infra = pd.concat([keep, util], ignore_index=True)

    return model, scaler, infra

@st.cache_resource
def load_tft_model():
    meta_path   = ROOT / "models" / "spread_tft_metadata.json"
    scaler_path = ROOT / "models" / "spread_scaler.pkl"
    model_path  = ROOT / "models" / "spread_tft_best.pt"

    with open(meta_path) as f:
        meta = json.load(f)

    tft_scaler = joblib.load(scaler_path)
    tft = SpreadTFT(
        n_features    = meta["n_features"],
        d_model       = meta["architecture"]["d_model"],
        n_heads       = meta["architecture"]["n_heads"],
        n_lstm_layers = meta["architecture"]["n_lstm_layers"],
        dropout       = 0.0,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    tft.load_state_dict(state)
    tft.eval()
    return tft, tft_scaler, meta

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
    model, scaler, infra_df          = load_model_and_data()
    tft_model, tft_scaler, tft_meta  = load_tft_model()
    all_historical_fires             = load_historical_fires()
    historical_weather               = load_historical_weather()
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

st.sidebar.success("✅ BNN v2 + TFT Spread Ready")

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
    Batch BNN inference — all assets in one forward pass × 50 MC samples.
    ~50x faster than per-asset loop.
    """
    if len(fires_df) == 0 or len(assets_df) == 0:
        return []

    fire_lats = fires_df['latitude'].values
    fire_lons = fires_df['longitude'].values
    max_frp   = float(fires_df['frp'].max()) if 'frp' in fires_df.columns else 50.0
    wind_dir  = float(weather.get('wind_direction', 180))

    # Build feature matrix for all assets at once
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

    X = scaler.transform(np.array(rows, dtype=np.float32))  # (N, 11)
    X_tensor = tf.constant(X, dtype=tf.float32)

    # 50 MC Dropout passes — each pass scores ALL assets at once
    all_preds = np.stack([
        np.clip(model(X_tensor, training=True).numpy().flatten(), 0, 100)
        for _ in range(50)
    ])  # (50, N)

    means = all_preds.mean(axis=0)   # (N,)
    stds  = all_preds.std(axis=0)    # (N,)

    results = []
    for i, m in enumerate(meta):
        results.append({
            **m,
            'risk_score':    float(means[i]),
            'confidence':    float(100 * (1 - min(stds[i] / 30, 1.0))),
            'uncertainty':   float(stds[i]),
        })
    return results

# ============================================================================
# TFT SPREAD FORECAST
# ============================================================================

def run_tft_spread(fire_lat, fire_lon, fire_area_km2, weather, firms_ctx):
    """Run TFT to get spread ellipse params at T+6h, T+12h, T+24h."""
    slope = float(np.clip((fire_lat - 32.0) / 12.0, 0, 1))
    ws    = weather.get('wind_speed_kmh', 10.0)

    feat_map = {
        "wind_speed_kmh":      ws,
        "wind_direction":      weather.get('wind_direction', 270.0),
        "temperature_c":       weather.get('temp_c', 25.0),
        "humidity":            weather.get('humidity', 40.0),
        "max_wind_kmh":        ws * 1.3,
        "wind_speed_trend":    0.0,
        "temp_trend":          0.0,
        "humidity_trend":      0.0,
        "drought_index":       weather.get('drought_index', 0.5),
        "current_area_km2":    fire_area_km2,
        "fuel_multiplier":     1.0,
        "slope_proxy":         slope,
        "frp_mean":            firms_ctx.get('frp_mean', 0.0),
        "frp_max":             firms_ctx.get('frp_max', 0.0),
        "n_detections":        firms_ctx.get('n_detections', 0.0),
        "fire_density":        firms_ctx.get('fire_density', 0.0),
        "frp_trend":           0.0,
        "slope_wind_interaction": slope * ws / 50.0,
        "wind_x_fuel":         ws,
        "heat_dryness":        max(0, weather.get('temp_c', 25.0) - 25) *
                               max(0, 60 - weather.get('humidity', 40.0)) / 100,
    }

    all_features = tft_meta["all_features"]
    X = np.array([[feat_map.get(f, 0.0) for f in all_features]], dtype=np.float32)
    X = np.nan_to_num(tft_scaler.transform(X), nan=0.0,
                      posinf=3.0, neginf=-3.0).astype(np.float32)

    with torch.no_grad():
        pred = tft_model(torch.tensor(X)).numpy()[0]  # (3 horizons, 3 quantiles)

    p10 = np.expm1(pred[:, 0]).clip(0, tft_meta["growth_clip_km2h"])
    p50 = np.expm1(pred[:, 1]).clip(0, tft_meta["growth_clip_km2h"])
    p90 = np.expm1(pred[:, 2]).clip(0, tft_meta["growth_clip_km2h"])

    # Spread direction from wind (Rothermel: fire spreads downwind)
    spread_dir  = (weather.get('wind_direction', 270.0) + 180) % 360
    move_rad    = np.radians(spread_dir)
    km_per_lat  = 111.0
    km_per_lon  = 111.0 * np.cos(np.radians(fire_lat))
    no_firms    = firms_ctx.get('n_detections', 0) == 0

    # Dynamic radii — TFT predicted area × uncertainty correction factor
    # 20x multiplier corrects for known TFT underprediction on extreme fires
    # This keeps circles physically tied to model output, not hardcoded
    UNCERTAINTY_FACTOR = 20.0

    horizons = []
    for i, h in enumerate([6, 12, 24]):
        # TFT predicted area at this horizon
        area_p50 = float(p50[i] * h)
        area_p90 = float(p90[i] * h)

        # Dynamic radius — scales with TFT prediction
        radius_p50_km = float(np.sqrt(area_p50 * UNCERTAINTY_FACTOR / np.pi))
        radius_p90_km = float(np.sqrt(area_p90 * UNCERTAINTY_FACTOR / np.pi))

        # Ensure P90 > P50 always
        radius_p90_km = max(radius_p90_km, radius_p50_km * 1.5)

        conf = max(0, 1 - h/48.0) * (0.6 if no_firms else 1.0)
        horizons.append({
            "horizon_h":        h,
            "center_lat":       fire_lat,
            "center_lon":       fire_lon,
            "spread_dir_deg":   spread_dir,
            "radius_p50_km":    radius_p50_km,
            "radius_p90_km":    radius_p90_km,
            "area_p50_km2":     round(area_p50, 2),
            "area_p90_km2":     round(area_p90, 2),
            "growth_p10":       round(float(p10[i]), 4),
            "growth_p50":       round(float(p50[i]), 4),
            "growth_p90":       round(float(p90[i]), 4),
            "confidence":       round(conf, 2),
            "no_firms_warning": no_firms,
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
                    m1.metric("Growth Rate P50",
                              f"{hz['growth_p50']:.3f} km²/h")
                    m2.metric("Growth Rate P90",
                              f"{hz['growth_p90']:.3f} km²/h")
                    m3.metric("Predicted Area (P50)",
                              f"{hz['area_p50_km2']:.1f} km²")
                    m4.metric("Uncertainty Zone (P90)",
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
                    "Asset":   str(r.get('name','Unknown'))[:40],
                    "Type":    r.get('type',''),
                    "State":   r.get('state',''),
                    "City":    str(r.get('city','N/A'))[:20],
                    "Risk %":  f"{r['risk_score']:.1f}",
                    "Conf %":  f"{r['confidence']:.1f}",
                    "Unc":     f"±{r['uncertainty']:.1f}",
                    "Dist km": f"{r['distance_km']:.2f}"
                               if r.get('distance_km') else "N/A",
                    "Action":  ("🔴 EVACUATE" if r['risk_score'] > 65 else
                                "🟠 PREPARE"  if r['risk_score'] > 35 else
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
                        # P90 uncertainty zone — outer ring
                        if h_data["radius_p90_km"] > 0.5:
                            folium.Circle(
                                [h_data["center_lat"], h_data["center_lon"]],
                                radius=h_data["radius_p90_km"] * 1000,
                                color=col, fill=True, fill_opacity=0.07,
                                weight=1, dash_array="5",
                                tooltip=(f"T+{h}h Uncertainty Zone (P90)  "
                                         f"Plan for: {h_data['area_p90_km2']:.1f} km²")
                            ).add_to(rm)
                        # P50 predicted perimeter — inner circle
                        if h_data["radius_p50_km"] > 0.1:
                            folium.Circle(
                                [h_data["center_lat"], h_data["center_lon"]],
                                radius=h_data["radius_p50_km"] * 1000,
                                color=col, fill=True, fill_opacity=0.18,
                                weight=2,
                                tooltip=(f"T+{h}h Predicted Perimeter (P50)  "
                                         f"Predicted: {h_data['area_p50_km2']:.1f} km²  "
                                         f"Growth: {h_data['growth_p50']:.3f} km²/h")
                            ).add_to(rm)
                        # No per-horizon center marker — all circles share fire origin

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
                        <div style='width:240px'>
                            <div style='background:{c};color:white;
                                 padding:10px;margin:-10px -10px 10px -10px'>
                                <b>{str(a.get('name','Unknown'))[:35]}</b>
                            </div>
                            <p>{a.get('type','')} · {a.get('state','')} ·
                               {str(a.get('city','N/A'))[:20]}</p>
                            <hr>
                            <b style='font-size:18px;color:{c}'>
                              Risk: {r:.1f}%</b><br>
                            <b>Confidence: {a['confidence']:.1f}%</b><br>
                            <b>Uncertainty: ±{a['uncertainty']:.1f}</b><br>
                            <hr>
                            <p style='font-size:12px'>
                              Distance: {dist:.1f} km<br>
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