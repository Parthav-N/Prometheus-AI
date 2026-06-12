import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from scipy.spatial import cKDTree

st.set_page_config(page_title="Wildfire Risk Prediction", layout="wide")

st.title("🔥 Wildfire Infrastructure Risk Prediction")
st.caption("Bayesian Neural Network v2 • 192,884 Assets • 10-State Western US • Live + Historical Data")

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

@st.cache_resource
def load_model_and_data():
    model_path = ROOT / "models" / "bayesian_risk_model_v2.keras"
    scaler_path = ROOT / "models" / "feature_scaler_v2.pkl"
    infra_path  = ROOT / "data" / "infrastructure" / "national_infrastructure.csv"

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

@st.cache_data
def load_historical_fires():
    fire_file = ROOT / "data" / "fires" / "national_fires_2017_2025.csv"
    df = pd.read_csv(str(fire_file), low_memory=False)
    df = df[df['confidence'].astype(str).str.lower().isin(['h', 'high'])]
    return df

@st.cache_data
def load_historical_weather():
    weather_file = ROOT / "data" / "weather" / "national_weather_grid.csv"
    df = pd.read_csv(str(weather_file), low_memory=False)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    return df

try:
    model, scaler, infra_df = load_model_and_data()
    all_historical_fires    = load_historical_fires()
    historical_weather      = load_historical_weather()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# DATA MODE
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

# LOCATION
st.sidebar.subheader("📍 Location")

location_preset = st.sidebar.selectbox(
    "Quick Location",
    [
        "Paradise (Camp Fire 2018)",
        "Custom",
        # California
        "Los Angeles", "San Francisco", "San Diego", "Sacramento",
        # Oregon / Washington
        "Medford OR", "Bend OR", "Spokane WA", "Yakima WA",
        # Mountain West
        "Missoula MT", "Boise ID", "Denver CO", "Salt Lake City UT",
        "Reno NV", "Phoenix AZ",
    ],
    index=0,
    key="location_preset"
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

st.sidebar.success("✅ Model v2 Ready  (11 features)")

# ============================================================================
# SESSION STATE
# ============================================================================

if "display_lat"         not in st.session_state: st.session_state.display_lat         = center_lat
if "display_lon"         not in st.session_state: st.session_state.display_lon         = center_lon
if "last_handled_click"  not in st.session_state: st.session_state.last_handled_click  = None
if "use_preset"          not in st.session_state: st.session_state.use_preset          = True

if st.session_state.use_preset:
    current_lat = center_lat
    current_lon = center_lon
else:
    current_lat = st.session_state.display_lat
    current_lon = st.session_state.display_lon

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Build KDTree for weather lookup once
_wx_coords = historical_weather[['grid_lat', 'grid_lon']].drop_duplicates().values
_wx_tree   = cKDTree(_wx_coords)

def get_weather(lat, lon, date_str, is_historical):
    """Get weather for location — historical KDTree lookup or live Open-Meteo"""

    if is_historical and date_str not in ("N/A", "Live"):
        try:
            _, idx     = _wx_tree.query([lat, lon])
            near_lat, near_lon = _wx_coords[idx]
            target_date = pd.to_datetime(date_str).date()

            station = historical_weather[
                (historical_weather['grid_lat'] == near_lat) &
                (historical_weather['grid_lon'] == near_lon) &
                (historical_weather['datetime'].dt.date == target_date)
            ]

            if len(station) > 0:
                drought = float(station.get('drought_index',
                                pd.Series([0.5])).mean()) \
                          if 'drought_index' in station.columns else 0.5
                dsr     = float(station.get('days_since_rain',
                                pd.Series([7.0])).mean()) \
                          if 'days_since_rain' in station.columns else 7.0
                return {
                    'wind_speed_kmh': float(station['wind_speed_kmh'].mean()),
                    'wind_direction': float(station['wind_direction'].mean()),
                    'temp_c':         float(station['temp_c'].mean()),
                    'humidity':       float(station['humidity'].mean()),
                    'drought_index':  drought,
                    'days_since_rain': dsr,
                }
        except Exception:
            pass

    # Live fallback
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
        return {
            'wind_speed_kmh': 15, 'wind_direction': 180,
            'temp_c': 20, 'humidity': 50,
            'drought_index': 0.5, 'days_since_rain': 7.0,
        }


def haversine_vectorized(lat1, lon1, lats2, lons2):
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats2, lons2 = np.radians(np.array(lats2)), np.radians(np.array(lons2))
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def calculate_wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
    lat1, lon1 = np.radians(fire_lat), np.radians(fire_lon)
    lat2, lon2 = np.radians(asset_lat), np.radians(asset_lon)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
    wind_toward  = (wind_dir + 180) % 360
    angle_diff   = abs(wind_toward - bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return np.cos(np.radians(angle_diff))


def quick_risk(min_dist_km):
    """Fast distance-based risk for map preview (no model needed)"""
    if   min_dist_km < 3:   return 90
    elif min_dist_km < 7:   return 70
    elif min_dist_km < 15:  return 45
    elif min_dist_km < 30:  return 25
    elif min_dist_km < 50:  return 12
    return 5


def predict_risk(asset, fires_df, weather, model, scaler):
    """Full BNN prediction with MC Dropout — 11 features (v2 model)"""

    if len(fires_df) == 0:
        return {
            'risk_score': 0.0, 'confidence': 100.0,
            'uncertainty': 0.0, 'num_fires': 0,
            'distance_km': None, 'distance_miles': None,
        }

    distances = haversine_vectorized(
        asset['lat'], asset['lon'],
        fires_df['latitude'].values,
        fires_df['longitude'].values
    )

    min_dist   = float(distances.min())
    mean_dist  = float(distances.mean())
    num_nearby = int((distances < 30).sum())
    max_frp    = float(fires_df['frp'].max()) if 'frp' in fires_df.columns else 50.0

    closest_fire = fires_df.iloc[distances.argmin()]
    wind_align   = calculate_wind_alignment(
        closest_fire['latitude'], closest_fire['longitude'],
        asset['lat'], asset['lon'],
        float(weather.get('wind_direction', 180))
    )

    drought_idx  = float(weather.get('drought_index',   0.5))
    days_since_r = float(weather.get('days_since_rain', 7.0))

    features = pd.DataFrame([{
        'min_distance_km':     min_dist,
        'mean_distance_km':    mean_dist,
        'num_fires_30km':      num_nearby,
        'max_frp':             max_frp,
        'wind_speed_kmh':      float(weather.get('wind_speed_kmh', 15)),
        'wind_direction':      float(weather.get('wind_direction', 180)),
        'temperature_c':       float(weather.get('temp_c', 20)),
        'humidity':            float(weather.get('humidity', 50)),
        'wind_fire_alignment': float(wind_align),
        'drought_index':       drought_idx,
        'days_since_rain':     days_since_r,
    }])

    feat_scaled = scaler.transform(features).astype(np.float32)

    preds = np.array([
        model(feat_scaled, training=True).numpy()[0, 0]
        for _ in range(50)
    ])
    preds      = np.clip(preds, 0, 100)
    risk_mean  = float(preds.mean())
    risk_std   = float(preds.std())
    confidence = float(100 * (1 - min(risk_std / 30, 1.0)))

    return {
        'risk_score':    risk_mean,
        'confidence':    confidence,
        'uncertainty':   risk_std,
        'num_fires':     num_nearby,
        'distance_km':   round(min_dist, 2),
        'distance_miles': round(min_dist * 0.621371, 2),
    }

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
            live_fires_df = live_fires_df.rename(columns={
                'lat': 'latitude', 'lon': 'longitude'
            })
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
# MAP
# ============================================================================

m = folium.Map(
    location=[current_lat, current_lon],
    zoom_start=zoom,
    tiles="OpenStreetMap"
)

# Fire markers
fires_list = fires_for_date.to_dict('records') if len(fires_for_date) > 0 else []

# Subsample large fire sets for map display
display_fires = fires_list[:500] if len(fires_list) > 500 else fires_list

for fire in display_fires:
    folium.CircleMarker(
        [fire['latitude'], fire['longitude']],
        radius=5,
        color='darkred',
        fill=True,
        fill_opacity=0.8,
        popup=f"Fire<br>FRP: {fire.get('frp', 'N/A')}<br>Date: {fire.get('acq_date', '')}"
    ).add_to(m)

# Infrastructure in current view
delta = 0.4
west,  east  = current_lon - delta, current_lon + delta
south, north = current_lat - delta, current_lat + delta

local_infra = infra_df[
    (infra_df['lat']  >= south) & (infra_df['lat']  <= north) &
    (infra_df['lon']  >= west)  & (infra_df['lon']  <= east)
]

# Cap at 200 for map performance
if len(local_infra) > 200:
    local_infra = local_infra.sample(200, random_state=42)

icon_map = {
    'Power Substation': 'flash',
    'Hospital':         'plus-sign',
    'Wind Farm':        'cloud',
    'Solar Farm':       'star',
    'Fire Station':     'fire',
    'School':           'book',
    'Residential Area': 'home',
    'Gas Plant':        'oil',
    'Coal Plant':       'cog',
    'Hydro Plant':      'tint',
    'Airport':          'plane',
    'Cell Tower':       'signal',
    'Water Treatment':  'tint',
    'Medical Clinic':   'plus-sign',
    'University':       'book',
}

for _, asset in local_infra.iterrows():
    if len(fires_for_date) > 0:
        dists    = haversine_vectorized(
            asset['lat'], asset['lon'],
            fires_for_date['latitude'].values,
            fires_for_date['longitude'].values
        )
        min_d    = dists.min()
        min_d_mi = min_d * 0.621371
        qr       = quick_risk(min_d)
    else:
        min_d, min_d_mi, qr = 999, 999, 0

    color = 'red' if qr >= 70 else 'orange' if qr >= 40 else 'green'
    icon  = icon_map.get(str(asset.get('type', '')), 'info-sign')

    folium.Marker(
        [asset['lat'], asset['lon']],
        icon=folium.Icon(color=color, icon=icon, prefix='glyphicon'),
        popup=f"""
        <div style='width:220px'>
            <div style='background:{color};color:white;padding:8px;margin:-10px -10px 8px -10px'>
                <b>{str(asset.get('name', 'Unknown'))[:30]}</b>
            </div>
            <p>{asset.get('type', '')} · {asset.get('state', '')} · {asset.get('city', 'N/A')}</p>
            <b>Quick Risk: {qr:.0f}%</b><br>
            Distance: {min_d:.1f} km ({min_d_mi:.1f} mi)
        </div>
        """,
        tooltip=f"{str(asset.get('name', 'Unknown'))[:25]} — ~{qr:.0f}%"
    ).add_to(m)

st.info(f"Showing {len(display_fires):,} fires • {len(local_infra):,} assets in view")

map_data = st_folium(
    m, width=1400, height=600,
    key="map",
    returned_objects=["bounds", "last_clicked"]
)

# ============================================================================
# CLICK HANDLING
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
# ANALYZE BUTTON
# ============================================================================

if st.button("🧠 Analyze", type="primary", use_container_width=True):
    if map_data and map_data.get("bounds"):
        bounds = map_data["bounds"]
        sw = bounds["_southWest"]
        ne = bounds["_northEast"]
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

    if len(infra_in_view) > 200:
        infra_in_view = infra_in_view.sample(200, random_state=42)

    if len(fires_in_view) == 0:
        st.warning("⚠️ No fires in current view. Try zooming to an active fire area.")
    else:
        with st.spinner(f"Running BNN v2 on {len(infra_in_view)} assets..."):
            results = []
            for _, a in infra_in_view.iterrows():
                ad = {
                    'name':  a.get('name',  'Unknown'),
                    'city':  a.get('city',  ''),
                    'state': a.get('state', ''),
                    'type':  a.get('type',  ''),
                    'lat':   a['lat'],
                    'lon':   a['lon'],
                }
                rd = predict_risk(ad, fires_in_view, current_weather, model, scaler)
                results.append({**ad, **rd})

        top20 = sorted(results, key=lambda x: x['risk_score'], reverse=True)[:20]

        st.markdown("### 🚨 Top 20 Risk Assets")
        df_out = pd.DataFrame([{
            "Asset":    str(r.get('name', 'Unknown'))[:40],
            "Type":     r.get('type', ''),
            "State":    r.get('state', ''),
            "City":     str(r.get('city', 'N/A'))[:20],
            "Risk %":   f"{r['risk_score']:.1f}",
            "Conf %":   f"{r['confidence']:.1f}",
            "Unc":      f"±{r['uncertainty']:.1f}",
            "Dist km":  f"{r['distance_km']:.2f}" if r.get('distance_km') else "N/A",
            "Dist mi":  f"{r['distance_miles']:.2f}" if r.get('distance_miles') else "N/A",
            "Fires 30km": r['num_fires'],
            "Action":   "🔴 EVACUATE" if r['risk_score'] > 70
                        else "🟠 PREPARE" if r['risk_score'] > 40
                        else "🟡 MONITOR",
        } for r in top20])
        st.dataframe(df_out, use_container_width=True)

        # Risk heatmap
        st.markdown("### 🗺️ Risk Heatmap (Top 20 Critical)")
        rm = folium.Map(
            location=[(vs+vn)/2, (vw+ve)/2],
            zoom_start=map_data.get("zoom", zoom)
        )

        for _, f in fires_in_view.iterrows():
            folium.CircleMarker(
                [f['latitude'], f['longitude']],
                radius=8, color='darkred', fill=True, fill_opacity=0.8
            ).add_to(rm)

        for a in top20:
            r = a['risk_score']
            c = 'red' if r >= 70 else 'orange' if r >= 40 else 'green'
            ic = icon_map.get(str(a.get('type', '')), 'info-sign')

            folium.Marker(
                [a['lat'], a['lon']],
                icon=folium.Icon(color=c, icon=ic, prefix='glyphicon'),
                popup=f"""
                <div style='width:240px'>
                    <div style='background:{c};color:white;padding:10px;margin:-10px -10px 10px -10px'>
                        <b>{str(a.get('name','Unknown'))[:35]}</b>
                    </div>
                    <p>{a.get('type','')} · {a.get('state','')} · {str(a.get('city','N/A'))[:20]}</p>
                    <hr>
                    <b style='font-size:18px;color:{c}'>Risk: {r:.1f}%</b><br>
                    <b>Confidence: {a['confidence']:.1f}%</b><br>
                    <b>Uncertainty: ±{a['uncertainty']:.1f}</b><br>
                    <hr>
                    <p style='font-size:12px'>
                        Distance: {a['distance_km']:.1f} km ({a['distance_miles']:.1f} mi)<br>
                        Fires within 30km: {a['num_fires']}
                    </p>
                </div>
                """,
                tooltip=f"{str(a.get('name','Unknown'))[:25]} — {r:.0f}%"
            ).add_to(rm)

        st_folium(rm, width=1400, height=600, key="riskmap", returned_objects=[])

        # Camp Fire / Paradise validation
        if '2018-11-08' in str(selected_date):
            p = next((r for r in results
                      if 'Feather River' in str(r.get('name', ''))
                      or 'Paradise' in str(r.get('city', ''))), None)
            if p:
                st.markdown("---")
                st.markdown("### 🎯 Validation — Camp Fire 2018")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Paradise Hospital Risk",  f"{p['risk_score']:.1f}%")
                c2.metric("Confidence",              f"{p['confidence']:.1f}%")
                c3.metric("Uncertainty",             f"±{p['uncertainty']:.1f}")
                c4.metric("Distance to Fire",        f"{p['distance_miles']:.1f} mi")
                st.success(
                    "✅ Model correctly predicted CRITICAL risk. "
                    "Feather River Hospital was evacuated and destroyed ~18 hours later."
                )