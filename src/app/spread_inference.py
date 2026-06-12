"""
spread_inference.py
====================
Loads TFT spread model + BNN risk model and runs inference.
Returns spread polygons at T+6h/12h/24h and infrastructure risk scores.

Used by streamlit_app_v3.py
"""

import json
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from spread_tft_model import SpreadTFT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEATHER_FEATURES = [
    "wind_speed_kmh", "wind_direction", "temperature_c", "humidity",
    "max_wind_kmh", "wind_speed_trend", "temp_trend", "humidity_trend",
    "drought_index",
]
STATIC_FEATURES = [
    "current_area_km2", "fuel_multiplier", "slope_proxy",
    "frp_mean", "frp_max", "n_detections", "fire_density", "frp_trend",
    "slope_wind_interaction", "wind_x_fuel", "heat_dryness",
]
BNN_FEATURES = [
    "min_distance_km", "mean_distance_km", "num_fires_30km", "max_frp",
    "wind_speed_kmh", "wind_direction", "temperature_c", "humidity",
    "wind_fire_alignment", "drought_index", "days_since_rain",
]

HORIZONS_H  = [6, 12, 24]
MC_SAMPLES  = 30

# Vulnerability weights per asset type
VULN = {
    "Power Substation": 65, "Wind Farm": 70, "Solar Farm": 60,
    "Gas Plant": 55,        "Coal Plant": 55, "Hydro Plant": 50,
    "Hospital": 45,         "Fire Station": 40, "School": 35,
    "Airport": 45,          "Water Treatment": 40, "Cell Tower": 30,
}

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class SpreadInference:

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._load_tft()
        self._load_bnn()

    def _load_tft(self):
        meta_path   = self.model_dir / "spread_tft_metadata.json"
        scaler_path = self.model_dir / "spread_scaler.pkl"
        model_path  = self.model_dir / "spread_tft_best.pt"

        with open(meta_path) as f:
            self.tft_meta = json.load(f)

        self.tft_scaler = joblib.load(scaler_path)
        self.tft_all_features = self.tft_meta["all_features"]
        self.growth_clip = self.tft_meta["growth_clip_km2h"]

        self.tft = SpreadTFT(
            n_features    = self.tft_meta["n_features"],
            d_model       = self.tft_meta["architecture"]["d_model"],
            n_heads       = self.tft_meta["architecture"]["n_heads"],
            n_lstm_layers = self.tft_meta["architecture"]["n_lstm_layers"],
            dropout       = 0.0,
        )
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        self.tft.load_state_dict(state)
        self.tft.eval()

    def _load_bnn(self):
        import tensorflow as tf
        self.bnn = tf.keras.models.load_model(
            str(self.model_dir / "bayesian_risk_model_v2.keras"),
            compile=False)
        self.bnn_scaler = joblib.load(
            self.model_dir / "feature_scaler_v2.pkl")

    # -----------------------------------------------------------------------
    # TFT spread prediction
    # -----------------------------------------------------------------------

    def predict_spread(self, fire_lat, fire_lon, fire_area_km2,
                       weather: dict, firms_context: dict) -> dict:
        """
        Returns spread ellipse parameters at T+6h, T+12h, T+24h.

        weather: {wind_speed_kmh, wind_direction, temperature_c, humidity,
                  max_wind_kmh, wind_speed_trend, temp_trend,
                  humidity_trend, drought_index}
        firms_context: {frp_mean, frp_max, n_detections,
                        fire_density, frp_trend}
        """
        slope = float(np.clip((fire_lat - 32.0) / 12.0, 0, 1))

        feat_map = {
            "wind_speed_kmh":      weather.get("wind_speed_kmh", 10.0),
            "wind_direction":      weather.get("wind_direction", 270.0),
            "temperature_c":       weather.get("temperature_c", 25.0),
            "humidity":            weather.get("humidity", 40.0),
            "max_wind_kmh":        weather.get("max_wind_kmh",
                                   weather.get("wind_speed_kmh", 10.0)),
            "wind_speed_trend":    weather.get("wind_speed_trend", 0.0),
            "temp_trend":          weather.get("temp_trend", 0.0),
            "humidity_trend":      weather.get("humidity_trend", 0.0),
            "drought_index":       weather.get("drought_index", 0.5),
            "current_area_km2":    fire_area_km2,
            "fuel_multiplier":     1.0,
            "slope_proxy":         slope,
            "frp_mean":            firms_context.get("frp_mean", 0.0),
            "frp_max":             firms_context.get("frp_max", 0.0),
            "n_detections":        firms_context.get("n_detections", 0.0),
            "fire_density":        firms_context.get("fire_density", 0.0),
            "frp_trend":           firms_context.get("frp_trend", 0.0),
            "slope_wind_interaction": slope * weather.get("wind_speed_kmh",
                                                          10.0) / 50.0,
            "wind_x_fuel":         weather.get("wind_speed_kmh", 10.0) * 1.0,
            "heat_dryness":        max(0, weather.get("temperature_c", 25.0)
                                       - 25) *
                                   max(0, 60 - weather.get("humidity", 40.0))
                                   / 100,
        }

        X = np.array([[feat_map.get(f, 0.0) for f in self.tft_all_features]],
                     dtype=np.float32)
        X = np.nan_to_num(
            self.tft_scaler.transform(X), nan=0.0,
            posinf=3.0, neginf=-3.0).astype(np.float32)

        with torch.no_grad():
            pred = self.tft(torch.tensor(X)).numpy()[0]  # (3, 3)

        p10 = np.expm1(pred[:, 0]).clip(0, self.growth_clip)
        p50 = np.expm1(pred[:, 1]).clip(0, self.growth_clip)
        p90 = np.expm1(pred[:, 2]).clip(0, self.growth_clip)

        # No FIRMS signal → elevated uncertainty flag
        no_firms = firms_context.get("n_detections", 0) == 0

        # Rothermel spread direction from wind
        wind_dir    = weather.get("wind_direction", 270.0)
        spread_dir  = (wind_dir + 180) % 360  # fire spreads downwind

        # Damped displacement at each horizon
        km_per_lat = 111.0
        km_per_lon = 111.0 * np.cos(np.radians(fire_lat))
        move_rad   = np.radians(spread_dir)

        horizons = []
        for i, h in enumerate(HORIZONS_H):
            # P50 → spread distance (km)
            tau      = 12.0
            disp_p50 = p50[i] * tau * (1 - np.exp(-h / tau))
            disp_p90 = p90[i] * tau * (1 - np.exp(-h / tau))

            # Center of spread ellipse
            center_lat = float(np.clip(
                fire_lat + (disp_p50 * np.cos(move_rad)) / km_per_lat,
                24.0, 50.0))
            center_lon = float(np.clip(
                fire_lon + (disp_p50 * np.sin(move_rad)) / km_per_lon,
                -125.0, -100.0))

            # Radius from fire origin at P50 and P90
            radius_p50_km = float(disp_p50)
            radius_p90_km = float(disp_p90)

            # Confidence degrades with horizon and no-FIRMS
            base_conf = max(0, 1 - h / 48.0)
            confidence = base_conf * (0.6 if no_firms else 1.0)

            horizons.append({
                "horizon_h":       h,
                "center_lat":      center_lat,
                "center_lon":      center_lon,
                "spread_dir_deg":  spread_dir,
                "radius_p50_km":   radius_p50_km,
                "radius_p90_km":   max(radius_p90_km, radius_p50_km + 1),
                "growth_p10":      round(float(p10[i]), 4),
                "growth_p50":      round(float(p50[i]), 4),
                "growth_p90":      round(float(p90[i]), 4),
                "confidence":      round(confidence, 2),
                "no_firms_warning":no_firms,
            })

        return {
            "fire_lat":    fire_lat,
            "fire_lon":    fire_lon,
            "spread_dir":  spread_dir,
            "wind_speed":  weather.get("wind_speed_kmh", 0),
            "horizons":    horizons,
            "no_firms":    no_firms,
        }

    # -----------------------------------------------------------------------
    # BNN infrastructure risk
    # -----------------------------------------------------------------------

    def score_assets(self, assets_df: pd.DataFrame,
                     fires_df: pd.DataFrame,
                     weather: dict,
                     projected_fire_lat: float = None,
                     projected_fire_lon: float = None) -> pd.DataFrame:
        """
        Score infrastructure assets using BNN.
        If projected_fire_lat/lon provided, uses projected position.
        Otherwise uses current fire positions.
        """
        if assets_df.empty or fires_df.empty:
            return pd.DataFrame()

        use_projected = (projected_fire_lat is not None and
                         projected_fire_lon is not None)

        results = []
        for _, asset in assets_df.iterrows():
            alat = float(asset["lat"])
            alon = float(asset["lon"])

            if use_projected:
                # Distance to projected fire center
                dlat = np.radians(projected_fire_lat - alat)
                dlon = np.radians(projected_fire_lon - alon)
                a = (np.sin(dlat/2)**2 +
                     np.cos(np.radians(alat)) *
                     np.cos(np.radians(projected_fire_lat)) *
                     np.sin(dlon/2)**2)
                min_dist = float(6371 * 2 * np.arcsin(np.sqrt(a)))
                mean_dist = min_dist * 1.3
                num_fires = 1
                max_frp = float(fires_df["frp"].max()) if "frp" in fires_df else 50.0
                wa = self._wind_alignment(projected_fire_lat,
                                          projected_fire_lon,
                                          alat, alon,
                                          weather.get("wind_direction", 270.0))
            else:
                fire_lats = fires_df["latitude"].values if "latitude" in fires_df.columns else fires_df["lat"].values
                fire_lons = fires_df["longitude"].values if "longitude" in fires_df.columns else fires_df["lon"].values
                dists = self._haversine_vec(alat, alon, fire_lats, fire_lons)
                min_dist  = float(dists.min())
                mean_dist = float(dists.mean())
                num_fires = int((dists < 30).sum())
                max_frp   = float(fires_df["frp"].max()) if "frp" in fires_df.columns else 50.0
                cf_idx    = dists.argmin()
                wa        = self._wind_alignment(
                    fire_lats[cf_idx], fire_lons[cf_idx],
                    alat, alon, weather.get("wind_direction", 270.0))

            feat = np.array([[
                min_dist,
                mean_dist,
                num_fires,
                max_frp,
                weather.get("wind_speed_kmh", 10.0),
                weather.get("wind_direction", 270.0),
                weather.get("temperature_c", 25.0),
                weather.get("humidity", 40.0),
                wa,
                weather.get("drought_index", 0.5),
                weather.get("days_since_rain", 7.0),
            ]], dtype=np.float32)

            feat_scaled = self.bnn_scaler.transform(feat).astype(np.float32)
            preds = np.array([
                self.bnn(feat_scaled, training=True).numpy().flatten()[0]
                for _ in range(MC_SAMPLES)
            ])
            mean_risk = float(np.clip(preds.mean(), 0, 100))
            unc       = float(preds.std())

            results.append({
                "name":        str(asset.get("name", "Unknown"))[:35],
                "type":        str(asset.get("type", "")),
                "lat":         alat,
                "lon":         alon,
                "risk_score":  round(mean_risk, 1),
                "uncertainty": round(unc, 1),
                "min_dist_km": round(min_dist, 1),
                "risk_level":  ("HIGH"   if mean_risk > 70 else
                                "MEDIUM" if mean_risk > 40 else "LOW"),
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("risk_score", ascending=False)
        return df

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _haversine_vec(lat1, lon1, lats2, lons2):
        R = 6371
        la1 = np.radians(lat1); lo1 = np.radians(lon1)
        la2 = np.radians(np.asarray(lats2, float))
        lo2 = np.radians(np.asarray(lons2, float))
        a = (np.sin((la2-la1)/2)**2 +
             np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2)
        return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    @staticmethod
    def _wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
        la1,lo1 = np.radians(fire_lat),  np.radians(fire_lon)
        la2,lo2 = np.radians(asset_lat), np.radians(asset_lon)
        dlon = lo2 - lo1
        x = np.sin(dlon)*np.cos(la2)
        y = np.cos(la1)*np.sin(la2) - np.sin(la1)*np.cos(la2)*np.cos(dlon)
        bearing     = (np.degrees(np.arctan2(x,y)) + 360) % 360
        wind_toward = (wind_dir + 180) % 360
        diff        = abs(wind_toward - bearing)
        if diff > 180: diff = 360 - diff
        return float(np.cos(np.radians(diff)))