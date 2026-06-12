"""
bayesian_risk_model.py  (v2)
==============================
Bayesian NN inference module for the national multi-hazard risk system.

Changes vs v1:
  - 11 features (adds drought_index, days_since_rain)
  - Wider architecture: 256-128-64-32
  - Loads v2 weights + scaler
  - Decision layer with action recommendations
  - Hazard attribution placeholder (wildfire only in Phase 1)

Used by: src/backend/main.py (FastAPI /risk endpoint)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"

FEATURE_COLS = [
    "min_distance_km",
    "mean_distance_km",
    "num_fires_30km",
    "max_frp",
    "wind_speed_kmh",
    "wind_direction",
    "temperature_c",
    "humidity",
    "wind_fire_alignment",
    "drought_index",
    "days_since_rain",
]

MC_SAMPLES = 50

# Decision table (matches config.yaml + problem statement)
# (min_risk, max_risk, min_conf, action)
DECISION_TABLE = [
    (90, 100, 0.0,  "Immediate action"),
    (75,  90, 0.6,  "Prepare shutdown"),
    (50,  75, 0.6,  "Pre-position crews"),
    (50,  75, 0.0,  "Monitor + flag"),
    ( 0,  50, 0.0,  "Monitor"),
]

# ---------------------------------------------------------------------------
_model  = None
_scaler = None


def _build_model():
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(len(FEATURE_COLS),))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64,  activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32,  activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _get_model():
    global _model, _scaler
    if _model is not None:
        return _model, _scaler

    try:
        print("Loading BNN v2...")
        _model = _build_model()
        _model.load_weights(str(MODEL_DIR / "bayesian_risk_model_v2.weights.h5"))
        _scaler = joblib.load(str(MODEL_DIR / "feature_scaler_v2.pkl"))
        print(f"  ✓ BNN v2 loaded from {MODEL_DIR}")
    except Exception as exc:
        print(f"  ✗ Model load failed: {exc} — using heuristic fallback")
        _model  = "FAILED"
        _scaler = None

    return _model, _scaler


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lats2, lons2):
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats2 = np.radians(np.asarray(lats2, dtype=float))
    lons2 = np.radians(np.asarray(lons2, dtype=float))
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lats2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
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


# ---------------------------------------------------------------------------
# Fallback heuristic (used when model weights not found)
# ---------------------------------------------------------------------------

def _heuristic(min_dist, num_fires, wind_speed, wind_align):
    if   min_dist < 1:   base = 95
    elif min_dist < 5:   base = 75
    elif min_dist < 15:  base = 45
    elif min_dist < 30:  base = 25
    elif min_dist < 100: base = 10
    else:                base = 3

    wind_factor = 1.0 + max(0, wind_speed - 20) / 40
    if wind_align > 0.5:
        wind_factor *= 1.2

    return float(min(base * wind_factor * (1 + min(num_fires/30, 0.3)), 100))


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------

def _get_action(risk_score: float, confidence: float) -> str:
    for min_r, max_r, min_c, action in DECISION_TABLE:
        if min_r <= risk_score < max_r and confidence >= min_c:
            return action
    return "Monitor"


def _risk_bucket(score: float) -> str:
    if score >= 75: return "high"
    if score >= 40: return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def compute_asset_risk(asset: Dict, fires: List[Dict], weather: Dict) -> Dict:
    """
    Compute probabilistic wildfire risk for one asset.

    Returns:
        risk_score      : 0.0 – 1.0  (normalised)
        confidence      : 0.0 – 1.0
        uncertainty     : raw std in risk points
        risk_bucket     : "low" | "medium" | "high"
        action          : recommended operational action
        hazard          : "wildfire" (Phase 1 only)
        model_used      : "bayesian_nn_v2" | "heuristic"
        features        : dict of computed input features
    """
    if not fires:
        return {
            "asset_id":   asset["id"],
            "risk_score": 0.0,
            "confidence": 1.0,
            "uncertainty": 0.0,
            "risk_bucket": "low",
            "action":      "Monitor",
            "hazard":      "wildfire",
            "model_used":  "bayesian_nn_v2",
            "features":    {},
        }

    # ── Geometry ─────────────────────────────────────────────────────────
    fire_lats = np.array([f["lat"] for f in fires])
    fire_lons = np.array([f["lon"] for f in fires])
    distances = _haversine(asset["lat"], asset["lon"], fire_lats, fire_lons)

    min_dist   = float(distances.min())
    mean_dist  = float(distances.mean())
    num_nearby = int((distances < 30).sum())
    max_frp    = float(max(
        f.get("frp") or f.get("brightness") or 50
        for f in fires
    ))

    closest_fire = fires[int(distances.argmin())]
    wind_dir     = float(weather.get("wind_direction_deg", 180))
    w_align      = _wind_alignment(
        closest_fire["lat"], closest_fire["lon"],
        asset["lat"], asset["lon"],
        wind_dir
    )

    # Drought proxy from weather (precipitation rolling deficit)
    # If not pre-computed, default to moderate drought
    drought_idx  = float(weather.get("drought_index",   0.5))
    days_since_r = float(weather.get("days_since_rain", 7.0))

    features_dict = {
        "min_dist_to_fire_km": round(min_dist,   2),
        "mean_dist_km":        round(mean_dist,  2),
        "num_fires_30km":      num_nearby,
        "max_frp":             round(max_frp,    2),
        "wind_speed_kmh":      round(float(weather.get("wind_speed_kmh", 15)), 2),
        "wind_alignment":      round(w_align,    3),
        "drought_index":       round(drought_idx, 3),
        "days_since_rain":     round(days_since_r, 1),
    }

    # ── ML inference ─────────────────────────────────────────────────────
    model, scaler = _get_model()

    if model != "FAILED" and model is not None:
        try:
            feat_df = pd.DataFrame([{
                "min_distance_km":     min_dist,
                "mean_distance_km":    mean_dist,
                "num_fires_30km":      num_nearby,
                "max_frp":             max_frp,
                "wind_speed_kmh":      float(weather.get("wind_speed_kmh",    15)),
                "wind_direction":      wind_dir,
                "temperature_c":       float(weather.get("temperature_c",     20)),
                "humidity":            float(weather.get("humidity_pct",      50)),
                "wind_fire_alignment": w_align,
                "drought_index":       drought_idx,
                "days_since_rain":     days_since_r,
            }])

            feat_scaled = scaler.transform(feat_df).astype(np.float32)

            preds = np.array([
                model(feat_scaled, training=True).numpy()[0, 0]
                for _ in range(MC_SAMPLES)
            ])

            risk_raw = float(np.clip(preds.mean(), 0, 100))
            unc      = float(preds.std())
            conf     = float(1.0 - min(unc / 30.0, 1.0))

            return {
                "asset_id":    asset["id"],
                "risk_score":  round(risk_raw / 100, 4),
                "confidence":  round(conf, 4),
                "uncertainty": round(unc, 2),
                "risk_bucket": _risk_bucket(risk_raw),
                "action":      _get_action(risk_raw, conf),
                "hazard":      "wildfire",
                "model_used":  "bayesian_nn_v2",
                "features":    features_dict,
            }

        except Exception as exc:
            print(f"  BNN inference failed: {exc} — falling back to heuristic")

    # ── Heuristic fallback ────────────────────────────────────────────────
    risk_raw = _heuristic(
        min_dist, num_nearby,
        float(weather.get("wind_speed_kmh", 15)),
        w_align
    )
    conf = 0.55

    return {
        "asset_id":    asset["id"],
        "risk_score":  round(risk_raw / 100, 4),
        "confidence":  round(conf, 4),
        "uncertainty": 15.0,
        "risk_bucket": _risk_bucket(risk_raw),
        "action":      _get_action(risk_raw, conf),
        "hazard":      "wildfire",
        "model_used":  "heuristic",
        "features":    features_dict,
    }