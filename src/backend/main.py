"""
main.py  (v2)
==============
FastAPI backend for the national multi-hazard risk system.

Endpoints:
  GET  /health
  GET  /fires          — live FIRMS fire detections
  GET  /assets         — infrastructure assets in bbox
  POST /risk           — full risk assessment
  GET  /scenario/{id}  — pre-built back-test scenarios
  GET  /states         — list supported states + bboxes

Run:
  uvicorn src.backend.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from backend.ingest.firms          import fetch_fires
from backend.ingest.weather        import fetch_weather_summary
from backend.ingest.infrastructure import fetch_assets_from_csv
from backend.ingest.osm            import fetch_assets as fetch_osm_assets
from backend.model.bayesian_risk_model import compute_asset_risk
from backend.cascade.impact        import compute_cascade_impacts

app = FastAPI(
    title="Multi-Hazard Infrastructure Risk API",
    version="2.0.0",
    description="Probabilistic wildfire risk for 10-state western US infrastructure",
)

SCENARIO_DIR = Path(__file__).resolve().parent / "data_cache" / "scenarios"

# Supported states with bboxes [west, south, east, north]
STATES = {
    "CA": (-124.5, 32.5,  -114.0, 42.0),
    "OR": (-124.6, 41.9,  -116.5, 46.3),
    "WA": (-124.7, 45.5,  -116.9, 49.0),
    "MT": (-116.0, 44.4,  -104.0, 49.0),
    "CO": (-109.1, 36.9,  -102.0, 41.0),
    "ID": (-117.2, 42.0,  -111.0, 49.0),
    "WY": (-111.1, 40.9,  -104.0, 45.0),
    "UT": (-114.1, 36.9,  -109.0, 42.0),
    "NV": (-120.0, 35.0,  -114.0, 42.0),
    "AZ": (-114.8, 31.3,  -109.0, 37.0),
}

WESTERN_US_BBOX = "-125.0,31.0,-102.0,49.5"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RiskRequest(BaseModel):
    bbox:                     str   = Field(..., description="west,south,east,north")
    horizon_hours:            int   = Field(default=24, ge=1,  le=72)
    firms_days:               int   = Field(default=1,  ge=1,  le=10)
    fire_source:              str   = Field(default="VIIRS_NOAA20_NRT")
    fire_confidence_threshold: float = Field(default=0.0, ge=0.0, le=100.0)
    weather_source:           str   = Field(default="openmeteo")
    asset_types:              Optional[List[str]] = None
    states:                   Optional[List[str]] = None
    max_assets:               int   = Field(default=2000, ge=1, le=5000)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict:
    return {"ok": True, "version": "2.0.0"}


@app.get("/states")
def list_states() -> Dict:
    return {
        "states": [
            {
                "code": code,
                "bbox": f"{w},{s},{e},{n}",
            }
            for code, (w, s, e, n) in STATES.items()
        ]
    }


@app.get("/fires")
def fires(
    bbox:           str   = Query(default=WESTERN_US_BBOX,
                                  description="west,south,east,north"),
    days:           int   = Query(default=1),
    source:         str   = Query(default="VIIRS_NOAA20_NRT"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=100.0),
) -> Dict:
    try:
        points = fetch_fires(
            bbox=bbox, days=days,
            source=source, min_confidence=min_confidence,
        )
    except Exception as exc:
        raise HTTPException(502, f"FIRMS fetch failed: {exc}")

    return {
        "bbox":           bbox,
        "count":          len(points),
        "source":         source,
        "min_confidence": min_confidence,
        "fires":          points,
    }


@app.get("/assets")
def assets(
    bbox:   str = Query(..., description="west,south,east,north"),
    states: str = Query(default=None, description="comma-separated state codes"),
) -> Dict:
    state_list = [s.strip().upper() for s in states.split(",")] \
                 if states else None
    try:
        results = fetch_assets_from_csv(bbox=bbox, states=state_list)
    except Exception as exc:
        raise HTTPException(502, f"Asset fetch failed: {exc}")

    return {"bbox": bbox, "count": len(results), "assets": results}


@app.post("/risk")
def risk(req: RiskRequest) -> Dict:
    # ── Fetch fires ───────────────────────────────────────────────────────
    try:
        fire_list = fetch_fires(
            bbox=req.bbox,
            days=req.firms_days,
            source=req.fire_source,
            min_confidence=req.fire_confidence_threshold,
        )
    except Exception as exc:
        raise HTTPException(502, f"Fire data error: {exc}")

    # ── Fetch assets ──────────────────────────────────────────────────────
    try:
        asset_list = fetch_assets_from_csv(
            bbox=req.bbox,
            asset_types=req.asset_types,
            states=req.states,
            max_assets=req.max_assets,
        )
    except Exception as exc:
        raise HTTPException(502, f"Asset data error: {exc}")

    # ── Fetch weather ─────────────────────────────────────────────────────
    west, south, east, north = [float(v) for v in req.bbox.split(",")]
    center_lat = (south + north) / 2.0
    center_lon = (west  + east)  / 2.0

    try:
        weather = fetch_weather_summary(
            lat=center_lat,
            lon=center_lon,
            hours=req.horizon_hours,
            source=req.weather_source,
        )
    except Exception as exc:
        raise HTTPException(502, f"Weather error: {exc}")

    # ── Score assets ──────────────────────────────────────────────────────
    scored_assets: List[Dict] = []
    risks_by_id:   Dict[str, Dict] = {}

    for asset in asset_list:
        result = compute_asset_risk(
            asset=asset, fires=fire_list, weather=weather
        )
        # Apply vulnerability weight — more critical assets get a bump
        vuln   = float(asset.get("vulnerability_weight", 1.0))
        raw    = result["risk_score"]
        result["risk_score"] = round(min(raw * vuln, 1.0), 4)

        merged = {**asset, **result}
        scored_assets.append(merged)
        risks_by_id[asset["id"]] = result

    # ── Cascade impacts ───────────────────────────────────────────────────
    cascade = compute_cascade_impacts(
        assets=asset_list,
        risks_by_asset_id=risks_by_id,
        fires=fire_list,
    )

    # ── Summary tier counts ───────────────────────────────────────────────
    tiers = {"high": 0, "medium": 0, "low": 0}
    for a in scored_assets:
        tiers[a.get("risk_bucket", "low")] += 1

    return {
        "bbox":                    req.bbox,
        "horizon_hours":           req.horizon_hours,
        "weather":                 weather,
        "weather_source":          weather.get("weather_source", req.weather_source),
        "fire_count":              len(fire_list),
        "asset_count":             len(asset_list),
        "fire_confidence_threshold": req.fire_confidence_threshold,
        "risk_tiers":              tiers,
        "assets":                  scored_assets,
        "cascade":                 cascade,
    }


@app.get("/scenario/{scenario_id}")
def scenario(scenario_id: str) -> Dict:
    path = SCENARIO_DIR / f"{scenario_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Scenario '{scenario_id}' not found. "
                                 f"Available: camp-fire-2018")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)