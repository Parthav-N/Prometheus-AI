"""
infrastructure.py  (v2)
========================
Loads national infrastructure assets from CSV with bbox + state filtering.

Key improvement: Solar Farm and Wind Farm entries are filtered to
utility-scale only (EIA-sourced OR capacity_mw >= 1.0 MW).
This removes parking lot solar canopies and individual wind turbines
that OSM tags identically to utility-scale installations.
"""

from __future__ import annotations

import pandas as pd
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

ROOT       = Path(__file__).resolve().parents[3]
INFRA_FILE = ROOT / "data" / "infrastructure" / "national_infrastructure.csv"

VULN_WEIGHTS: Dict[str, float] = {
    "Power Substation":  1.0,
    "Power Plant":       1.0,
    "Wind Farm":         0.9,
    "Solar Farm":        0.7,
    "Gas Plant":         0.95,
    "Coal Plant":        0.9,
    "Hydro Plant":       0.85,
    "Nuclear Plant":     1.0,
    "Hospital":          1.0,
    "Medical Clinic":    0.85,
    "Nursing Home":      0.9,
    "Fire Station":      0.9,
    "Police Station":    0.8,
    "School":            0.8,
    "University":        0.75,
    "Water Tower":       0.75,
    "Water Treatment":   0.85,
    "Cell Tower":        0.7,
    "Airport":           0.85,
    "Residential Area":  0.6,
}


@lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    """Load full national infrastructure CSV once, cache in memory."""
    df = pd.read_csv(INFRA_FILE, low_memory=False)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure required columns exist
    for col in ["id", "lat", "lon", "type", "name", "state"]:
        if col not in df.columns:
            df[col] = None

    df["lat"]         = pd.to_numeric(df["lat"],         errors="coerce")
    df["lon"]         = pd.to_numeric(df["lon"],         errors="coerce")
    df["capacity_mw"] = pd.to_numeric(
        df["capacity_mw"] if "capacity_mw" in df.columns else None,
        errors="coerce"
    )
    df = df.dropna(subset=["lat", "lon"])

    before = len(df)

    # ── Filter Solar Farm + Wind Farm to utility-scale only ──────────────
    # OSM tags parking lot solar canopies and individual turbines with the
    # same tags as utility-scale installations. We keep only:
    #   - EIA-sourced entries (always utility-scale)
    #   - OSM entries that explicitly declare capacity_mw >= 1.0 MW
    # Everything else (Walmart canopies, single turbines) is dropped.
    renewable_mask = df["type"].isin(["Solar Farm", "Wind Farm"])
    non_renewable  = df[~renewable_mask]
    renewable      = df[renewable_mask]

    utility_scale = renewable[
        (renewable.get("source", pd.Series(dtype=str)) == "EIA") |
        (renewable["capacity_mw"] >= 1.0)
    ] if "source" in renewable.columns else renewable[
        renewable["capacity_mw"] >= 1.0
    ]

    df      = pd.concat([non_renewable, utility_scale], ignore_index=True)
    dropped = before - len(df)

    print(f"  ✓ Infrastructure loaded  : {len(df):,} assets")
    print(f"  ✓ Sub-utility removed    : {dropped:,} "
          f"(parking lot solar canopies, single turbines)")
    return df


def fetch_assets_from_csv(
    bbox:        str,
    asset_types: Optional[List[str]] = None,
    states:      Optional[List[str]] = None,
    max_assets:  int = 2000,
) -> List[Dict]:
    """
    Return infrastructure assets within bounding box.

    Args:
        bbox        : "west,south,east,north"
        asset_types : optional list of types to filter
        states      : optional list of state codes e.g. ["CA", "OR"]
        max_assets  : cap to avoid sending too many to the risk engine
    """
    west, south, east, north = [float(v) for v in bbox.split(",")]
    df = _load_df()

    mask = (
        (df["lat"] >= south) & (df["lat"] <= north) &
        (df["lon"] >= west)  & (df["lon"] <= east)
    )
    filtered = df[mask].copy()

    if asset_types:
        filtered = filtered[filtered["type"].isin(asset_types)]

    if states:
        filtered = filtered[filtered["state"].isin(states)]

    # Cap results — prioritise critical infrastructure
    if len(filtered) > max_assets:
        priority = [
            "Power Substation", "Power Plant", "Wind Farm", "Gas Plant",
            "Hospital", "Fire Station", "Nuclear Plant", "Water Treatment",
        ]
        p_mask  = filtered["type"].isin(priority)
        p_df    = filtered[p_mask]
        rest_df = filtered[~p_mask]

        n_rest = max(0, max_assets - len(p_df))
        if n_rest > 0:
            rest_df = rest_df.sample(
                n=min(n_rest, len(rest_df)), random_state=42
            )
        filtered = pd.concat([p_df, rest_df], ignore_index=True)

    assets = []
    for idx, row in filtered.iterrows():
        vuln = float(
            row.get("vulnerability_weight") or
            VULN_WEIGHTS.get(str(row["type"]), 0.7)
        )
        assets.append({
            "id":                   str(row["id"]) if row["id"] else f"{row['type']}_{idx}",
            "lat":                  float(row["lat"]),
            "lon":                  float(row["lon"]),
            "asset_type":           str(row["type"]),
            "name":                 str(row.get("name") or row["type"]),
            "city":                 str(row.get("city") or ""),
            "state":                str(row.get("state") or ""),
            "vulnerability_weight": vuln,
        })

    return assets