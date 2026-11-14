import streamlit as st
import av
import cv2
import os
import time
import tempfile
import subprocess
import threading
import pandas as pd
import numpy as np

from collections import deque
from itertools import islice
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)
import imageio_ffmpeg


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CrowdGuard",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# Theme: soft mint / sage canvas, deep ocean teal for
# navigation + key sections, white cards, muted coral for
# alerts, lavender + dusty pink as light accents.
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --cg-bg:#eaf4ee;
    --cg-bg2:#dcebe2;
    --cg-card:#ffffff;
    --cg-card-soft:#f4faf6;
    --cg-border:#d3e4da;
    --cg-text:#17353b;
    --cg-muted:#5f7b80;
    --cg-teal:#14606b;
    --cg-teal-dark:#0e4854;
    --cg-teal-soft:#1f7f86;
    --cg-mint:#bfe5d3;
    --cg-coral:#e2806f;
    --cg-lavender:#c9bdee;
    --cg-pink:#e3b5c4;
    --cg-shadow:0 4px 18px rgba(20,96,107,.08);
}

/* ---------- Base ---------- */
.stApp {
    background: linear-gradient(160deg, #f1f8f3 0%, #e6f2ea 45%, #dcebe2 100%);
    color: var(--cg-text);
    font-family: 'DM Sans', sans-serif;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 {
    font-family: 'Manrope','DM Sans',sans-serif !important;
    letter-spacing: -.025em;
    color: var(--cg-text);
}
.stApp p, .stApp li, .stApp label, .stApp span { color: var(--cg-text); }
.block-container { padding-top: 3.2rem; padding-bottom: 2.2rem; max-width: 1400px; }

/* Remove the stray frame / line around the top of the page */
[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stDecoration"] { display: none; }
[data-testid="stMain"], [data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"], section.main {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stAppDeployButton"], .stDeployButton { display: none; }

/* ---------- Sidebar (deep ocean teal) ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f5260 0%, #0b3f4b 100%);
    border-right: none;
}
/* Navigation: clean menu items (no radio dots), readable light text */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 10px 14px;
    border-radius: 10px;
    transition: background .15s ease;
    cursor: pointer;
    width: 100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label span {
    color: #d5ebe6 !important;
    font-weight: 600;
    font-size: .95rem;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: rgba(191,229,211,.20); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .cg-eyebrow { color: #9cc9c3 !important; }

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: var(--cg-shadow);
}
[data-testid="stMetricLabel"] p { color: var(--cg-muted) !important; font-weight: 500; }
[data-testid="stMetricValue"] { color: var(--cg-teal-dark) !important; font-family: 'Manrope',sans-serif; }
[data-testid="stMetricDelta"] { color: var(--cg-teal-soft) !important; }

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 10px;
    border: none;
    background: linear-gradient(100deg, #14606b, #1f7f86);
    color: #ffffff;
    font-weight: 700;
    min-height: 2.7rem;
    box-shadow: 0 4px 12px rgba(20,96,107,.22);
}
.stButton > button p { color: #ffffff !important; }
.stButton > button:hover {
    background: linear-gradient(100deg, #0e4854, #14606b);
    color: #ffffff;
    border: none;
}

/* ---------- Progress / uploader / tables ---------- */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #1f7f86, #8fd0b4); }
[data-testid="stFileUploader"] section {
    background: var(--cg-card-soft);
    border: 1.5px dashed #9cc7b5;
    border-radius: 14px;
}
[data-testid="stFileUploader"] section button { border-radius: 8px; }
[data-testid="stDataFrame"] {
    border: 1px solid var(--cg-border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--cg-card);
}
[data-testid="stExpander"] {
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: 12px;
}
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: var(--cg-border) !important; }

/* ---------- Progress bar (teal, regardless of Streamlit theme) ---------- */
div[role="progressbar"] > div > div { background: linear-gradient(90deg, #1f7f86, #8fd0b4) !important; }

/* ---------- Video input card ---------- */
.st-key-cg_video_card {
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: 17px;
    padding: 20px 22px;
    box-shadow: var(--cg-shadow);
}

/* ---------- Header with compact tags ---------- */
.cg-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 16px;
    flex-wrap: wrap;
}
.cg-tags { display: flex; gap: 8px; flex-wrap: wrap; }
.cg-tag {
    background: rgba(255,255,255,.75);
    border: 1px solid var(--cg-border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: .76rem;
    font-weight: 600;
    color: var(--cg-muted);
    white-space: nowrap;
}
.cg-tag b { color: var(--cg-teal-dark); font-weight: 700; }
.cg-ready {
    display: inline-block;
    background: #d6f1e4;
    color: var(--cg-teal);
    border-radius: 20px;
    padding: 5px 13px;
    font-size: .8rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.cg-hint {
    background: var(--cg-card-soft);
    border: 1px dashed #b5d3c4;
    border-radius: 12px;
    padding: 14px 16px;
    color: var(--cg-muted);
    font-size: .86rem;
    line-height: 1.6;
}
.cg-hint b { color: var(--cg-teal-dark); }

/* ---------- Custom components ---------- */
.cg-eyebrow {
    color: var(--cg-teal-soft);
    text-transform: uppercase;
    letter-spacing: .14em;
    font-size: .72rem;
    font-weight: 700;
}
.cg-title {
    font-family: 'Manrope',sans-serif;
    color: var(--cg-teal-dark);
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -.04em;
    margin: 0;
}
.cg-subtitle { color: var(--cg-muted); margin-top: 3px; font-size: .96rem; }
.cg-panel {
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: 17px;
    padding: 17px 18px;
    box-shadow: var(--cg-shadow);
}
.cg-panel-title {
    color: var(--cg-teal-dark);
    font-family: 'Manrope',sans-serif;
    font-size: 1.03rem;
    font-weight: 800;
    margin-bottom: 12px;
}
.cg-zone {
    border-radius: 14px;
    padding: 14px 12px;
    text-align: center;
    color: var(--cg-teal-dark);
    min-height: 132px;
    border: 1px solid rgba(20,96,107,.08);
}
.cg-zone-a { background: linear-gradient(145deg, #d6f1e4, #bfe5d3); }
.cg-zone-b { background: linear-gradient(145deg, #d3e8ee, #b8d8e2); }
.cg-zone-c { background: linear-gradient(145deg, #e6e0f6, #d3c9ef); }
.cg-zone-name { font-weight: 800; font-size: .9rem; color: var(--cg-teal-dark); }
.cg-zone-number {
    font-family: 'Manrope',sans-serif;
    font-size: 2rem;
    line-height: 1.2;
    font-weight: 800;
    margin: 6px 0 1px;
    color: var(--cg-teal-dark);
}
.cg-zone-note { font-size: .76rem; color: var(--cg-teal-dark); opacity: .75; }
.cg-chip {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: .72rem;
    font-weight: 700;
    background: rgba(255,255,255,.7);
    color: var(--cg-teal-dark);
    margin-top: 8px;
}
.cg-risk-value { font-family:'Manrope',sans-serif; font-size: 1.9rem; font-weight: 800; }
.cg-risk-note { color: var(--cg-muted); font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

ZONE_NAMES = ["Zone A", "Zone B", "Zone C"]

# =========================================================
# EXPLAINABLE MULTI-FACTOR CROWD RISK INDEX CONFIGURATION
# (Engineered mathematical risk model: 0 - 100)
# =========================================================
# Four normalized dynamic factors:
# 1. Density Concentration (Weight: 40%) - monocular crowd volume
# 2. Velocity Hazard / Sudden Stoppage (Weight: 30%) - crush bottleneck
# 3. Movement Direction Turbulence (Weight: 20%) - counter-flow collision
# 4. Congestion Persistence Flag (Weight: 10%) - sustained low mobility

RISK_LEVEL_BOUNDARIES = [
    (75, "Critical", "#cc5a5a"),
    (50, "High", "#e2806f"),
    (25, "Medium", "#c99a3b"),
    (0, "Low", "#2f8f7a")
]

RISK_THRESHOLDS = {
    "Medium": 5,
    "High": 10,
    "Critical": 15
}

RISK_ORDER = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3
}

# Colours used for the on-page risk indicator (matches the theme).
RISK_UI_COLORS = {
    "Low": "#2f8f7a",
    "Medium": "#c99a3b",
    "High": "#e2806f",
    "Critical": "#cc5a5a"
}

def score_to_risk_level(score):
    """Map numeric risk score (0-100) to human-readable risk category and theme color."""
    for threshold, level, color in RISK_LEVEL_BOUNDARIES:
        if score >= threshold:
            return level, color
    return "Low", "#2f8f7a"


def compute_zone_risk_index(count, average_speed, directions=None, congestion=False, capacity_ref=15):
    """
    Explainable multi-factor Crowd Risk Index (0 - 100).
    Produces normalized sub-scores, overall score, risk category, and contributing factors.
    """
    if directions is None:
        directions = {"Left": 0, "Right": 0, "Up": 0, "Down": 0, "Stationary": 0}

    if count <= 0:
        return {
            "score": 0,
            "level": "Low",
            "color": "#2f8f7a",
            "primary_factor": "Zone is clear (0 people)",
            "breakdown": {
                "density_subscore": 0.0,
                "velocity_subscore": 0.0,
                "turbulence_subscore": 0.0,
                "congestion_subscore": 0.0
            }
        }

    # Factor 1: Normalized Crowd Density (40% weight)
    norm_density = min(1.0, count / float(capacity_ref))
    density_subscore = 40.0 * norm_density

    # Factor 2: Velocity Hazard & Stoppage (30% weight)
    # Scaled by crowd presence so 1 isolated person doesn't trigger false alarm
    presence_scale = min(1.0, count / 5.0)
    if average_speed is None or average_speed <= 5.0:
        norm_velocity = 1.0 * presence_scale
    elif average_speed < 35.0:
        norm_velocity = ((35.0 - average_speed) / 30.0) * presence_scale
    else:
        norm_velocity = 0.05 * presence_scale
    velocity_subscore = 30.0 * norm_velocity

    # Factor 3: Turbulence & Counter-Flow Conflict (20% weight)
    left = directions.get("Left", 0)
    right = directions.get("Right", 0)
    up = directions.get("Up", 0)
    down = directions.get("Down", 0)

    h_total = left + right
    h_conflict = (2.0 * min(left, right) / float(h_total)) if h_total >= 2 else 0.0

    v_total = up + down
    v_conflict = (2.0 * min(up, down) / float(v_total)) if v_total >= 2 else 0.0

    norm_turbulence = min(1.0, max(h_conflict, v_conflict)) if count >= 2 else 0.0
    turbulence_subscore = 20.0 * norm_turbulence

    # Factor 4: Congestion Persistence (10% weight)
    congestion_subscore = 10.0 if congestion else 0.0

    total_score = int(np.clip(round(density_subscore + velocity_subscore + turbulence_subscore + congestion_subscore), 0, 100))
    level, color = score_to_risk_level(total_score)

    factors = []
    if density_subscore >= 18.0:
        factors.append(f"High Density ({density_subscore:.0f}/40)")
    if velocity_subscore >= 14.0:
        factors.append(f"Slowed Movement ({velocity_subscore:.0f}/30)")
    if turbulence_subscore >= 8.0:
        factors.append(f"Counter-flow Conflict ({turbulence_subscore:.0f}/20)")
    if congestion_subscore >= 8.0:
        factors.append("Congestion Bottleneck (10/10)")

    if not factors:
        if total_score < 25:
            primary_factor = "Normal crowd flow & low concentration"
        else:
            primary_factor = "Moderate crowd movement"
    else:
        primary_factor = " + ".join(factors)

    return {
        "score": total_score,
        "level": level,
        "color": color,
        "primary_factor": primary_factor,
        "breakdown": {
            "density_subscore": round(density_subscore, 1),
            "velocity_subscore": round(velocity_subscore, 1),
            "turbulence_subscore": round(turbulence
