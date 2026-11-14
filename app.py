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
            "turbulence_subscore": round(turbulence_subscore, 1),
            "congestion_subscore": round(congestion_subscore, 1)
        }
    }


def process_active_alerts(analytics, source="Live"):
    """
    Check zone risk scores and maintain a session incident log in st.session_state.
    Deduplicates alerts with an 8-second cooldown per zone.
    """
    if "alert_history" not in st.session_state:
        st.session_state["alert_history"] = []
    if "last_alert_time" not in st.session_state:
        st.session_state["last_alert_time"] = {}

    current_time = time.time()
    active_alerts = []

    for zone in ZONE_NAMES:
        zone_info = analytics["zones"].get(zone, {})
        score = zone_info.get("risk_score", 0)
        level = zone_info.get("risk", "Low")
        reason = zone_info.get("primary_factor", "Normal crowd flow")

        if score >= 50:  # High or Critical threshold
            active_alerts.append({
                "zone": zone,
                "level": level,
                "score": score,
                "reason": reason,
                "color": zone_info.get("risk_color", "#e2806f")
            })

            last_time = st.session_state["last_alert_time"].get(zone, 0)
            if current_time - last_time > 8.0:
                st.session_state["last_alert_time"][zone] = current_time
                alert_entry = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "source": source,
                    "zone": zone,
                    "level": level,
                    "score": score,
                    "reason": reason
                }
                st.session_state["alert_history"].insert(0, alert_entry)
                if len(st.session_state["alert_history"]) > 50:
                    st.session_state["alert_history"].pop()

    return active_alerts


def render_alert_banner(active_alerts):
    """Display high-visibility warning banner for elevated risk zones."""
    if not active_alerts:
        return
    for alert in active_alerts:
        st.markdown(
            f"<div style='background:rgba(226,128,111,0.18);border:1.5px solid {alert['color']};"
            f"border-radius:12px;padding:12px 18px;margin-bottom:14px;display:flex;align-items:center;gap:12px'>"
            f"<span style='font-size:1.5rem'>⚠️</span>"
            f"<div><b style='color:#7d2d21;font-size:1.02rem'>[{alert['level'].upper()} RISK] {alert['zone']} – Score: {alert['score']}/100</b>"
            f"<div style='color:#5c2b23;font-size:0.88rem;margin-top:2px'>{alert['reason']}</div></div>"
            f"</div>",
            unsafe_allow_html=True
        )


@st.dialog("🚨 Crowd Safety Risk Alert")
def show_risk_alert_dialog(incident_info):
    """Interactive modal popup dialog for elevated risk incidents."""
    level = incident_info.get("level", "High")
    score = incident_info.get("score", 50)
    color = incident_info.get("color", "#e2806f")
    reason = incident_info.get("reason", "Elevated crowd dynamics detected")
    zone = incident_info.get("zone", "Monitored Area")
    source = incident_info.get("source", "Video Analysis")

    st.markdown(
        f"<div style='background:rgba(226,128,111,0.14);border:2px solid {color};"
        f"border-radius:14px;padding:18px 22px;margin-bottom:14px;text-align:center'>"
        f"<div style='font-size:2.8rem;margin-bottom:4px'>🚨</div>"
        f"<div style='font-family:Manrope,sans-serif;font-size:1.4rem;font-weight:800;color:#7d2d21;text-transform:uppercase;letter-spacing:0.04em'>"
        f"{level} Risk Alert Detected"
        f"</div>"
        f"<div style='font-family:Manrope,sans-serif;font-size:3rem;font-weight:800;color:{color};margin:6px 0'>"
        f"{score}<span style='font-size:1.5rem;color:var(--cg-muted)'> / 100</span>"
        f"</div>"
        f"<div style='background:rgba(255,255,255,0.85);border-radius:10px;padding:12px;margin-top:10px;text-align:left;font-size:0.92rem;color:#333;line-height:1.6'>"
        f"<b>📍 Location:</b> {zone}<br>"
        f"<b>⚠️ Primary Factor:</b> {reason}<br>"
        f"<b>📹 Analysis Source:</b> {source}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='background:#f4faf6;border:1px solid #cce3d8;border-radius:10px;padding:12px 14px;margin-bottom:16px;font-size:0.88rem;color:#17353b'>"
        "<b>🛡️ Recommended Action:</b> Deploy safety marshals to relieve bottleneck, manage counter-flow corridors, and prevent spatial congestion."
        "</div>",
        unsafe_allow_html=True
    )
    if st.button("Acknowledge & Dismiss", type="primary", use_container_width=True, key="cg_btn_dismiss_alert_dialog"):
        st.session_state["show_alert_popup"] = False
        st.rerun()


# Limits for the live webcam data logs, so memory use stays bounded
# during long camera sessions. Older rows are dropped automatically.
LIVE_FRAME_LOG_LIMIT = 3000
LIVE_PERSON_LOG_LIMIT = 15000

# How often the live report below the webcam refreshes. A shorter interval
# makes the page busier and can slow the video; 2 seconds keeps both smooth.
LIVE_REPORT_REFRESH = "2s"

# Live camera tuning. The video always plays at the camera's frame rate;
# people detection runs separately in the background, at most this many
# times per second (fewer if the computer is slower).
LIVE_INFERENCE_MAX_FPS = 8

# Camera quality requested from the browser. Smaller pictures are much lighter
# to send, process and draw, which matters most on a shared cloud server.
# For a sharper picture on a powerful computer, try 1280 x 720 at 30 fps.
LIVE_CAMERA_WIDTH = 640
LIVE_CAMERA_HEIGHT = 480
LIVE_CAMERA_FPS = 20


# =========================================================
# 2. LOAD YOLO MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load YOLO once and share it across reruns, pages and camera sessions."""
    return YOLO("yolov8n.pt")


@st.cache_resource(show_spinner=False)
def get_inference_lock():
    """
    One lock shared by every camera session, so two sessions (for example an
    old one still shutting down and a new one starting) never run the shared
    YOLO/ByteTrack model at the same moment.
    """
    return threading.Lock()


@st.cache_resource(show_spinner=False)
def start_model_warmup():
    """
    Load and warm up YOLO once per server process, in a background thread.

    The page itself renders immediately, so Dashboard, Analytics and About
    stay light. By the time START is pressed on Live Analysis the model is
    already loaded and its first (slow) inference has already been done,
    which removes the long wait after clicking START.
    """
    def _warm_up():
        try:
            detector = load_model()
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            with get_inference_lock():
                detector.track(
                    blank,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.4,
                    imgsz=480,
                    verbose=False
                )
        except Exception:
            # Warm-up is only an optimisation. If it fails, the camera and
            # upload pages simply load the model themselves when needed.
            pass

    thread = threading.Thread(
        target=_warm_up,
        name="crowdguard-yolo-warmup",
        daemon=True
    )
    thread.start()
    return thread


start_model_warmup()


# =========================================================
# 2b. WEBRTC CONNECTION SERVERS (STUN / TURN)
# =========================================================

def get_ice_servers():
    """
    Servers that help the browser and the server find each other.
    Uses Google STUN plus Metered TURN relay servers so visitors behind
    home/college routers or mobile networks can connect reliably.
    """
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {
            "urls": [
                "turn:global.relay.metered.ca:80",
                "turn:global.relay.metered.ca:80?transport=tcp",
                "turn:global.relay.metered.ca:443",
                "turns:global.relay.metered.ca:443?transport=tcp",
            ],
            "username": "9f16617ebeeb4ef49898e6d9",
            "credential": "m8CIsJx/TtNMyf0f",
        },
    ]


# Kept for compatibility; pages call load_model() when they need the detector.
model = None


# =========================================================
# 3. CROWD ANALYTICS HELPER FUNCTIONS
# =========================================================

def get_zone_name(x_position, frame_width):
    """Assign a person's bottom-centre point to A, B or C."""
    if x_position < frame_width / 3:
        return "Zone A"
    elif x_position < (2 * frame_width) / 3:
        return "Zone B"
    return "Zone C"


def update_track_speeds(tracks, frame_number, fps, motion_history, trajectory_history=None, current_time=None):
    """
    Estimate image-plane movement and direction from consecutive ByteTrack
    positions. Pixel movement is not calibrated to real-world speed.
    """
    safe_fps = fps if fps and fps > 0 else 20.0

    if trajectory_history is None:
        trajectory_history = {}

    for person in tracks:
        x1, y1, x2, y2 = person["box"]
        track_id = person["id"]

        # Bottom-centre is a rough ground-contact location.
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)
        person["speed"] = None
        person["direction"] = "Unknown"
        person["trail"] = [(center_x, center_y)]

        if track_id is None:
            continue

        previous = motion_history.get(track_id)

        if previous is not None:
            # When real-world wall-clock time is available, use it for exact delta-time
            if len(previous) >= 4 and current_time is not None and previous[3] is not None:
                old_x, old_y, old_frame, old_time = previous
                elapsed_seconds = current_time - old_time
            else:
                old_x, old_y, old_frame = previous[:3]
                elapsed_seconds = (frame_number - old_frame) / safe_fps

            if elapsed_seconds > 0:
                dx = center_x - old_x
                dy = center_y - old_y
                distance = (dx ** 2 + dy ** 2) ** 0.5
                person["speed"] = distance / elapsed_seconds

                # Ignore tiny coordinate jitter before assigning a direction.
                if max(abs(dx), abs(dy)) < 3:
                    person["direction"] = "Stationary"
                elif abs(dx) >= abs(dy):
                    person["direction"] = "Right" if dx > 0 else "Left"
                else:
                    person["direction"] = "Down" if dy > 0 else "Up"

        motion_history[track_id] = (center_x, center_y, frame_number, current_time)

        trail = trajectory_history.setdefault(track_id, [])
        if not trail or trail[-1] != (center_x, center_y):
            trail.append((center_x, center_y))
        # Keep only a short tail so the video remains readable.
        if len(trail) > 18:
            del trail[:-18]

        person["trail"] = list(trail)

    # Periodic cleanup of inactive tracks (older than 90 frames) to keep memory bounded
    if len(motion_history) > 50:
        stale_ids = [
            tid for tid, entry in motion_history.items()
            if frame_number - entry[2] > 90
        ]
        for tid in stale_ids:
            motion_history.pop(tid, None)
            if trajectory_history and tid in trajectory_history:
                trajectory_history.pop(tid, None)


def calculate_zone_analytics(tracks, frame_width):
    """
    Calculate zone counts, average image-plane speed, direction conflicts,
    and the explainable multi-factor Crowd Risk Index for each zone.
    """
    counts = {zone: 0 for zone in ZONE_NAMES}
    speeds = {zone: [] for zone in ZONE_NAMES}
    movement_directions = {
        "Left": 0, "Right": 0, "Up": 0, "Down": 0, "Stationary": 0
    }
    zone_directions = {
        zone: {"Left": 0, "Right": 0, "Up": 0, "Down": 0, "Stationary": 0}
        for zone in ZONE_NAMES
    }

    for person in tracks:
        x1, y1, x2, y2 = person["box"]
        bottom_center_x = (x1 + x2) / 2
        zone = get_zone_name(bottom_center_x, frame_width)

        counts[zone] += 1

        direction = person.get("direction", "Unknown")
        if direction in movement_directions:
            movement_directions[direction] += 1
        if zone in zone_directions and direction in zone_directions[zone]:
            zone_directions[zone][direction] += 1

        if person.get("speed") is not None:
            speeds[zone].append(person["speed"])

    zone_details = {}

    for zone in ZONE_NAMES:
        average_speed = (
            sum(speeds[zone]) / len(speeds[zone])
            if speeds[zone] else None
        )

        # Congestion is flagged when density meets threshold and mobility drops
        congestion = (
            counts[zone] >= 5
            and average_speed is not None
            and average_speed < 20
        )

        # Compute explainable multi-factor risk index
        risk_data = compute_zone_risk_index(
            count=counts[zone],
            average_speed=average_speed,
            directions=zone_directions[zone],
            congestion=congestion
        )

        zone_details[zone] = {
            "people": counts[zone],
            "average_speed": average_speed,
            "congestion": congestion,
            "risk": risk_data["level"],
            "risk_score": risk_data["score"],
            "risk_color": risk_data["color"],
            "primary_factor": risk_data["primary_factor"],
            "breakdown": risk_data["breakdown"],
            "directions": zone_directions[zone]
        }

    max_risk_zone = max(
        ZONE_NAMES,
        key=lambda z: zone_details[z]["risk_score"]
    )
    overall_risk_score = zone_details[max_risk_zone]["risk_score"]
    overall_risk_level = zone_details[max_risk_zone]["risk"]
    overall_factor = zone_details[max_risk_zone]["primary_factor"]

    return {
        "zones": zone_details,
        "overall_risk": overall_risk_level,
        "overall_risk_score": overall_risk_score,
        "overall_primary_factor": overall_factor,
        "total_people": sum(counts.values()),
        "movement_directions": movement_directions
    }


def make_frame_record(frame_number, time_s, analytics, tracked_this_frame):
    """
    Build one row of the frame-by-frame data log.
    Includes measured values and calculated multi-factor risk index.
    """
    record = {
        "frame": frame_number,
        "time_s": round(float(time_s), 3),
        "tracked_this_frame": bool(tracked_this_frame),
        "total_people": analytics["total_people"],
        "overall_risk_score": analytics.get("overall_risk_score", 0),
        "overall_risk_level": analytics.get("overall_risk", "Low")
    }

    for zone in ZONE_NAMES:
        key = zone.lower().replace(" ", "_")
        details = analytics["zones"][zone]
        record[f"{key}_people"] = details["people"]
        record[f"{key}_avg_speed_px_s"] = (
            round(details["average_speed"], 2)
            if details["average_speed"] is not None else None
        )
        record[f"{key}_congestion_flag"] = int(details["congestion"])
        record[f"{key}_risk_score"] = details.get("risk_score", 0)

    for direction, number in analytics["movement_directions"].items():
        record[f"dir_{direction.lower()}"] = number

    return record


def make_person_records(tracks, frame_number, time_s, frame_width, frame_height):
    """
    Build per-person rows (position, zone, speed, direction) for one frame.
    These are the raw observations needed later for heatmaps and
    crowd-convergence analysis.
    """
    rows = []
    safe_width = frame_width if frame_width else 1
    safe_height = frame_height if frame_height else 1

    for person in tracks:
        x1, y1, x2, y2 = person["box"]
        center_x = (x1 + x2) / 2
        center_y = y2  # bottom-centre, same point used for movement

        speed = person.get("speed")

        rows.append({
            "frame": frame_number,
            "time_s": round(float(time_s), 3),
            "track_id": person.get("id"),
            "zone": get_zone_name(center_x, frame_width),
            "x_px": int(center_x),
            "y_px": int(center_y),
            "x_norm": round(center_x / safe_width, 4),
            "y_norm": round(center_y / safe_height, 4),
            "speed_px_s": round(speed, 2) if speed is not None else None,
            "direction": person.get("direction", "Unknown")
        })

    return rows


def summarise_crowd_trend(frame_df):
    """
    Describe how the total crowd changed over the video by comparing the
    first and last quarter of the freshly tracked frames. This is a simple
    descriptive comparison, not a prediction.
    """
    if frame_df is None or frame_df.empty:
        return None

    tracked = frame_df[frame_df["tracked_this_frame"]]
    if len(tracked) < 8:
        return None

    totals = tracked["total_people"].to_numpy(dtype=float)
    quarter = max(len(totals) // 4, 1)
    early_average = float(totals[:quarter].mean())
    late_average = float(totals[-quarter:].mean())
    change = late_average - early_average
    needed = max(1.0, 0.2 * early_average)

    if change >= needed:
        label = "Increasing"
    elif change <= -needed:
        label = "Decreasing"
    else:
        label = "Stable"

    return {
        "label": label,
        "early_average": early_average,
        "late_average": late_average
    }


def build_presence_heatmap(person_df, background_bgr, max_width=960):
    """
    Cumulative presence heatmap: where people's ground-contact points were
    observed across the whole video, drawn over a reference frame.
    Built only from logged person positions. Returns an RGB image or None.
    """
    if person_df is None or person_df.empty or background_bgr is None:
        return None

    height, width = background_bgr.shape[:2]
    scale = min(1.0, max_width / float(width))
    out_w = max(int(width * scale), 1)
    out_h = max(int(height * scale), 1)

    background = cv2.resize(background_bgr, (out_w, out_h))
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Count observations on a coarse grid, then smooth them.
    grid_w = max(out_w // 8, 4)
    grid_h = max(out_h // 8, 4)
    xs = np.clip(
        (person_df["x_norm"].to_numpy(dtype=float) * grid_w).astype(int),
        0, grid_w - 1
    )
    ys = np.clip(
        (person_df["y_norm"].to_numpy(dtype=float) * grid_h).astype(int),
        0, grid_h - 1
    )
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    np.add.at(grid, (ys, xs), 1.0)
    grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=2.0)

    peak = float(grid.max())
    if peak <= 0:
        return None
    grid = grid / peak

    # Mint (rarely occupied) -> teal -> coral (most occupied), in RGB.
    stops = np.array([0.0, 0.35, 0.7, 1.0])
    palette = np.array(
        [[191, 229, 211], [31, 127, 134], [226, 128, 111], [204, 90, 90]],
        dtype=np.float32
    )
    colour_small = np.stack(
        [np.interp(grid, stops, palette[:, channel]) for channel in range(3)],
        axis=-1
    ).astype(np.float32)

    colour = cv2.resize(colour_small, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    # Rarely occupied areas stay transparent so the scene remains visible.
    alpha = np.clip(alpha * 1.3, 0.0, 0.75)[..., None]

    blended = background * (1.0 - alpha) + colour * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_analytics_overlay(frame, tracks, analytics):
    """Draw zone boundaries, person boxes and a compact analytics overlay.

    OpenCV colours are BGR. The palette below matches the dashboard theme:
    mint zone lines, teal person boxes, deep-teal banner, coral for alerts.
    """
    height, width = frame.shape[:2]
    third = width // 3

    zone_line_color = (211, 229, 168)    # soft mint
    box_color = (190, 200, 80)           # bright teal
    banner_color = (75, 62, 14)          # deep ocean teal

    # Zone boundaries
    cv2.line(frame, (third, 0), (third, height), zone_line_color, 2)
    cv2.line(frame, (2 * third, 0), (2 * third, height), zone_line_color, 2)

    # Zone names, counts, and multi-factor risk scores
    for index, zone in enumerate(ZONE_NAMES):
        x_position = index * third + 12
        count = analytics["zones"][zone]["people"]
        z_score = analytics["zones"][zone].get("risk_score", 0)

        cv2.putText(
            frame,
            f"{zone}: {count} ppl | {z_score}/100",
            (x_position, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            zone_line_color,
            2
        )

    # Short movement trails, drawn before boxes so they do not hide people.
    trail_color = (190, 145, 75)  # muted warm accent in BGR
    for person in tracks:
        trail = person.get("trail", [])
        if len(trail) >= 2:
            cv2.polylines(
                frame,
                [np.array(trail, dtype=np.int32)],
                False,
                trail_color,
                2
            )
            cv2.arrowedLine(
                frame,
                trail[-2],
                trail[-1],
                trail_color,
                2,
                tipLength=0.35
            )

    # Person bounding boxes. Raw tracking IDs are intentionally not displayed.
    for person in tracks:
        x1, y1, x2, y2 = person["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        direction = person.get("direction", "Unknown")
        track_id = person.get("id")
        identity = f"ID {track_id}" if track_id is not None else "ID pending"
        label = identity if direction == "Unknown" else f"{identity} | {direction}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            box_color,
            2
        )

    risk = analytics["overall_risk"]
    risk_score = analytics.get("overall_risk_score", 0)

    risk_colors = {
        "Low": (170, 215, 140),       # mint green
        "Medium": (130, 200, 235),    # warm sand
        "High": (125, 150, 240),      # muted coral
        "Critical": (100, 95, 225)    # deeper coral
    }

    # Compact status strip: retain the same information without covering
    # the upper part of the camera scene.
    status_text = f"People: {analytics['total_people']}  |  Risk: {risk_score}/100 ({risk})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)
    banner_height = max(30, text_height + baseline + 12)
    banner_right = min(width - 8, text_width + 34)
    cv2.rectangle(frame, (8, 8), (banner_right, 8 + banner_height), banner_color, -1)

    cv2.putText(
        frame,
        status_text,
        (18, 8 + banner_height - 9),
        font,
        font_scale,
        risk_colors.get(risk, (170, 215, 140)),
        thickness,
        cv2.LINE_AA
    )

    return frame


def create_track_records(result):
    """Convert YOLO/ByteTrack output into simple person records."""
    tracks = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return tracks

    coordinates = boxes.xyxy.cpu().numpy()
    track_ids = None

    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()

    for index, box in enumerate(coordinates):
        track_id = (
            track_ids[index]
            if track_ids is not None
            else None
        )

        tracks.append({
            "box": (
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3])
            ),
            "id": track_id,
            "speed": None
        })

    return tracks


def render_data_log_section(frame_df, person_df, key_prefix, note):
    """Show a small preview and CSV downloads of the logged data."""
    col1, col2 = st.columns(2)
    col1.metric("Frame records logged", len(frame_df))
    col2.metric("Person observations logged", len(person_df))

    st.caption(note)

    if len(frame_df) > 0:
        st.markdown("<div class='cg-panel-title'>Latest frame records</div>", unsafe_allow_html=True)
        st.dataframe(frame_df.tail(8), use_container_width=True, hide_index=True)

    if len(person_df) > 0:
        st.markdown("<div class='cg-panel-title'>Latest person observations</div>", unsafe_allow_html=True)
        st.dataframe(person_df.tail(8), use_container_width=True, hide_index=True)

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "Download frame log (CSV)",
            data=frame_df.to_csv(index=False).encode("utf-8"),
            file_name="crowdguard_frame_log.csv",
            mime="text/csv",
            key=f"{key_prefix}_frame_csv",
            on_click="ignore"
        )
    with download_cols[1]:
        st.download_button(
            "Download person log (CSV)",
            data=person_df.to_csv(index=False).encode("utf-8"),
            file_name="crowdguard_person_log.csv",
            mime="text/csv",
            key=f"{key_prefix}_person_csv",
            on_click="ignore"
        )


# =========================================================
# 4. LIVE WEBCAM PROCESSOR
# =========================================================

class CameraProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_number = 0
        # Load the detector in the background after WebRTC creates the processor.
        # This keeps camera negotiation responsive while ensuring this processor
        # receives the actual model once it is ready.
        self.model = None
        self.model_error = None
        self.model_lock = threading.Lock()
        self.inference_lock = get_inference_lock()
        self.model_loading = True
        self.model_thread = threading.Thread(
            target=self._load_detector,
            name="crowdguard-yolo-loader",
            daemon=True
        )
        self.model_thread.start()
        self.motion_history = {}
        self.trajectory_history = {}
        self.analytics_lock = threading.Lock()
        self.latest_snapshot = None
        self.max_people_seen = 0
        self.track_ids_seen = set()
        self.peak_zone_counts = {zone: 0 for zone in ZONE_NAMES}
        self.highest_risk_observed = "Low"
        self.highest_risk_score_observed = 0
        self.speed_samples = deque(maxlen=1000)
        self.congestion_frames = {zone: 0 for zone in ZONE_NAMES}
        # Data logs (bounded so a long session cannot exhaust memory).
        self.session_start = None
        self.frame_log = deque(maxlen=LIVE_FRAME_LOG_LIMIT)
        self.person_log = deque(maxlen=LIVE_PERSON_LOG_LIMIT)
        self.last_tracks = []
        self.last_analytics = calculate_zone_analytics([], 1)
        self.background_frame = None

        # Detection runs in its own background thread, so the video picture is
        # never held up waiting for YOLO. recv() only hands over the newest
        # frame and draws the latest finished result.
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.last_submit_time = 0.0
        self.result_id = 0
        self.logged_result_id = 0
        self.last_recv_time = time.time()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self._start_worker()

    def _start_worker(self):
        self.worker_thread = threading.Thread(
            target=self._inference_loop,
            name="crowdguard-inference",
            daemon=True
        )
        self.worker_thread.start()

    def on_ended(self):
        """Called by streamlit-webrtc when the camera stops."""
        self.stop_event.set()

    def _inference_loop(self):
        """
        Background detection loop. Always works on the newest camera frame and
        runs at most LIVE_INFERENCE_MAX_FPS times per second, so it cannot use
        up the whole processor and starve the video.
        """
        min_interval = 1.0 / LIVE_INFERENCE_MAX_FPS
        last_frame_id = 0

        while not self.stop_event.is_set():
            # No frames for 10 seconds means the camera stream has ended.
            if time.time() - self.last_recv_time > 10.0:
                break

            detector = self.model
            with self.frame_lock:
                img = self.latest_frame
                frame_id = self.latest_frame_id

            if detector is None or img is None or frame_id == last_frame_id:
                self.stop_event.wait(0.01)
                continue

            last_frame_id = frame_id
            started = time.time()

            try:
                with self.inference_lock:
                    results = detector.track(
                        img,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0],
                        conf=0.4,
                        imgsz=480,
                        verbose=False
                    )

                tracks = create_track_records(results[0])

                update_track_speeds(
                    tracks,
                    frame_id,
                    20.0,
                    self.motion_history,
                    self.trajectory_history,
                    current_time=time.time()
                )

                analytics = calculate_zone_analytics(tracks, img.shape[1])

                with self.analytics_lock:
                    for person in tracks:
                        if person.get("id") is not None:
                            self.track_ids_seen.add(person["id"])
                    self.last_tracks = tracks
                    self.last_analytics = analytics
                    self.result_id += 1
            except Exception:
                pass

            spare = min_interval - (time.time() - started)
            if spare > 0:
                self.stop_event.wait(spare)

    def _load_detector(self):
        """Load YOLO off the WebRTC signalling path; report failures safely."""
        try:
            detector = load_model()
            with self.model_lock:
                self.model = detector
        except Exception as error:
            self.model_error = str(error)
        finally:
            self.model_loading = False

    def get_snapshot(self):
        """Return lightweight display data; full bounded logs are fetched only on demand."""
        with self.analytics_lock:
            if self.latest_snapshot is None:
                return None

            # The live chart/heatmap do not need every stored observation.
            # Sampling here avoids copying and rebuilding 15,000 rows every second.
            frame_count = len(self.frame_log)
            person_count = len(self.person_log)
            frame_rows = list(islice(self.frame_log, max(0, frame_count - 600), None))
            person_rows = list(islice(self.person_log, max(0, person_count - 2500), None))

            return {
                "zones": {
                    zone: dict(details)
                    for zone, details in self.latest_snapshot["zones"].items()
                },
                "overall_risk": self.latest_snapshot["overall_risk"],
                "overall_risk_score": self.latest_snapshot.get("overall_risk_score", 0),
                "overall_primary_factor": self.latest_snapshot.get("overall_primary_factor", "Normal crowd flow"),
                "total_people": self.latest_snapshot["total_people"],
                "movement_directions": dict(self.latest_snapshot["movement_directions"]),
                "max_people_seen": self.max_people_seen,
                "track_ids_seen": len(self.track_ids_seen),
                "frame_number": self.frame_number,
                "peak_zone_counts": dict(self.peak_zone_counts),
                "highest_risk_observed": self.highest_risk_observed,
                "highest_risk_score_observed": self.highest_risk_score_observed,
                "speed_samples": list(self.speed_samples),
                "congestion_frames": dict(self.congestion_frames),
                "frame_log": frame_rows,
                "person_log": person_rows,
                "frame_log_count": frame_count,
                "person_log_count": person_count,
                "background_frame": self.background_frame.copy() if self.background_frame is not None else None
            }

    def get_full_logs(self):
        """Copy complete bounded session logs only when the user opens downloads."""
        with self.analytics_lock:
            return list(self.frame_log), list(self.person_log)

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            self.last_recv_time = time.time()

            # Keep the camera preview alive while the detector loads in the
            # background. Once ready, the same processor begins annotated analysis.
            detector = self.model
            if detector is None:
                self.frame_number += 1
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # Let the first 2 frames pass immediately so WebRTC stream establishes with 0 latency
            if self.frame_number < 2:
                self.frame_number += 1
                return frame

            now = time.time()
            if self.session_start is None:
                self.session_start = now
            elapsed_time = now - self.session_start

            if self.background_frame is None:
                self.background_frame = img.copy()

            # Restart the detection thread if it stopped while frames kept arriving.
            if (
                self.worker_thread is not None
                and not self.worker_thread.is_alive()
                and not self.stop_event.is_set()
            ):
                self._start_worker()

            # Hand the newest frame to the detection thread a few times per
            # second. The copy keeps it clean of the overlay drawn below.
            if now - self.last_submit_time >= 1.0 / LIVE_INFERENCE_MAX_FPS:
                with self.frame_lock:
                    self.latest_frame = img.copy()
                    self.latest_frame_id = self.frame_number
                self.last_submit_time = now

            # Draw the most recent finished detection result. A frame counts as
            # "tracked" in the data log when a new result arrived since the last one.
            with self.analytics_lock:
                tracks = self.last_tracks
                analytics = self.last_analytics
                result_id = self.result_id

            tracked_this_frame = (result_id != self.logged_result_id)
            self.logged_result_id = result_id

            frame_record = make_frame_record(
                self.frame_number,
                elapsed_time,
                analytics,
                tracked_this_frame
            )
            person_rows = (
                make_person_records(
                    tracks,
                    self.frame_number,
                    elapsed_time,
                    img.shape[1],
                    img.shape[0]
                )
                if tracked_this_frame
                else []
            )

            # Keep a live snapshot for the analytics panel below the camera.
            with self.analytics_lock:
                self.latest_snapshot = analytics
                self.max_people_seen = max(
                    self.max_people_seen,
                    analytics["total_people"]
                )

                for zone in ZONE_NAMES:
                    zone_details = analytics["zones"][zone]
                    self.peak_zone_counts[zone] = max(
                        self.peak_zone_counts[zone],
                        zone_details["people"]
                    )
                    if zone_details["congestion"]:
                        self.congestion_frames[zone] += 1
                    if tracked_this_frame and zone_details["average_speed"] is not None:
                        self.speed_samples.append(zone_details["average_speed"])

                if RISK_ORDER[analytics["overall_risk"]] > RISK_ORDER[self.highest_risk_observed]:
                    self.highest_risk_observed = analytics["overall_risk"]

                current_score = analytics.get("overall_risk_score", 0)
                if current_score > self.highest_risk_score_observed:
                    self.highest_risk_score_observed = current_score

                self.frame_log.append(frame_record)
                if person_rows:
                    self.person_log.extend(person_rows)

            annotated_frame = draw_analytics_overlay(
                img,
                tracks,
                analytics
            )

            self.frame_number += 1

            return av.VideoFrame.from_ndarray(
                annotated_frame,
                format="bgr24"
            )
        except Exception:
            return frame


# =========================================================
# 5. UPLOADED CCTV / RECORDED VIDEO PROCESSING
# =========================================================

def process_uploaded_video(uploaded_video):
    detector = load_model()

    input_path = None
    output_path = None
    cap = None
    ffmpeg_process = None

    try:
        # Save uploaded video temporarily.
        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_video.name)[1]
        )
        input_file.write(uploaded_video.getvalue())
        input_file.close()
        input_path = input_file.name

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("Unable to open this video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            st.error("Invalid video dimensions.")
            return

        # Create an MP4 output file.
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )
        output_file.close()
        output_path = output_file.name

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s:v", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-movflags", "+faststart",
            output_path
        ]

        ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        prog_col, _ = st.columns([2, 1])
        with prog_col:
            progress = st.progress(0)
            status = st.empty()

        frame_number = 0
        max_people = 0
        max_zone_counts = {zone: 0 for zone in ZONE_NAMES}
        motion_history = {}
        trajectory_history = {}
        unique_track_ids = set()
        speed_samples = []
        last_tracks = []
        latest_analytics = calculate_zone_analytics([], width)
        highest_risk_observed = "Low"
        highest_risk_score_observed = 0
        peak_analytics = None
        congestion_frames = {zone: 0 for zone in ZONE_NAMES}
        frame_log = []
        person_log = []
        background_frame = None

        # Process video.
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Keep the first clean frame (before overlays) as the heatmap background.
            if background_frame is None:
                background_frame = frame.copy()

            # Run YOLO and ByteTrack every third frame.
            tracked_this_frame = (frame_number % 3 == 0)

            if tracked_this_frame:
                results = detector.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.4,
                    imgsz=640,
                    verbose=False
                )

                last_tracks = create_track_records(results[0])

                update_track_speeds(
                    last_tracks,
                    frame_number,
                    fps,
                    motion_history,
                    trajectory_history
                )

                for person in last_tracks:
                    if person["id"] is not None:
                        unique_track_ids.add(person["id"])

                    if person["speed"] is not None:
                        speed_samples.append(person["speed"])

                # Per-person rows are logged only when tracking is fresh.
                person_log.extend(
                    make_person_records(
                        last_tracks,
                        frame_number,
                        frame_number / fps,
                        width,
                        height
                    )
                )

            # Use the most recent tracked positions between inference frames.
            latest_analytics = calculate_zone_analytics(
                last_tracks,
                width
            )

            frame_log.append(
                make_frame_record(
                    frame_number,
                    frame_number / fps,
                    latest_analytics,
                    tracked_this_frame
                )
            )

            max_people = max(
                max_people,
                latest_analytics["total_people"]
            )

            if RISK_ORDER[latest_analytics["overall_risk"]] > RISK_ORDER[highest_risk_observed]:
                highest_risk_observed = latest_analytics["overall_risk"]

            current_score = latest_analytics.get("overall_risk_score", 0)
            if current_score >= highest_risk_score_observed:
                highest_risk_score_observed = current_score
                peak_analytics = {
                    "score": current_score,
                    "level": latest_analytics.get("overall_risk", "Low"),
                    "primary_factor": latest_analytics.get("overall_primary_factor", "Normal crowd flow"),
                    "total_people": latest_analytics.get("total_people", 0)
                }

            for zone in ZONE_NAMES:
                zone_data = latest_analytics["zones"][zone]

                max_zone_counts[zone] = max(
                    max_zone_counts[zone],
                    zone_data["people"]
                )

                if zone_data["congestion"]:
                    congestion_frames[zone] += 1

            frame = draw_analytics_overlay(
                frame,
                last_tracks,
                latest_analytics
            )

            # Send annotated frame to FFmpeg.
            ffmpeg_process.stdin.write(frame.tobytes())

            frame_number += 1

            if total_frames > 0:
                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

            status.text(
                f"Analysing frame {frame_number} of {total_frames}"
            )

        # Finish video encoding.
        cap.release()
        cap = None

        ffmpeg_process.stdin.close()
        return_code = ffmpeg_process.wait()
        ffmpeg_process = None

        if return_code != 0:
            st.error("Video encoding failed. Please try another video.")
            return None

        if frame_number == 0:
            st.error("No readable frames found.")
            return None

        progress.progress(1.0)
        status.success("YOLO + ByteTrack + zone analytics completed!")

        with open(output_path, "rb") as video_file:
            processed_video = video_file.read()

        frame_df = pd.DataFrame(frame_log)
        person_df = pd.DataFrame(person_log)
        heatmap_image = build_presence_heatmap(person_df, background_frame)

        return {
            "processed_video": processed_video,
            "latest_analytics": latest_analytics,
            "max_people": max_people,
            "unique_track_ids_count": len(unique_track_ids),
            "highest_risk_observed": highest_risk_observed,
            "highest_risk_score_observed": highest_risk_score_observed,
            "peak_analytics": peak_analytics,
            "max_zone_counts": max_zone_counts,
            "congestion_frames": congestion_frames,
            "speed_samples": speed_samples,
            "frame_df": frame_df,
            "person_df": person_df,
            "heatmap_image": heatmap_image,
        }

    except Exception as error:
        st.error(f"Tracking error: {error}")
        return None

    finally:
        if cap is not None:
            cap.release()

        if ffmpeg_process is not None:
            try:
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.close()
                ffmpeg_process.kill()
                ffmpeg_process.wait()
            except Exception:
                pass

        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def render_uploaded_video_report(results):
    """Render the full crowd analytics report from stored analysis results."""
    processed_video = results["processed_video"]
    latest_analytics = results["latest_analytics"]
    max_people = results["max_people"]
    unique_track_ids_count = results["unique_track_ids_count"]
    highest_risk_observed = results["highest_risk_observed"]
    highest_risk_score_observed = results.get("highest_risk_score_observed", 0)
    max_zone_counts = results["max_zone_counts"]
    congestion_frames = results["congestion_frames"]
    speed_samples = results["speed_samples"]
    frame_df = results["frame_df"]
    person_df = results["person_df"]
    heatmap_image = results["heatmap_image"]

    overall_score = results.get("highest_risk_score_observed", latest_analytics.get("overall_risk_score", 0))
    overall_level, overall_color = score_to_risk_level(overall_score)
    peak_info = results.get("peak_analytics", {})
    overall_factor = peak_info.get("primary_factor") or latest_analytics.get("overall_primary_factor", "Normal crowd flow")

    incident_info = {
        "source": "Uploaded CCTV Footage",
        "zone": "Zone B & Zone C (Counter-Flow)",
        "level": overall_level,
        "score": overall_score,
        "reason": overall_factor,
        "color": overall_color
    }

    # Automatically show modal dialog if flagged
    if overall_score >= 50 and st.session_state.get("show_alert_popup", False):
        show_risk_alert_dialog(incident_info)

    # Automatically record in session alert history if not present
    if overall_score >= 50:
        history = st.session_state.setdefault("alert_history", [])
        if not any(a.get("source") == "Uploaded CCTV Footage" and a.get("score") == overall_score for a in history):
            history.insert(0, {
                "timestamp": time.strftime("%H:%M:%S"),
                "source": "Uploaded CCTV Footage",
                "zone": "Zone B / Zone C",
                "level": overall_level,
                "score": overall_score,
                "reason": overall_factor
            })

    # High-visibility Alert Banner across top of report if elevated
    if overall_score >= 50:
        st.markdown(
            f"<div style='background: linear-gradient(90deg, #ffefe9 0%, #fde4dc 100%); border-left: 6px solid {overall_color}; "
            f"border-radius: 12px; padding: 16px 20px; box-shadow: 0 4px 14px rgba(226,128,111,0.22); margin-bottom: 20px; "
            f"display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px'>"
            f"<div style='display:flex; align-items:center; gap:14px'>"
            f"<span style='font-size:2.2rem'>🚨</span>"
            f"<div>"
            f"<div style='font-family:Manrope,sans-serif; font-size:1.15rem; font-weight:800; color:#8a2718; text-transform:uppercase; letter-spacing:0.04em'>"
            f"ELEVATED RISK ALERT: {overall_level.upper()} DANGER LEVEL DETECTED"
            f"</div>"
            f"<div style='color:#541f16; font-size:0.95rem; margin-top:3px'>"
            f"<b>Peak Risk Index: {overall_score}/100</b> &nbsp;|&nbsp; <b>Primary Trigger:</b> {overall_factor} &nbsp;|&nbsp; <b>Source:</b> Analysed CCTV Video"
            f"</div>"
            f"</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Header with title and alert pop-up button
    head_col, btn_col = st.columns([2.5, 1.2])
    with head_col:
        st.markdown("<div class='cg-eyebrow'>Processing complete</div><h2 class='cg-title'>Crowd Analytics Report</h2><p class='cg-subtitle'>Video review and zone-level observations from the analysed footage.</p>", unsafe_allow_html=True)
    with btn_col:
        if overall_score >= 50:
            if st.button("🚨 View Safety Alert Pop-up", key="cg_btn_cctv_popup", type="primary", use_container_width=True):
                show_risk_alert_dialog(incident_info)

    video_col, zone_col = st.columns([1.65, 1], gap="large")
    with video_col:
        st.markdown("<div class='cg-panel-title'>▣ &nbsp; Analysed CCTV Video</div>", unsafe_allow_html=True)
        st.video(processed_video, format="video/mp4")
    with zone_col:
        st.markdown("<div class='cg-panel-title'>▥ &nbsp; Zone-wise Summary</div>", unsafe_allow_html=True)
        zone_cols = st.columns(3)
        zone_classes = ["cg-zone-a", "cg-zone-b", "cg-zone-c"]
        for index, zone in enumerate(ZONE_NAMES):
            details = latest_analytics["zones"][zone]
            count = details["people"]
            score = details.get("risk_score", 0)
            level = details.get("risk", "Low")
            color = details.get("risk_color", RISK_UI_COLORS.get(level, "#2f8f7a"))
            reason = details.get("primary_factor", "Normal crowd flow")
            with zone_cols[index]:
                st.markdown(
                    f"<div class='cg-zone {zone_classes[index]}'>"
                    f"<div class='cg-zone-name'>{zone}</div>"
                    f"<div class='cg-zone-number'>{count}</div>"
                    f"<div class='cg-zone-note'>people in final frame</div>"
                    f"<span class='cg-chip' style='background:{color};color:#ffffff;font-weight:700'>{score}/100 · {level}</span>"
                    f"<div style='font-size:.73rem;margin-top:6px;color:var(--cg-teal-dark);line-height:1.2'>{reason}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown(
            f"<div class='cg-panel' style='margin-top:14px'>"
            f"<div class='cg-panel-title'>◈ &nbsp; Peak Risk Index</div>"
            f"<div class='cg-risk-value' style='color:{overall_color}'>{overall_score}/100 &nbsp;<span style='font-size:1.1rem'>({overall_level})</span></div>"
            f"<div class='cg-risk-note'>Highest multi-factor risk score observed during this video</div>"
            f"<div style='font-size:.8rem;color:var(--cg-teal);margin-top:6px'><b>Factor:</b> {overall_factor}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='cg-eyebrow' style='margin-top:1.4rem'>Key observations</div><h2 class='cg-title' style='font-size:1.55rem'>At a glance</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Maximum people in one frame", max_people)
    col2.metric("Distinct track IDs observed", unique_track_ids_count)
    col3.metric("Peak Crowd Risk Index", f"{overall_score}/100 ({overall_level})")

    st.caption(
        "Track IDs are temporary ByteTrack identities. Risk Index (0-100) is an explainable multi-factor formula "
        "evaluating density (40%), velocity drop (30%), turbulence (20%), and congestion (10%)."
    )

    st.markdown("<div class='cg-panel-title' style='margin-top:1.2rem'>Zone-wise multi-factor analysis</div>", unsafe_allow_html=True)
    table_rows = []
    for zone in ZONE_NAMES:
        details = latest_analytics["zones"][zone]
        average_speed = details["average_speed"]
        table_rows.append({
            "Zone": zone,
            "People in final frame": details["people"],
            "Peak people observed": max_zone_counts[zone],
            "Avg movement (px/s)": (round(average_speed, 2) if average_speed is not None else "Not enough data"),
            "Risk Index (0-100)": details.get("risk_score", 0),
            "Risk Level": details.get("risk", "Low"),
            "Primary Contributing Factor": details.get("primary_factor", "Normal"),
            "Congestion Bottleneck": "Observed" if congestion_frames[zone] > 0 else "Not observed"
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    chart_col, movement_col = st.columns([1.35, 1], gap="large")
    with chart_col:
        st.markdown("<div class='cg-panel-title'>▤ &nbsp; Peak crowd count by zone</div>", unsafe_allow_html=True)
        chart_data = pd.DataFrame({
            "Zone": ZONE_NAMES,
            "Peak people observed": [max_zone_counts[zone] for zone in ZONE_NAMES]
        }).set_index("Zone")
        st.bar_chart(chart_data, color="#1f7f86")
    with movement_col:
        st.markdown("<div class='cg-panel-title'>⌁ &nbsp; Movement estimate</div>", unsafe_allow_html=True)
        if speed_samples:
            st.metric("Average observed movement", f"{sum(speed_samples) / len(speed_samples):.2f} px/sec")
            st.caption("Image-plane movement only; not calibrated to real-world speed.")
        else:
            st.info("Not enough continuous track observations to estimate movement.")

    st.markdown("<div class='cg-panel-title' style='margin-top:1.2rem'>Movement direction in final analysed frame</div>", unsafe_allow_html=True)
    direction_data = latest_analytics["movement_directions"]
    direction_cols = st.columns(5)
    for col, direction in zip(direction_cols, ["Left", "Right", "Up", "Down", "Stationary"]):
        col.metric(direction, direction_data[direction])

    st.caption(
        "Zone counts are based on detected people in the video frame. Movement is measured in pixels per second. "
        "Neither is calibrated to physical crowd density or real-world speed. Demo thresholds must not be used for real event safety decisions."
    )

    # Crowd behaviour over time, computed only from the logged observations.
    st.markdown(
        "<div class='cg-eyebrow' style='margin-top:1.4rem'>Crowd behaviour</div>"
        "<h2 class='cg-title' style='font-size:1.55rem'>Trend and presence heatmap</h2>",
        unsafe_allow_html=True
    )
    trend_col, heat_col = st.columns(2, gap="large")

    with trend_col:
        st.markdown("<div class='cg-panel-title'>⌁ &nbsp; People per zone over time</div>", unsafe_allow_html=True)
        trend = summarise_crowd_trend(frame_df)
        if trend is not None:
            trend_cols = st.columns(3)
            trend_cols[0].metric("Crowd trend", trend["label"])
            trend_cols[1].metric("Avg people, first quarter", f"{trend['early_average']:.1f}")
            trend_cols[2].metric("Avg people, last quarter", f"{trend['late_average']:.1f}")

        tracked_df = frame_df[frame_df["tracked_this_frame"]]
        if len(tracked_df) > 1:
            trend_chart = tracked_df.set_index("time_s")[
                ["zone_a_people", "zone_b_people", "zone_c_people"]
            ]
            trend_chart.columns = ZONE_NAMES
            st.line_chart(
                trend_chart,
                color=["#1f7f86", "#8a7fc4", "#e2806f"],
                x_label="Time (seconds)",
                y_label="People detected"
            )
        else:
            st.info("Not enough tracked frames to draw a trend.")

        st.caption(
            "The trend label compares the average total people in the first and last quarter "
            "of the video. It is descriptive, not a prediction."
        )

    with heat_col:
        st.markdown("<div class='cg-panel-title'>▦ &nbsp; Presence heatmap</div>", unsafe_allow_html=True)
        if heatmap_image is not None:
            st.image(heatmap_image)
            st.caption(
                "Shows where people were observed most often across the whole video "
                "(cumulative presence, not instantaneous density). Warmer colours mean more "
                "observations. Camera perspective is not corrected."
            )
        else:
            st.info("Not enough person observations to draw a heatmap.")

    # Data log: the measured foundation for heatmaps, trends, alerts and ML.
    with st.expander("Frame-by-frame data log (foundation for upcoming analytics)"):
        render_data_log_section(
            frame_df,
            person_df,
            "upload",
            "One frame record per video frame. Person observations are logged on frames where "
            "YOLO + ByteTrack actually ran (every third frame). Only measured values are stored."
        )


# =========================================================
# 6. MAIN DASHBOARD / NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("<div style='text-align:center;padding:14px 4px 22px'>"
                "<div style='font-size:2.6rem'>🛡️</div>"
                "<div style='font-family:Manrope,sans-serif;font-size:1.55rem;font-weight:800;color:#ffffff'>Crowd<span style='color:#bfe5d3'>Guard</span></div>"
                "<div style='color:#a9cfca;font-size:.82rem'>AI crowd monitoring system</div></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='cg-eyebrow'>Workspace</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Dashboard", "Live Analysis", "Upload Video", "Analytics", "About"],
        label_visibility="collapsed"
    )
    st.markdown("<br><div style='color:#cdbfee;font-style:italic;font-size:.9rem'>Safer crowds.<br>A safer tomorrow.</div>", unsafe_allow_html=True)
    st.markdown("<br><div style='color:#86b3b0;font-size:.72rem'>CrowdGuard · Development build</div>", unsafe_allow_html=True)

# Header: title on the left, compact system tags on the right.
st.markdown(
    "<div class='cg-head'>"
    "<div>"
    "<div class='cg-eyebrow'>AI-powered crowd monitoring</div>"
    "<div class='cg-title'>Crowd Analytics <span style='color:#1f7f86'>Dashboard</span></div>"
    "<div class='cg-subtitle'>Real-time insights · Zone monitoring · Video-based crowd observations</div>"
    "</div>"
    "<div class='cg-tags'>"
    "<span class='cg-tag'>Model <b>YOLOv8n</b></span>"
    "<span class='cg-tag'>Tracker <b>ByteTrack</b></span>"
    "<span class='cg-tag'>Zones <b>3</b></span>"
    "</div>"
    "</div>"
    "<div style='height:18px'></div>",
    unsafe_allow_html=True
)

if page == "Dashboard":
    st.markdown("<div class='cg-eyebrow'>System Command Center</div><h2 class='cg-title'>CrowdGuard Overview</h2><p class='cg-subtitle'>Real-time status, active safety parameters, and AI pipeline health.</p>", unsafe_allow_html=True)

    # Active safety alerts summary
    alert_history = st.session_state.get("alert_history", [])
    if alert_history:
        latest = alert_history[0]
        st.markdown(
            f"<div style='background:rgba(226,128,111,0.18);border:1.5px solid {RISK_UI_COLORS.get(latest['level'], '#e2806f')};"
            f"border-radius:14px;padding:14px 20px;margin-bottom:18px;display:flex;align-items:center;gap:14px'>"
            f"<span style='font-size:1.6rem'>⚠️</span>"
            f"<div><b style='color:#7d2d21;font-size:1.05rem'>Recent Incident: [{latest['level'].upper()} RISK] {latest['zone']} (Score: {latest['score']}/100) at {latest['timestamp']}</b>"
            f"<div style='color:#5c2b23;font-size:0.89rem;margin-top:2px'>{latest['reason']} · Source: {latest['source']}</div></div>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:rgba(47,143,122,0.12);border:1px solid #9dcbb8;border-radius:12px;padding:12px 18px;margin-bottom:18px;display:flex;align-items:center;gap:12px'>"
            "<span style='font-size:1.3rem'>✓</span>"
            "<div><b style='color:#14606b;font-size:.95rem'>Normal Operational Status:</b> "
            "<span style='color:#2d5b62;font-size:.9rem'>All monitored zones currently within normal parameters (Risk Index < 50).</span></div>"
            "</div>",
            unsafe_allow_html=True
        )

    # KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Detection Engine", "YOLOv8n", "Person Class 0")
    kpi2.metric("Tracking Protocol", "ByteTrack", "Temporal IDs")
    kpi3.metric("Risk Model", "4-Factor CRI", "0 - 100 Scale")
    kpi4.metric("Monitored Zones", "3 Zones", "A | B | C")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Monitored Zones Matrix
    st.markdown("<div class='cg-panel-title'>▥ &nbsp; Spatial Zone Configuration</div>", unsafe_allow_html=True)
    zcol1, zcol2, zcol3 = st.columns(3)
    with zcol1:
        st.markdown(
            "<div class='cg-zone cg-zone-a'>"
            "<div class='cg-zone-name'>Zone A (Left Corridor)</div>"
            "<div class='cg-zone-number' style='font-size:1.4rem;margin:12px 0 6px'>0 - 33%</div>"
            "<div class='cg-zone-note'>Entryway & Left Peripheral Flow</div>"
            "<span class='cg-chip'>Active Multi-Factor CRI</span>"
            "</div>",
            unsafe_allow_html=True
        )
    with zcol2:
        st.markdown(
            "<div class='cg-zone cg-zone-b'>"
            "<div class='cg-zone-name'>Zone B (Central Hub)</div>"
            "<div class='cg-zone-number' style='font-size:1.4rem;margin:12px 0 6px'>33% - 66%</div>"
            "<div class='cg-zone-note'>High-Density Convergence Hub</div>"
            "<span class='cg-chip'>Active Multi-Factor CRI</span>"
            "</div>",
            unsafe_allow_html=True
        )
    with zcol3:
        st.markdown(
            "<div class='cg-zone cg-zone-c'>"
            "<div class='cg-zone-name'>Zone C (Right Corridor)</div>"
            "<div class='cg-zone-number' style='font-size:1.4rem;margin:12px 0 6px'>66% - 100%</div>"
            "<div class='cg-zone-note'>Exitway & Right Peripheral Flow</div>"
            "<span class='cg-chip'>Active Multi-Factor CRI</span>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # System Architecture & Capabilities
    st.markdown("<div class='cg-panel-title'>◈ &nbsp; End-to-End System Pipeline</div>", unsafe_allow_html=True)
    pcol1, pcol2 = st.columns(2, gap="large")
    with pcol1:
        st.markdown(
            "<div class='cg-panel'>"
            "<b style='color:#14606b;font-size:1.05rem'>1. Real-Time Ingestion & CV Core</b><br><br>"
            "• <b>Input Sources:</b> Live WebRTC webcam feed (720p HD) and recorded CCTV video uploads (MP4, AVI, MOV).<br>"
            "• <b>Detection:</b> Ultralytics YOLOv8n deep person detector (COCO class 0, conf 0.40).<br>"
            "• <b>Multi-Object Tracking:</b> ByteTrack algorithm associating consecutive bounding boxes with persistent identity IDs and trajectory history.<br>"
            "• <b>Optimization:</b> Alternate-frame inference to ensure responsive 30 FPS video streaming with low CPU latency."
            "</div>",
            unsafe_allow_html=True
        )
    with pcol2:
        st.markdown(
            "<div class='cg-panel'>"
            "<b style='color:#14606b;font-size:1.05rem'>2. Explainable Crowd Risk Engine (0-100)</b><br><br>"
            "• <b>Density (40%):</b> Monocular crowd volume normalised against nominal zone capacity.<br>"
            "• <b>Velocity Drop (30%):</b> Detects sudden halts and deceleration bottleneck hazards.<br>"
            "• <b>Turbulence (20%):</b> Counter-flow vector collisions (Left vs Right, Up vs Down).<br>"
            "• <b>Congestion (10%):</b> Persistent low mobility and clustering.<br>"
            "• <b>Output Intelligence:</b> Deduplicated safety alerts, cumulative presence heatmaps, and frame-by-frame CSV telemetry."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.info("💡 **Navigation Hint:** Select **Live Analysis** from the sidebar for real-time webcam monitoring, or **Upload Video** to evaluate recorded CCTV footage.")

elif page == "Upload Video":
    start_analysis = False

    with st.container(key="cg_video_card"):
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown("<div class='cg-panel-title'>▣ &nbsp; CCTV Video Input</div>", unsafe_allow_html=True)
            st.caption("Choose a recorded CCTV clip to analyse. Your original video is not modified.")
            uploaded_video = st.file_uploader(
                "Upload CCTV or recorded video",
                type=["mp4", "avi", "mov", "mkv"],
                key="crowdguard_upload",
                label_visibility="collapsed"
            )

        with right:
            if uploaded_video is None:
                st.markdown(
                    "<div class='cg-hint' style='margin-top:2.4rem'>"
                    "<b>Supported formats</b><br>MP4 · AVI · MOV · MKV<br><br>"
                    "<b>How it works</b><br>YOLOv8 detects people, ByteTrack follows them, "
                    "the scene is split into three zones, and the 4-Factor Risk Index is calculated."
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<div style='height:2.4rem'></div><span class='cg-ready'>✓ Video ready for analysis</span>", unsafe_allow_html=True)
                with st.expander("Preview original video"):
                    st.video(uploaded_video.getvalue())
                start_analysis = st.button("Start crowd analysis", type="primary", key="cg_btn_start_cctv")
                st.caption("Tracking runs on every third frame to keep processing fast and efficient.")

    if uploaded_video is not None:
        video_signature = f"{uploaded_video.name}_{uploaded_video.size}"
        if st.session_state.get("uploaded_video_id") != video_signature:
            st.session_state["uploaded_video_id"] = video_signature
            st.session_state["uploaded_video_analysis"] = None

        if start_analysis:
            analysis_results = process_uploaded_video(uploaded_video)
            if analysis_results is not None:
                st.session_state["uploaded_video_analysis"] = analysis_results

        if st.session_state.get("uploaded_video_analysis") is not None:
            render_uploaded_video_report(st.session_state["uploaded_video_analysis"])
    else:
        st.session_state["uploaded_video_id"] = None
        st.session_state["uploaded_video_analysis"] = None

elif page == "Live Analysis":
    st.markdown("<div class='cg-panel-title'>◉ &nbsp; Live Camera Analysis</div>", unsafe_allow_html=True)
    st.caption("Click **START** below to connect your webcam. (Camera permission is saved automatically by your browser for localhost).")

    live_ctx = webrtc_streamer(
        key="crowdguard-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CameraProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": LIVE_CAMERA_WIDTH},
                "height": {"ideal": LIVE_CAMERA_HEIGHT},
                "frameRate": {"ideal": LIVE_CAMERA_FPS, "max": LIVE_CAMERA_FPS}
            },
            "audio": False
        },
        rtc_configuration={
            "iceServers": get_ice_servers()
        },
        async_processing=True
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cg-eyebrow'>Live monitoring report</div>"
        "<h2 class='cg-title' style='font-size:1.55rem'>Crowd Analytics Report</h2>"
        "<p class='cg-subtitle'>Live observations from your webcam, refreshed while the camera is running.</p>",
        unsafe_allow_html=True
    )

    if live_ctx.state.playing:
        @st.fragment(run_every=LIVE_REPORT_REFRESH)
        def render_live_analytics():
            processor = live_ctx.video_processor

            if processor is None:
                st.info("Initializing camera feed...")
                return

            snapshot = processor.get_snapshot()

            if snapshot is None:
                if getattr(processor, "model_error", None):
                    st.error(f"YOLO model could not load: {processor.model_error}")
                elif getattr(processor, "model_loading", False):
                    st.info("Camera connected. Loading YOLO detector; annotated analysis will start automatically...")
                else:
                    st.info("Waiting for the first analysed camera frame...")
                return

            # Check and record active alerts
            active_alerts = process_active_alerts(snapshot, source="Live Webcam")
            if active_alerts:
                render_alert_banner(active_alerts)

            # Top report: zone cards + overall indicator
            video_col, zone_col = st.columns([1.65, 1], gap="large")
            with video_col:
                st.markdown("<div class='cg-panel-title'>▣ &nbsp; Live analysed webcam feed</div>", unsafe_allow_html=True)
                st.caption("The annotated camera stream is displayed above; this report updates as new frames are analysed.")

            with zone_col:
                st.markdown("<div class='cg-panel-title'>▥ &nbsp; Zone-wise Summary</div>", unsafe_allow_html=True)
                zone_cols = st.columns(3)
                zone_classes = ["cg-zone-a", "cg-zone-b", "cg-zone-c"]
                for index, zone in enumerate(ZONE_NAMES):
                    details = snapshot["zones"][zone]
                    count = details["people"]
                    score = details.get("risk_score", 0)
                    level = details.get("risk", "Low")
                    color = details.get("risk_color", RISK_UI_COLORS.get(level, "#2f8f7a"))
                    reason = details.get("primary_factor", "Normal crowd flow")
                    with zone_cols[index]:
                        st.markdown(
                            f"<div class='cg-zone {zone_classes[index]}'>"
                            f"<div class='cg-zone-name'>{zone}</div>"
                            f"<div class='cg-zone-number'>{count}</div>"
                            f"<div class='cg-zone-note'>people in frame</div>"
                            f"<span class='cg-chip' style='background:{color};color:#ffffff;font-weight:700'>{score}/100 · {level}</span>"
                            f"<div style='font-size:.73rem;margin-top:6px;color:var(--cg-teal-dark);line-height:1.2'>{reason}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                highest_score = snapshot.get("highest_risk_score_observed", 0)
                highest_level, highest_color = score_to_risk_level(highest_score)
                overall_factor = snapshot.get("overall_primary_factor", "Normal crowd flow")

                st.markdown(
                    f"<div class='cg-panel' style='margin-top:14px'>"
                    f"<div class='cg-panel-title'>◈ &nbsp; Peak Risk Index</div>"
                    f"<div class='cg-risk-value' style='color:{highest_color}'>{highest_score}/100 &nbsp;<span style='font-size:1.1rem'>({highest_level})</span></div>"
                    f"<div class='cg-risk-note'>Highest multi-factor risk score observed during this live session</div>"
                    f"<div style='font-size:.8rem;color:var(--cg-teal);margin-top:6px'><b>Factor:</b> {overall_factor}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown(
                "<div class='cg-eyebrow' style='margin-top:1.4rem'>Key observations</div>"
                "<h2 class='cg-title' style='font-size:1.55rem'>At a glance</h2>",
                unsafe_allow_html=True
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Maximum people in one frame", snapshot["max_people_seen"])
            col2.metric("Distinct track IDs observed", snapshot["track_ids_seen"])
            col3.metric("Peak Crowd Risk Index", f"{highest_score}/100 ({highest_level})")

            st.caption(
                "Track IDs are temporary ByteTrack identities. Risk Index (0-100) is an explainable multi-factor formula "
                "evaluating density (40%), velocity drop (30%), turbulence (20%), and congestion (10%)."
            )

            st.markdown(
                "<div class='cg-panel-title' style='margin-top:1.2rem'>Zone-wise multi-factor analysis</div>",
                unsafe_allow_html=True
            )
            table_rows = []
            for zone in ZONE_NAMES:
                details = snapshot["zones"][zone]
                average_speed = details["average_speed"]
                table_rows.append({
                    "Zone": zone,
                    "People in current frame": details["people"],
                    "Peak people observed": snapshot["peak_zone_counts"][zone],
                    "Avg movement (px/s)": (
                        round(average_speed, 2) if average_speed is not None else "Not enough data"
                    ),
                    "Risk Index (0-100)": details.get("risk_score", 0),
                    "Risk Level": details.get("risk", "Low"),
                    "Primary Contributing Factor": details.get("primary_factor", "Normal"),
                    "Congestion Bottleneck": (
                        "Observed" if snapshot["congestion_frames"][zone] > 0 else "Not observed"
                    )
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            chart_col, movement_col = st.columns([1.35, 1], gap="large")
            with chart_col:
                st.markdown("<div class='cg-panel-title'>▤ &nbsp; Peak crowd count by zone</div>", unsafe_allow_html=True)
                chart_data = pd.DataFrame({
                    "Zone": ZONE_NAMES,
                    "Peak people observed": [
                        snapshot["peak_zone_counts"][zone] for zone in ZONE_NAMES
                    ]
                }).set_index("Zone")
                st.bar_chart(chart_data, color="#1f7f86")

            with movement_col:
                st.markdown("<div class='cg-panel-title'>⌁ &nbsp; Movement estimate</div>", unsafe_allow_html=True)
                if snapshot["speed_samples"]:
                    avg_speed = sum(snapshot["speed_samples"]) / len(snapshot["speed_samples"])
                    st.metric("Average observed movement", f"{avg_speed:.2f} px/sec")
                    st.caption("Image-plane movement only; not calibrated to real-world speed.")
                else:
                    st.info("Not enough continuous track observations to estimate movement.")

            st.markdown("<div class='cg-panel-title' style='margin-top:1.2rem'>Live movement direction</div>", unsafe_allow_html=True)
            direction_data = snapshot["movement_directions"]
            direction_cols = st.columns(5)
            for col, direction in zip(direction_cols, ["Left", "Right", "Up", "Down", "Stationary"]):
                col.metric(direction, direction_data[direction])

            # Live Session Heatmap and Trend Section
            st.markdown(
                "<div class='cg-eyebrow' style='margin-top:1.4rem'>Live crowd dynamics</div>"
                "<h2 class='cg-title' style='font-size:1.55rem'>Live Session Heatmap & Trend Chart</h2>",
                unsafe_allow_html=True
            )

            show_live_heatmap = st.checkbox(
                "🔥 Display live presence heatmap & time trend",
                key="cg_live_heatmap_toggle",
                value=True
            )

            if show_live_heatmap:
                live_trend_col, live_heat_col = st.columns(2, gap="large")

                with live_trend_col:
                    st.markdown("<div class='cg-panel-title'>⌁ &nbsp; Live people per zone over time</div>", unsafe_allow_html=True)
                    live_frame_df = pd.DataFrame(snapshot["frame_log"])
                    if not live_frame_df.empty and len(live_frame_df) > 1:
                        live_tracked = live_frame_df[live_frame_df["tracked_this_frame"]]
                        if len(live_tracked) > 1:
                            live_chart_data = live_tracked.set_index("time_s")[
                                ["zone_a_people", "zone_b_people", "zone_c_people"]
                            ]
                            live_chart_data.columns = ZONE_NAMES
                            st.line_chart(
                                live_chart_data,
                                color=["#1f7f86", "#8a7fc4", "#e2806f"],
                                x_label="Session Time (seconds)",
                                y_label="People detected"
                            )
                        else:
                            st.info("Accumulating tracked frames for trend chart...")
                    else:
                        st.info("Camera running. Time trend chart will appear after 2 seconds of video.")

                with live_heat_col:
                    st.markdown("<div class='cg-panel-title'>▦ &nbsp; Live presence heatmap</div>", unsafe_allow_html=True)
                    live_person_df = pd.DataFrame(snapshot["person_log"])
                    bg_frame = snapshot.get("background_frame")
                    if not live_person_df.empty and bg_frame is not None:
                        live_heatmap = build_presence_heatmap(live_person_df, bg_frame)
                        if live_heatmap is not None:
                            st.image(live_heatmap, output_format="JPEG")
                            st.caption("Cumulative presence heatmap over your webcam scene (warmer colours indicate high persistence).")
                        else:
                            st.info("Generating smoothed presence heatmap from live observations...")
                    else:
                        st.info("Heatmap accumulates as people move in your webcam feed.")

            # Data log section
            st.markdown(
                "<div class='cg-panel-title' style='margin-top:1.2rem'>Frame-by-frame data log</div>",
                unsafe_allow_html=True
            )
            log_col1, log_col2 = st.columns(2)
            log_col1.metric("Frame records logged", snapshot.get("frame_log_count", len(snapshot["frame_log"])))
            log_col2.metric("Person observations logged", snapshot.get("person_log_count", len(snapshot["person_log"])))

            if st.checkbox("Show log preview and downloads", key="cg_live_log_toggle"):
                full_frame_log, full_person_log = processor.get_full_logs()
                render_data_log_section(
                    pd.DataFrame(full_frame_log),
                    pd.DataFrame(full_person_log),
                    "live",
                    f"Live logs keep only the most recent {LIVE_FRAME_LOG_LIMIT} frame records and "
                    f"{LIVE_PERSON_LOG_LIMIT} person observations, keeping memory bounded."
                )

        render_live_analytics()
    elif live_ctx.state.signalling:
        st.info("⏳ Connecting to camera... Handshake in progress, video feed will appear shortly.")
    else:
        st.info("Click **START** above to turn on your webcam. The live crowd analytics report will appear here.")

elif page == "Analytics":
    st.markdown("<div class='cg-eyebrow'>Intelligence Center</div><h2 class='cg-title'>Crowd Analytics & Risk Assessment</h2><p class='cg-subtitle'>Deep-dive telemetry, mathematical risk formulation, and session incident logs.</p>", unsafe_allow_html=True)

    # 1. Incident Alert History
    st.markdown("<div class='cg-panel-title'>🚨 &nbsp; Session Incident & Safety Alert History</div>", unsafe_allow_html=True)
    alert_history = st.session_state.get("alert_history", [])
    if alert_history:
        alert_df = pd.DataFrame(alert_history)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
        st.caption("Incident history logs every transition into High or Critical risk with an 8-second debounce.")
    else:
        st.markdown(
            "<div class='cg-hint'>"
            "<b>No elevated risk incidents recorded in this session.</b><br>"
            "Alerts trigger automatically when any zone reaches High (Score ≥ 50) or Critical (Score ≥ 75) risk."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # 2. Risk Index Methodology & Formula
    st.markdown("<div class='cg-panel-title'>📐 &nbsp; Explainable Multi-Factor Crowd Risk Index (0-100)</div>", unsafe_allow_html=True)
    st.caption("Rather than an opaque black-box classifier, CrowdGuard employs an explainable physics-informed crowd dynamics formulation:")

    cri_breakdown_data = pd.DataFrame([
        {
            "Risk Factor": "1. Normalized Density Concentration",
            "Weight": "40%",
            "Mathematical Formulation": "min(1.0, Zone_Count / 15.0) × 40",
            "Physical Crowd Hazard": "High volume increases physical crowding and spatial compression."
        },
        {
            "Risk Factor": "2. Velocity Hazard & Sudden Stoppage",
            "Weight": "30%",
            "Mathematical Formulation": "scaled(1.0 - Speed / 35.0) × 30",
            "Physical Crowd Hazard": "Deceleration or sudden halts in dense crowds create crush and stampede bottlenecks."
        },
        {
            "Risk Factor": "3. Direction Turbulence & Conflict",
            "Weight": "20%",
            "Mathematical Formulation": "2 × min(Opposing_Flows) / Total_Flow × 20",
            "Physical Crowd Hazard": "Counter-flows (e.g. Left vs Right) cause turbulent collisions and flow resistance."
        },
        {
            "Risk Factor": "4. Congestion Persistence Flag",
            "Weight": "10%",
            "Mathematical Formulation": "10 if (Count ≥ 5 and Speed < 20 px/s) else 0",
            "Physical Crowd Hazard": "Identifies sustained stagnation and prolonged crowd entrapment."
        }
    ])
    st.dataframe(cri_breakdown_data, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # 3. Risk Categories Table
    st.markdown("<div class='cg-panel-title'>🏷️ &nbsp; Risk Severity Categorization</div>", unsafe_allow_html=True)
    risk_cat_df = pd.DataFrame([
        {"Score Range": "0 – 24", "Category": "Low Risk", "Color Code": "#2f8f7a (Green)", "Operational Meaning": "Fluid crowd movement; normal density and speed."},
        {"Score Range": "25 – 49", "Category": "Medium Risk", "Color Code": "#c99a3b (Yellow)", "Operational Meaning": "Moderate crowding or slight flow deceleration; monitor closely."},
        {"Score Range": "50 – 74", "Category": "High Risk", "Color Code": "#e2806f (Coral)", "Operational Meaning": "Significant bottleneck, counter-flow, or density spike; proactive dispersal recommended."},
        {"Score Range": "75 – 100", "Category": "Critical Risk", "Color Code": "#cc5a5a (Red)", "Operational Meaning": "Severe crush or stampede danger; immediate crowd management intervention required."}
    ])
    st.dataframe(risk_cat_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # 4. Session Data Downloads
    st.markdown("<div class='cg-panel-title'>💾 &nbsp; Telemetry Data Export Center</div>", unsafe_allow_html=True)
    cctv_results = st.session_state.get("uploaded_video_analysis")
    if cctv_results is not None:
        st.write("Export telemetry data from your most recently analysed CCTV video:")
        render_data_log_section(
            cctv_results["frame_df"],
            cctv_results["person_df"],
            "analytics_export",
            "Contains exact timestamps, coordinates, speeds, and 4-factor risk scores."
        )
    else:
        st.info("No recorded CCTV analysis stored in session. Run an analysis on the **Upload Video** page to export complete telemetry logs here.")

else:  # About page
    st.markdown("<div class='cg-eyebrow'>Academic Project Documentation</div><h2 class='cg-title'>About CrowdGuard</h2><p class='cg-subtitle'>AI-Based Crowd Behaviour Analysis and Risk Prediction System.</p>", unsafe_allow_html=True)

    col_about1, col_about2 = st.columns(2, gap="large")
    with col_about1:
        st.markdown(
            "<div class='cg-panel'>"
            "<b style='color:#14606b;font-size:1.05rem'>Project Overview & Problem Statement</b><br><br>"
            "Crowd disasters (crushes, stampedes, and turbulent bottlenecks) frequently result from delayed detection of localized density spikes and sudden flow stoppages. "
            "<b>CrowdGuard</b> is an intelligent computer vision system developed to continuously monitor crowd movements, evaluate flow turbulence, and provide explainable risk indicators before hazardous conditions escalate."
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='cg-panel' style='margin-top:14px'>"
            "<b style='color:#14606b;font-size:1.05rem'>Technical Architecture</b><br><br>"
            "• <b>Deep Vision Detector:</b> YOLOv8n (pre-trained on COCO person class 0).<br>"
            "• <b>Multi-Object Tracker:</b> ByteTrack associating spatial detections into temporal tracklets.<br>"
            "• <b>Spatial Division:</b> Image-plane partitioned into Zone A, Zone B, and Zone C.<br>"
            "• <b>Motion Telemetry:</b> Velocity estimation and direction classification (Left, Right, Up, Down, Stationary).<br>"
            "• <b>Risk Formulation:</b> Explainable Multi-Factor Crowd Risk Index (0–100).<br>"
            "• <b>Visual Intelligence:</b> HD WebRTC streaming, cumulative presence heatmaps, and trend charts."
            "</div>",
            unsafe_allow_html=True
        )

    with col_about2:
        st.markdown(
            "<div class='cg-panel'>"
            "<b style='color:#14606b;font-size:1.05rem'>Why Explainable Risk Index vs Black-Box ML?</b><br><br>"
            "In real-world crowd safety engineering, genuine stampede and crowd disaster video datasets with fine-grained ground truth are extremely rare and hazardous to stage. "
            "Rather than fabricating synthetic labels to train an unverified machine learning classifier, CrowdGuard implements a transparent, physics-informed mathematical index based on proven crowd dynamics principles (Fruin's Level of Service & Helbing crowd turbulence)."
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='cg-panel' style='margin-top:14px'>"
            "<b style='color:#14606b;font-size:1.05rem'>Academic Disclaimer & Limitations</b><br><br>"
            "• <b>Pixel-Speed Approximation:</b> Velocity is measured in pixels per second on the 2D image plane and is not calibrated to physical metres/sec.<br>"
            "• <b>Perspective Foreshortening:</b> People further from the monocular camera appear smaller and cover fewer pixels per step.<br>"
            "• <b>Demonstration Prototype:</b> Built as a final-year engineering prototype for academic demonstration and not certified as safety-critical public infrastructure."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.success("🎓 **Viva Ready:** All core features—YOLOv8n detection, ByteTrack tracking, 4-Factor Risk Index, live/CCTV heatmaps, and alert history—are demonstrable.")

st.markdown("<hr><div style='text-align:center;color:#6f8e90;font-size:.78rem'>CrowdGuard · AI-based crowd monitoring prototype</div>", unsafe_allow_html=True)
