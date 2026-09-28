import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.energy_manager import compute_energy_balance


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Energy AI | Smart Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS — KEEP YOUR EXISTING PROJECT STRUCTURE
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEMAND_CSV_PATH = os.path.join(PROJECT_ROOT, "output", "demand_forecast.csv")
SOLAR_CSV_PATH = os.path.join(
    PROJECT_ROOT, "model2", "predictions", "test_predictions.csv"
)


# ============================================================
# GLOBAL CSS — FUTURISTIC ENERGY COMMAND CENTER
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

        :root {
            --bg: #061017;
            --bg-2: #081720;
            --panel: rgba(12, 29, 39, 0.78);
            --panel-solid: #0b202b;
            --border: rgba(123, 184, 205, 0.14);
            --border-bright: rgba(77, 219, 239, 0.28);
            --cyan: #42d9ee;
            --cyan-soft: rgba(66, 217, 238, 0.12);
            --green: #49e6a1;
            --yellow: #f5c85b;
            --blue: #5b9dff;
            --red: #ff6b6b;
            --text: #eaf6f8;
            --muted: #78939e;
            --muted-2: #526b75;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 75% 5%, rgba(31, 141, 168, 0.11), transparent 28%),
                radial-gradient(circle at 12% 75%, rgba(38, 102, 148, 0.08), transparent 28%),
                linear-gradient(135deg, #040c11 0%, #061017 45%, #07151d 100%);
            color: var(--text);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(110, 190, 205, 0.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(110, 190, 205, 0.018) 1px, transparent 1px);
            background-size: 48px 48px;
            mask-image: linear-gradient(to bottom, black, transparent 85%);
        }

        header[data-testid="stHeader"] {
            background: rgba(4, 12, 17, 0.82);
        }

        .block-container {
            max-width: 1540px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(5, 18, 25, 0.98), rgba(4, 13, 19, 0.98));
            border-right: 1px solid rgba(90, 180, 201, 0.12);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.25rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 6px 8px 22px 8px;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #061017;
            background: linear-gradient(135deg, #55e8f6, #2ab9d3);
            box-shadow: 0 0 28px rgba(66, 217, 238, 0.25);
            font-size: 22px;
            font-weight: 800;
        }

        .brand-name {
            font-family: 'Manrope', sans-serif;
            font-weight: 800;
            letter-spacing: .04em;
            font-size: 15px;
            color: #eaf9fb;
        }

        .brand-sub {
            font-size: 10px;
            color: var(--muted);
            margin-top: 2px;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .side-label {
            color: #526f79;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin: 8px 8px 10px;
        }

        div[data-testid="stRadio"] > label {
            display: none;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 4px;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 11px;
            padding: 9px 11px;
            color: #6e8994;
            transition: all .2s ease;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
            background: rgba(66, 217, 238, 0.06);
            border-color: rgba(66, 217, 238, 0.10);
            color: #bceef4;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(90deg, rgba(66, 217, 238, .13), rgba(66, 217, 238, .025));
            border-color: rgba(66, 217, 238, .24);
            color: #e5fbff;
            box-shadow: inset 3px 0 0 var(--cyan);
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
            display: none;
        }

        .system-card {
            margin: 24px 6px 0;
            padding: 15px;
            border-radius: 14px;
            border: 1px solid rgba(73, 230, 161, .15);
            background: linear-gradient(145deg, rgba(19, 52, 54, .60), rgba(8, 27, 34, .55));
        }

        .system-title {
            font-size: 10px;
            color: #6d8991;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .system-online {
            margin-top: 8px;
            font-size: 16px;
            font-weight: 800;
            color: var(--green);
        }

        .pulse {
            display: inline-block;
            width: 7px;
            height: 7px;
            background: var(--green);
            border-radius: 50%;
            margin-right: 7px;
            box-shadow: 0 0 12px rgba(73,230,161,.75);
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0%,100% { opacity: 1; box-shadow: 0 0 12px rgba(73,230,161,.75); }
            50% { opacity: .42; box-shadow: 0 0 4px rgba(73,230,161,.2); }
        }

        .system-desc {
            color: #5e7881;
            font-size: 10px;
            margin-top: 3px;
        }

        /* Header */
        .topline {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
            margin: 4px 0 25px;
        }

        .eyebrow {
            color: var(--cyan);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .17em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .main-title {
            font-family: 'Manrope', sans-serif;
            font-size: clamp(28px, 3.2vw, 46px);
            line-height: 1.05;
            letter-spacing: -.045em;
            font-weight: 800;
            color: #f0fbfc;
            margin: 0;
        }

        .main-subtitle {
            color: #718b95;
            font-size: 13px;
            margin-top: 10px;
            max-width: 760px;
            line-height: 1.7;
        }

        .status-cluster {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .status-pill {
            border: 1px solid rgba(95, 158, 175, .17);
            background: rgba(10, 28, 37, .75);
            border-radius: 999px;
            padding: 8px 11px;
            font-size: 10px;
            color: #77939d;
        }

        .status-pill strong {
            color: #d7f7fa;
            font-weight: 600;
        }

        /* Cards */
        .card {
            border: 1px solid var(--border);
            background:
                linear-gradient(145deg, rgba(16, 38, 48, .76), rgba(7, 21, 28, .72));
            border-radius: 18px;
            padding: 19px;
            box-shadow:
                0 18px 45px rgba(0,0,0,.16),
                inset 0 1px 0 rgba(255,255,255,.018);
            transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
        }

        .card:hover {
            border-color: var(--border-bright);
            box-shadow:
                0 20px 55px rgba(0,0,0,.20),
                0 0 30px rgba(66,217,238,.035);
        }

        .section-card {
            margin-top: 18px;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 13px;
        }

        .card-title {
            font-family: 'Manrope', sans-serif;
            font-size: 15px;
            font-weight: 800;
            color: #dff5f7;
            letter-spacing: -.01em;
        }

        .card-caption {
            color: #607a84;
            font-size: 10px;
            margin-top: 3px;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(66,217,238,.16);
            background: rgba(66,217,238,.06);
            color: #70ddeb;
            border-radius: 999px;
            padding: 5px 8px;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        /* KPI */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 20px 0;
        }

        .kpi {
            position: relative;
            overflow: hidden;
            min-height: 142px;
            border-radius: 17px;
            border: 1px solid var(--border);
            padding: 17px;
            background:
                radial-gradient(circle at 90% 0%, rgba(66,217,238,.075), transparent 38%),
                linear-gradient(145deg, rgba(14,35,45,.88), rgba(7,20,27,.88));
            box-shadow: 0 15px 35px rgba(0,0,0,.14);
        }

        .kpi::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(66,217,238,.35), transparent);
        }

        .kpi-label {
            color: #66818b;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .11em;
            font-weight: 700;
        }

        .kpi-value {
            margin-top: 15px;
            font-family: 'Manrope', sans-serif;
            color: #effcfd;
            font-size: 27px;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -.045em;
        }

        .kpi-unit {
            font-size: 11px;
            color: #78949d;
            margin-left: 4px;
            letter-spacing: 0;
        }

        .kpi-meta {
            margin-top: 12px;
            color: #58727b;
            font-size: 10px;
        }

        .dot-cyan { color: var(--cyan); }
        .dot-yellow { color: var(--yellow); }
        .dot-blue { color: var(--blue); }
        .dot-green { color: var(--green); }

        /* Battery */
        .battery-shell {
            width: 100%;
            height: 22px;
            padding: 3px;
            border-radius: 8px;
            border: 1px solid rgba(73,230,161,.2);
            background: #07141b;
            margin: 12px 0 7px;
        }

        .battery-fill {
            height: 100%;
            border-radius: 5px;
            background: linear-gradient(90deg, #24c987, #62edb0);
            box-shadow: 0 0 18px rgba(73,230,161,.2);
            transition: width .4s ease;
        }

        .battery-pct {
            font-family: 'Manrope', sans-serif;
            font-size: 34px;
            font-weight: 800;
            color: #eafff6;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 9px;
            margin-top: 14px;
        }

        .mini-stat {
            border: 1px solid rgba(111,157,170,.10);
            border-radius: 11px;
            padding: 10px;
            background: rgba(2,12,17,.28);
        }

        .mini-label {
            font-size: 9px;
            color: #59747e;
            text-transform: uppercase;
            letter-spacing: .07em;
        }

        .mini-value {
            margin-top: 5px;
            font-size: 13px;
            font-weight: 700;
            color: #d9edf0;
        }

        /* Grid status */
        .grid-state {
            display: flex;
            align-items: center;
            gap: 9px;
            margin: 6px 0 16px;
            font-size: 18px;
            font-weight: 800;
            color: #dffaf0;
        }

        .state-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 14px rgba(73,230,161,.65);
        }

        .state-dot.warn {
            background: var(--yellow);
            box-shadow: 0 0 14px rgba(245,200,91,.55);
        }

        .state-dot.alert {
            background: var(--red);
            box-shadow: 0 0 14px rgba(255,107,107,.55);
        }

        /* Model cards */
        .model-card {
            min-height: 245px;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px;
            background: linear-gradient(145deg, rgba(14,34,44,.82), rgba(7,19,26,.82));
        }

        .model-name {
            color: #e9f8fa;
            font-family: 'Manrope', sans-serif;
            font-size: 17px;
            font-weight: 800;
        }

        .model-type {
            color: var(--cyan);
            font-size: 10px;
            margin-top: 5px;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .model-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 20px;
        }

        .model-metric {
            padding: 10px;
            border-radius: 10px;
            background: rgba(3,13,18,.35);
            border: 1px solid rgba(105,153,166,.09);
        }

        .model-metric-label {
            font-size: 8px;
            color: #5e7881;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .model-metric-value {
            margin-top: 5px;
            font-size: 14px;
            font-weight: 800;
            color: #e6f6f8;
        }

        .predictors {
            margin-top: 17px;
            color: #718b95;
            font-size: 10px;
            line-height: 1.7;
        }

        /* Controls */
        div[data-testid="stSlider"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stDateInput"] label,
        div[data-testid="stRadio"] label {
            color: #829ca5 !important;
            font-size: 10px !important;
            font-weight: 600 !important;
            letter-spacing: .05em;
        }

        div[data-baseweb="select"] > div {
            background: #091a22 !important;
            border-color: rgba(105,153,166,.15) !important;
            border-radius: 10px !important;
        }

        input {
            background: #091a22 !important;
            color: #e5f7f9 !important;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            border-radius: 10px !important;
            border: 1px solid rgba(66,217,238,.24) !important;
            background: linear-gradient(135deg, rgba(66,217,238,.16), rgba(66,217,238,.05)) !important;
            color: #dffcff !important;
            font-weight: 700 !important;
            font-size: 11px !important;
            min-height: 40px !important;
            transition: all .2s ease !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(66,217,238,.55) !important;
            background: linear-gradient(135deg, rgba(66,217,238,.24), rgba(66,217,238,.08)) !important;
            box-shadow: 0 0 24px rgba(66,217,238,.08);
            transform: translateY(-1px);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background: rgba(4,14,20,.62);
            padding: 5px;
            border: 1px solid rgba(105,153,166,.10);
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 37px;
            color: #69848e;
            border-radius: 8px;
            padding: 0 15px;
            font-size: 10px;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            color: #dffbfe !important;
            background: rgba(66,217,238,.10);
        }

        /* Dataframe */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(105,153,166,.11);
            border-radius: 13px;
            overflow: hidden;
        }

        /* Alerts */
        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid rgba(105,153,166,.12);
        }

        /* Hide Streamlit decoration */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Mobile */
        @media (max-width: 900px) {
            .kpi-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .topline {
                flex-direction: column;
            }
            .status-cluster {
                justify-content: flex-start;
            }
        }

        @media (max-width: 600px) {
            .kpi-grid {
                grid-template-columns: 1fr;
            }
            .mini-grid,
            .model-metrics {
                grid-template-columns: 1fr;
            }
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_datasets():
    df_demand = None
    df_solar = None

    if os.path.exists(DEMAND_CSV_PATH):
        df_demand = pd.read_csv(DEMAND_CSV_PATH)
        if "timestamp" in df_demand.columns:
            df_demand["timestamp"] = pd.to_datetime(df_demand["timestamp"])

    if os.path.exists(SOLAR_CSV_PATH):
        df_solar = pd.read_csv(SOLAR_CSV_PATH)
        solar_time_col = (
            "Timestamp" if "Timestamp" in df_solar.columns else "timestamp"
        )
        if solar_time_col in df_solar.columns:
            df_solar["timestamp"] = pd.to_datetime(df_solar[solar_time_col])

    return df_demand, df_solar


# ============================================================
# HELPERS
# ============================================================
def fmt_num(value, decimals=1):
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "—"


def safe_sum(df, col):
    if df is None or col not in df.columns:
        return 0.0
    return float(df[col].sum())


def render_brand():
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">⚡</div>
            <div>
                <div class="brand-name">ENERGY AI</div>
                <div class="brand-sub">Smart Grid Intelligence</div>
            </div>
        </div>
        <div class="side-label">Operations</div>
        """,
        unsafe_allow_html=True,
    )


def render_system_status():
    st.markdown(
        """
        <div class="system-card">
            <div class="system-title">System Status</div>
            <div class="system-online"><span class="pulse"></span>ONLINE</div>
            <div class="system-desc">All energy services nominal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(mode):
    st.markdown(
        f"""
        <div class="topline">
            <div>
                <div class="eyebrow">Workspace / Command Center</div>
                <div class="main-title">Smart Energy Intelligence</div>
                <div class="main-subtitle">
                    AI-powered electricity demand forecasting, renewable generation
                    prediction and intelligent energy storage management.
                </div>
            </div>
            <div class="status-cluster">
                <div class="status-pill">● <strong>Live environment</strong></div>
                <div class="status-pill">Mode: <strong>{mode}</strong></div>
                <div class="status-pill">Engine: <strong>Energy AI v2.4</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(kpis):
    demand = kpis.get("total_demand_kWh", 0)
    solar = kpis.get("total_solar_kWh", 0)
    grid = kpis.get("total_grid_import_kWh", 0)
    renewable = kpis.get("overall_renewable_contribution_pct", 0)

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi">
                <div class="kpi-label"><span class="dot-cyan">●</span> Total Demand</div>
                <div class="kpi-value">{fmt_num(demand)}<span class="kpi-unit">kWh</span></div>
                <div class="kpi-meta">Predicted electricity requirement</div>
            </div>
            <div class="kpi">
                <div class="kpi-label"><span class="dot-yellow">●</span> Solar Generation</div>
                <div class="kpi-value">{fmt_num(solar)}<span class="kpi-unit">kWh</span></div>
                <div class="kpi-meta">Forecast renewable production</div>
            </div>
            <div class="kpi">
                <div class="kpi-label"><span class="dot-blue">●</span> Grid Import</div>
                <div class="kpi-value">{fmt_num(grid)}<span class="kpi-unit">kWh</span></div>
                <div class="kpi-meta">Net external grid requirement</div>
            </div>
            <div class="kpi">
                <div class="kpi-label"><span class="dot-green">●</span> Renewable Share</div>
                <div class="kpi-value">{fmt_num(renewable)}<span class="kpi-unit">%</span></div>
                <div class="kpi-meta">Contribution of renewable energy</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_theme(fig, height=460):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,15,21,0.35)",
        font=dict(family="Inter", color="#78939e", size=10),
        hovermode="x unified",
        margin=dict(l=12, r=12, t=18, b=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10, color="#89a3ad"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#0a202a",
            bordercolor="#23434e",
            font=dict(color="#eaf8fa", size=11),
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(105,153,166,.10)",
        tickfont=dict(color="#607983", size=9),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(105,153,166,.075)",
        zeroline=False,
        linecolor="rgba(105,153,166,.10)",
        tickfont=dict(color="#607983", size=9),
    )
    return fig


def render_energy_chart(df_res):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_res["timestamp"],
            y=df_res["demand_kWh"],
            name="Demand",
            mode="lines",
            line=dict(color="#4ddbec", width=2.8),
            hovertemplate="Demand: %{y:,.1f} kWh<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_res["timestamp"],
            y=df_res["solar_kWh"],
            name="Solar",
            mode="lines",
            line=dict(color="#f5c85b", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(245,200,91,.09)",
            hovertemplate="Solar: %{y:,.1f} kWh<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_res["timestamp"],
            y=df_res["net_grid_import_kWh"],
            name="Grid Import",
            mode="lines",
            line=dict(color="#649cff", width=2.2, dash="dot"),
            hovertemplate="Grid: %{y:,.1f} kWh<extra></extra>",
        ),
        secondary_y=False,
    )

    if "renewable_contribution_pct" in df_res.columns:
        fig.add_trace(
            go.Scatter(
                x=df_res["timestamp"],
                y=df_res["renewable_contribution_pct"],
                name="Renewable %",
                mode="lines",
                line=dict(color="#49e6a1", width=1.5, dash="dash"),
                visible="legendonly",
                hovertemplate="Renewable: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    plot_theme(fig, 500)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Energy (kWh / hour)", secondary_y=False)
    fig.update_yaxes(
        title_text="Renewable Share (%)",
        range=[0, 105],
        secondary_y=True,
        showgrid=False,
    )
    return fig


def render_battery_chart(df_res):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.56, 0.44],
    )

    fig.add_trace(
        go.Scatter(
            x=df_res["timestamp"],
            y=df_res["battery_soc_pct"],
            name="Battery SoC",
            mode="lines",
            line=dict(color="#49e6a1", width=2.7),
            fill="tozeroy",
            fillcolor="rgba(73,230,161,.10)",
            hovertemplate="SoC: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df_res["timestamp"],
            y=df_res["battery_charge_kWh"],
            name="Charge",
            marker_color="#49e6a1",
            opacity=.82,
            hovertemplate="Charge: %{y:,.1f} kWh<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df_res["timestamp"],
            y=-df_res["battery_discharge_kWh"],
            name="Discharge",
            marker_color="#ff6b6b",
            opacity=.82,
            hovertemplate="Discharge: %{y:,.1f} kWh<extra></extra>",
        ),
        row=2,
        col=1,
    )

    plot_theme(fig, 500)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="SoC (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Charge / Discharge", row=2, col=1)
    return fig


def render_battery_panel(df_res, kpis, bess_capacity):
    latest_soc = float(df_res["battery_soc_pct"].iloc[-1])
    total_charged = safe_sum(df_res, "battery_charge_kWh")
    total_discharged = safe_sum(df_res, "battery_discharge_kWh")
    surplus = float(kpis.get("total_surplus_export_kWh", 0))

    with st.container():
        st.markdown(
            """
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Battery Intelligence</div>
                        <div class="card-caption">Energy storage state and dispatch telemetry</div>
                    </div>
                    <div class="tag">BESS</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            f"""
            <div class="card" style="margin-top:12px; min-height:500px;">
                <div class="card-caption">STATE OF CHARGE</div>
                <div class="battery-pct">{latest_soc:.1f}%</div>
                <div class="battery-shell">
                    <div class="battery-fill" style="width:{max(0,min(100,latest_soc))}%;"></div>
                </div>
                <div class="card-caption">Current battery reserve</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            render_battery_chart(df_res),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )

    with right:
        st.markdown(
            f"""
            <div class="card" style="margin-top:12px;">
                <div class="card-header">
                    <div class="card-title">Storage Telemetry</div>
                    <div class="tag">LIVE</div>
                </div>
                <div class="mini-grid">
                    <div class="mini-stat">
                        <div class="mini-label">Capacity</div>
                        <div class="mini-value">{fmt_num(bess_capacity)} kWh</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Current SoC</div>
                        <div class="mini-value">{latest_soc:.1f}%</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Total Charged</div>
                        <div class="mini-value">{fmt_num(total_charged)} kWh</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Total Discharged</div>
                        <div class="mini-value">{fmt_num(total_discharged)} kWh</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Solar Export</div>
                        <div class="mini-value">{fmt_num(surplus)} kWh</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Dispatch</div>
                        <div class="mini-value">{df_res['dispatch_status'].iloc[-1] if 'dispatch_status' in df_res.columns else 'Active'}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        last_rows = df_res[
            [
                c
                for c in [
                    "timestamp",
                    "demand_kWh",
                    "solar_kWh",
                    "battery_soc_pct",
                    "dispatch_status",
                ]
                if c in df_res.columns
            ]
        ].tail(8)

        st.markdown(
            """
            <div class="card" style="margin-top:12px;">
                <div class="card-title">Recent Dispatch Log</div>
                <div class="card-caption">Latest energy-management events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(last_rows, use_container_width=True, hide_index=True, height=310)


def render_grid_status(df_res, kpis):
    grid = float(kpis.get("total_grid_import_kWh", 0))
    renewable = float(kpis.get("overall_renewable_contribution_pct", 0))
    surplus = float(kpis.get("total_surplus_export_kWh", 0))

    latest_status = (
        str(df_res["dispatch_status"].iloc[-1])
        if "dispatch_status" in df_res.columns
        else "Grid serving load"
    )

    if renewable >= 60:
        state = "STABLE"
        dot_class = ""
    elif renewable >= 30:
        state = "BALANCED"
        dot_class = "warn"
    else:
        state = "GRID DEPENDENT"
        dot_class = "alert"

    st.markdown(
        f"""
        <div class="card" style="height:100%;">
            <div class="card-header">
                <div>
                    <div class="card-title">Grid Status</div>
                    <div class="card-caption">Integrated power-flow condition</div>
                </div>
                <div class="tag">GRID</div>
            </div>

            <div class="grid-state">
                <span class="state-dot {dot_class}"></span>{state}
            </div>

            <div class="mini-grid">
                <div class="mini-stat">
                    <div class="mini-label">Grid Import</div>
                    <div class="mini-value">{fmt_num(grid)} kWh</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-label">Renewable</div>
                    <div class="mini-value">{renewable:.1f}%</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-label">Solar Surplus</div>
                    <div class="mini-value">{fmt_num(surplus)} kWh</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-label">Latest Action</div>
                    <div class="mini-value">{latest_status}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(
    title,
    model_type,
    r2,
    mae,
    rmse,
    predictors,
):
    rmse_html = (
        f'<div class="model-metric"><div class="model-metric-label">RMSE</div>'
        f'<div class="model-metric-value">{rmse}</div></div>'
        if rmse is not None
        else ""
    )

    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-name">{title}</div>
            <div class="model-type">{model_type}</div>

            <div class="model-metrics">
                <div class="model-metric">
                    <div class="model-metric-label">R² Score</div>
                    <div class="model-metric-value">{r2}</div>
                </div>
                <div class="model-metric">
                    <div class="model-metric-label">MAE</div>
                    <div class="model-metric-value">{mae}</div>
                </div>
                {rmse_html}
            </div>

            <div class="predictors">
                <strong style="color:#9bb3bb;">Key predictors:</strong><br>
                {predictors}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    render_brand()

    nav_options = [
        "Overview",
        "Energy Forecast",
        "Solar Analytics",
        "Battery & Storage",
        "AI Models",
        "Data Explorer",
        "Scenario Simulator",
    ]

    selected_nav = st.radio(
        "Navigation",
        nav_options,
        index=0,
        key="navigation",
    )

    st.markdown("---")
    st.markdown('<div class="side-label">Operation Mode</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Select Operation Mode",
        ["Historical Benchmark Analysis", "Live Scenario Simulator"],
        index=0,
        key="input_mode",
    )

    st.markdown("---")
    st.markdown('<div class="side-label">Battery Configuration</div>', unsafe_allow_html=True)

    bess_capacity = st.slider(
        "Battery Capacity (kWh)",
        min_value=100.0,
        max_value=2000.0,
        value=500.0,
        step=50.0,
        key="bess_capacity",
    )

    max_charge_rate = st.slider(
        "Max Charge / Discharge (kW)",
        min_value=50.0,
        max_value=500.0,
        value=150.0,
        step=25.0,
        key="max_charge_rate",
    )

    initial_soc = st.slider(
        "Initial State of Charge (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=10.0,
        key="initial_soc",
    )

    render_system_status()

    return (
        selected_nav,
        input_mode,
        bess_capacity,
        max_charge_rate,
        initial_soc,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    inject_css()

    (
        selected_nav,
        input_mode,
        bess_capacity,
        max_charge_rate,
        initial_soc,
    ) = render_sidebar()

    df_demand, df_solar = load_datasets()

    # --------------------------------------------------------
    # DATA / SCENARIO PREPARATION
    # --------------------------------------------------------
    if input_mode == "Historical Benchmark Analysis":
        if df_demand is None or df_demand.empty:
            st.error(
                "Demand forecast dataset not found. "
                "Please run `python src/train_demand_model.py` first."
            )
            return

        min_date = df_demand["timestamp"].min().date()
        max_date = df_demand["timestamp"].max().date()

        # Historical controls are placed in a compact top card.
        with st.container():
            st.markdown(
                """
                <div class="card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">Historical Forecast Controls</div>
                            <div class="card-caption">Select the period used by the integration engine</div>
                        </div>
                        <div class="tag">HISTORICAL</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns([1.4, 1])

        with c1:
            selected_date = st.date_input(
                "Forecast Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="selected_date",
            )

        with c2:
            horizon_hours = st.selectbox(
                "Forecast Horizon",
                [24, 72, 168],
                format_func=lambda x: f"{x} Hours  /  {x // 24} Day{'s' if x > 24 else ''}",
                key="horizon_hours",
            )

        start_ts = pd.to_datetime(selected_date)
        end_ts = start_ts + pd.Timedelta(hours=horizon_hours)

        df_sub_demand = df_demand[
            (df_demand["timestamp"] >= start_ts)
            & (df_demand["timestamp"] < end_ts)
        ].copy()

        df_sub_solar = None
        if df_solar is not None and not df_solar.empty:
            df_sub_solar = df_solar[
                (df_solar["timestamp"] >= start_ts)
                & (df_solar["timestamp"] < end_ts)
            ].copy()

        if df_sub_demand.empty:
            st.warning(
                "No demand data is available for the selected date range. "
                "Showing the default dataset window."
            )
            df_sub_demand = df_demand.iloc[:horizon_hours].copy()
            if df_solar is not None:
                df_sub_solar = df_solar.iloc[: horizon_hours * 4].copy()

        df_res, kpis = compute_energy_balance(
            df_sub_demand,
            df_sub_solar,
            battery_capacity_kwh=bess_capacity,
            max_charge_rate_kw=max_charge_rate,
            initial_soc_pct=initial_soc,
        )

    else:
        # ----------------------------------------------------
        # LIVE SCENARIO SIMULATOR
        # ----------------------------------------------------
        st.markdown(
            """
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Scenario Simulator</div>
                        <div class="card-caption">Create an operating scenario and run the integration engine</div>
                    </div>
                    <div class="tag">SIMULATION</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            sim_hours = st.slider(
                "Simulation Duration (Hours)",
                min_value=12,
                max_value=48,
                value=24,
                key="sim_hours",
            )

        with c2:
            ghi_peak = st.slider(
                "Peak Solar Irradiance (W/m²)",
                min_value=0,
                max_value=1000,
                value=750,
                step=50,
                key="ghi_peak",
            )

        with c3:
            temp_ambient = st.slider(
                "Ambient Temperature (°C)",
                min_value=10.0,
                max_value=45.0,
                value=28.0,
                key="temp_ambient",
            )

        with c4:
            demand_multiplier = st.slider(
                "Facility Load Factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="demand_multiplier",
            )

        # Kept for interface continuity / scenario context.
        _ = temp_ambient

        time_range = pd.date_range(
            start="2026-10-01 00:00:00",
            periods=sim_hours,
            freq="1h",
        )

        hours = time_range.hour

        base_demand = (
            1800
            + 800 * np.sin((hours - 6) * np.pi / 12.0)
            + np.random.normal(0, 30, sim_hours)
        )
        base_demand = np.maximum(500, base_demand) * demand_multiplier

        solar_gen = np.where(
            (hours >= 6) & (hours <= 18),
            (ghi_peak * 1.8) * np.sin((hours - 6) * np.pi / 12.0),
            0.0,
        )

        df_sub_demand = pd.DataFrame(
            {
                "timestamp": time_range,
                "predicted_demand_kWh": base_demand,
            }
        )

        df_sub_solar = pd.DataFrame(
            {
                "timestamp": time_range,
                "predicted_solar_kWh": solar_gen,
            }
        )

        run_scenario = st.button(
            "RUN SCENARIO",
            type="primary",
            use_container_width=True,
            key="run_scenario",
        )

        # Streamlit reruns on every control change. The button gives a
        # prominent action without changing the existing calculation flow.
        if run_scenario:
            st.toast("Scenario executed successfully.", icon="⚡")

        df_res, kpis = compute_energy_balance(
            df_sub_demand,
            df_sub_solar,
            battery_capacity_kwh=bess_capacity,
            max_charge_rate_kw=max_charge_rate,
            initial_soc_pct=initial_soc,
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------
    render_header(input_mode)

    # --------------------------------------------------------
    # OVERVIEW / TOP KPI AREA
    # --------------------------------------------------------
    render_kpis(kpis)

    # --------------------------------------------------------
    # NAVIGATION / CONTENT
    # --------------------------------------------------------
    # The sidebar navigation controls which primary panel is emphasized.
    # The tabs below keep all existing functionality available.
    tab1, tab2, tab3 = st.tabs(
        [
            "OVERVIEW & FORECAST",
            "BATTERY & MICROGRID",
            "AI MODELS & DATA",
        ]
    )

    # --------------------------------------------------------
    # TAB 1
    # --------------------------------------------------------
    with tab1:
        if selected_nav in ["Overview", "Energy Forecast", "Solar Analytics", "Scenario Simulator"]:
            st.markdown(
                """
                <div class="card section-card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">Live Energy Forecast</div>
                            <div class="card-caption">
                                Integrated demand, solar generation and grid requirement
                            </div>
                        </div>
                        <div class="tag">FORECAST ENGINE</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                render_energy_chart(df_res),
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "responsive": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )

            g1, g2 = st.columns([1.25, 1])

            with g1:
                st.markdown(
                    """
                    <div class="card">
                        <div class="card-header">
                            <div>
                                <div class="card-title">Energy Flow</div>
                                <div class="card-caption">How forecast energy is balanced across the system</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # A compact flow summary using actual calculated values.
                st.markdown(
                    f"""
                    <div class="mini-grid">
                        <div class="mini-stat">
                            <div class="mini-label">Demand</div>
                            <div class="mini-value">{fmt_num(kpis.get('total_demand_kWh', 0))} kWh</div>
                        </div>
                        <div class="mini-stat">
                            <div class="mini-label">Solar</div>
                            <div class="mini-value">{fmt_num(kpis.get('total_solar_kWh', 0))} kWh</div>
                        </div>
                        <div class="mini-stat">
                            <div class="mini-label">Grid</div>
                            <div class="mini-value">{fmt_num(kpis.get('total_grid_import_kWh', 0))} kWh</div>
                        </div>
                        <div class="mini-stat">
                            <div class="mini-label">Renewable</div>
                            <div class="mini-value">{fmt_num(kpis.get('overall_renewable_contribution_pct', 0))}%</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with g2:
                render_grid_status(df_res, kpis)

        else:
            st.info(
                "Use the Overview, Energy Forecast, Solar Analytics or Scenario Simulator "
                "navigation items to focus this workspace."
            )

    # --------------------------------------------------------
    # TAB 2
    # --------------------------------------------------------
    with tab2:
        render_battery_panel(df_res, kpis, bess_capacity)

    # --------------------------------------------------------
    # TAB 3
    # --------------------------------------------------------
    with tab3:
        st.markdown(
            """
            <div class="card section-card">
                <div class="card-header">
                    <div>
                        <div class="card-title">AI Model Performance</div>
                        <div class="card-caption">Predictive intelligence behind the energy platform</div>
                    </div>
                    <div class="tag">XGBOOST</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)

        with m1:
            render_model_card(
                "Electricity Demand Model",
                "XGBoost Regressor",
                "92.28%",
                "75.50 kWh",
                "126.24 kWh",
                "Lagged load (`lag_1h`, `lag_168h`), time of day, harmonic sine/cosine hour encodings.",
            )

        with m2:
            render_model_card(
                "Solar PV Generation Model",
                "XGBoost Regressor",
                "~96%",
                "0.010 kWh / 15-min",
                None,
                "Global Horizontal Irradiance (GHI), solar elevation, trailing generation.",
            )

        st.markdown(
            """
            <div class="card section-card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Raw Forecast Data</div>
                        <div class="card-caption">Integrated output generated by the energy-balance engine</div>
                    </div>
                    <div class="tag">DATA EXPLORER</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Search/filter without changing the source dataframe.
        search = st.text_input(
            "Filter timestamps or dispatch status",
            placeholder="Search...",
            label_visibility="collapsed",
            key="data_search",
        )

        display_df = df_res.copy()

        if search:
            search_lower = search.lower()
            mask = pd.Series(False, index=display_df.index)

            for col in display_df.columns:
                mask = mask | display_df[col].astype(str).str.lower().str.contains(
                    search_lower,
                    na=False,
                )

            display_df = display_df[mask]

        st.caption(f"Showing {len(display_df):,} of {len(df_res):,} rows")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=460,
        )

        csv_data = df_res.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="EXPORT INTEGRATED FORECAST CSV",
            data=csv_data,
            file_name="smart_energy_forecast_integration.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------
    st.markdown(
        """
        <div style="
            margin-top:34px;
            padding-top:14px;
            border-top:1px solid rgba(105,153,166,.08);
            display:flex;
            justify-content:space-between;
            color:#4f6871;
            font-size:9px;
            letter-spacing:.05em;">
            <span>ENERGY AI • SMART GRID INTELLIGENCE PLATFORM</span>
            <span>Forecast engine • BESS • Renewable integration</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
