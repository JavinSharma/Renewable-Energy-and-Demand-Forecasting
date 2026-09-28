import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.energy_manager import compute_energy_balance

st.set_page_config(
    page_title="AI Energy Forecasting & Renewable Integration",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEMAND_CSV_PATH = os.path.join(PROJECT_ROOT, "output", "demand_forecast.csv")
SOLAR_CSV_PATH = os.path.join(PROJECT_ROOT, "model2", "predictions", "test_predictions.csv")

@st.cache_data
def load_datasets():
    df_demand = None
    df_solar = None
    
    if os.path.exists(DEMAND_CSV_PATH):
        df_demand = pd.read_csv(DEMAND_CSV_PATH)
        df_demand['timestamp'] = pd.to_datetime(df_demand['timestamp'])
        
    if os.path.exists(SOLAR_CSV_PATH):
        df_solar = pd.read_csv(SOLAR_CSV_PATH)
        df_solar['timestamp'] = pd.to_datetime(df_solar['Timestamp'] if 'Timestamp' in df_solar.columns else df_solar['timestamp'])
        
    return df_demand, df_solar

def main():
    # Application Title & Banner
    st.title("⚡ AI-Based Energy Demand Forecasting for Renewable Integration")
    st.markdown("""
    **Capstone Project System Architecture**: 
    - 👤 **Member 1**: Electricity Demand Forecasting (XGBoost)
    - 👤 **Member 2**: Solar Generation Forecasting (XGBoost)
    - 👤 **Member 3**: Energy Balance, Grid Requirement Optimization & Dashboard
    """)
    st.divider()

    df_demand, df_solar = load_datasets()

    # Sidebar Controls
    st.sidebar.header("⚙️ Simulation Controls")
    
    input_mode = st.sidebar.radio(
        "Select Operation Mode:",
        ["📅 Historical Test Data Benchmark", "🎛️ Interactive 'What-If' Simulator"]
    )
    
    st.sidebar.subheader("🔋 Battery Storage Configuration")
    bess_capacity = st.sidebar.slider("Battery Capacity (kWh)", min_value=100.0, max_value=2000.0, value=500.0, step=50.0)
    max_charge_rate = st.sidebar.slider("Max Charge/Discharge Rate (kW)", min_value=50.0, max_value=500.0, value=150.0, step=25.0)
    initial_soc = st.sidebar.slider("Initial State of Charge (%)", min_value=0.0, max_value=100.0, value=50.0, step=10.0)

    if input_mode == "📅 Historical Test Data Benchmark":
        if df_demand is None or df_demand.empty:
            st.error("Demand forecast dataset not found. Please run `python src/train_demand_model.py` first.")
            return

        st.sidebar.subheader("📅 Date & Horizon Selection")
        min_date = df_demand['timestamp'].min().date()
        max_date = df_demand['timestamp'].max().date()
        
        selected_date = st.sidebar.date_input(
            "Forecast Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        horizon_hours = st.sidebar.selectbox("Forecast Horizon", [24, 72, 168], format_func=lambda x: f"{x} Hours ({x//24} Days)")

        # Filter dataset for chosen date range
        start_ts = pd.to_datetime(selected_date)
        end_ts = start_ts + pd.Timedelta(hours=horizon_hours)
        
        df_sub_demand = df_demand[(df_demand['timestamp'] >= start_ts) & (df_demand['timestamp'] < end_ts)].copy()
        df_sub_solar = None
        if df_solar is not None:
            df_sub_solar = df_solar[(df_solar['timestamp'] >= start_ts) & (df_solar['timestamp'] < end_ts)].copy()

        if df_sub_demand.empty:
            st.warning("No demand data available for selected date range. Showing default start date window.")
            df_sub_demand = df_demand.iloc[:horizon_hours].copy()
            if df_solar is not None:
                df_sub_solar = df_solar.iloc[:horizon_hours * 4].copy()

        # Compute energy balance logic via Member 3 Engine
        df_res, kpis = compute_energy_balance(
            df_sub_demand,
            df_sub_solar,
            battery_capacity_kwh=bess_capacity,
            max_charge_rate_kw=max_charge_rate,
            initial_soc_pct=initial_soc
        )

    else:
        # Interactive Simulator Mode
        st.sidebar.subheader("🎛️ Environment Parameters")
        sim_hours = st.sidebar.slider("Simulation Duration (Hours)", min_value=12, max_value=48, value=24)
        ghi_peak = st.sidebar.slider("Peak Solar Irradiance GHI (W/m²)", min_value=0, max_value=1000, value=750, step=50)
        temp_ambient = st.sidebar.slider("Ambient Temperature (°C)", min_value=10.0, max_value=45.0, value=28.0)
        demand_multiplier = st.sidebar.slider("Campus Activity Level Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

        # Synthetic curve generation for scenario testing
        time_range = pd.date_range(start="2026-10-01 00:00:00", periods=sim_hours, freq="1h")
        hours = time_range.hour

        # Base diurnal demand curve (peak at 2 PM, low at night)
        base_demand = 1800 + 800 * np.sin((hours - 6) * np.pi / 12.0) + np.random.normal(0, 30, sim_hours)
        base_demand = np.maximum(500, base_demand) * demand_multiplier

        # Base diurnal solar curve (bell curve daylight between 6 AM and 6 PM)
        solar_gen = np.where(
            (hours >= 6) & (hours <= 18),
            (ghi_peak * 1.8) * np.sin((hours - 6) * np.pi / 12.0),
            0.0
        )

        df_sub_demand = pd.DataFrame({'timestamp': time_range, 'predicted_demand_kWh': base_demand})
        df_sub_solar = pd.DataFrame({'timestamp': time_range, 'predicted_solar_kWh': solar_gen})

        df_res, kpis = compute_energy_balance(
            df_sub_demand,
            df_sub_solar,
            battery_capacity_kwh=bess_capacity,
            max_charge_rate_kw=max_charge_rate,
            initial_soc_pct=initial_soc
        )

    # Render Dashboard Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Renewable Integration Dashboard",
        "🔋 Battery Storage & Grid Management (Member 3)",
        "🤖 Model Performance & Raw Data"
    ])

    with tab1:
        st.subheader("💡 Key Energy Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("⚡ Total Expected Demand", f"{kpis['total_demand_kWh']:,.1f} kWh")
        col2.metric("☀️ Total Solar Generation", f"{kpis['total_solar_kWh']:,.1f} kWh")
        col3.metric("🔌 Net Grid Import Required", f"{kpis['total_grid_import_kWh']:,.1f} kWh")
        col4.metric("🌱 Renewable Contribution", f"{kpis['overall_renewable_contribution_pct']:.1f}%")

        st.divider()
        st.subheader("📈 Energy Balance Timeline (Demand vs Solar vs Net Grid Import)")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=df_res['timestamp'], y=df_res['demand_kWh'],
                name="⚡ Predicted Demand (Member 1)",
                line=dict(color="#EF553B", width=2.5)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_res['timestamp'], y=df_res['solar_kWh'],
                name="☀️ Predicted Solar (Member 2)",
                line=dict(color="#FECB52", width=2.5),
                fill='tozeroy', fillcolor='rgba(254, 203, 82, 0.15)'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_res['timestamp'], y=df_res['net_grid_import_kWh'],
                name="🔌 Net Grid Import (Member 3)",
                line=dict(color="#636EFA", width=2.5, dash="dash")
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_res['timestamp'], y=df_res['renewable_contribution_pct'],
                name="🌱 Renewable %",
                line=dict(color="#00CC96", width=1.5, dash="dot"),
                visible="legendonly"
            ),
            secondary_y=True
        )

        fig.update_layout(
            height=480,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_xaxes(title_text="Timestamp")
        fig.update_yaxes(title_text="Energy (kWh per hour)", secondary_y=False)
        fig.update_yaxes(title_text="Renewable Share (%)", range=[0, 105], secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🔋 Battery Energy Storage System (BESS) Dispatch & State of Charge")
        
        col_b1, col_b2 = st.columns([2, 1])
        
        with col_b1:
            fig_bess = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            
            # State of Charge plot
            fig_bess.add_trace(
                go.Scatter(
                    x=df_res['timestamp'], y=df_res['battery_soc_pct'],
                    name="Battery SoC (%)",
                    line=dict(color="#00CC96", width=2.5),
                    fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.15)'
                ),
                row=1, col=1
            )
            
            # Charge / Discharge plot
            fig_bess.add_trace(
                go.Bar(
                    x=df_res['timestamp'], y=df_res['battery_charge_kWh'],
                    name="Charge (Surplus Solar)",
                    marker_color="#2CA02C"
                ),
                row=2, col=1
            )
            fig_bess.add_trace(
                go.Bar(
                    x=df_res['timestamp'], y=-df_res['battery_discharge_kWh'],
                    name="Discharge (Deficit Offset)",
                    marker_color="#D62728"
                ),
                row=2, col=1
            )
            
            fig_bess.update_layout(height=450, showlegend=True, barmode="relative")
            fig_bess.update_yaxes(title_text="SoC (%)", range=[0, 105], row=1, col=1)
            fig_bess.update_yaxes(title_text="Charge (+)/Discharge (-) kWh", row=2, col=1)
            
            st.plotly_chart(fig_bess, use_container_width=True)
            
        with col_b2:
            st.markdown("### 📋 Dispatch Summary")
            total_charged = df_res['battery_charge_kWh'].sum()
            total_discharged = df_res['battery_discharge_kWh'].sum()
            
            st.info(f"**Total Battery Charged**: `{total_charged:,.1f} kWh`")
            st.success(f"**Total Battery Discharged**: `{total_discharged:,.1f} kWh`")
            st.warning(f"**Surplus Solar Exported to Grid**: `{kpis['total_surplus_export_kWh']:,.1f} kWh`")
            
            st.markdown("#### Operational Log Sample")
            st.dataframe(
                df_res[['timestamp', 'demand_kWh', 'solar_kWh', 'battery_soc_pct', 'dispatch_status']].head(10),
                hide_index=True
            )

    with tab3:
        st.subheader("🤖 Individual ML Model Metrics & Diagnostics")
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown("### ⚡ Member 1 — Demand Forecasting Model")
            st.markdown("""
            - **Algorithm**: XGBoost Regressor (`models/demand_model.pkl`)
            - **Evaluation Metrics**:
              - **R² Score**: `0.9228` (92.3% variance explained)
              - **MAE**: `75.50 kWh`
              - **RMSE**: `126.24 kWh`
            - **Top Features**: `lag_1h`, `lag_168h`, `hour`, `sin_hour`, `cos_hour`
            """)
            
        with c_m2:
            st.markdown("### ☀️ Member 2 — Solar Generation Forecasting Model")
            st.markdown("""
            - **Algorithm**: XGBoost Regressor (`model2/final_model/xgboost_model.json`)
            - **Evaluation Metrics**:
              - **R² Score**: `~0.96` (High Daylight Accuracy)
              - **MAE**: `0.010 kWh / 15-min`
            - **Top Features**: `Ghi`, `solar_elevation`, `lag_1`, `rolling_mean_1h`
            """)
            
        st.divider()
        st.subheader("📄 Exportable Predictions & Energy Balance Table")
        
        st.dataframe(df_res, use_container_width=True)
        
        csv_data = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast Results CSV",
            data=csv_data,
            file_name="renewable_integration_forecast.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
