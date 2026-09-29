import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.energy_manager import compute_energy_balance, compute_financial_payback

st.set_page_config(
    page_title="Smart Energy Forecasting & Renewable Payoff Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEMAND_CSV_PATH = os.path.join(PROJECT_ROOT, "output", "demand_forecast.csv")
SOLAR_CSV_PATH = os.path.join(PROJECT_ROOT, "model2", "predictions", "test_predictions.csv")

USER_PROFILES = {
    "🏡 Residential Home": {
        "description": "Small household (Single-phase / 3-phase)",
        "demand_multiplier": 0.02,
        "solar_kw": 5.0,
        "battery_kwh": 10.0,
        "max_charge_kw": 5.0
    },
    "🏢 Commercial Office": {
        "description": "Medium commercial building / office complex",
        "demand_multiplier": 0.30,
        "solar_kw": 100.0,
        "battery_kwh": 150.0,
        "max_charge_kw": 50.0
    },
    "🎓 University Campus / Industrial": {
        "description": "Large institution / manufacturing facility",
        "demand_multiplier": 1.00,
        "solar_kw": 500.0,
        "battery_kwh": 500.0,
        "max_charge_kw": 150.0
    }
}

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
    # Header Banner
    st.title("⚡ Smart Energy Demand Forecasting & Renewable Payoff Platform")
    st.caption("AI load forecasting, solar generation prediction, renewable fulfillment reporting, and investment payback visualizer.")
    st.divider()

    df_demand, df_solar = load_datasets()

    # User Profile Selector Buttons / Cards
    st.subheader("👤 Select User Profile & Consumption Scale")
    profile_cols = st.columns(3)
    
    if 'selected_profile' not in st.session_state:
        st.session_state.selected_profile = "🎓 University Campus / Industrial"

    for idx, (p_name, p_info) in enumerate(USER_PROFILES.items()):
        with profile_cols[idx]:
            is_active = (st.session_state.selected_profile == p_name)
            btn_label = f"✅ {p_name}" if is_active else p_name
            if st.button(btn_label, key=f"btn_{idx}", use_container_width=True):
                st.session_state.selected_profile = p_name
                st.rerun()

    active_p_data = USER_PROFILES[st.session_state.selected_profile]
    st.info(f"Active Profile: **{st.session_state.selected_profile}** — *{active_p_data['description']}*")

    st.divider()

    # Sidebar Controls
    st.sidebar.header("⚙️ Simulation & Financial Controls")
    
    input_mode = st.sidebar.radio(
        "Select Operation Mode:",
        ["📅 Historical Benchmark Analysis", "🎛️ Live Scenario Simulator"]
    )

    st.sidebar.subheader("🔌 Energy & System Capacity")
    solar_kw_cap = st.sidebar.slider("Solar PV Array Size (kW)", min_value=1.0, max_value=1000.0, value=float(active_p_data['solar_kw']), step=1.0)
    bess_capacity = st.sidebar.slider("Battery Storage Capacity (kWh)", min_value=0.0, max_value=2000.0, value=float(active_p_data['battery_kwh']), step=5.0)
    max_charge_rate = st.sidebar.slider("Max Battery Charge Rate (kW)", min_value=1.0, max_value=500.0, value=float(active_p_data['max_charge_kw']), step=5.0)
    
    st.sidebar.subheader("💰 Financial Rates & Costs (₹ in INR)")
    grid_tariff_rs = st.sidebar.slider("Grid Electricity Rate (₹/kWh)", min_value=4.0, max_value=20.0, value=8.5, step=0.5)
    solar_cost_per_kw = st.sidebar.slider("Solar Panel System Cost (₹/kW)", min_value=30000.0, max_value=100000.0, value=55000.0, step=2500.0)
    battery_cost_per_kwh = st.sidebar.slider("Battery Storage Cost (₹/kWh)", min_value=8000.0, max_value=35000.0, value=18000.0, step=1000.0)

    if input_mode == "📅 Historical Benchmark Analysis":
        if df_demand is None or df_demand.empty:
            st.error("Demand dataset not found. Run `python src/train_demand_model.py` first.")
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

        start_ts = pd.to_datetime(selected_date)
        end_ts = start_ts + pd.Timedelta(hours=horizon_hours)
        
        df_sub_demand = df_demand[(df_demand['timestamp'] >= start_ts) & (df_demand['timestamp'] < end_ts)].copy()
        df_sub_solar = None
        if df_solar is not None:
            df_sub_solar = df_solar[(df_solar['timestamp'] >= start_ts) & (df_solar['timestamp'] < end_ts)].copy()

        if df_sub_demand.empty:
            df_sub_demand = df_demand.iloc[:horizon_hours].copy()
            if df_solar is not None:
                df_sub_solar = df_solar.iloc[:horizon_hours * 4].copy()

        # Apply user profile demand multiplier
        df_sub_demand['predicted_demand_kWh'] = df_sub_demand['actual_demand_kWh'] * active_p_data['demand_multiplier'] if 'actual_demand_kWh' in df_sub_demand.columns else df_sub_demand['predicted_demand_kWh'] * active_p_data['demand_multiplier']
        
        # Scale solar according to solar array size ratio
        if df_sub_solar is not None and not df_sub_solar.empty:
            solar_col = 'Predicted_Generation_XGBoost' if 'Predicted_Generation_XGBoost' in df_sub_solar.columns else 'predicted_solar_kWh'
            df_sub_solar[solar_col] = df_sub_solar[solar_col] * (solar_kw_cap / 100.0)

        df_res, kpis = compute_energy_balance(
            df_sub_demand,
            df_sub_solar,
            battery_capacity_kwh=bess_capacity,
            max_charge_rate_kw=max_charge_rate,
            initial_soc_pct=50.0
        )

    else:
        # Live Simulator Mode
        st.sidebar.subheader("🎛️ Load & Environment Simulation")
        sim_hours = st.sidebar.slider("Simulation Horizon (Hours)", min_value=12, max_value=48, value=24)
        ghi_peak = st.sidebar.slider("Peak Solar Irradiance GHI (W/m²)", min_value=0, max_value=1000, value=750, step=50)

        time_range = pd.date_range(start="2026-10-01 00:00:00", periods=sim_hours, freq="1h")
        hours = time_range.hour

        base_demand = (1800 + 800 * np.sin((hours - 6) * np.pi / 12.0) + np.random.normal(0, 30, sim_hours)) * active_p_data['demand_multiplier']
        base_demand = np.maximum(5.0, base_demand)

        solar_gen = np.where(
            (hours >= 6) & (hours <= 18),
            (ghi_peak / 1000.0) * solar_kw_cap * 0.85 * np.sin((hours - 6) * np.pi / 12.0),
            0.0
        )

        df_sub_demand = pd.DataFrame({'timestamp': time_range, 'predicted_demand_kWh': base_demand})
        df_sub_solar = pd.DataFrame({'timestamp': time_range, 'predicted_solar_kWh': solar_gen})

        df_res, kpis = compute_energy_balance(
            df_sub_demand,
            df_sub_solar,
            battery_capacity_kwh=bess_capacity,
            max_charge_rate_kw=max_charge_rate,
            initial_soc_pct=50.0
        )

    # Financial Payback Calculation
    fin_results = compute_financial_payback(
        kpis,
        solar_kw_capacity=solar_kw_cap,
        battery_kwh_capacity=bess_capacity,
        grid_rate_rs_per_kwh=grid_tariff_rs,
        solar_cost_rs_per_kw=solar_cost_per_kw,
        battery_cost_rs_per_kwh=battery_cost_per_kwh
    )

    # Display Dashboard Tabs
    tab1, tab2, tab3 = st.tabs([
        "🌱 User Renewable Fulfillment & Payoff Visualizer",
        "📈 Integrated Load & Solar Curves",
        "🤖 Data & System Diagnostics"
    ])

    with tab1:
        st.subheader(f"📋 Renewable Fulfillment & Payoff Summary for {st.session_state.selected_profile}")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("🌱 Renewable Energy Share", f"{kpis['overall_renewable_contribution_pct']:.1f}%")
        m_col2.metric("🔌 Grid Import Reliance", f"{100.0 - kpis['overall_renewable_contribution_pct']:.1f}%")
        m_col3.metric("⚡ Period Consumption", f"{kpis['total_demand_kWh']:,.1f} kWh")
        m_col4.metric("💰 Grid Electricity Rate", f"₹{grid_tariff_rs:.2f} / kWh")

        st.divider()

        # Financial Investment & Payback Summary Banner
        st.subheader("💳 Investment Summary & Payoff Period")
        
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        capex_lakhs = fin_results['total_capex_rs'] / 100000.0
        savings_lakhs = fin_results['annual_bill_savings_rs'] / 100000.0
        
        f_col1.metric("🏗️ Capital Investment (CAPEX)", f"₹{capex_lakhs:,.2f} Lakhs")
        f_col2.metric("💵 Annual Bill Savings", f"₹{savings_lakhs:,.2f} Lakhs / yr")
        f_col3.metric("📉 Reduced Annual Bill", f"₹{fin_results['annual_bill_with_renewables_rs']/100000.0:,.2f} Lakhs")
        f_col4.metric("⏳ Estimated Payoff Period", fin_results['payback_str'], delta="Return on Investment")

        st.success(f"**Key Takeaway**: With a **{solar_kw_cap:.1f} kW Solar Array** and **{bess_capacity:.1f} kWh Battery Storage**, the **{st.session_state.selected_profile}** achieves **{kpis['overall_renewable_contribution_pct']:.1f}% renewable energy fulfillment**. The total investment of **₹{fin_results['total_capex_rs']:,.0f}** pays for itself in **{fin_results['payback_str']}**.")

        st.divider()

        # Visual Demonstration Graphs
        st.subheader("📊 Visual Breakdown & Payoff Trajectory")

        g_col1, g_col2 = st.columns(2)

        with g_col1:
            st.markdown("##### 1. Renewable Energy Fulfillment Mix (Donut Chart)")
            
            # Donut chart data
            mix_labels = ['Direct Solar Used', 'Battery Discharge Used', 'Net Grid Import Needed']
            mix_values = [
                kpis['total_direct_solar_kWh'],
                kpis['total_battery_discharge_kWh'],
                kpis['total_grid_import_kWh']
            ]
            mix_colors = ['#FECB52', '#00CC96', '#636EFA']
            
            fig_mix = go.Figure(data=[go.Pie(
                labels=mix_labels,
                values=mix_values,
                hole=.45,
                marker=dict(colors=mix_colors),
                textinfo='label+percent',
                insidetextorientation='radial'
            )])
            fig_mix.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15)
            )
            st.plotly_chart(fig_mix, use_container_width=True)

        with g_col2:
            st.markdown("##### 2. Cumulative Cash Flow & Payoff Point (12-Year Curve)")
            
            # Cumulative Cash Flow Line Chart
            fig_cf = go.Figure()
            
            fig_cf.add_trace(go.Scatter(
                x=fin_results['cash_flow_years'],
                y=fin_results['cash_flow_lakhs'],
                mode='lines+markers',
                name='Cumulative Balance (₹ Lakhs)',
                line=dict(color='#00CC96', width=3),
                marker=dict(size=7)
            ))
            
            # Zero Break-Even Reference Line
            fig_cf.add_shape(
                type="line", x0=0, y0=0, x1=12, y1=0,
                line=dict(color="Red", width=2, dash="dash")
            )
            
            # Annotation for Payoff Year
            if fin_results['payback_float_years'] <= 12:
                fig_cf.add_annotation(
                    x=fin_results['payback_float_years'],
                    y=0,
                    text=f"Payoff: {fin_results['payback_str']}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    ax=0,
                    ay=-40
                )

            fig_cf.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Years",
                yaxis_title="Net Cumulative Balance (₹ Lakhs)",
                hovermode="x"
            )
            st.plotly_chart(fig_cf, use_container_width=True)

        st.markdown("##### 3. Annual Electricity Bill Comparison (₹ Lakhs / Year)")
        
        # Financial Comparison Bar Chart
        bill_labels = ['Bill WITHOUT Renewables', 'Bill WITH Renewables & Battery', 'Annual Net Savings']
        bill_values = [
            fin_results['annual_bill_without_renewables_rs'] / 100000.0,
            fin_results['annual_bill_with_renewables_rs'] / 100000.0,
            fin_results['annual_bill_savings_rs'] / 100000.0
        ]
        bill_colors = ['#EF553B', '#636EFA', '#00CC96']
        
        fig_bill = go.Figure([go.Bar(
            x=bill_labels,
            y=bill_values,
            marker_color=bill_colors,
            text=[f"₹{v:.2f} Lakhs" for v in bill_values],
            textposition='auto'
        )])
        fig_bill.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Amount (₹ Lakhs)"
        )
        st.plotly_chart(fig_bill, use_container_width=True)

    with tab2:
        st.subheader("📈 Hourly Energy Generation & Storage Profile")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=df_res['timestamp'], y=df_res['demand_kWh'],
                name="⚡ Demand (kWh)",
                line=dict(color="#EF553B", width=2.5)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_res['timestamp'], y=df_res['solar_kWh'],
                name="☀️ Solar Output (kWh)",
                line=dict(color="#FECB52", width=2.5),
                fill='tozeroy', fillcolor='rgba(254, 203, 82, 0.15)'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_res['timestamp'], y=df_res['net_grid_import_kWh'],
                name="🔌 Net Grid Import Required",
                line=dict(color="#636EFA", width=2.5, dash="dash")
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_res['timestamp'], y=df_res['battery_soc_pct'],
                name="🔋 Battery State of Charge (%)",
                line=dict(color="#00CC96", width=2.0, dash="dot")
            ),
            secondary_y=True
        )

        fig.update_layout(
            height=480,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="Timestamp")
        fig.update_yaxes(title_text="Energy (kWh)", secondary_y=False)
        fig.update_yaxes(title_text="Battery SoC (%)", range=[0, 105], secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📄 Raw Forecast Data & CSV Download")
        st.dataframe(df_res, use_container_width=True)
        
        csv_data = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download User Report CSV",
            data=csv_data,
            file_name="user_renewable_fulfillment_report.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
