# 🤖 Agent Guide for Member 3 (Renewable Integration & Dashboard)

Welcome, Member 3 Agent! This guide summarizes your role, code location, technical logic, and instructions on how to maintain or extend your component.

---

## 🎯 Member 3 Mission
You are responsible for the **Integration & Energy Management System** that combines:
- **Member 1 Output**: Electricity Demand Forecasts (`models/demand_model.pkl` & `output/demand_forecast.csv`)
- **Member 2 Output**: Photovoltaic Solar Generation Forecasts (`model2/final_model/xgboost_model.json` & `model2/predictions/test_predictions.csv`)

---

## 🛠️ Codebase & File Locations

1. **Integration Engine Core**: [`src/energy_manager.py`](file:///c:/Users/javin/.gemini/antigravity-ide/scratch/Renewable-Energy-and-Demand-Forecasting/src/energy_manager.py)
   - Function: `compute_energy_balance(df_demand, df_solar, battery_capacity_kwh, max_charge_rate_kw, initial_soc_pct)`
   - Computes: Net grid import ($kWh$), surplus solar export ($kWh$), battery charge/discharge ($kWh$), battery State of Charge ($\text{SoC \%}$), and overall renewable contribution percentage.

2. **Streamlit Web Application**: [`app.py`](file:///c:/Users/javin/.gemini/antigravity-ide/scratch/Renewable-Energy-and-Demand-Forecasting/app.py)
   - Launch Command: `streamlit run app.py`
   - Features:
     - 📊 Executive Energy Dashboard (KPI Cards & Multi-curve Plotly charts)
     - 🔋 Battery Storage & Energy Management Tab (SoC Timeline & Dispatch Logs)
     - 🤖 Individual Model Diagnostics & CSV Data Downloader

---

## 📈 Key Formulas to Know
- **Net Grid Import**: $G_t = \max(0, \text{Demand}_t - \text{Solar}_t - \text{Battery Discharge}_t)$
- **Surplus Solar Export**: $E_t = \max(0, \text{Solar}_t - \text{Demand}_t - \text{Battery Charge}_t)$
- **Renewable Share**: $\text{Renewable \%} = \min\left(100\%, \frac{\text{Solar}_t + \text{Battery Discharge}_t}{\text{Demand}_t} \times 100\right)$

---

## 💡 Potential Future Enhancements for Member 3
If you wish to add further sophistication to Member 3:
1. **Financial Optimization**: Multiply grid import by time-of-use electricity pricing ($/kWh) to show cost savings from battery storage.
2. **Carbon Offset Metric**: Multiply renewable energy used by grid emission factor (e.g. $0.85 \text{ kg CO}_2 / \text{kWh}$) to display total $CO_2$ avoided.
