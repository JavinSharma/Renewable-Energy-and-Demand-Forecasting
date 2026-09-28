# 🔌 Model 3 — Renewable Energy Integration & Dashboard Engine

## 📌 Overview & Main Responsibilities
**Member 3** is responsible for turning the individual forecast outputs of **Member 1 (Demand Forecasting)** and **Member 2 (Solar Forecasting)** into a cohesive, actionable **Renewable Integration & Energy Management System**.

### Core Deliverables:
1. **Energy Balance Calculation Engine**: Quantifies Net Grid Requirement and Renewable Contribution Percentage.
2. **Battery Energy Storage System (BESS) Optimizer**: Simulates battery charge/discharge behavior to absorb solar surplus and buffer grid deficits.
3. **Interactive Streamlit Dashboard (`app.py`)**: Web application UI for demonstration and submission.

---

## 🧮 Integration Logic & Core Equations

### 1. Net Grid Requirement ($G_t$)
Power imported from the external electrical grid at hour $t$:
$$G_t = \max\left(0, D_t - S_t - B_t^{\text{discharge}}\right)$$
Where:
- $D_t$: Predicted electricity demand ($kWh$) from Member 1
- $S_t$: Predicted solar generation ($kWh$) from Member 2
- $B_t^{\text{discharge}}$: Battery energy discharged ($kWh$)

---

### 2. Renewable Contribution Share (%)
Percentage of demand covered by clean renewable energy:
$$\text{Renewable \%}_t = \min\left(100\%, \frac{\min(D_t, S_t + B_t^{\text{discharge}})}{D_t} \times 100\right)$$

---

### 3. Battery Storage Dispatch Algorithm
When **Solar > Demand** (Surplus State $\Delta_t = S_t - D_t > 0$):
$$B_t^{\text{charge}} = \min\left(\Delta_t, P_{\text{max\_charge}}, C_{\text{max}} - \text{SoC}_{t-1}\right)$$
$$\text{Grid Export}_t = \Delta_t - B_t^{\text{charge}}$$

When **Demand > Solar** (Deficit State $\Delta_t = D_t - S_t > 0$):
$$B_t^{\text{discharge}} = \min\left(\Delta_t, P_{\text{max\_discharge}}, \text{SoC}_{t-1}\right)$$
$$G_t = \Delta_t - B_t^{\text{discharge}}$$

---

## 📁 Member 3 Directory Structure

```
model3/
├── README.md                 # Technical specification and system architecture (this file)
└── MEMBER_3_GUIDE.md          # Comprehensive Agent Guide & Quick Start instructions
src/
└── energy_manager.py         # Member 3 python engine functions
app.py                        # Streamlit web application frontend
```

---

## 🚀 How to Run & Demonstrate
```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Launch Streamlit Application
streamlit run app.py
```
