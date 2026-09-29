# Model 2 — Photovoltaic Solar Generation Forecasting (15-Minute Ahead)

**Project:** AI-Based Energy Demand Forecasting for Renewable Integration  
**Component:** Model 2 — Renewable Generation Subsystem  
**Lead Contributor:** Member 2 (Solar & Renewable Generation Modeling)  
**Academic Target:** College Senior Capstone Project  

---

## 1. Problem Definition
Model 2 predicts the electrical energy output of campus rooftop photovoltaic (PV) solar installations for the **next 15-minute interval ($t+1$)** using only telemetry and meteorological data available at or before time $t$:

$$\hat{y}_{i, t+1} = f(\mathbf{x}_{i, t})$$

- **Prediction Horizon**: 15 minutes ahead ($t+1$)
- **Resolution**: 15-minute discrete intervals
- **Target Unit**: Kilowatt-hours ($	ext{kWh}$) generated per 15-minute interval
- **Spatial Coverage**: 42 rooftop PV systems across 5 Australian university campuses

---

## 2. Dataset Overview
Out of 16 original CSV files provided in the project archive, a rigorous data audit isolated the **5 relevant files** for solar generation forecasting:
1. `unisolar/unisolar/Solar_Energy_Generation.csv`: Primary 15-minute inverter telemetry (2,731,946 observations).
2. `unisolar/unisolar/Solar_Irradiance.csv`: Satellite-derived hourly Global Horizontal Irradiance ($	ext{GHI}$) and Cloud Opacity.
3. `unisolar/unisolar/Weather_Data_reordered_all.csv`: 15-minute local surface weather (temperature, humidity, wind).
4. `unisolar/unisolar/Solar_Site_Details.csv`: Installed capacity ($kW_p$) and GPS coordinates for all 42 arrays.
5. `archive/campus_meta.csv`: Relational key lookup between numerical IDs and campus names.

The 11 remaining files in `archive/` belong strictly to building consumption and grid utility metering (the domain of Model 1: Demand Forecasting).

---

## 3. Data Preprocessing & Leakage Prevention
- **Astronomical Nighttime Imputation**: An astronomical solar position audit proved that $99.81\%$ of raw missing values occurred during nighttime ($	ext{solar elevation} \le 0^\circ$). These $1,266,483$ intervals were legitimately imputed as $0.00	ext{ kWh}$.
- **Unresolved Daytime Outages**: Daylight missing values ($185,907$ intervals, $12.10\%$ of nulls) were confirmed to be multi-day hardware inverter outages and were preserved as `NaN` (excluded from target loss to prevent teaching the model that clear noon sun produces 0 energy).
- **Timezone Synchronization**: UTC satellite timestamps were converted to local `Australia/Melbourne` time, eliminating a catastrophic 10-hour phase error and increasing irradiance-generation correlation from $r = 0.02$ to $r = 0.834$.
- **Strict Chronological Splitting**:
  - **Training Set (70%)**: 1,723,450 rows (2020-01-01 to 2021-10-01)
  - **Validation Set (15%)**: 369,303 rows (2021-10-01 to 2022-01-12)
  - **Test Set (15%)**: 369,333 rows (2022-01-12 to 2022-04-23)
  - *No shuffling; test data remained strictly untouched during training.*

---

## 4. Feature Engineering (33 Input Features)
All features represent information strictly observed at or before prediction time $t$:
- **Group A (Historical Generation)**: $lag_1$ ($y_t$), $lag_2$, $lag_4$, $lag_{96}$ (yesterday), $lag_{672}$ (last week), trailing rolling statistics on $lag_1$ ($1	ext{h}$ mean/std/min/max, $4	ext{h}$ mean, $24	ext{h}$ mean).
- **Group B (Solar Geometry)**: Astronomical solar elevation angle, solar azimuth, clear-sky theoretical radiation ($	ext{GHI}_	ext{clear}$).
- **Group C (Satellite Irradiance)**: Forward-filled satellite $	ext{GHI}$ and Cloud Opacity.
- **Group D (Surface Weather)**: Ambient air temperature, relative humidity, wind speed, circular sine/cosine wind direction.
- **Group E (Temporal Cycles)**: Continuous sine/cosine harmonics of hour-of-day and day-of-year.
- **Group F (Site Metadata)**: Nameplate capacity ($kW_p$), latitude, longitude, and site identifier.

---

## 5. Model Selection & Empirical Results

Five models were trained and benchmarked across the 369,333-row test partition:
1. **Persistence Baseline**: $\hat{y}_{t+1} = y_t$ ($lag_1$)
2. **Previous-Day Baseline**: $\hat{y}_{t+1} = y_{t-95}$ ($lag_{96}$)
3. **Ridge Linear Regression**: Scaled linear combination of all features
4. **Random Forest Regressor**: 60 trees, depth 12
5. **XGBoost Regressor**: 350 estimators, learning rate 0.06, depth 7, histogram binning

### Key Test Set Performance Across Solar Regimes

| Evaluation Regime | Persistence ($	ext{lag}_1$) MAE | Linear Regression MAE | Random Forest MAE | **XGBoost MAE** | XGBoost RMSE | XGBoost $R^2$ |
|---|---:|---:|---:|---:|---:|---:|
| **Overall (24h Test)** | 1.087 kWh | 1.085 kWh | 0.647 kWh | **0.657 kWh** | **2.065 kWh** | **0.9522** |
| **Daylight Only (Active)** | 2.064 kWh | 1.927 kWh | 1.229 kWh | **1.228 kWh** | **2.846 kWh** | **0.9432** |
| **High Generation ($\ge 7.48$ kWh)** | 4.453 kWh | 3.784 kWh | 2.579 kWh | **2.623 kWh** | **4.928 kWh** | **0.9099** |
| **Nighttime** | 0.002 kWh | 0.149 kWh | 0.000 kWh | **0.021 kWh** | **0.024 kWh** | — |

> **Selection Rationale**: XGBoost and Random Forest delivered virtually identical accuracy (~$40.5\%$ MAE reduction over persistence during daylight). **XGBoost was selected as the final production model** due to its computational efficiency ($41.4	ext{s}$ training time vs $88.7	ext{s}$ on a subset for RF), compact serialized footprint ($4.8	ext{ MB}$ JSON), and native handling of missing surface weather readings during deployment.

---

## 6. How to Use the Prediction Function

The model provides a simple, standalone prediction interface in `predict_solar.py`:

```python
from predict_solar import predict_solar_generation

# Example real-time telemetry observation at time t
input_payload = {
    "SiteKey": 14,
    "CampusKey": 1,
    "lag_1": 10.41,               # Current generation at time t (kWh)
    "rolling_mean_1h": 9.95,      # Trailing 1-hour average generation (kWh)
    "solar_elevation": 58.4,      # Solar elevation angle (degrees)
    "Ghi": 820.0,                 # Surface solar irradiance (W/m^2)
    "CloudOpacity": 5.0,          # Cloud opacity (%)
    "AirTemperature": 24.5,       # Ambient air temperature (°C)
    "RelativeHumidity": 42.0,     # Relative humidity (%)
    "kWp": 66.0                   # Nameplate capacity (kWp)
}

result = predict_solar_generation(input_payload)
print(result)
```

**Expected JSON Response:**
```json
{
  "predicted_solar_generation": 10.0525,
  "forecast_horizon_minutes": 15,
  "unit": "kWh per 15-minute interval",
  "is_daylight": true
}
```

---

## 7. Model Limitations
1. **Persistence Strength**: At ultra-short 15-minute horizons, atmospheric persistence is strong ($R^2 = 0.85$ for persistence alone). While XGBoost improves daylight MAE by $40.5\%$, ML models cannot eliminate the irreducible stochastic noise caused by fast cloud edges passing between 15-minute timesteps.
2. **Satellite Scan Latency**: Hourly satellite irradiance was forward-filled. Operational deployment requires ingesting satellite feeds within 15 minutes of acquisition.
3. **Site 19 Metadata Typo**: Site 19 has an administrative typo in its capacity field ($kW_p = 34.32$ listed vs $168.88	ext{ kW}$ observed peak). Predictions remain accurate because the model predicts absolute kWh, but cross-site capacity normalization must treat Site 19 as an exception.
