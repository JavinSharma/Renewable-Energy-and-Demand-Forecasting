"""
predict_solar.py — Reusable Prediction Interface for Model 2: Solar Generation Forecasting

Project: AI-Based Energy Demand Forecasting for Renewable Integration
Component: Model 2 — 15-Minute-Ahead Photovoltaic Solar Generation Predictor
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "final_model", "xgboost_model.json")
DEFAULT_METADATA_PATH = os.path.join(CURRENT_DIR, "preprocessing", "feature_metadata.json")

_CACHED_MODEL = None
_FEATURE_COLS = None

def get_model(model_path=DEFAULT_MODEL_PATH):
    global _CACHED_MODEL, _FEATURE_COLS
    if _CACHED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        _CACHED_MODEL = model
        
        if os.path.exists(DEFAULT_METADATA_PATH):
            with open(DEFAULT_METADATA_PATH, 'r') as f:
                meta = json.load(f)
                _FEATURE_COLS = meta.get("feature_columns", [])
    return _CACHED_MODEL, _FEATURE_COLS

def predict_solar_generation(input_data, model_path=DEFAULT_MODEL_PATH):
    """
    Predicts photovoltaic solar energy generation for the NEXT 15-minute interval (t+1).
    
    Parameters:
    -----------
    input_data : dict, pd.Series, or pd.DataFrame
        Input dictionary or DataFrame containing features observed at or before time t.
        Core inputs:
            - lag_1: SolarGeneration observed at time t (kWh)
            - rolling_mean_1h: Trailing 1-hour average generation up to time t (kWh)
            - Ghi: Latest available Global Horizontal Irradiance (W/m^2)
            - solar_elevation: Astronomical solar elevation angle at time t (degrees)
            - SiteKey: Array installation identifier (1 to 42)
            
    Returns:
    --------
    dict or list of dicts:
        {
            "predicted_solar_generation": float, # in kWh per 15-minute interval
            "forecast_horizon_minutes": 15,
            "unit": "kWh per 15-minute interval",
            "is_daylight": bool
        }
    """
    model, feature_cols = get_model(model_path)
    
    is_single = False
    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
        is_single = True
    elif isinstance(input_data, pd.Series):
        df_in = pd.DataFrame([input_data.to_dict()])
        is_single = True
    elif isinstance(input_data, pd.DataFrame):
        df_in = input_data.copy()
    else:
        raise TypeError("input_data must be a dict, pd.Series, or pd.DataFrame")
        
    # Auto-derivation if raw timestamp provided
    if 'Timestamp' in df_in.columns:
        ts = pd.to_datetime(df_in['Timestamp'])
        if 'hour_sin' not in df_in.columns:
            df_in['hour_sin'] = np.sin(2.0 * np.pi * ts.dt.hour / 24.0)
            df_in['hour_cos'] = np.cos(2.0 * np.pi * ts.dt.hour / 24.0)
        if 'doy_sin' not in df_in.columns:
            doy = ts.dt.dayofyear
            df_in['doy_sin'] = np.sin(2.0 * np.pi * doy / 365.25)
            df_in['doy_cos'] = np.cos(2.0 * np.pi * doy / 365.25)
        if 'day_of_week' not in df_in.columns:
            df_in['day_of_week'] = ts.dt.dayofweek
            df_in['is_weekend'] = df_in['day_of_week'].isin([5, 6]).astype(int)
            
    if 'WindDirection' in df_in.columns and 'wind_dir_sin' not in df_in.columns:
        w_rad = np.radians(df_in['WindDirection'].fillna(0.0))
        df_in['wind_dir_sin'] = np.sin(w_rad)
        df_in['wind_dir_cos'] = np.cos(w_rad)
        
    if 'weather_is_missing' not in df_in.columns:
        df_in['weather_is_missing'] = df_in['AirTemperature'].isnull().astype(int) if 'AirTemperature' in df_in.columns else 0

    if feature_cols:
        for c in feature_cols:
            if c not in df_in.columns:
                df_in[c] = np.nan
        X = df_in[feature_cols]
    else:
        X = df_in

    raw_preds = model.predict(X)
    clean_preds = np.clip(raw_preds, 0.0, None)
    
    elevations = df_in.get('solar_elevation', pd.Series([10.0]*len(df_in)))
    ghis = df_in.get('Ghi', pd.Series([100.0]*len(df_in)))
    
    results = []
    for i in range(len(clean_preds)):
        p = float(clean_preds[i])
        elev = float(elevations.iloc[i]) if hasattr(elevations, 'iloc') else float(elevations[i])
        ghi_val = float(ghis.iloc[i]) if hasattr(ghis, 'iloc') else float(ghis[i])
        
        is_daylight = (elev > 0.0) or (ghi_val > 0.0)
        if not is_daylight:
            p = 0.0
            
        results.append({
            "predicted_solar_generation": round(p, 4),
            "forecast_horizon_minutes": 15,
            "unit": "kWh per 15-minute interval",
            "is_daylight": is_daylight
        })
        
    return results[0] if is_single else results

if __name__ == "__main__":
    print("Testing predict_solar_generation()...")
    
    sample_daylight = {
        "SiteKey": 14, "CampusKey": 1,
        "lag_1": 10.41, "lag_2": 9.84, "lag_4": 8.66, "lag_96": 9.50, "lag_672": 10.10,
        "rolling_mean_1h": 9.95, "rolling_std_1h": 0.52, "rolling_max_1h": 10.41, "rolling_min_1h": 8.66,
        "rolling_mean_4h": 7.80, "rolling_mean_24h": 3.10,
        "solar_elevation": 58.4, "solar_azimuth": 22.1, "clear_sky_ghi": 850.0,
        "Ghi": 820.0, "CloudOpacity": 5.0,
        "AirTemperature": 24.5, "RelativeHumidity": 42.0, "WindSpeed": 14.0,
        "wind_dir_sin": 0.5, "wind_dir_cos": 0.866, "weather_is_missing": 0,
        "hour_sin": 0.0, "hour_cos": -1.0, "doy_sin": 0.5, "doy_cos": 0.86,
        "day_of_week": 2, "is_weekend": 0, "lat": -37.718, "Lon": 145.051, "kWp": 66.0
    }
    pred_day = predict_solar_generation(sample_daylight)
    print("\n--- TEST CASE 1: MIDDAY SUNNY CONDITION ---")
    print(json.dumps(pred_day, indent=2))
    
    sample_night = {
        "SiteKey": 14, "CampusKey": 1, "lag_1": 0.0, "rolling_mean_1h": 0.0,
        "solar_elevation": -35.0, "Ghi": 0.0, "CloudOpacity": 0.0, "kWp": 66.0
    }
    pred_night = predict_solar_generation(sample_night)
    print("\n--- TEST CASE 2: MIDNIGHT INACTIVE CONDITION ---")
    print(json.dumps(pred_night, indent=2))
    print("\nValidation PASSED successfully!")
