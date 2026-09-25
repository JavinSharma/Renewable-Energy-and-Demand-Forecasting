import os
import numpy as np
import pandas as pd

def preprocess_data(data_dir, output_dir):
    print("[1/5] Loading building consumption data...")
    bldg_path = os.path.join(data_dir, "building_consumption.csv")
    df_bldg = pd.read_csv(bldg_path, usecols=['campus_id', 'meter_id', 'timestamp', 'consumption'])
    
    df_bldg['timestamp'] = pd.to_datetime(df_bldg['timestamp'])
    df_bldg['hour'] = df_bldg['timestamp'].dt.floor('h')
    
    print("[2/5] Aggregating 15-minute readings to hourly total demand (kWh)...")
    df_hourly = df_bldg.groupby('hour')['consumption'].sum().reset_index().rename(
        columns={'hour': 'timestamp', 'consumption': 'demand_kWh'}
    )
    df_hourly = df_hourly.sort_values('timestamp').reset_index(drop=True)
    
    print("[3/5] Merging calendar features...")
    cal_path = os.path.join(data_dir, "calender.csv")
    if os.path.exists(cal_path):
        df_cal = pd.read_csv(cal_path)
        df_cal['date'] = pd.to_datetime(df_cal['date']).dt.date
        df_hourly['date'] = df_hourly['timestamp'].dt.date
        df_hourly = pd.merge(df_hourly, df_cal, on='date', how='left')
        df_hourly.drop(columns=['date'], inplace=True)
        df_hourly['is_holiday'] = df_hourly['is_holiday'].fillna(0).astype(int)
        df_hourly['is_semester'] = df_hourly['is_semester'].fillna(0).astype(int)
        df_hourly['is_exam'] = df_hourly['is_exam'].fillna(0).astype(int)
    
    print("[4/5] Creating time features...")
    df_hourly['hour'] = df_hourly['timestamp'].dt.hour
    df_hourly['day_of_week'] = df_hourly['timestamp'].dt.dayofweek
    df_hourly['day_of_month'] = df_hourly['timestamp'].dt.day
    df_hourly['month'] = df_hourly['timestamp'].dt.month
    df_hourly['is_weekend'] = (df_hourly['day_of_week'] >= 5).astype(int)
    
    # Cyclic time encodings
    df_hourly['sin_hour'] = np.sin(2 * np.pi * df_hourly['hour'] / 24.0)
    df_hourly['cos_hour'] = np.cos(2 * np.pi * df_hourly['hour'] / 24.0)
    
    print("[5/5] Creating lag and rolling window features...")
    df_hourly['lag_1h'] = df_hourly['demand_kWh'].shift(1)
    df_hourly['lag_2h'] = df_hourly['demand_kWh'].shift(2)
    df_hourly['lag_24h'] = df_hourly['demand_kWh'].shift(24)
    df_hourly['lag_168h'] = df_hourly['demand_kWh'].shift(168)
    
    # Rolling 24-hour statistics (shifted by 1 to prevent data leakage)
    df_hourly['rolling_mean_24h'] = df_hourly['demand_kWh'].shift(1).rolling(window=24).mean()
    df_hourly['rolling_std_24h'] = df_hourly['demand_kWh'].shift(1).rolling(window=24).std()
    
    # Clean up NaNs from initial lag window
    before_dropna = len(df_hourly)
    df_hourly = df_hourly.dropna().reset_index(drop=True)
    after_dropna = len(df_hourly)
    print(f"Cleaned initial lag window NaNs. Rows retained: {after_dropna} (dropped {before_dropna - after_dropna})")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_demand.csv")
    df_hourly.to_csv(output_file, index=False)
    print(f"SUCCESS: Preprocessed dataset saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive_dir = os.path.join(project_root, "archive")
    data_dir = os.path.join(project_root, "data")
    preprocess_data(archive_dir, data_dir)
