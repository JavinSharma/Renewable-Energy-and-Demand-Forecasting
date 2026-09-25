import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def train_demand_model():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, "data", "processed_demand.csv")
    models_dir = os.path.join(project_root, "models")
    output_dir = os.path.join(project_root, "output")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Processed dataset not found at {data_file}. Run preprocess.py first.")
        
    print(f"[1/5] Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature columns definition
    feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'sin_hour', 'cos_hour', 'is_holiday', 'is_semester', 'is_exam',
        'lag_1h', 'lag_2h', 'lag_24h', 'lag_168h',
        'rolling_mean_24h', 'rolling_std_24h'
    ]
    
    # Ensure all feature columns exist in dataset
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = 'demand_kWh'
    
    X = df[feature_cols]
    y = df[target_col]
    timestamps = df['timestamp']
    
    # 80/20 Time-series split (strictly chronological)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    ts_train, ts_test = timestamps.iloc[:split_idx], timestamps.iloc[split_idx:]
    
    print(f"[2/5] Dataset Split: {len(X_train)} Train samples | {len(X_test)} Test samples")
    print(f"      Training range: {ts_train.min()} to {ts_train.max()}")
    print(f"      Testing range:  {ts_test.min()} to {ts_test.max()}")
    
    # Model Initialization & Training
    print("\n[3/5] Training XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions & Evaluation
    print("[4/5] Evaluating model on Test set...")
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print("\n" + "="*50)
    print("  MEMBER 1 - DEMAND FORECASTING PERFORMANCE  ")
    print("="*50)
    print(f"  * MAE  (Mean Absolute Error):     {mae:.2f} kWh")
    print(f"  * RMSE (Root Mean Squared Error): {rmse:.2f} kWh")
    print(f"  * R2   (Variance Explained):      {r2:.4f}")
    print("="*50)
    
    # Feature Importances
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 Predictive Features:")
    for idx, row in importances.head(5).iterrows():
        print(f"  - {row['Feature']:<20}: {row['Importance']:.4f}")
        
    # Save Model Artifact
    model_path = os.path.join(models_dir, "demand_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n[5/5] Model artifact saved to: {model_path}")
    
    # Save Forecast Output CSV
    results_df = pd.DataFrame({
        'timestamp': ts_test,
        'actual_demand_kWh': y_test.values,
        'predicted_demand_kWh': preds
    })
    results_path = os.path.join(output_dir, "demand_forecast.csv")
    results_df.to_csv(results_path, index=False)
    print(f"SUCCESS: Forecast output exported to: {results_path}")
    
    return model, results_df

if __name__ == "__main__":
    train_demand_model()
