import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def verify():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "demand_model.pkl")
    forecast_path = os.path.join(project_root, "output", "demand_forecast.csv")
    plot_path = os.path.join(project_root, "output", "forecast_comparison.png")
    
    # Check if files exist
    print("Checking model artifact...")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"SUCCESS: Successfully loaded trained model artifact from '{model_path}'")
    else:
        print(f"ERROR: Model file not found at '{model_path}'")
        return
        
    print("\nChecking generated forecast CSV...")
    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"SUCCESS: Loaded {len(df)} forecast rows from '{forecast_path}'")
        
        # Display sample 5 rows
        print("\nSample Predictions (First 5 hours of test set):")
        print("-" * 65)
        print(f"{'Timestamp':<22} | {'Actual (kWh)':<15} | {'Predicted (kWh)':<15}")
        print("-" * 65)
        for _, row in df.head(5).iterrows():
            ts_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
            print(f"{ts_str:<22} | {row['actual_demand_kWh']:<15.2f} | {row['predicted_demand_kWh']:<15.2f}")
        print("-" * 65)
        
        # Generate visual plot of 7-day sample window (168 hours)
        sample_7d = df.iloc[:168]
        plt.figure(figsize=(12, 5))
        plt.plot(sample_7d['timestamp'], sample_7d['actual_demand_kWh'], label='Actual Demand', color='#1f77b4', linewidth=1.8)
        plt.plot(sample_7d['timestamp'], sample_7d['predicted_demand_kWh'], label='XGBoost Forecast', color='#ff7f0e', linestyle='--', linewidth=1.8)
        plt.title("Member 1 — Energy Demand Forecasting (7-Day Sample: Actual vs Predicted)", fontsize=13, fontweight='bold')
        plt.xlabel("Timestamp", fontsize=10)
        plt.ylabel("Electricity Demand (kWh)", fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nSUCCESS: Generated 7-day comparison chart saved to '{plot_path}'")
        
    else:
        print(f"ERROR: Forecast output CSV not found at '{forecast_path}'")

if __name__ == "__main__":
    verify()
