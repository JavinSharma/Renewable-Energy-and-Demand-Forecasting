"""
energy_manager.py — Member 3 Integration, Energy Management & Financial Payback Engine

Calculates grid requirements, renewable contribution percentages, surplus/deficit states,
battery energy storage system (BESS) dispatch, and payback period in Rupees (₹).
"""

import numpy as np
import pandas as pd

def compute_energy_balance(
    df_demand,
    df_solar=None,
    battery_capacity_kwh=500.0,
    max_charge_rate_kw=100.0,
    initial_soc_pct=50.0
):
    """
    Computes hourly energy balance, grid requirement, renewable contribution,
    and battery storage dispatch given demand and solar predictions.
    """
    df = df_demand.copy()
    
    # Standardize demand timestamp column
    if 'timestamp' not in df.columns and 'Timestamp' in df.columns:
        df['timestamp'] = df['Timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Standardize demand target column
    if 'predicted_demand_kWh' in df.columns:
        df['demand_kWh'] = df['predicted_demand_kWh']
    elif 'actual_demand_kWh' in df.columns and 'demand_kWh' not in df.columns:
        df['demand_kWh'] = df['actual_demand_kWh']
        
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Merge solar data if available
    if df_solar is not None and not df_solar.empty:
        df_s = df_solar.copy()
        
        if 'timestamp' not in df_s.columns and 'Timestamp' in df_s.columns:
            df_s['timestamp'] = df_s['Timestamp']
        df_s['timestamp'] = pd.to_datetime(df_s['timestamp'])

        if 'predicted_solar_kWh' in df_s.columns:
            df_s['solar_kWh'] = df_s['predicted_solar_kWh']
        elif 'Predicted_Generation_XGBoost' in df_s.columns:
            df_s['solar_kWh'] = df_s['Predicted_Generation_XGBoost']
        elif 'Actual_Generation_t_plus_1' in df_s.columns and 'solar_kWh' not in df_s.columns:
            df_s['solar_kWh'] = df_s['Actual_Generation_t_plus_1']
            
        df_s = df_s.sort_values('timestamp').reset_index(drop=True)
        
        # If solar is in 15-min intervals, resample to hourly
        if len(df_s) > len(df) * 2:
            df_s_hourly = df_s.set_index('timestamp').resample('1h')['solar_kWh'].sum().reset_index()
            df = pd.merge_asof(df, df_s_hourly, on='timestamp', direction='nearest')
        else:
            df = pd.merge(df, df_s[['timestamp', 'solar_kWh']], on='timestamp', how='left')
            
        df['solar_kWh'] = df['solar_kWh'].fillna(0.0)
    else:
        df['solar_kWh'] = 0.0

    # Ensure non-negative predictions
    df['demand_kWh'] = np.maximum(0.0, df['demand_kWh'])
    df['solar_kWh'] = np.maximum(0.0, df['solar_kWh'])

    # Initialize battery simulation tracking arrays
    n = len(df)
    soc_kwh = np.zeros(n)
    charge_kwh = np.zeros(n)
    discharge_kwh = np.zeros(n)
    grid_import_kwh = np.zeros(n)
    grid_export_kwh = np.zeros(n)
    dispatch_status = []

    current_soc = (initial_soc_pct / 100.0) * battery_capacity_kwh

    for i in range(n):
        demand = df.loc[i, 'demand_kWh']
        solar = df.loc[i, 'solar_kWh']
        net_diff = solar - demand  # Positive = Surplus, Negative = Deficit

        if net_diff > 0:
            # SOLAR SURPLUS: Demand is fully met by solar directly.
            surplus = net_diff
            
            # Charge battery with surplus solar up to max charge rate and max capacity
            max_possible_charge = min(surplus, max_charge_rate_kw, battery_capacity_kwh - current_soc)
            charge_kwh[i] = max(0.0, max_possible_charge)
            discharge_kwh[i] = 0.0
            current_soc += charge_kwh[i]
            
            # Excess remaining surplus after battery is charged can be exported to grid
            grid_export_kwh[i] = surplus - charge_kwh[i]
            grid_import_kwh[i] = 0.0
            
            if charge_kwh[i] > 0:
                dispatch_status.append("CHARGING (SOLAR SURPLUS)")
            else:
                dispatch_status.append("BATTERY FULL (GRID EXPORT)")
                
        else:
            # DEMAND DEFICIT: Solar is insufficient to cover demand directly.
            deficit = -net_diff  # Amount of extra energy needed
            
            # Discharge battery to cover deficit up to max discharge rate and current SoC
            max_possible_discharge = min(deficit, max_charge_rate_kw, current_soc)
            discharge_kwh[i] = max(0.0, max_possible_discharge)
            charge_kwh[i] = 0.0
            current_soc -= discharge_kwh[i]
            
            # Remaining deficit after battery discharge must be imported from the grid
            remaining_deficit = deficit - discharge_kwh[i]
            grid_import_kwh[i] = max(0.0, remaining_deficit)
            grid_export_kwh[i] = 0.0
            
            if discharge_kwh[i] > 0 and grid_import_kwh[i] > 0:
                dispatch_status.append("DISCHARGING + GRID IMPORT")
            elif discharge_kwh[i] > 0:
                dispatch_status.append("DISCHARGING (COVERED BY BESS)")
            else:
                dispatch_status.append("GRID IMPORT REQUIRED")

        soc_kwh[i] = current_soc

    df['net_grid_import_kWh'] = grid_import_kwh
    df['surplus_export_kWh'] = grid_export_kwh
    df['battery_charge_kWh'] = charge_kwh
    df['battery_discharge_kWh'] = discharge_kwh
    df['battery_soc_kWh'] = soc_kwh
    df['battery_soc_pct'] = (soc_kwh / max(1.0, battery_capacity_kwh)) * 100.0
    df['dispatch_status'] = dispatch_status

    # Renewable contribution calculation: (Direct Solar + Battery Discharge) / Demand * 100
    df['renewable_kWh_used'] = np.minimum(df['demand_kWh'], df['solar_kWh'] + df['battery_discharge_kWh'])
    df['renewable_contribution_pct'] = np.where(
        df['demand_kWh'] > 0,
        np.minimum(100.0, (df['renewable_kWh_used'] / df['demand_kWh']) * 100.0),
        100.0
    )

    # Compute Summary KPIs
    total_demand = df['demand_kWh'].sum()
    total_solar = df['solar_kWh'].sum()
    total_grid_import = df['net_grid_import_kWh'].sum()
    total_surplus_export = df['surplus_export_kWh'].sum()
    total_renewable_used = df['renewable_kWh_used'].sum()
    avg_renewable_pct = (total_renewable_used / max(1.0, total_demand)) * 100.0

    kpis = {
        'total_demand_kWh': float(total_demand),
        'total_solar_kWh': float(total_solar),
        'total_grid_import_kWh': float(total_grid_import),
        'total_surplus_export_kWh': float(total_surplus_export),
        'total_renewable_used_kWh': float(total_renewable_used),
        'overall_renewable_contribution_pct': float(avg_renewable_pct),
        'peak_demand_kWh': float(df['demand_kWh'].max()),
        'peak_solar_kWh': float(df['solar_kWh'].max()),
        'battery_capacity_kwh': battery_capacity_kwh,
        'total_hours': len(df)
    }

    return df, kpis


def compute_financial_payback(
    kpis,
    solar_kw_capacity,
    battery_kwh_capacity,
    grid_rate_rs_per_kwh=8.5,
    solar_cost_rs_per_kw=55000.0,
    battery_cost_rs_per_kwh=18000.0
):
    """
    Calculates initial capital investment, annual electricity bill savings,
    and payback period (Years & Months) in Rupees (₹).
    """
    hours_in_period = max(1, kpis.get('total_hours', 24))
    annual_factor = 8760.0 / hours_in_period
    
    annual_demand_kwh = kpis['total_demand_kWh'] * annual_factor
    annual_renewable_kwh_used = kpis['total_renewable_used_kWh'] * annual_factor
    
    # Financial CAPEX Calculation
    solar_capex_rs = solar_kw_capacity * solar_cost_rs_per_kw
    battery_capex_rs = battery_kwh_capacity * battery_cost_rs_per_kwh
    total_capex_rs = solar_capex_rs + battery_capex_rs
    
    # Financial Savings Calculation
    annual_bill_without_renewables_rs = annual_demand_kwh * grid_rate_rs_per_kwh
    annual_bill_savings_rs = annual_renewable_kwh_used * grid_rate_rs_per_kwh
    annual_bill_with_renewables_rs = max(0.0, annual_bill_without_renewables_rs - annual_bill_savings_rs)
    
    # Payback Period Calculation
    if annual_bill_savings_rs > 0:
        payback_float_years = total_capex_rs / annual_bill_savings_rs
        years = int(payback_float_years)
        months = int(round((payback_float_years - years) * 12))
        if months == 12:
            years += 1
            months = 0
    else:
        years = 99
        months = 0
        payback_float_years = 99.0
        
    return {
        'solar_capex_rs': float(solar_capex_rs),
        'battery_capex_rs': float(battery_capex_rs),
        'total_capex_rs': float(total_capex_rs),
        'annual_demand_kwh': float(annual_demand_kwh),
        'annual_renewable_kwh_used': float(annual_renewable_kwh_used),
        'annual_bill_without_renewables_rs': float(annual_bill_without_renewables_rs),
        'annual_bill_savings_rs': float(annual_bill_savings_rs),
        'annual_bill_with_renewables_rs': float(annual_bill_with_renewables_rs),
        'payback_float_years': float(payback_float_years),
        'payback_years': years,
        'payback_months': months,
        'payback_str': f"{years} Years, {months} Months" if years < 50 else "N/A (No Savings)"
    }
