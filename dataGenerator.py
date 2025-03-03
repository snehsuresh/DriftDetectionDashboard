import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_reference_data(n=1000, start_date='2023-01-01'):
    np.random.seed(42)
    dates = pd.date_range(start_date, periods=n, freq='D')
    feature1 = np.random.normal(loc=50, scale=10, size=n)
    feature2 = np.random.normal(loc=30, scale=5, size=n)
    noise = np.random.normal(0, 5, size=n)
    target = 2 * feature1 + 3 * feature2 + noise
    df = pd.DataFrame({
        'date': dates, 
        'feature1': feature1, 
        'feature2': feature2, 
        'target': target
    })
    return df

def generate_current_data(n=200, start_date='2025-09-01'):
    # Introduce drift: change in feature distributions and target relationship.
    np.random.seed(24)
    dates = pd.date_range(start_date, periods=n, freq='D')
    feature1 = np.random.normal(loc=55, scale=12, size=n)  # mean and variance shifted
    feature2 = np.random.normal(loc=28, scale=6, size=n)
    noise = np.random.normal(0, 7, size=n)
    # Drifted target relationship: coefficients changed
    target = 1.5 * feature1 + 3.5 * feature2 + noise
    df = pd.DataFrame({
        'date': dates, 
        'feature1': feature1, 
        'feature2': feature2, 
        'target': target
    })
    return df

if __name__ == "__main__":
    ref_data = generate_reference_data()
    curr_data = generate_current_data()
    ref_data.to_csv('reference_data.csv', index=False)
    curr_data.to_csv('current_data.csv', index=False)
    print("Data generated and saved to reference_data.csv and current_data.csv")
