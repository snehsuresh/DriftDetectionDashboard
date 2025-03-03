import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model():
    df = pd.read_csv('reference_data.csv', parse_dates=['date'])
    # Use feature1 and feature2 to predict target
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Evaluate on training data (for information)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Training MSE: {mse:.2f}")
    
    # Save the trained model
    joblib.dump(model, 'model.pkl')
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_model()
