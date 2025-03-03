import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    df = pd.read_csv('reference_data.csv', parse_dates=['date'])
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    # Define hyperparameter grid for Random Forest.
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Hyperparameter tuning using RandomizedSearchCV.
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    
    # Evaluate using cross-validation.
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_cv = -cv_scores.mean()
    logging.info(f"Cross-validated MSE: {mse_cv:.2f}")
    
    # Train on the full dataset.
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    mse_train = mean_squared_error(y, y_pred)
    logging.info(f"Training MSE on full data: {mse_train:.2f}")
    
    # Save the best model.
    joblib.dump(best_model, 'model.pkl')
    logging.info("Best model saved to 'model.pkl'.")
    print("Model training and hyperparameter tuning completed.")

if __name__ == "__main__":
    train_model()
