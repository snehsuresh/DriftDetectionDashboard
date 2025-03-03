import pandas as pd
import joblib
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import ColumnDriftMetric
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import logging

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    ref = pd.read_csv('reference_data.csv', parse_dates=['date'])
    curr = pd.read_csv('current_data.csv', parse_dates=['date'])
    return ref, curr

def add_predictions(df, model):
    X = df[['feature1', 'feature2']]
    df = df.copy()
    df['prediction'] = model.predict(X)
    return df

def generate_evidently_report(ref, curr):
    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset(),
        ColumnDriftMetric(column_name='feature1'),
        ColumnDriftMetric(column_name='feature2')
    ])
    
    report.run(reference_data=ref, current_data=curr)
    report.save_html('evidently_report.html')
    logging.info("Evidently report saved to 'evidently_report.html'.")

def compute_feature_importance_drift(model, ref, curr):
    X_ref = ref[['feature1', 'feature2']]
    y_ref = ref['target']
    X_curr = curr[['feature1', 'feature2']]
    y_curr = curr['target']
    
    result_ref = permutation_importance(model, X_ref, y_ref, n_repeats=10, random_state=42, n_jobs=-1)
    result_curr = permutation_importance(model, X_curr, y_curr, n_repeats=10, random_state=42, n_jobs=-1)
    
    features = ['feature1', 'feature2']
    ref_importance = [result_ref.importances_mean[i] for i in range(len(features))]
    curr_importance = [result_curr.importances_mean[i] for i in range(len(features))]
    
    x = range(len(features))
    plt.figure(figsize=(8, 6))
    plt.bar([p - 0.2 for p in x], ref_importance, width=0.4, label='Reference')
    plt.bar([p + 0.2 for p in x], curr_importance, width=0.4, label='Current')
    plt.xticks(x, features)
    plt.ylabel("Permutation Importance")
    plt.title("Feature Importance Drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importance_drift.png')
    logging.info("Feature importance drift plot saved to 'feature_importance_drift.png'.")

def main():
    ref, curr = load_data()
    model = joblib.load('model.pkl')
    
    ref = add_predictions(ref, model)
    curr = add_predictions(curr, model)
    
    generate_evidently_report(ref, curr)
    compute_feature_importance_drift(model, ref, curr)
    
    mse_ref = mean_squared_error(ref['target'], ref['prediction'])
    mse_curr = mean_squared_error(curr['target'], curr['prediction'])
    
    logging.info(f"Reference MSE: {mse_ref:.2f}")
    logging.info(f"Current MSE: {mse_curr:.2f}")
    
    if mse_curr > 1.5 * mse_ref:
        logging.error("ALERT: Performance degradation detected!")
    else:
        logging.info("Model performance is stable.")

if __name__ == "__main__":
    main()
