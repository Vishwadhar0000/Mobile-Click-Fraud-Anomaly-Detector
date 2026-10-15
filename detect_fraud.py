import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- 1. Synthetic Telemetry Generator (Simulating High-Frequency Mobile Ad / Transaction Fraud) ---
def generate_fraud_telemetry(n_samples=10000, fraud_ratio=0.03, random_state=42):
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # Legitimate user patterns: Natural touch durations, low burst frequencies, regular coords
    touch_duration_norm = np.random.normal(loc=120, scale=25, size=n_normal).clip(30, 300)
    click_interval_norm = np.random.exponential(scale=3500, size=n_normal) + 500
    coord_variance_norm = np.random.normal(loc=45.0, scale=10.0, size=n_normal).clip(10, 100)
    device_velocity_norm = np.random.normal(loc=1.2, scale=0.4, size=n_normal).clip(0.1, 4.0)

    # Fraudulent/Bot patterns: Micro-second bursts, zero coordinate variance, repetitive intervals
    touch_duration_fraud = np.random.exponential(scale=15, size=n_fraud).clip(5, 50)
    click_interval_fraud = np.random.normal(loc=80, scale=15, size=n_fraud).clip(20, 200)
    coord_variance_fraud = np.random.exponential(scale=1.5, size=n_fraud).clip(0, 5)
    device_velocity_fraud = np.random.normal(loc=8.5, scale=2.0, size=n_fraud).clip(4.5, 15.0)

    X_norm = np.column_stack([touch_duration_norm, click_interval_norm, coord_variance_norm, device_velocity_norm])
    X_fraud = np.column_stack([touch_duration_fraud, click_interval_fraud, coord_variance_fraud, device_velocity_fraud])

    X = np.vstack([X_norm, X_fraud])
    y = np.array([0] * n_normal + [1] * n_fraud)

    df = pd.DataFrame(X, columns=['touch_duration_ms', 'inter_click_interval_ms', 'touch_coord_variance', 'device_velocity'])
    df['is_fraud'] = y
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# --- 2. Training and Evaluation Pipeline ---
def run_pipeline():
    print("Generating simulated mobile telemetry dataset...")
    df = generate_fraud_telemetry(n_samples=12000, fraud_ratio=0.04)

    features = ['touch_duration_ms', 'inter_click_interval_ms', 'touch_coord_variance', 'device_velocity']
    X = df[features]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Training Supervised Anomaly Classifier (Random Forest) ---")
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print("\nClassification Report (Handling Imbalanced Fraud Stream):")
    print(classification_report(y_test, y_pred, target_names=['Legitimate Click', 'Fraudulent Bot']))

    # Feature Importance
    importances = clf.feature_importances_
    print("\nFeature Importances:")
    for feat, imp in zip(features, importances):
        print(f"  - {feat}: {imp:.4f}")

if __name__ == '__main__':
    run_pipeline()
