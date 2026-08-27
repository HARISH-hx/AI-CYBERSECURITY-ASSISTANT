import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

data = pd.read_csv("user_activity.csv")


features = [
    "login_hour",
    "login_duration",
    "failed_attempts",
    "device_change",
    "location_change"
]

X = data[features]


model = IsolationForest(
    n_estimators=100,
    contamination=0.25,
    random_state=42
)


model.fit(X)


joblib.dump(model, "anomaly_model.pkl")

print("=" * 50)
print("USER BEHAVIOR ANOMALY DETECTION MODEL")
print("=" * 50)

print(f"Dataset records: {len(data)}")
print("Features used:")
for feature in features:
    print(f"- {feature}")

print("\nModel: Isolation Forest")
print("Training completed successfully.")
print("Model saved as anomaly_model.pkl")