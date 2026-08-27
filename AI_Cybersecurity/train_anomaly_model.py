import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest


# -----------------------------------------
# Load dataset
# -----------------------------------------
data = pd.read_csv("user_activity.csv")


# -----------------------------------------
# Features used by the model
# -----------------------------------------
features = [
    "login_hour",
    "login_duration",
    "failed_attempts",
    "device_change",
    "location_change"
]


# -----------------------------------------
# Check required columns
# -----------------------------------------
missing_columns = [column for column in features if column not in data.columns]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# -----------------------------------------
# Prepare training data
# -----------------------------------------
X = data[features].copy()


# Make sure all values are numeric
for column in features:
    X[column] = pd.to_numeric(X[column], errors="raise")


# -----------------------------------------
# Train Isolation Forest
# -----------------------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=0.20,
    random_state=42
)


model.fit(X)


# -----------------------------------------
# Save model
# -----------------------------------------
joblib.dump(model, "anomaly_model.pkl")


# -----------------------------------------
# Training information
# -----------------------------------------
print("=" * 50)
print("USER BEHAVIOR ANOMALY DETECTION MODEL")
print("=" * 50)

print(f"Dataset records: {len(data)}")

print("\nFeatures used:")
for feature in features:
    print(f"- {feature}")

print("\nModel: Isolation Forest")
print("Trees: 200")
print("Expected anomaly rate: 20%")

print("\nTraining completed successfully.")
print("Model saved as anomaly_model.pkl")
print("=" * 50)