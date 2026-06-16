import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pickle

# Load cleaned dataset
df = pd.read_csv("ames_model.csv")

# Separate target and predictors
# Drop sale_price (raw target) and log_sale_price (target we're predicting)
X = df.drop(columns=["sale_price", "log_sale_price"])
y = df["log_sale_price"]

# Convert categorical columns to dummies — same as R's model.matrix
X = pd.get_dummies(X, drop_first=True)

print(f"Features after dummies: {X.shape[1]}")

# Standardize — Lasso requires features on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Lasso with cross-validation to find best alpha (same as lambda in R)
model = LassoCV(cv=10, random_state=42, max_iter=10000)
model.fit(X_scaled, y)

print(f"Best alpha: {model.alpha_}")
print(f"R² on training data: {model.score(X_scaled, y):.4f}")

# Save model, scaler, and feature names for the API
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

feature_names = X.columns.tolist()
with open("features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("Model saved successfully")