import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------
# Step 1: Load and Preprocess Data
# ------------------------------
path = "datasets/uber.csv"
data = pd.read_csv(path)

# Drop rows with null values
data = data.dropna()

# Extract datetime features
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], errors='coerce')
data = data.dropna(subset=['pickup_datetime'])

data['hour'] = data['pickup_datetime'].dt.hour
data['day'] = data['pickup_datetime'].dt.day
data['month'] = data['pickup_datetime'].dt.month


# Compute distance between pickup & dropoff using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


data['distance'] = haversine(
    data['pickup_latitude'], data['pickup_longitude'],
    data['dropoff_latitude'], data['dropoff_longitude']
)

# Keep relevant features
features = ['distance', 'passenger_count', 'hour', 'day', 'month']
X = data[features].values
y = data['fare_amount'].values

# Standardize features
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]

# ------------------------------
# Step 2: Outlier Detection
# ------------------------------
# Simple rule: Remove fares < 0 or > 200
mask = (y > 0) & (y < 200) & (data['distance'] < 100)
X, y = X[mask], y[mask]

# ------------------------------
# Step 3: Correlation Check
# ------------------------------
corr = pd.DataFrame(np.c_[X[:, 1:], y], columns=features + ['fare']).corr()
print("Correlation with fare:\n", corr['fare'])


# ------------------------------
# Step 4: Implement Regression Models
# ------------------------------

def train_linear_regression(X, y, lr=0.01, epochs=1000):
    """
    Gradient Descent for Linear Regression
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    for _ in range(epochs):
        y_pred = X.dot(theta)
        error = y_pred - y
        grad = (1 / n_samples) * X.T.dot(error)
        theta -= lr * grad
    return theta


def train_ridge(X, y, alpha=1.0, lr=0.01, epochs=1000):
    """
    Ridge Regression (L2 regularization)
    Cost = MSE + alpha * ||theta||^2
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    for _ in range(epochs):
        y_pred = X.dot(theta)
        error = y_pred - y
        grad = (1 / n_samples) * (X.T.dot(error) + alpha * theta)
        grad[0] -= alpha * theta[0]  # don't penalize bias
        theta -= lr * grad
    return theta


def train_lasso(X, y, alpha=0.01, lr=0.01, epochs=1000):
    """
    Lasso Regression (L1 regularization)
    Cost = MSE + alpha * ||theta||
    Uses sub-gradient for L1 term
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    for _ in range(epochs):
        y_pred = X.dot(theta)
        error = y_pred - y
        grad = (1 / n_samples) * X.T.dot(error)

        # Add L1 penalty (subgradient)
        grad += alpha * np.sign(theta)
        grad[0] -= alpha * np.sign(theta[0])  # don't penalize bias
        theta -= lr * grad
    return theta


# ------------------------------
# Step 5: Model Evaluation
# ------------------------------
def predict(X, theta):
    return X.dot(theta)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Train models
theta_lin = train_linear_regression(X, y)
theta_ridge = train_ridge(X, y, alpha=1.0)
theta_lasso = train_lasso(X, y, alpha=0.01)

# Predictions
y_pred_lin = predict(X, theta_lin)
y_pred_ridge = predict(X, theta_ridge)
y_pred_lasso = predict(X, theta_lasso)

# Scores
print("\nModel Evaluation:")
print("Linear Regression: R2 =", r2_score(y, y_pred_lin), ", RMSE =", rmse(y, y_pred_lin))
print("Ridge Regression:  R2 =", r2_score(y, y_pred_ridge), ", RMSE =", rmse(y, y_pred_ridge))
print("Lasso Regression:  R2 =", r2_score(y, y_pred_lasso), ", RMSE =", rmse(y, y_pred_lasso))
