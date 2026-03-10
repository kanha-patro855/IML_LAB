import numpy as np

# Design matrix
X = np.array([
    [1,1,2],
    [1,2,3],
    [1,3,4]
])

# Target values
y = np.array([1,2,3])

# Regularization parameter
lam = 1

# Identity matrix
I = np.eye(X.shape[1])

# Ridge regression formula
beta = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y

print("Ridge Regression Coefficients:")
print(beta)

# Model equation
print("\nRegression Model:")
print(f"y = {beta[0]} + {beta[1]}*x1 + {beta[2]}*x2")

# Prediction
y_pred = X @ beta

print("\nPredicted values:")
print(y_pred) 