import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dataset (2 input features)
X = np.array([
    [1, 1],
    [1, 2],
    [2, 2]
])

# Output feature
y = np.array([6, 8, 9])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Model parameters
b0 = model.intercept_
b1, b2 = model.coef_

print("Regression Equation:")
print(f"y = {b0:.2f} + {b1:.2f}*x1 + {b2:.2f}*x2")

# Predictions
y_pred = model.predict(X)

print("\nActual values:", y)
print("Predicted values:", y_pred)

# Error metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nError Metrics")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

# 3D Plot (since two inputs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = X[:,0]
x2 = X[:,1]

ax.scatter(x1, x2, y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")

plt.title("Multiple Linear Regression Data Points")
plt.show()