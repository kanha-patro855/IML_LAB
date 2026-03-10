import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
y = np.array([2, 4, 5, 4, 5])

# Create Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Model parameters
slope = model.coef_[0]
intercept = model.intercept_

print("Regression Equation:")
print(f"y = {slope:.2f}x + {intercept:.2f}")

# Predictions
y_pred = model.predict(X)

print("\nActual Values:", y)
print("Predicted Values:", y_pred)

# Error Metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nError Metrics")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# Graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression Model")
plt.legend()

plt.show()