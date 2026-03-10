import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Input feature
X = np.array([1,2,3]).reshape(-1,1)

# Output feature
Y = np.array([
    [2,3],
    [4,5],
    [6,7]
])

# Create regression model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Extract parameters
intercept = model.intercept_
coef = model.coef_

print("Regression Equations:")
print(f"y1 = {intercept[0]:.2f} + {coef[0][0]:.2f}x")
print(f"y2 = {intercept[1]:.2f} + {coef[1][0]:.2f}x")

# Predictions
Y_pred = model.predict(X)

print("\nActual Values:")
print(Y)

print("\nPredicted Values:")
print(Y_pred)

# Error Metrics
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("\nError Metrics")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

# Separate outputs
y1 = Y[:,0]
y2 = Y[:,1]

y1_pred = Y_pred[:,0]
y2_pred = Y_pred[:,1]

# Graph for y1
plt.figure()
plt.scatter(X, y1, label="Actual y1")
plt.plot(X, y1_pred, label="Regression line")
plt.xlabel("X")
plt.ylabel("Y1")
plt.title("Regression Graph for y1")
plt.legend()
plt.show()

# Graph for y2
plt.figure()
plt.scatter(X, y2, label="Actual y2")
plt.plot(X, y2_pred, label="Regression line")
plt.xlabel("X")
plt.ylabel("Y2")
plt.title("Regression Graph for y2")
plt.legend()
plt.show()