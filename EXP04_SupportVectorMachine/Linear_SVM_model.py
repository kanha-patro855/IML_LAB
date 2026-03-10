import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Dataset
X = np.array([
    [1,1],
    [2,1],
    [2,3],
    [3,3]
])

# Labels
y = np.array([1,1,-1,-1])

# Train SVM model
model = SVC(kernel='linear', C=1000000)
model.fit(X, y)

# Extract parameters
w = model.coef_[0]
b = model.intercept_[0]

print("Weight vector (w):", w)
print("Bias (b):", b)

# Margin
margin = 2 / np.linalg.norm(w)
print("Margin:", margin)

# Predictions
pred = model.predict(X)
print("Predicted labels:", pred)

# Create decision boundary line
x_vals = np.linspace(0,4,100)
y_vals = -(w[0]*x_vals + b) / w[1]

# Plot graph
plt.scatter(X[:,0], X[:,1], label="Data points")
plt.plot(x_vals, y_vals, label="Decision Boundary")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("SVM Decision Boundary")
plt.legend()

plt.show()