import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given values
alpha = np.array([1,1,1])
y = np.array([1,-1,-1])

# Support vectors
SV = np.array([
    [0,-1,1],
    [0,2,-1],
    [-1,0,2]
])

# Compute weight vector
w = np.sum((alpha * y).reshape(-1,1) * SV, axis=0)

# Bias (not provided → assume 0)
b = 0

print("Weight vector w:", w)
print("Bias b:", b)

# Input feature vector
x = np.array([0.2,0.1,0.4])

# Decision function
f = np.dot(w, x) + b

# Predicted label
prediction = np.sign(f)

print("Decision value:", f)
print("Predicted class label:", prediction)

# 3D Graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot support vectors
ax.scatter(SV[:,0], SV[:,1], SV[:,2])

# Plot input point
ax.scatter(x[0], x[1], x[2])

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("SVM Support Vectors and Input Point")

plt.show()