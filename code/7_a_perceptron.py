import numpy as np
import matplotlib.pyplot as plt

# Perceptron model
X = np.array([[2, 3], [4, 6], [1, 1], [3, 4]])  # Training data
y = np.array([0, 1, 0, 1])  # Labels
weights = np.zeros(3)  # Initial weights (including bias)

# Perceptron training
for _ in range(1000):
    for i in range(len(X)):
        prediction = 1 if np.dot(X[i], weights[1:]) + weights[0] >= 0 else 0
        weights[1:] += 0.1 * (y[i] - prediction) * X[i]  # Update weights
        weights[0] += 0.1 * (y[i] - prediction)  # Update bias

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
x_vals = np.linspace(0, 5, 100)
plt.plot(x_vals, -(weights[1] * x_vals + weights[0]) / weights[2], label="Decision Boundary")
plt.show()
