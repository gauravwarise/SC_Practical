import numpy as np

# Step function as activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Delta rule learning function
def delta_rule(inputs, outputs, weights, learning_rate=0.1, epochs=10):
    for _ in range(epochs):
        for x, y_true in zip(inputs, outputs):
            y_pred = step_function(np.dot(x, weights))  # Predicted output
            error = y_true - y_pred  # Error
            weights += learning_rate * error * x  # Update weights
    return weights

# AND gate input/output
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])  # AND outputs

# Initialize weights
weights = np.zeros(2)

# Train using delta rule
weights = delta_rule(inputs, outputs, weights)

# Test the trained model
for x in inputs:
    print(f"Input: {x}, Predicted Output: {step_function(np.dot(x, weights))}")
