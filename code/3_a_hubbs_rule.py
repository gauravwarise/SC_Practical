import numpy as np

# Hebb's rule learning function
def hebbian_learning(inputs, outputs, weights, learning_rate=0.1):
    # Update the weights according to Hebb's rule
    for x, y in zip(inputs, outputs):
        weights += learning_rate * x * y  # Weight update rule
    return weights

# Example: Train a neuron to learn an AND operation
# Inputs (x1, x2) and expected outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])  # AND gate output

# Initialize weights and learning rate (ensure weights are float type)
weights = np.array([0.0, 0.0], dtype=float)  # Use float type for weights
learning_rate = 0.1

# Train the neuron using Hebb's rule
for epoch in range(10):  # 10 epochs
    weights = hebbian_learning(inputs, outputs, weights, learning_rate)
    print(f"Epoch {epoch+1}: Weights = {weights}")

# Test the trained neuron with the learned weights
print("\nTesting learned neuron:")
for x in inputs:
    output = np.dot(x, weights)  # Calculate output using learned weights
    print(f"Input: {x}, Output: {1 if output >= 0 else 0}")
