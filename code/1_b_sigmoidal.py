import numpy as np

# Define the binary sigmoid function
def binary_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the bipolar sigmoid function
def bipolar_sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1

# Example values for input, weight, and bias
x = 2
w = 0.5
b = -1

# Compute the weighted sum
z = w * x + b

# Calculate the outputs
output_binary = binary_sigmoid(z)
output_bipolar = bipolar_sigmoid(z)

# Print the results
print(f"Binary Sigmoid Output: {output_binary}")
print(f"Bipolar Sigmoid Output: {output_bipolar}")
