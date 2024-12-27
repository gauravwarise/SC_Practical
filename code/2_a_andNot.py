import numpy as np

# McCulloch-Pitts Neuron
def mcculloch_pitts(inputs, weights, bias, threshold):
    # Weighted sum of inputs
    weighted_sum = np.dot(inputs, weights) + bias
    # Apply threshold function
    return 1 if weighted_sum >= threshold else 0

# AND Function
def and_function(x1, x2):
    weights = np.array([1, 1])  # Weights for AND
    bias = -1.5  # Bias for AND
    threshold = 1  # Threshold for AND
    return mcculloch_pitts(np.array([x1, x2]), weights, bias, threshold)

# NOT Function
def not_function(x):
    weights = np.array([-1])  # Weight for NOT
    bias = 0.5  # Bias for NOT
    threshold = 0  # Threshold for NOT
    return mcculloch_pitts(np.array([x]), weights, bias, threshold)

# Test AND function
print("AND Function:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"AND({x1}, {x2}) = {and_function(x1, x2)}")

# Test NOT function
print("\nNOT Function:")
for x in [0, 1]:
    print(f"NOT({x}) = {not_function(x)}")
