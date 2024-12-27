import numpy as np

# McCulloch-Pitts Neuron
def mcculloch_pitts(inputs, weights, bias, threshold):
    weighted_sum = np.dot(inputs, weights) + bias
    return 1 if weighted_sum >= threshold else 0

# AND Function
def and_function(x1, x2):
    weights = np.array([1, 1])
    bias = -1.5
    threshold = 1
    return mcculloch_pitts(np.array([x1, x2]), weights, bias, threshold)

# OR Function
def or_function(x1, x2):
    weights = np.array([1, 1])
    bias = -0.5
    threshold = 1
    return mcculloch_pitts(np.array([x1, x2]), weights, bias, threshold)

# XOR Function using AND, OR, and NOT
def xor_function(x1, x2):
    # Step 1: Compute intermediate results
    and_out = and_function(x1, x2)
    or_out = or_function(x1, x2)
    
    # Step 2: XOR is (A OR B) AND NOT(A AND B)
    xor_out = and_function(or_out, not_function(and_out))  # NOT AND is used here
    return xor_out

# NOT Function
def not_function(x):
    weights = np.array([-1])
    bias = 0.5
    threshold = 0
    return mcculloch_pitts(np.array([x]), weights, bias, threshold)

# Test XOR function
print("XOR Function:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"XOR({x1}, {x2}) = {xor_function(x1, x2)}")
