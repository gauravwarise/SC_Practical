import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Backpropagation Algorithm
def backpropagation(inputs, outputs, hidden_size=2, learning_rate=0.1, epochs=10000):
    # Initialize weights
    input_size = inputs.shape[1]
    output_size = outputs.shape[1]
    
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    
    for epoch in range(epochs):
        # Forward pass
        hidden_input = np.dot(inputs, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        
        final_input = np.dot(hidden_output, weights_hidden_output)
        final_output = sigmoid(final_input)
        
        # Error calculation (Mean Squared Error)
        error = outputs - final_output
        
        # Backward pass (Gradient Descent)
        d_output = error * sigmoid_derivative(final_output)
        d_hidden = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)
        
        # Update weights
        weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
        weights_input_hidden += inputs.T.dot(d_hidden) * learning_rate

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

    return weights_input_hidden, weights_hidden_output

# XOR Input and Output
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Train the neural network using Backpropagation
weights_input_hidden, weights_hidden_output = backpropagation(inputs, outputs)

# Testing the trained network
hidden_input = np.dot(inputs, weights_input_hidden)
hidden_output = sigmoid(hidden_input)
final_input = np.dot(hidden_output, weights_hidden_output)
final_output = sigmoid(final_input)

print("\nFinal Outputs after training:")
print(final_output)
