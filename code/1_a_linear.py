import numpy as np

# Define the linear model
def linear_model(x, w, b):
    return np.dot(x, w) + b  # y = wx + b

# Mean Squared Error (MSE) Loss function
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Training function (Gradient Descent)
def train(x_train, y_train, w, b, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        # Forward pass: calculate predicted output
        y_pred = linear_model(x_train, w, b)
        
        # Calculate the loss
        loss = mse_loss(y_pred, y_train)
        
        # Backpropagation: compute gradients
        w_grad = np.dot(x_train.T, (y_pred - y_train)) / len(x_train)
        b_grad = np.mean(y_pred - y_train)
        
        # Update weights and bias using gradient descent
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad
        
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return w, b

# Example data (simple linear relationship: y = 2x + 1)
x_train = np.array([[1], [2], [3], [4]])  # Input features (1D)
y_train = np.array([3, 5, 7, 9])  # Target values

# Initialize weights and bias
w = np.random.randn(1)  # Random initial weight
b = np.random.randn(1)  # Random initial bias

# Train the model
w, b = train(x_train, y_train, w, b, learning_rate=0.01, epochs=1000)

# Final weights and bias after training
print("Trained weight:", w)
print("Trained bias:", b)

# Make a prediction with the trained model
x_test = np.array([5])
prediction = linear_model(x_test, w, b)
print("Prediction for input 5:", prediction)


