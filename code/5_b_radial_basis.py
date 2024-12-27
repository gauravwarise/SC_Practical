import numpy as np

# Gaussian Radial Basis Function
def gaussian_rbf(x, center, spread=1.0):
    return np.exp(-np.linalg.norm(x - center)**2 / (2 * spread**2))

# RBF Network class
class RBFNetwork:
    def __init__(self, n_input, n_hidden):
        self.centers = np.random.randn(n_hidden, n_input)  # Random centers
        self.weights = np.random.randn(n_hidden)  # Weights from hidden to output

    # Forward pass (output)
    def forward(self, X):
        hidden_activations = np.array([gaussian_rbf(X, center) for center in self.centers])
        return np.dot(hidden_activations, self.weights)

# Example XOR patterns
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR outputs

# Create and train RBF network (simplified, no proper training process)
rbf = RBFNetwork(n_input=2, n_hidden=3)

# Test the network
for x in X:
    output = rbf.forward(x)
    print(f"Input: {x}, Output: {output}")
