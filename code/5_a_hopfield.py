import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    # Train using Hebbian learning
    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    # Recall the stored pattern
    def recall(self, pattern, steps=5):
        for _ in range(steps):
            pattern = np.sign(np.dot(self.weights, pattern))  # Update state
        return pattern

# Example patterns
pattern1 = [1, -1, 1, -1]
pattern2 = [-1, -1, 1, 1]

# Initialize and train
hopfield = HopfieldNetwork(4)
hopfield.train([pattern1, pattern2])

# Test with noisy input
noisy_input = [1, -1, -1, -1]
recalled = hopfield.recall(noisy_input)

print("Noisy Input:", noisy_input)
print("Recalled Pattern:", recalled)
