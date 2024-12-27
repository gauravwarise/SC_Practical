import numpy as np

class HopfieldNetwork:
    def __init__(self, size): self.weights = np.zeros((size, size))
    def train(self, patterns): 
        for p in patterns: self.weights += np.outer(2*np.array(p)-1, 2*np.array(p)-1)
        np.fill_diagonal(self.weights, 0)
    def recall(self, pattern, steps=10):
        state = 2*np.array(pattern)-1
        for _ in range(steps): 
            for i in range(len(state)): state[i] = 1 if np.dot(self.weights[i], state) > 0 else -1
        return (state + 1) // 2

# Example usage
hopfield = HopfieldNetwork(4)
hopfield.train([[1, 1, 1, 1], [1, 0, 0, 1]])
print(hopfield.recall([1, 0, 0, 1]))
