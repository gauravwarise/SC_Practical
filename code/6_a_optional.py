import numpy as np
import matplotlib.pyplot as plt

class KohonenSOM:
    def __init__(self, grid_size, input_dim, learning_rate=0.1, radius=1, max_iter=1000):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.radius = radius
        self.max_iter = max_iter
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)

    def _distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def _find_bmu(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        return np.unravel_index(np.argmin(distances), self.grid_size)

    def _update_weights(self, input_vector, bmu_coords, iter_count):
        learning_rate = self.learning_rate * (1 - iter_count / self.max_iter)
        radius = self.radius * (1 - iter_count / self.max_iter)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                dist_to_bmu = self._distance(np.array([i, j]), np.array(bmu_coords))
                if dist_to_bmu < radius:
                    influence = np.exp(-dist_to_bmu**2 / (2 * radius**2))
                    self.weights[i, j] += learning_rate * influence * (input_vector - self.weights[i, j])

    def train(self, data):
        for iter_count in range(self.max_iter):
            for input_vector in data:
                bmu_coords = self._find_bmu(input_vector)
                self._update_weights(input_vector, bmu_coords, iter_count)

    def visualize(self):
        plt.imshow(self.weights[:, :, 0], cmap='coolwarm')
        plt.colorbar()
        plt.title("SOM Visualization (First Feature)")
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Sample 2D data points
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Initialize and train the SOM
    som = KohonenSOM(grid_size=(3, 3), input_dim=2, learning_rate=0.1, max_iter=1000)
    som.train(data)

    # Visualize the trained SOM
    som.visualize()
