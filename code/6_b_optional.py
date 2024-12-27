import numpy as np

class ART1:
    def __init__(self, input_dim, max_categories=3, vigilance=0.5, beta=0.1):
        self.input_dim = input_dim
        self.max_categories = max_categories
        self.vigilance = vigilance
        self.beta = beta
        self.weights = np.random.rand(max_categories, input_dim)  # Initialize random weights
        self.categories = 0

    def _calculate_vigilance(self, input_vector, category_idx):
        # Calculate how well the input matches the category (using Manhattan distance)
        return np.sum(np.abs(input_vector - self.weights[category_idx])) / np.sum(input_vector)

    def train(self, input_data):
        for input_vector in input_data:
            matched = False
            for category_idx in range(self.categories):
                if self._calculate_vigilance(input_vector, category_idx) < self.vigilance:
                    # Update the weights of the matching category
                    self.weights[category_idx] += self.beta * (input_vector - self.weights[category_idx])
                    matched = True
                    break
            if not matched and self.categories < self.max_categories:
                # Create a new category if no match found
                self.weights[self.categories] = input_vector
                self.categories += 1

    def predict(self, input_vector):
        for category_idx in range(self.categories):
            if self._calculate_vigilance(input_vector, category_idx) < self.vigilance:
                return category_idx  # Return the index of the matched category
        return None  # No category matched

# Example usage
if __name__ == "__main__":
    data = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ])

    art = ART1(input_dim=3)
    art.train(data)

    test_input = np.array([1, 0, 0])
    category = art.predict(test_input)
    print(f"Test Input {test_input} matched category {category}")
