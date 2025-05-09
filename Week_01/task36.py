import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinearModel:

    def __init__(self):
        self.weight1, self.weight2, self.bias = self.initialize_weights(0, 1)

    @staticmethod
    def initialize_weights(x: float, y: float):
        return random.uniform(x, y), random.uniform(x, y), random.uniform(x, y)

    def predict(self, x1, x2):
        return sigmoid(self.weight1 * x1 + self.weight2 * x2 + self.bias)

    def calculate_loss(self, dataset):
        total_loss = 0
        for x1, x2, true_output in dataset:
            predicted_output = self.predict(x1, x2)
            total_loss += (predicted_output - true_output)**2
        return total_loss / len(dataset)


def train(dataset, epochs=100000):
    model = LinearModel()
    losses = []
    for epoch in range(epochs):
        loss = model.calculate_loss(dataset)
        losses.append(loss)
        print(
            f"Epoch {epoch}: Loss = {loss:.4f}, w1 = {model.weight1:.4f}, w2 = {model.weight2:.4f}, bias = {model.bias:.4f}"
        )

    return model, losses


def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses, label="Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':

    and_dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    or_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]

    print("Training AND model:")
    and_model, and_losses = train(and_dataset)

    print("\nTraining OR model:")
    or_model, or_losses = train(or_dataset)

    print("\nAND Gate predictions:")
    for x1, x2, true_output in and_dataset:
        prediction = and_model.predict(x1, x2)
        print(
            f"Input: ({x1}, {x2}) -> Prediction: {prediction:.4f}, True Output: {true_output}"
        )

    print("\nOR Gate predictions:")
    for x1, x2, true_output in or_dataset:
        prediction = or_model.predict(x1, x2)
        print(
            f"Input: ({x1}, {x2}) -> Prediction: {prediction:.4f}, True Output: {true_output}"
        )
    plot_loss(and_losses)
    plot_loss(or_losses)

# Comparing predictions:
# Without sigmoid:
# AND Gate predictions:
# Input: (0, 0) -> Prediction: 5.3154, True Output: 0
# Input: (0, 1) -> Prediction: 6.9909, True Output: 0
# Input: (1, 0) -> Prediction: 6.9940, True Output: 0
# Input: (1, 1) -> Prediction: 8.6695, True Output: 1
# OR Gate predictions:
# Input: (0, 0) -> Prediction: 0.9557, True Output: 0
# Input: (0, 1) -> Prediction: 9.7126, True Output: 1
# Input: (1, 0) -> Prediction: 6.7131, True Output: 1
# Input: (1, 1) -> Prediction: 15.4699, True Output: 1

# With sigmoid:
# AND Gate predictions:
# Input: (0, 0) -> Prediction: 0.9982, True Output: 0
# Input: (0, 1) -> Prediction: 0.9999, True Output: 0
# Input: (1, 0) -> Prediction: 1.0000, True Output: 0
# Input: (1, 1) -> Prediction: 1.0000, True Output: 1

# OR Gate predictions:
# Input: (0, 0) -> Prediction: 0.9988, True Output: 0
# Input: (0, 1) -> Prediction: 1.0000, True Output: 1
# Input: (1, 0) -> Prediction: 1.0000, True Output: 1
# Input: (1, 1) -> Prediction: 1.0000, True Output: 1

#
# More closer predictions and the losse is static
#
