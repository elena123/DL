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

    nand_dataset = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    print("\nTraining NAND model:")
    nand_model, nand_losses = train(nand_dataset)

    print("\nNAND Gate predictions:")
    for x1, x2, true_output in nand_dataset:
        prediction = nand_model.predict(x1, x2)
        print(
            f"Input: ({x1}, {x2}) -> Prediction: {prediction:.4f}, True Output: {true_output}"
        )
    plot_loss(nand_losses)
