import random
import numpy as np
from matplotlib import pyplot as plt


class LinearModel:

    def __init__(self, x: float, y: float):
        random.seed(42)
        self.weight = random.uniform(x, y)

    def predict(self, x):
        return self.weight * x  # Model: y = w * x


def create_dataset(n: int):
    return [(i, 2 * i) for i in range(n)]


def calculate_loss(model, dataset):
    total_loss = 0
    for x, true_y in dataset:
        predicted_y = model.predict(x)
        total_loss += (predicted_y - true_y)**2
    return total_loss / len(dataset)


if __name__ == '__main__':
    dataset = create_dataset(6)
    model = LinearModel(0, 10)

    initial_loss = calculate_loss(model, dataset)
    print(f"Initial weight: {model.weight:.4f}")
    print(f"Initial loss: {initial_loss:.4f}")

    # Experiment with different weight variations
    weight_variations = [
        model.weight + 0.001 * 2, model.weight + 0.001, model.weight - 0.001,
        model.weight - 0.001 * 2
    ]

    for i, w in enumerate(weight_variations, start=1):
        model.weight = w
        loss = calculate_loss(model, dataset)
        print(f"Loss for weight w_{i}: {loss:.4f}")

# The loss depends on the weight. The greater the weight is, the greater the loss is.
