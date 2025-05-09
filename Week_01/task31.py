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
    total_loss = sum((model.predict(x) - true_y)**2 for x, true_y in dataset)
    return total_loss / len(dataset)


# Compute finite difference derivative
def finite_difference_derivative(model, dataset, h=0.001):
    original_weight = model.weight

    model.weight = original_weight + h
    loss_w_plus_h = calculate_loss(model, dataset)

    model.weight = original_weight
    loss_w = calculate_loss(model, dataset)

    return (loss_w_plus_h - loss_w) / h


if __name__ == '__main__':
    dataset = create_dataset(6)
    model = LinearModel(0, 10)

    learning_rate = 0.001
    iterations = 10
    loss_history = []

    for i in range(iterations):
        loss_before = calculate_loss(model, dataset)
        loss_history.append(loss_before)
        print(f"Iteration {i + 1}: Loss before update = {loss_before:.4f}")

        derivative = finite_difference_derivative(model, dataset)

        model.weight -= learning_rate * derivative

        loss_after = calculate_loss(model, dataset)
        print(f"Iteration {i + 1}: Updated weight = {model.weight:.4f}")
        print(f"Iteration {i + 1}: Loss after update = {loss_after:.4f}\n")

    # Plot loss reduction over iterations
    plt.plot(range(1, iterations + 1), loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Reduction Over Iterations")
    plt.grid()
    plt.show()
