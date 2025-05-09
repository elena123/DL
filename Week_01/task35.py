import math
import random
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x_values = np.linspace(-10, 10, 400)
    y_values = sigmoid(x_values)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label="Sigmoid Function", color="blue")
    plt.axhline(y=0.5, color="red", linestyle="dashed", label="y = 0.5")
    plt.axvline(x=0, color="gray", linestyle="dotted", label="x = 0")

    plt.xlabel("x")
    plt.ylabel("Sigmoid(x)")
    plt.title("Sigmoid Function Plot")
    plt.legend()
    plt.grid()

    plt.show()
