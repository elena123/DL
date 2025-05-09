import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class XorModel:

    def __init__(self):
        self.w1 = random.uniform(0, 1)
        self.w2 = random.uniform(0, 1)
        self.b1 = 1

        self.w3 = random.uniform(0, 1)
        self.w4 = random.uniform(0, 1)
        self.b2 = 1

        # Output layer weights and bias
        self.w5 = random.uniform(0, 1)
        self.w6 = random.uniform(0, 1)
        self.b3 = 1

    def forward(self, x1, x2):
        h1 = sigmoid(self.w1 * x1 + self.w2 * x2 + self.b1)
        h2 = sigmoid(self.w3 * x1 + self.w4 * x2 + self.b2)
        output = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return output

    def calculate_loss(self, dataset):
        total_loss = 0
        for x1, x2, true_output in dataset:
            predicted_output = self.predict(x1, x2)
            total_loss += (predicted_output - true_output)**2
        return total_loss / len(dataset)


def train(dataset, epochs=100000, learning_rate=0.1):
    model = XorModel()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, true_output in dataset:
            h1 = sigmoid(model.w1 * x1 + model.w2 * x2 + model.b1)
            h2 = sigmoid(model.w3 * x1 + model.w4 * x2 + model.b2)
            output = sigmoid(model.w5 * h1 + model.w6 * h2 + model.b3)

            error = (output - true_output)**2
            total_loss += error

        losses.append(total_loss / len(dataset))

        if epoch % 10000 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(dataset):.4f}")

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

    xor_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    print("Training XOR model:")
    xor_model, xor_losses = train(xor_dataset)

    print("\nXOR Gate predictions:")
    for x1, x2, true_output in xor_dataset:
        prediction = xor_model.forward(x1, x2)
        print(
            f"Input: ({x1}, {x2}) -> Prediction: {prediction:.4f}, True Output: {true_output}"
        )

    plot_loss(xor_losses)
