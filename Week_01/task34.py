import random


class LinearModel:

    def __init__(self):
        self.weight1, self.weight2, self.bias = self.initialize_weights(0, 10)

    @staticmethod
    def initialize_weights(x: float, y: float):
        return random.uniform(x, y), random.uniform(x, y), random.uniform(x, y)

    def predict(self, x1, x2):
        return self.weight1 * x1 + self.weight2 * x2 + self.bias

    def calculate_loss(self, dataset):
        total_loss = 0
        for x1, x2, true_output in dataset:
            predicted_output = self.predict(x1, x2)
            total_loss += (predicted_output - true_output)**2
        return total_loss / len(dataset)


def train(dataset, epochs=100000):
    model = LinearModel()

    for epoch in range(epochs):
        loss = model.calculate_loss(dataset)
        print(
            f"Epoch {epoch}: Loss = {loss:.4f}, w1 = {model.weight1:.4f}, w2 = {model.weight2:.4f}, bias = {model.bias:.4f}"
        )

    return model


if __name__ == '__main__':

    and_dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    or_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]

    print("Training AND model:")
    and_model = train(and_dataset)

    print("\nTraining OR model:")
    or_model = train(or_dataset)

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

    # I don't think the model is confident in predictions
