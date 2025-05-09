import random


class LinearModel:

    def __init__(self, min_val=0, max_val=10):
        self.w1, self.w2 = random.uniform(min_val, max_val), random.uniform(
            min_val, max_val)

    def predict(self, x1, x2):
        return self.w1 * x1 + self.w2 * x2

    def calculate_loss(self, dataset):
        total_loss = 0
        for x1, x2, true_output in dataset:
            predicted_output = self.predict(x1, x2)
            total_loss += (predicted_output - true_output)**2
        return total_loss / len(dataset)

    def train(self, dataset, epochs=100000, learning_rate=0.01):
        for epoch in range(epochs):
            grad_w1, grad_w2 = 0, 0

            # Compute gradient (finite differences method)
            for x1, x2, true_output in dataset:
                prediction = self.predict(x1, x2)
                error = prediction - true_output
                grad_w1 += error * x1
                grad_w2 += error * x2

            # Average gradient over dataset
            grad_w1 /= len(dataset)
            grad_w2 /= len(dataset)

            # Update weights
            self.w1 -= learning_rate * grad_w1
            self.w2 -= learning_rate * grad_w2

            loss = self.calculate_loss(dataset)
            print(
                f"Epoch {epoch}: Loss = {loss:.4f}, w1 = {self.w1:.4f}, w2 = {self.w2:.4f}"
            )


if __name__ == '__main__':

    and_dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    or_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]

    print("Training AND model:")
    and_model = LinearModel()
    and_model.train(and_dataset)

    print("\nTraining OR model:")
    or_model = LinearModel()
    or_model.train(or_dataset)

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

# Both models have 2 parameters
# The values that model predicts are close to the real values, but still 0 and 1 are not reached.
# I don't think it's confident in values
