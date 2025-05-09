import random
import numpy as np

if __name__ == '__main__':

    def create_dataset(n: int):
        dataset = []
        for i in range(n):
            dataset.append((i, 2 * i))
        return dataset

    def initialize_weights(x: float, y: float) -> float:
        w = random.uniform(x, y)
        b = random.uniform(x, y)
        return w, b


print(create_dataset(4))

print(initialize_weights(0, 100))
print(initialize_weights(0, 10))

# A model consists of one level, 2 parameters and can be represented as linear function: y=w⋅x+b
