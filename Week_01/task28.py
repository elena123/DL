import random
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    random.seed(123)
    all_walks = []

    for i in range(500):
        step = 0
        steps_history = [step]

        random_float = random.random()

        for j in range(100):
            dice = random.randint(1, 6)

            if dice in [1, 2]:
                step -= 1
            elif dice in [3, 4, 5]:
                step += 1
            else:
                extra_roll = random.randint(1, 6)
                step += extra_roll

            if random.random() <= 0.005:
                step = 0

            steps_history.append(step)
        all_walks.append(steps_history[-1])

    print(all_walks)

    all_walks_np = np.array(all_walks)

    plt.figure(figsize=(10, 6))
    plt.hist(all_walks_np)
    plt.xlabel("Final Step Count")
    plt.ylabel("Frequency")
    plt.title("Random walks (500 Simulations)")
    plt.show()

success_count = np.sum(all_walks_np >= 60)
probability = success_count / 500

print(f"Odds of reaching 60 steps: {probability:.2%}")
