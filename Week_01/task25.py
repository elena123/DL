import random
from matplotlib import pyplot as plt

if __name__ == '__main__':

    random.seed(123)
    all_walks = []

    for i in range(5):
        step = 0
        steps_history = [step]

        for j in range(100):
            dice = random.randint(1, 6)

            if dice in [1, 2]:
                step -= 1
            elif dice in [3, 4, 5]:
                step += 1
            else:
                extra_roll = random.randint(1, 6)
                step += extra_roll

            steps_history.append(step)
        all_walks.append(steps_history)

    print(all_walks)
