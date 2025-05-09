import random

if __name__ == '__main__':

    random.seed(123)

    # Start at step 0
    step = 0
    steps_history = [step]

    # Corrected loop (no step reassignment)
    for i in range(100):
        dice = random.randint(1, 6)

        if dice in [1, 2]:
            step -= 1
        elif dice in [3, 4, 5]:
            step += 1
        else:
            extra_roll = random.randint(1, 6)
            step += extra_roll

        steps_history.append(step)

    print(steps_history)
