import numpy as np
import random
from matplotlib import pyplot as plt

if __name__ == '__main__':

    random.seed(123)

    random_float = random.random()
    print(f"Random float: {random_float}")

roll_1 = random.randint(1, 6)
roll_2 = random.randint(1, 6)
print(f"Random integer 1: {roll_1}")
print(f"Random integer 2: {roll_2}")

# Start at step 50
step = 50
print(f"Before throw step: {step}")

dice = random.randint(1, 6)
print(f"After throw dice: {dice}")

if dice in [1, 2]:
    step -= 1
elif dice in [3, 4, 5]:
    step += 1
else:
    extra_roll = random.randint(1, 6)
    print(f"After throw step: {extra_roll}")
    step += extra_roll

print(f"After throw step: {step}")
