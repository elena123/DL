import numpy as np


def generate_hyperparam():
    # Generate 10 random pairs (learning rate, momentum)
    hyperparam_tuple = []

    for i in range(1, 11):
        lr = np.random.uniform(0.0001, 0.01)
        mom = np.random.uniform(0.85, 0.99)
        hyperparam_tuple.append((lr, mom))

    print(hyperparam_tuple)


def main():
    generate_hyperparam()


if __name__ == '__main__':
    main()
