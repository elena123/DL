import torch.nn as nn


# Function to calculate the number of parameters
def count_parameters(model):
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
    return total


# Define a 3-layer linear network with < 120 parameters
model_3layer = nn.Sequential(
    nn.Linear(8, 5),
    nn.Linear(5, 3),
    nn.Linear(3, 2)
)

# Define a 4-layer linear network with > 120 parameters
model_4layer = nn.Sequential(
    nn.Linear(8, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 4),
    nn.Linear(4, 2)
)


def main():
    print("Number of parameters in network 1:", count_parameters(model_3layer))
    print("Number of parameters in network 2:", count_parameters(model_4layer))


if __name__ == '__main__':
    main()
