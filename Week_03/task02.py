import torch
from torch import nn


def main():
    model = nn.Sequential(
        nn.Linear(8, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    temperature_observation = torch.tensor([[2, 3, 6, 7, 9, 3, 2, 1]], dtype=torch.float32)
    logit_output = model(temperature_observation)

    print(logit_output)


if __name__ == '__main__':
    main()
