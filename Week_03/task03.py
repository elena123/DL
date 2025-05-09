import torch
from torch import nn


def main():
    model = nn.Sequential(
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    temperature_observation = torch.tensor([[3, 4, 6, 2, 3, 6, 8, 9]], dtype=torch.float32)

    confidence_output = model(temperature_observation)

    print(confidence_output.item())


# As we ae using a sigmoid, the output will be always between 0 and 1
# So B is false

if __name__ == '__main__':
    main()
