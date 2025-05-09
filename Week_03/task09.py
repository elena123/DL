import torch.nn as nn
import torch.nn.init as init

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.Linear(16, 12),
    nn.Linear(12, 8),
    nn.Linear(8, 2)
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, -0.1, 0.1)
        init.uniform_(m.bias, -0.1, 0.1)


def main():
    model.apply(init_weights)

    # Freeze the first two layers
    for param in list(model[0].parameters()) + list(model[1].parameters()):
        param.requires_grad = False

    # Output weights and biases for the first five neurons in all layers
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            print(f"Layer {i + 1} :\n", layer.weight[:5], "\n", layer.bias[:5])


if __name__ == '__main__':
    main()
