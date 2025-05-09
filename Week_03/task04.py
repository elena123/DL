import torch
from torch.nn import CrossEntropyLoss


def main():
    scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]], dtype=torch.float64)
    one_hot_target = torch.tensor([2])

    criterion = CrossEntropyLoss()
    loss = criterion(scores, one_hot_target)
    print(loss)


if __name__ == '__main__':
    main()
