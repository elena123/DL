import numpy as np
import torch
from torch.utils.data import TensorDataset


def main():
    data_array = np.random.rand(12, 9)

    data_array_tensor = torch.tensor(data_array, dtype=torch.float64)

    # All columns except the last
    features = data_array_tensor[:, :-1]

    # Last column as target
    targets = data_array_tensor[:, -1]

    dataset = TensorDataset(features, targets)

    last_sample, last_label = dataset[-1]
    print("Last sample:", last_sample)
    print("Last label:", last_label.unsqueeze(0))


if __name__ == '__main__':
    main()
