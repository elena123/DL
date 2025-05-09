import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


# Resources:
# https://medium.com/data-science/introduction-to-pytorch-e0235570a080

class WaterDataset(Dataset):
    def __init__(self, water_file):
        super(WaterDataset).__init__()
        self.water_data = pd.read_csv(water_file)
        self.data_array = np.array(self.water_data)

    def __len__(self):
        return len(self.water_data)

    def __getitem__(self, idx):
        array = self.data_array[idx]
        features = array[:-1]
        label = array[-1]
        return features, label


def load_data():
    water_file = '../DATA/water_train.csv'
    dataset = WaterDataset(water_file)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    first_batch = next(iter(dataloader))
    features, labels = first_batch

    print(features)
    print(labels)


def main():
    water_file = '../DATA/water_train.csv'
    dataset = WaterDataset(water_file)

    print(f"Number of instances: {len(dataset)}")

    fifth_item = dataset[4]
    print(f"Fifth item: {fifth_item}")
    load_data()


if __name__ == '__main__':
    main()
