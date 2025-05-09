import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

torch.manual_seed(42)


class WaterDataset(Dataset):
    def __init__(self, water_file):
        super(WaterDataset).__init__()
        self.water_data = pd.read_csv(water_file, dtype=np.float32)
        self.data_array = np.array(self.water_data)

    def __len__(self):
        return len(self.water_data)

    def __getitem__(self, idx):
        array = self.data_array[idx]
        features = array[:-1]
        label = array[-1]
        return features, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.sigmoid(self.fc3(x))
        return x


def train_model(dataloader_train, optimizer, net, num_epochs, create_plot=False):
    criterion = nn.BCELoss()
    loss_values = []

    for epoch in tqdm(range(num_epochs), desc="Training progress", ncols=100, colour="green"):
        epoch_loss = 0
        net.train()
        for features, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader_train)
        loss_values.append(avg_epoch_loss)

    if create_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, num_epochs + 1), loss_values, label="Loss per Epoch", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.legend()
        plt.grid(True)
        plt.show()

    return sum(loss_values) / len(loss_values)


def compare_optimizers(dataloader_train, net):
    optimizers = [
        optim.SGD(net.parameters(), lr=0.0005),
        optim.RMSprop(net.parameters(), lr=0.0005),
        optim.Adam(net.parameters(), lr=0.0005),
        optim.AdamW(net.parameters(), lr=0.0005)
    ]

    for optimizer in optimizers:
        print(f"Using the {optimizer.__class__.__name__} optimizer: ")
        avg_loss = train_model(dataloader_train, optimizer, net, num_epochs=10)
        print(f"Average loss: {avg_loss:.4f}")


def main():
    water_file = '../DATA/water_train.csv'
    dataset_train = WaterDataset(water_file)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)

    net = Net()

    compare_optimizers(dataloader_train, net)

    optimizer = optim.AdamW(net.parameters(), lr=0.0005)
    avg_loss = train_model(dataloader_train, optimizer, net, num_epochs=1000, create_plot=True)

    water_test_file = '../DATA/water_test.csv'
    dataset_test = WaterDataset(water_test_file)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False)

    net.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in dataloader_test:
            outputs = net(features)
            preds = (outputs.squeeze() > 0.5).int()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).cpu()
    all_labels = torch.cat(all_labels).cpu()
    f1_score = BinaryF1Score()
    f1 = f1_score(all_labels, all_preds)
    print(f"F1 score on the test set: {f1}")

    # Answer: Correct statement is B
    # B. Batch normalization normalizes a layer's inputs to a standard normal distribution and passes these normalized values further.


if __name__ == "__main__":
    main()
