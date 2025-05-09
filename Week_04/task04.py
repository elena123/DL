import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryF1Score

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
        self.fc1 = nn.Linear(9, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.4)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='sigmoid')

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


def train_model_with_validation(net, train_loader, val_loader, num_epochs=1000):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        net.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_outputs = net(val_features)
                val_loss += criterion(val_outputs.squeeze(), val_labels).item()
                val_preds.append(val_outputs.squeeze() > 0.5)
                val_true.append(val_labels)

        val_preds = torch.cat(val_preds).cpu()
        val_true = torch.cat(val_true).cpu()

        avg_val_loss = val_loss / len(val_loader)
        f1_score = BinaryF1Score()
        f1_val = f1_score(val_true, val_preds)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | F1 (val): {f1_val:.4f}")
        scheduler.step(avg_val_loss)

def evaluate_test_set(net, dataloader_test):
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


def main():
    water_file = '../DATA/water_train.csv'
    dataset_train = WaterDataset(water_file)

    # Split into training and validation sets
    train_dataset, val_dataset = train_test_split(
        dataset_train, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    net = Net()

    train_model_with_validation(net, train_loader, val_loader)

    water_test_file = '../DATA/water_test.csv'
    dataset_test = WaterDataset(water_test_file)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
    evaluate_test_set(net, dataloader_test)

if __name__ == "__main__":
    main()


