import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


def main():
    test_data = pd.read_csv('../DATA/water_test.csv').dropna()

    train_data = pd.read_csv('../DATA/water_train.csv').dropna()

    # Separate features and target
    X_train = torch.tensor(train_data.drop('Potability', axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(train_data['Potability'].values, dtype=torch.float32).unsqueeze(1)

    X_test = torch.tensor(test_data.drop('Potability', axis=1).values, dtype=torch.float32)
    y_test = torch.tensor(test_data['Potability'].values, dtype=torch.float32).unsqueeze(1)

    # Split train data into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(X_train))
    val_size = len(X_train) - train_size

    train_dataset, val_dataset = random_split(TensorDataset(X_train, y_train), [train_size, val_size])
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Display target distribution
    for name, dataset in [('Train', train_dataset), ('Validation', val_dataset), ('Test', test_dataset)]:
        targets = [y.item() for _, y in dataset]
        print(f"{name} target distribution: {np.bincount(targets)}")

    # Define the model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 30
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        train_losses.append(epoch_loss / len(train_loader))
        train_metrics.append(correct / total)

        # Validation loop
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_losses.append(val_loss / len(val_loader))
        val_metrics.append(correct / total)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss per epoch
    axes[0].plot(range(1, epochs + 1), train_losses, label="Train loss", color="blue")
    axes[0].plot(range(1, epochs + 1), val_losses, label="Validation loss", color="orange")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Metric per epoch
    axes[1].plot(range(1, epochs + 1), train_metrics, label="Train", color="blue")
    axes[1].plot(range(1, epochs + 1), val_metrics, label="Validation", color="orange")
    axes[1].set_title("Metric per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Test set evaluation
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print(f"Test loss: {test_loss / len(test_loader):.4f}")
    print(f"Test accuracy: {correct / total:.4f}")


if __name__ == '__main__':
    main()
