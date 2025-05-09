import os

import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, optim, device
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score
from torchvision import transforms
from tqdm import tqdm


# https://medium.com/data-science/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a

class OmniglotDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        label_map = {}
        class_idx = 0

        for alphabet in os.listdir(image_path):
            alphabet_path = os.path.join(image_path, alphabet)
            if os.path.isdir(alphabet_path):
                for character in os.listdir(alphabet_path):
                    char_path = os.path.join(alphabet_path, character)
                    if os.path.isdir(char_path):
                        if char_path not in label_map:
                            label_map[char_path] = class_idx
                            class_idx += 1

                        for img_file in os.listdir(char_path):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data.append(os.path.join(char_path, img_file))
                                self.labels.append(label_map[char_path])

        self.num_classes = class_idx
        if len(self.data) == 0:
            raise ValueError(f"No image files found in {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        label_one_hot = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return image, label_one_hot, label


class Net(nn.Module):
    def __init__(self, num_classes=964):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, padding=1, kernel_size=10),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=7),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.vector = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 16 * 16),
        )
        self.classfier = nn.Linear(25600, num_classes)

    def forward(self, image, vector):
        x1 = self.cnn(image)
        x2 = self.vector(vector)
        x = torch.cat((x1, x2), dim=1)
        return self.classfier(x)


def get_class_distribution(dataset, scores_list):
    idx2class = {v: k for k, v in dataset.class_to_idx.items()}
    return {idx2class[i]: score.item() for i, score in enumerate(scores_list)}


def main():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    dataset_train = OmniglotDataset(image_path='../DATA/omniglot_train', transform=train_transforms)
    dataset_test = OmniglotDataset(image_path='../DATA/omniglot_test', transform=train_transforms)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    net = Net(num_classes=dataset_train.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 3
    train_losses, val_losses = [], []
    train_preds, train_true = [], []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        net.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for image, vector, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100, colour="green"):
            image = image.to(device)
            vector = vector.to(device)
            label = label.to(device)

            output = net.forward(image, vector)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(output, dim=1)
            train_preds.append(preds)
            train_true.append(label)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        f1_global = F1Score(task="multiclass", num_classes=964, average='micro')
        f1_score_train = f1_global(torch.cat(train_preds).cpu(), torch.cat(train_true).cpu())
        val_running_loss = 0.0
        net.eval()

        val_preds, val_true = [], []
        with torch.no_grad():
            for image, vector, label in val_loader:
                image = image.to(device)
                vector = vector.to(device)
                label = label.to(device)
                outputs = net.forward(image, vector)

                loss = criterion(outputs, label)
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                val_preds.append(preds)
                val_true.append(label)

        all_preds = torch.cat(val_preds).cpu()
        all_labels = torch.cat(val_true).cpu()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        f1_global = F1Score(task="multiclass", num_classes=964, average='micro')
        f1_score_val = f1_global(all_preds, all_labels)

        train_f1_scores.append(f1_score_train.item())
        val_f1_scores.append(f1_score_val.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}]:")
        print(f" Average training loss: {avg_train_loss:.6f}")
        print(f" Average validation loss: {avg_val_loss:.6f}")
        print(f" Training metric score: {f1_score_train.item()}")
        print(f" Validation metric score: {f1_score_val.item()}")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].plot(range(1, num_epochs + 1), train_losses, label="Training loss")
    axes[0].plot(range(1, num_epochs + 1), val_losses, label="Validation loss")
    axes[0].set_title("Model loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(range(1, num_epochs + 1), train_f1_scores, label="Training metric")
    axes[1].plot(range(1, num_epochs + 1), val_f1_scores, label="Validation metric")
    axes[1].set_title("Model performance")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Metric value")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses


if __name__ == "__main__":
    main()
