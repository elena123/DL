import os

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score
from torchvision import transforms
from tqdm import tqdm


class OmniglotDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.transform = transform
        self.data = []
        self.char_labels = []
        self.alphabets_labels = []
        char_label_map = {}
        alphabet_label_map = {}
        char_class_idx = 0
        alphabet_class_idx = 0

        for alphabet in os.listdir(image_path):
            alphabet_path = os.path.join(image_path, alphabet)
            if alphabet not in alphabet_label_map:
                alphabet_label_map[alphabet] = alphabet_class_idx
                alphabet_class_idx += 1
            if os.path.isdir(alphabet_path):
                if os.path.isdir(alphabet_path):
                    for character in os.listdir(alphabet_path):
                        char_path = os.path.join(alphabet_path, character)
                        if os.path.isdir(char_path):
                            if char_path not in char_label_map:
                                char_label_map[char_path] = char_class_idx
                                char_class_idx += 1

                            for img_file in os.listdir(char_path):
                                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    self.data.append(os.path.join(char_path, img_file))
                                    self.char_labels.append(char_label_map[char_path])
                                    self.alphabets_labels.append(alphabet_label_map[alphabet])

        self.num_classes = char_class_idx
        self.num_alphabets = alphabet_class_idx

        if len(self.data) == 0:
            raise ValueError(f"No image files found in {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        char_label = self.char_labels[idx]
        alphabet_label = self.alphabets_labels[idx]

        return image, char_label, alphabet_label


class Net(nn.Module):
    def __init__(self, num_classes=964, num_alphabets=30):
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
        self.classfier_char = nn.Linear(9216, num_classes)
        self.classfier_alphabet = nn.Linear(9216, num_alphabets)

    def forward(self, image):
        x = self.cnn(image)
        return self.classfier_char(x), self.classfier_alphabet(x)


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
    train_preds_char, train_true_char = [], []
    train_preds_alphabet, train_true_alphabet = [], []
    train_f1_scores_char, train_f1_scores_alphabet = [], []
    val_f1_scores_char, val_f1_scores_alphabet = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        net.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for image, char_label, alphabet_label in tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100, colour="green"):
            image = image.to(device)
            char_label = char_label.to(device)
            alphabet_label = alphabet_label.to(device)

            char_output, alphabet_output = net.forward(image)
            char_loss = criterion(char_output, char_label)
            alphabet_loss = criterion(alphabet_output, alphabet_label)

            loss = char_loss + alphabet_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds_char = torch.max(char_output, dim=1)
            _, preds_alphabet = torch.max(alphabet_output, dim=1)

            train_preds_char.append(preds_char)
            train_preds_alphabet.append(preds_alphabet)

            train_true_char.append(char_label)
            train_true_alphabet.append(alphabet_label)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        f1_global = F1Score(task="multiclass", num_classes=964, average='micro')
        f1_score_train_char = f1_global(torch.cat(train_preds_char).cpu(), torch.cat(train_true_char).cpu())
        f1_score_train_alphabet = f1_global(torch.cat(train_preds_alphabet).cpu(), torch.cat(train_true_alphabet).cpu())

        val_running_loss = 0.0
        net.eval()

        val_preds_char, val_true_char = [], []
        val_preds_alphabet, val_true_alphabet = [], []
        with torch.no_grad():
            for image, char_label, alphabet_label in val_loader:
                image = image.to(device)

                char_label = char_label.to(device)
                alphabet_label = alphabet_label.to(device)
                char_output, alphabet_output = net.forward(image)

                char_loss = criterion(char_output, char_label)
                alphabet_loss = criterion(alphabet_output, alphabet_label)
                loss = char_loss + alphabet_loss
                val_running_loss += loss.item()
                _, preds_char= torch.max(char_output, dim=1)
                _, preds_alphabet = torch.max(alphabet_output, dim=1)

                val_preds_char.append(preds_char)
                val_true_char.append(char_label)

                val_preds_alphabet.append(preds_alphabet)
                val_true_alphabet.append(alphabet_label)

        all_preds_char = torch.cat(val_preds_char).cpu()
        all_labels_char = torch.cat(val_true_char).cpu()

        all_preds_alphabet = torch.cat(val_preds_alphabet).cpu()
        all_labels_alphabet = torch.cat(val_true_alphabet).cpu()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        f1_global = F1Score(task="multiclass", num_classes=964, average='micro')
        f1_score_val_char = f1_global(all_preds_char, all_labels_char)
        f1_score_val_alphabet = f1_global(all_preds_alphabet, all_labels_alphabet)

        train_f1_scores_char.append(f1_score_train_char.item())
        train_f1_scores_alphabet.append(f1_score_train_alphabet.item())

        val_f1_scores_char.append(f1_score_val_char.item())
        val_f1_scores_alphabet.append(f1_score_val_alphabet.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}]:")
        print(f" Average training loss: {avg_train_loss:.6f}")
        print(f" Average validation loss: {avg_val_loss:.6f}")
        print(f" Training metric score characters: {f1_score_train_char.item()}")
        print(f" Validation metric score characters: {f1_score_val_char.item()}")
        print(f" Training metric score alphabets: {f1_score_train_alphabet.item()}")
        print(f" Validation metric score alphabets: {f1_score_val_alphabet.item()}")



    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    axes[0].plot(range(1, num_epochs + 1), train_losses, label="Training loss")
    axes[0].plot(range(1, num_epochs + 1), val_losses, label="Validation loss")
    axes[0].set_title("Model loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(range(1, num_epochs + 1), train_f1_scores_char, label="Training metric")
    axes[1].plot(range(1, num_epochs + 1), val_f1_scores_char, label="Validation metric")
    axes[1].set_title("Model performance on characters")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Metric value")
    axes[1].legend()

    axes[2].plot(range(1, num_epochs + 1), train_f1_scores_alphabet, label="Training metric")
    axes[2].plot(range(1, num_epochs + 1), val_f1_scores_alphabet, label="Validation metric")
    axes[2].set_title("Model performance on alphabets")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Metric value")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses


if __name__ == "__main__":
    main()
