import time
from pprint import pprint

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# https://discuss.pytorch.org/t/runtimeerror-mat1-and-mat2-shapes-cannot-be-multiplied-64x13056-and-153600x2048/101315

class CloudCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CloudCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16384, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def get_class_distribution(dataset, scores_list):
    idx2class = {v: k for k, v in dataset.class_to_idx.items()}
    return {idx2class[i]: score.item() for i, score in enumerate(scores_list)}


def main():
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    start_time = time.time()
    dataset_train = ImageFolder(
        '../DATA/clouds/clouds_test',
        transform=train_transforms,
    )

    # dataloader_train = DataLoader(
    #     dataset_train,
    #     shuffle=True,
    #     batch_size=16,
    # )

    train_dataset, val_dataset = train_test_split(
        dataset_train, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    image, label = next(iter(train_loader))
    image = image[0].permute(1, 2, 0)
    print(image.shape)

    model = CloudCNN(in_channels=3, num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    loss_values = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100, colour="green"):
            images, labels = images.to(torch.float32), labels
            scores = model(images)
            loss = criterion(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        model.eval()
        val_loss = 0
        val_preds, val_true = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, dim=1)
                val_preds.append(preds)
                val_true.append(labels)

        total_time = time.time() - start_time

        all_preds = torch.cat(val_preds).cpu()
        all_labels = torch.cat(val_true).cpu()

        avg_batch_loss = epoch_loss / num_batches
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_epoch_loss)
        print(f"Average training loss per batch: {avg_batch_loss}")

    precision_global = Precision(task="multiclass", num_classes=7, average='macro')
    recall_global = Recall(task="multiclass", num_classes=7, average='macro')
    f1_global = F1Score(task="multiclass", num_classes=7, average='macro')
    f1_per_class = F1Score(task="multiclass", num_classes=7, average=None)

    precision_score = precision_global(all_preds, all_labels)
    recall_score = recall_global(all_preds, all_labels)
    f1_score = f1_global(all_preds, all_labels)
    f1_per_class_score = f1_per_class(all_preds, all_labels)

    per_class_items = get_class_distribution(dataset_train, f1_per_class_score)
    print("Summary statistics:")
    summary_stats = {
        "Average training loss per epoch": sum(loss_values) / len(loss_values),
        "Precision": precision_score.item(),
        "Recall": recall_score.item(),
        "F1": f1_score.item(),
        "Total time taken to train the model in seconds": total_time,
    }
    pprint(summary_stats)
    print(f"Per class F1 score: {per_class_items}")


if __name__ == "__main__":
    main()
