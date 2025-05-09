import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# https://www.datacamp.com/tutorial/pytorch-cnn-tutorial

class CloudCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CloudCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # 64 * (64 // 4) * (64 // 4)
        self.classifier = nn.Linear(16384, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def main():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0, 45)),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    dataset_train = ImageFolder(
        '../DATA/clouds/clouds_train',
        transform=train_transforms,
    )

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=16,
    )
    image, label = next(iter(dataloader_train))
    image = image[0].permute(1, 2, 0)
    print(image.shape)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    model = CloudCNN(in_channels=3, num_classes=10)
    # print(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    loss_values = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for images, labels in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}", ncols=100, colour="green"):
            images, labels = images.to(torch.float32), labels
            scores = model(images)
            loss = criterion(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / len(dataloader_train)
        loss_values.append(avg_loss)

        avg_batch_loss = epoch_loss / num_batches
        print(f"Average training loss per batch: {avg_batch_loss}")

    print(f"Average training loss per epoch: {sum(loss_values) / len(loss_values)}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), loss_values, label="Loss per Epoch", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
