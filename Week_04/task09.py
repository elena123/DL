import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


# https://discuss.pytorch.org/t/simple-efficient-way-to-create-dataset/65178
# https://discuss.pytorch.org/t/how-to-initialize-one-hot-encoding-in-a-class-in-pytorch/151540/2

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

def main():

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    dataset_train = OmniglotDataset(image_path='../DATA/omniglot_train', transform=train_transforms)
    print("Number of items in dataset:", len(dataset_train))
    dataloader_test = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=2,
    )

    count = len(dataset_train)
    print(f"Number of instances: {count}")
    char_image, char_label, alphabet_one_hot = dataset_train[count-1]
    print(f"Last item: ({char_image}, {char_label}, {alphabet_one_hot})")
    print(f"Shape of the last image: {char_image.shape}")


if __name__ == "__main__":
    main()
