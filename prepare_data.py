import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class dataset:
    def __init__(self, split, path, transforms = None):
        self.split = split
        self.path = path
        self.transform = transforms

        self.images = []
        self.labels = []

        for img_path in os.listdir(self.path):
            self.images.append(os.path.join(self.path, img_path))
            self.labels.append(int(img_path[-5]))

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")
        label = self.labels[idx]
        if self.transform != None:
            img = self.transform(img)
        tensor = transform.ToTensor()
        img = tensor(img)
        normalize = transform.Normalize(mean = [0.335], std = [0.0132])
        img = normalize(img)
        return img, label

    def __len__(self):
        return len(self.images)

img_transform = transform.Compose([
    transform.Resize(52),
    transform.CenterCrop((52, 52)),
    transform.RandomHorizontalFlip(p = 0.4),
    transform.RandomVerticalFlip(p = 0.4)
])


def prep(batch_size, shuffle = True):
    train_loader = DataLoader(dataset('train', r'data\train', img_transform), batch_size= batch_size, shuffle= shuffle)
    val_loader = DataLoader(dataset('val', r'data\val', None), batch_size= 1)
    test_loader = DataLoader(dataset('val', r'data\test', None), batch_size= 1)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader = DataLoader(dataset('train', r'data\train', img_transform), batch_size= 1)
    rand = np.random.randint(15000)
    print(rand)

    for idx, data in enumerate(train_loader):
        img = data[0].squeeze().numpy()
        label = data[1].item()

        if idx == rand:
            print(label)
            plt.figure()
            plt.imshow(img, cmap = 'gray')  
            plt.show()
            break