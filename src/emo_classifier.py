import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import csv
import os

class EmoDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)

        return image, self.labels[idx]

class EmoClassifier(nn.Module):
    def __init__(self):
        super(EmoClassifier, self).__init__()

        self.linear1 = nn.Linear(100, 48)
        self.linear2 = nn.Linear(48, 1)

    def forward(self, input):
        output = self.linear1(input)
        output = F.relu(output)
        output = self.linear2(output)
        return 0
    

def generate_dataset(images, output_csv, train=False):
    emotions = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_path', 'emotion'])

        for emo, label in emotions.items():
            folder = os.path.join(images, emo)

            if os.path.isdir(folder):
                for img in os.listdir(folder):
                    path = os.path.join(folder, img)
                    writer.writerow([path, label])

if __name__ == '__main__':
    # Modify as needed
    train_image_folder = 'res/data/emotions/train'
    test_image_folder = 'res/data/emotions/test'

    train_csv = 'res/csvs/emotion_train.csv'
    test_csv = 'res/csvs/emotion_test.csv'

    # Generate training dataset
    generate_dataset(train_image_folder, train_csv, train=True)

    # Generate testing dataset
    generate_dataset(test_image_folder, test_csv, train=False)
