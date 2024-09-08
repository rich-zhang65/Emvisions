import torch
import torch.nn as nn
import torch.nn.functional as F

import csv
import os

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

    train_csv = 'res/datasets/emotion_train.csv'
    test_csv = 'res/datasets/emotion_test.csv'

    # Generate training dataset
    generate_dataset(train_image_folder, train_csv, train=True)

    # Generate testing dataset
    generate_dataset(test_image_folder, test_csv, train=False)
