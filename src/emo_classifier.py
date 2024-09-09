import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import csv
import os
import argparse

class Config:
    model_path = "res/models/model.pth"
    batch_size = 20
    num_epochs = 20
    learning_rate = 1e-3
    weight_decay = 0
    num_targets = 1

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
    def __init__(self, classes=7):
        super(EmoClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = 32 * 6 * 6

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Helper functions for training

def generate_dataset(images, output_csv):
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

def train_model(model, train_dataloader, criterion, optimizer, num_epochs=Config.num_epochs, validation_dataloader=None):
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        trained_batches = 0
        total_correct = 0
        total_labels = 0

        for batch_idx, (image, label) in enumerate(tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')):
            optimizer.zero_grad()

            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)

            outputs = model(image)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            trained_batches += 1

            _, predicted = torch.max(outputs, 1)
            total_labels += label.size(0)
            total_correct += (predicted == label).sum().item()

        avg_loss = total_loss / trained_batches
        accuracy = total_correct * 100 / total_labels
        print(f"Training Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}, Accuracy: {accuracy:0.4f}%")

        # Evaluate model on validation set
        model.eval()
        if validation_dataloader:
            valid_loss = 0.0
            valid_batches = 0
            valid_correct = 0
            valid_labels = 0

            for batch_idx, (image, label) in enumerate(tqdm(validation_dataloader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}')):
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                with torch.no_grad():
                    outputs = model(image)

                    loss = criterion(outputs, label)
            
                    valid_loss += loss.item()
                    valid_batches += 1

                    _, predicted = torch.max(outputs, 1)
                    valid_labels += label.size(0)
                    valid_correct += (predicted == label).sum().item()

            avg_valid_loss = valid_loss / valid_batches
            valid_acc = valid_correct * 100 / valid_labels
            print(f"Validation Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_valid_loss}, Accuracy: {valid_acc:0.4f}%")

def evaluate_model(model, test_dataloader):
    model.eval()
    
    total_correct = 0
    total_labels = 0

    for batch_idx, (image, label) in enumerate(tqdm(test_dataloader, desc='Testing')):
        image = image.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(image)
        
            _, predicted = torch.max(outputs, 1)
            total_labels += label.size(0)
            total_correct += (predicted == label).sum().item()

    accuracy = total_correct * 100 / total_labels
    print(f"Testing Evaluation - Accuracy: {accuracy:0.4f}%")

if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    # Modify as needed
    train_image_folder = 'res/data/emotions/train/'
    test_image_folder = 'res/data/emotions/test/'

    train_csv = 'res/csvs/emotion_train.csv'
    test_csv = 'res/csvs/emotion_test.csv'

    # Generate training dataset
    generate_dataset(train_image_folder, train_csv)

    # Generate testing dataset
    generate_dataset(test_image_folder, test_csv)

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])

    train_data = pd.read_csv('res/csvs/emotion_train.csv')
    test_data = pd.read_csv('res/csvs/emotion_test.csv')

    train_dataset = EmoDataset(train_data['file_path'], train_data['emotion'], transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    test_dataset = EmoDataset(test_data['file_path'], test_data['emotion'], transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    model = EmoClassifier()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    if not args.test:
        train_model(model, train_dataloader, criterion, optimizer, Config.num_epochs, test_dataloader)
        torch.save(model.state_dict(), Config.model_path)
    
    model.load_state_dict(torch.load(Config.model_path))
    evaluate_model(model, test_dataloader)
