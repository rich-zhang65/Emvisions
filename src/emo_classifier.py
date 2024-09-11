import cv2
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

from utils import get_circle_dim

class Config:
    model_path = "res/models/model.pth"
    emoji_folder = "res/emojis/ios/"
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
    def __init__(self, emotions):
        super(EmoClassifier, self).__init__()

        self.emotions = emotions

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = 32 * 6 * 6

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, len(self.emotions))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _classify(self, face, device):
        face = cv2.resize(face, (48, 48))
        face = face / 255
        face = np.reshape(face, (1, 1, 48, 48))

        face_tensor = torch.tensor(face, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = self.forward(face_tensor)
            prediction = torch.argmax(output, dim=1).item()

        return prediction
    
    def show_emotion_text(self, frame, faces, device):
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces:
            center, _ = get_circle_dim(x, y, w, h)

            face = grayed[y:y+h, x:x+w]
            label = self._classify(face, device)
            text = self.emotions[label]

            # Debug
            # cv2.imshow('Detected Face', face)

            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            cv2.putText(frame, text, (center[0] - text_width // 2, y - text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    def get_filtered_frame(self, frame, faces, device):
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces:
            center, radius = get_circle_dim(x, y, w, h)

            face = grayed[y:y+h, x:x+w]
            label = self._classify(face, device)
            text = self.emotions[label]
            image_path = os.path.join(Config.emoji_folder, text + '.png')

            filter = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            filter = cv2.resize(filter, (2*radius, 2*radius))

            overlay_rgb = filter[:, :, :3]
            mask = filter[:, :, 3] / 255
            
            min_x = center[0] - radius
            min_y = center[1] - radius
            max_x = min_x + 2 * radius
            max_y = min_y + 2 * radius

            roi = frame[min_y:max_y, min_x:max_x]
            mask = mask[:, :, np.newaxis]
            blended = (roi * (1 - mask) + overlay_rgb * mask)

            # Place the blended result back into the frame
            frame[min_y:max_y, min_x:max_x] = blended

        return frame
    
# Helper functions for training

def generate_dataset(images, output_csv):
    emotions = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_path', 'emotion'])

        for label, emo in emotions.items():
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
