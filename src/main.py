import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from camera import Camera
from face_detector import FaceDetector
from emo_classifier import EmoDataset, EmoClassifier

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH_TO_CASCADES = 'res/cascades/'

class Config:
    model_path = "res/models/model.pth"
    batch_size = 20
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 0
    num_targets = 1

def train_model(model, train_dataloader, criterion, optimizer, num_epochs=Config.num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        trained_batches = 0

        for batch_idx, (image, label) in enumerate(tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')):
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)

            outputs = model(image)

            loss = criterion(outputs, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            trained_batches += 1

        avg_loss = total_loss / trained_batches
        # Also calculate accuracy here
        print(f"Training Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

    torch.save(model.state_dict(), Config.model_path)

def main():
    TRAINING_IMAGES_PATH = "res/data/emotions/train/"
    TEST_IMAGES_PATH = "res/data/emotions/test/"

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    train_data = pd.read_csv('res/csvs/emotion_train.csv')
    test_data = pd.read_csv('res/csvs/emotion_test.csv')
    print(train_data.head(5))

    train_dataset = EmoDataset(train_data['file_path'], train_data['emotion'], transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    test_dataset = EmoDataset(train_data['file_path'], train_data['emotion'], transform=transform)
    test_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    model = EmoClassifier()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    criterion = torch.nn.MSELoss()

    train_model(model, train_dataloader, criterion, optimizer, Config.num_epochs)

    cam = Camera()
    face_detector = FaceDetector(margin=10, cascade=PATH_TO_CASCADES+'pretrained.xml')

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        faces = face_detector.detect_face(frame)
        face_detector.show_bounding_box(frame, faces)

        if not cam.show_frame(frame):
            break
    
    cam.release()

if __name__ == '__main__':
    main()
