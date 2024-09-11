import torch

from camera import Camera
from face_detector import FaceDetector
from emo_classifier import EmoClassifier

PATH_TO_CASCADES = 'res/cascades/'
MODEL_PATH = 'res/models/model.pth'

EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    cam = Camera()
    face_detector = FaceDetector(margin=10, cascade=PATH_TO_CASCADES+'pretrained.xml')

    emo_classifier = EmoClassifier(EMOTIONS)
    emo_classifier.to(device)
    emo_classifier.load_state_dict(torch.load(MODEL_PATH))

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        faces = face_detector.detect_face(frame)
        # TODO: ideally don't check for emotion EVERY frame, might flicker and be too computationally expensive
        # i think detection each frame is still okay
        # face_detector.show_bounding_box(frame, faces)

        # emo_classifier.show_emotion_text(frame, faces, device)

        frame = emo_classifier.get_filtered_frame(frame, faces, device)

        if not cam.show_frame(frame):
            break
    
    cam.release()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("Not using CUDA")

    main()
