import cv2

from utils import get_circle_dim

class FaceDetector:
    def __init__(self, margin=10, cascade='res/cascades/pretrained.xml'):
        self.margin = margin

        self.cascade = cv2.CascadeClassifier(cascade)

    def detect_face(self, frame):
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(grayed, 1.3, 5)
        return faces
    
    def show_bounding_box(self, frame, faces):
        for (x, y, w, h) in faces:
            center, radius = get_circle_dim(x, y, w, h)
            cv2.circle(frame, center, radius + self.margin, (255, 0, 0), 2)
            # cv2.rectangle(frame, (x-self.margin, y-self.margin), (x+w+self.margin, y+h+self.margin), (255, 0, 0), 2)
