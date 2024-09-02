import cv2
from utils import get_circle_dim

class Camera:
    def __init__(self, margin=10, cascade='res/cascades/pretrained.xml'):
        self.vid = cv2.VideoCapture(0)

        self.margin = margin

        self.cascade = cv2.CascadeClassifier(cascade)
        
    def get_frame(self):
        ret, frame = self.vid.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.vid.release()
        cv2.destroyAllWindows()

    def show_frame(self, frame):
        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(grayed, 1.3, 5)

        for (x, y, w, h) in faces:
            center, radius = get_circle_dim(x, y, w, h)
            cv2.circle(frame, center, radius + self.margin, (255, 0, 0), 2)
            # cv2.rectangle(frame, (x-self.margin, y-self.margin), (x+w+self.margin, y+h+self.margin), (255, 0, 0), 2)
        
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

if __name__ == '__main__':
    cam = Camera()
    while True:
        frame = cam.get_frame()
        if frame is None:
            break
        if not cam.show_frame(frame):
            break
    cam.release()
