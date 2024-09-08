import cv2

class Camera:
    def __init__(self):
        self.vid = cv2.VideoCapture(0)
        
    def get_frame(self):
        ret, frame = self.vid.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.vid.release()
        cv2.destroyAllWindows()

    def show_frame(self, frame):
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
