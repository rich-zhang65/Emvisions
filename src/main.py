from camera import Camera
from face_detector import FaceDetector

PATH_TO_CASCADES = 'res/cascades/'

def main():
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
