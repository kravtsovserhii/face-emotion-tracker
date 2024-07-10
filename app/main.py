from camera import Camera
from face_detector import FaceDetector
from emotion_classifier import EmotionClassifier

def main():
    camera = Camera()
    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier()

    while True:
        frame = camera.get_frame()
        faces = face_detector.detect_faces(frame)
        # increase the y value by 150 to get the face region
        faces = [(x-100, y-100, w+150, h+150) for (x, y, w, h) in faces]
        emotions = emotion_classifier.classify_emotions(frame, faces)

        camera.display_frame(frame, faces, emotions)

if __name__ == "__main__":
    main()