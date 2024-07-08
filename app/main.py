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
        emotions = emotion_classifier.classify_emotions(frame, faces)

        camera.display_frame(frame, faces, emotions)

if __name__ == "__main__":
    main()