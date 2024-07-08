import torch
import torchvision.transforms as transforms
from PIL import Image
from models import lightning_emotion_vgg

class EmotionClassifier:
    def __init__(self):
        self.model = lightning_emotion_vgg.EmotionVGG.load_from_checkpoint('models/emotion_classification_model.ckpt', config={}, map_location=torch.device('cpu'))
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def classify_emotions(self, frame, faces):
        emotions = []
        for (x, y, w, h) in faces:
            face_img = Image.fromarray(frame[y:y+h, x:x+w])
            face_img = self.transform(face_img).unsqueeze(0)
            with torch.no_grad():
                output = self.model(face_img)
                emotion = self.emotion_labels[torch.argmax(output)]
                emotions.append(emotion)
        return emotions