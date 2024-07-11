import torch
import torchvision.transforms as transforms
from PIL import Image
from models import lightning_effnet

class EmotionClassifier:
    def __init__(self):
        self.model = lightning_effnet.EmotionClassifier.load_from_checkpoint('models/emotion_classification_model.ckpt', config={}, map_location=torch.device('cpu'))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    def classify_emotions(self, frame, faces):
        emotions = []
        for (x, y, w, h) in faces:
            face_img = Image.fromarray(frame[y:y+h, x:x+w])
            face_img = self.transform(face_img).unsqueeze(0)
            with torch.no_grad():
                output= self.model(face_img)
                emotion = self.emotion_labels[torch.argmax(output)]
                emotions.append(emotion)
        return emotions