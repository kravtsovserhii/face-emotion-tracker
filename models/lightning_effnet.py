import torch
from torch import nn
from torch import optim
import torchmetrics
from torchvision import models
import pytorch_lightning as pl


class EmotionClassifier(pl.LightningModule):
    """  
    Pretrained EfficientNet model for EmotionClassifier classification.
    """
    def __init__(self, config=None):
        super(EmotionClassifier, self).__init__()
        self.loss_fn = config.get('loss_fn',
                                  nn.CrossEntropyLoss(weight=torch.tensor([0.7630, 0.9761, 1.0144, 1.1574, 1.1759, 1.0857, 1.2310, 0.8122])))
        self.num_classes = config.get('num_classes', 8)
        self.metric = config.get('metric', torchmetrics.AUROC(num_classes=self.num_classes, task='multiclass'))
        self.learning_rate = config.get('learning_rate', 0.001)
        b1, b2 = config.get('b1', True), config.get('b2', False)
        # Define Feature part
        if b1:
            self.model = models.efficientnet_b1(weights='DEFAULT')
        elif b2:
            self.model = models.efficientnet_b2(weights='DEFAULT')
        else:
            self.model = models.efficientnet_b0(weights='DEFAULT')
        
        # Define Classification part
        if b1:
            self.classification = nn.Sequential(
                nn.Dropout(0.2, inplace=True),
                nn.Linear(1280, self.num_classes))
        elif b2:
            self.classification = nn.Sequential(
                nn.Dropout(0.2, inplace=True),
                nn.Linear(1408, self.num_classes)
            )
        else:
            self.classification = nn.Sequential(
                nn.Dropout(0.2, inplace=True),
                nn.Linear(1280, self.num_classes)
            )
        self.model.classifier = self.classification
        
        
    def forward(self, image):
        out = self.model(image)
        return out

    def _common_step(self, batch, batch_idx):
        """
        Common step for both training and validation steps.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        """
        Training step of 
        """
        loss, y_hat, y = self._common_step(batch, batch_idx)
        metric_score = self.metric(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_metric', metric_score, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the EmotionClassifier model.
        """
        loss, y_hat, y = self._common_step(batch, batch_idx)
        metric_score = self.metric(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_metric', metric_score, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step of the EmotionClassifier model.
        """
        loss, y_hat, y = self._common_step(batch, batch_idx)
        metric_score = self.metric(y_hat, y)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_metric', metric_score, on_epoch=True)
        return loss
    

    def predict_step(self, batch):
        """
        Prediction step of the EmotionClassifier model.
        """
        x = batch
        y_hat = self(x)
        predict = torch.argmax(y_hat, dim=1)
        prob = torch.nn.functional.softmax(y_hat, dim=1)
        return predict, prob

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for EmotionClassifier model.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}