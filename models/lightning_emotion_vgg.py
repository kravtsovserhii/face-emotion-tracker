import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics

import torch.optim as optim


class VGGBlock(pl.LightningModule):
    """
    VGG Block module consisting of convolutional layers, batch normalization (optional), and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, num_conv_layers, batch_norm=False):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the VGG Block module.
        """
        return self.block(x)


class EmotionVGG(pl.LightningModule):
    """
    EmotionVGG model consisting of VGG Blocks and a classifier for image classification.
    """

    def __init__(self, config=None):
        super(EmotionVGG, self).__init__()
        self.loss_fn = config.get('loss_fn', nn.CrossEntropyLoss(weight=torch.tensor([0.00027716, 0.00030826, 0.00028802, 0.00032862, 0.00023063,
       0.00034953, 0.00033389, 0.00021664])))
        self.batch_norm = config.get('batch_norm', True)
        self.num_classes = config.get('num_classes', 7)
        self.metric = config.get('metric', torchmetrics.AUROC(num_classes=self.num_classes, task='multiclass'))
        self.in_channels = config.get('in_channels', 3)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.features = nn.Sequential(
            VGGBlock(self.in_channels, 64, 2, batch_norm=self.batch_norm),  
            VGGBlock(64, 128, 2, batch_norm=self.batch_norm),  
            VGGBlock(128, 256, 3, batch_norm=self.batch_norm), 
            VGGBlock(256, 512, 3, batch_norm=self.batch_norm), 
            VGGBlock(512, 512, 3, batch_norm=self.batch_norm),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, self.num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the EmotionVGG model.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

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
        Training step of the EmotionVGG model.
        """
        loss, y_hat, y = self._common_step(batch, batch_idx)
        metric_score = self.metric(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_metric', metric_score, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the EmotionVGG model.
        """
        loss, y_hat, y = self._common_step(batch, batch_idx)
        metric_score = self.metric(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_metric', metric_score, on_epoch=True)
        return loss

    def predict_step(self, batch):
        """
        Prediction step of the EmotionVGG model.
        """
        x = batch
        y_hat = self(x)
        predict = torch.argmax(y_hat, dim=1)
        return predict

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for the EmotionVGG model.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
