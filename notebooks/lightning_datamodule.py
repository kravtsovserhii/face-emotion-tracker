import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class EmotionDataModule(pl.LightningDataModule):
    """
    LightningDataModule for emotion classification dataset.
    """

    def __init__(self, train_dir, test_dir, batch_size=32, num_workers=4):
        """
        Initializes the EmotionDataModule.

        Args:
            train_dir (str): Directory path to the training dataset.
            test_dir (str): Directory path to the testing dataset.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            num_workers (int, optional): Number of workers for data loaders. Defaults to 4.
        """
        super(EmotionDataModule, self).__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define the transformations for training and testing datasets
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48, 48)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage):
        """
        Set up the datasets for training and validation.

        Args:
            stage (str): Current stage (fit or test).
        """
        if stage == "fit" or stage is None:
            # Create training and validation datasets
            self.train_dataset = ImageFolder(self.train_dir, transform=self.train_transform)
            self.val_dataset = ImageFolder(self.test_dir, transform=self.test_transform)

    def train_dataloader(self):
        """
        Returns the data loader for training dataset.

        Returns:
            torch.utils.data.DataLoader: Data loader for training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        """
        Returns the data loader for validation dataset.

        Returns:
            torch.utils.data.DataLoader: Data loader for validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
