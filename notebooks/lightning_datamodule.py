import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import cv2

class EmotionDataModule(pl.LightningDataModule):
    """
    LightningDataModule for emotion classification dataset.
    """

    def __init__(self, root_dir, labels_dataframe, batch_size=32, num_workers=4):
        """
        Initializes the EmotionDataModule.

        Args:
            train_dir (str): Directory path to the training dataset.
            test_dir (str): Directory path to the testing dataset.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            num_workers (int, optional): Number of workers for data loaders. Defaults to 4.
        """
        super(EmotionDataModule, self).__init__()
        self.root_dir = root_dir
        self.labels_dataframe = labels_dataframe
        # join the root directory with the image paths
        self.labels_dataframe.iloc[:, 0] = self.root_dir + '/' + self.labels_dataframe.iloc[:, 0]
        # create a dictionary mapping labels to indices
        self.label2idx = {label: idx for idx, label in enumerate(self.labels_dataframe.iloc[:, 1].unique())}
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        # stratified split of the dataset
        self.train_df, self.val_df = train_test_split(self.labels_dataframe, test_size=0.2, stratify=self.labels_dataframe.iloc[:, 1])
        self.val_df, self.test_df = train_test_split(self.val_df, test_size=0.2, stratify=self.val_df.iloc[:, 1])

        # Define the transformations for training and testing datasets
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage):
        """
        Set up the datasets for training and validation.

        Args:
            stage (str): Current stage (fit or test).
        """
        if stage == "fit" or stage is None:
            self.train_dataset = EmotionDataset(self.train_df, self.label2idx, self.train_transforms)
            self.val_dataset = EmotionDataset(self.val_df, self.label2idx, self.val_transforms)
        elif stage == "test" or stage == "validate":
            self.test_dataset = EmotionDataset(self.test_df, self.label2idx, self.val_transforms)
            
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
            shuffle=False
        )
    
    def test_dataloader(self):
        """
        Returns the data loader for testing dataset.

        Returns:
            torch.utils.data.DataLoader: Data loader for testing dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )



class EmotionDataset(Dataset):
    """
    Dataset class for emotion classification dataset.
    """
    def __init__(self, dataframe, label2idx, transforms=None):
        """
        Initializes the EmotionDataset.

        Args:
            dataframe (pandas.DataFrame): DataFrame containing image paths and labels.
            transforms (torchvision.transforms.Compose, optional): Transforms to apply to the images. Defaults to None.
        """
        self.dataframe = dataframe
        self.transforms = transforms
        self.label2idx = label2idx

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing the image and label.
        """
        img_path = self.dataframe.iloc[idx, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.dataframe.iloc[idx, 1]
        if self.transforms:
            img = self.transforms(img)
    
        return img, self.label2idx[label]
        