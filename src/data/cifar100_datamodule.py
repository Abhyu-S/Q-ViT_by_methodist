"""
CIFAR-100 DataModule for PyTorch Lightning
Implements proper Train/Val/Test splitting to prevent overfitting.
"""
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
import pytorch_lightning as pl
from typing import Optional

class CIFAR100DataModule(pl.LightningDataModule):
    """
    DataModule for CIFAR-100 dataset
    Splits: 45k Train, 5k Validation, 10k Test
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1, # 10% of training data for validation
        image_size: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
        # Training Transform (Augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # Val/Test Transform (No Augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def prepare_data(self):
        """Download dataset if not already present"""
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up train, validation, and test datasets.
        Crucial: Splits 'train' data into Train/Val, keeping 'test' data separate.
        """
        if stage == "fit" or stage is None:
            # 1. Load the full training data to get indices
            full_train_data = CIFAR100(self.data_dir, train=True)
            total_len = len(full_train_data)
            val_len = int(total_len * self.val_split)
            train_len = total_len - val_len
            
            # 2. Generate randomized indices
            generator = torch.Generator().manual_seed(42)
            train_subset, val_subset = random_split(
                full_train_data, [train_len, val_len], generator=generator
            )
            
            # 3. Create Subsets with CORRECT transforms
            # We instantiate two datasets, one with train_transform and one with val_transform
            # Then we use the indices from random_split to pick the correct images
            train_data_aug = CIFAR100(self.data_dir, train=True, transform=self.train_transform)
            val_data_clean = CIFAR100(self.data_dir, train=True, transform=self.val_transform)
            
            self.train_dataset = Subset(train_data_aug, train_subset.indices)
            self.val_dataset = Subset(val_data_clean, val_subset.indices)

        if stage == "test" or stage is None:
            # 4. Load the official Test set (completely unseen during training)
            self.test_dataset = CIFAR100(
                self.data_dir,
                train=False,
                transform=self.val_transform,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )