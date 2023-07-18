import pytorch_lightning as pl
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
import torch
import os


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, train_valid_split: float = 0.8):
        """
        Pytorch lightning conform MNIST dataset.
        :param batch_size: batch size of the respective dataloaders
        :param train_valid_split: proportion between train and validation data
        """
        super().__init__()
        path = os.path.join(os.path.dirname(__file__), '../..', 'data')
        if not os.path.exists(path):
            os.makedirs(path)
        self.batch_size = batch_size
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        self.test_set = datasets.MNIST(root=path, train=False, download=False, transform=transforms)
        self.train_set = datasets.MNIST(root=path, train=True, download=False, transform=transforms)
        len_train = int(train_valid_split * len(self.train_set))
        len_valid = len(self.train_set) - len_train
        self.train_set, self.valid_set = torch.utils.data.random_split(self.train_set, [len_train, len_valid])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def validation_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, batch_size=len(self.valid_set), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)
