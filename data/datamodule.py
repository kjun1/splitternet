import os
import pytorch_lightning as pl
from .dataset import JVSDataset, McJVSDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


root = os.path.join("../",  'jvs_r9y9_ver1')
with open(root+ '/female_f0range.txt') as f:
    lines = f.readlines()
    FEMALE_SPEAKERS =  [i[0:6] for i in lines[1:]]
FEMALE_SPEAKERS= [FEMALE_SPEAKERS[0]]

class JVSDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "jvs_r9y9_ver1", batch_size: int = 512, data_type="mel"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 24
        self.data_type = data_type
        
    def setup(self, stage):
        if self.data_type=="mc":
            dataset = McJVSDataset("melceps")
        else:
            dataset = JVSDataset(self.data_dir, data_type=self.data_type, speakers=FEMALE_SPEAKERS)
        self.train, valid_dataset = train_test_split(dataset, train_size=0.8, shuffle=True)
        self.valid, self.test = train_test_split(valid_dataset, train_size=0.9, shuffle=True)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            )

    

