import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from model.main import SplitterNet
from data.datamodule import JVSDataModule



device = 'cuda:0'
model = SplitterNet()

checkpoint_callback = ModelCheckpoint(save_top_k=10, monitor="val_loss")

trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=[TensorBoardLogger("logs/")],
        max_epochs=100000,
        callbacks=[checkpoint_callback],
        )


root = "melceps" #os.path.join("../",  'jvs_r9y9_ver1')

data_module = JVSDataModule(root, data_type="mc")

trainer.fit(model, data_module)
