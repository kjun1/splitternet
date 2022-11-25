import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from model.crossmodal.main import Model, params
from data.datamodule import CrossJVSDataModule

device = 'cuda:0'
model = Model(params)# .load_from_checkpoint("crosslogs/lightning_logs/version_10/checkpoints/last.ckpt")


checkpoint_callback = ModelCheckpoint(
    save_last=True,
    save_top_k=10,
    every_n_epochs=10,
    monitor="val_loss")

trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=[TensorBoardLogger("crosslogs/")],
        max_epochs=10000,
        callbacks=[checkpoint_callback],
        )


data_module = CrossJVSDataModule(audio_data_dir="data/36_40_melceps", image_data_dir="data/images")

trainer.fit(model, data_module)
