import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model.main import SplitterNet
from data.dataset import JVS_Dataset

device = 'cuda:0'
model = SplitterNet()

trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=[TensorBoardLogger("logs/")],
        max_epochs=100000,)


root = os.path.join("../",  'jvs_r9y9_ver1')

speakers = ['jvs{0:03}'.format(i) for i in range(1, 101)]
dataset = JVS_Dataset(root, data_type='mel', speakers=speakers)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=512,
    shuffle=True,
    num_workers=24
)

trainer.fit(model, dataloader)
