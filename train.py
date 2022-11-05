import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model.main import SplitterNet
from data.dataset import JVS_Dataset

device = 'cuda:0'
model = SplitterNet()

trainer = pl.Trainer(gpus=1)


root = os.path.join(os.environ['DATA'],  'jvs_r9y9_ver1')

speakers = ['jvs{0:03}'.format(i) for i in range(1, 101)]
dataset = JVS_Dataset(root, data_type='mel', speakers=speakers)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,
    num_workers=10
)

trainer.fit(model, dataloader)
