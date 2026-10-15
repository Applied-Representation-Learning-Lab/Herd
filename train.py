import os
import pandas as pd

import pytorch_lightning as pl
from lightly.transforms.simclr_transform import SimCLRViewTransform
from torch.utils.data import DataLoader

from herd.models.herd import Herd
from herd.dataset.herd import HerdDataset, pretrain_collate_fn

df = pd.read_csv("data/ammonitor/metadata.csv")

# get all the downloaded images first
paths = [os.path.join(root.split(os.sep)[-1], fname) for root, _, fnames in os.walk("data/ammonitor/images") for fname in fnames]
df = df[df["path"].isin(paths)]
df = df[df["label"]!="empty"]
labels = df["label"].unique().tolist()

TRAIN_TRANSFORM = SimCLRViewTransform()
train_dataset = HerdDataset(root="data/ammonitor/images", df=df, transform=TRAIN_TRANSFORM, min_window=10)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=pretrain_collate_fn)

model = Herd(labels, TRAIN_TRANSFORM)

trainer = pl.Trainer(
    max_epochs=100, 
    accelerator="gpu",
    devices=[0],
    strategy="auto",
    precision="16-mixed",
    sync_batchnorm=True,
)

trainer.fit(model, train_dataloaders=train_dataloader)

