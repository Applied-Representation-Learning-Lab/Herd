import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image

import open_clip

from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from lightly.models.utils import deactivate_requires_grad, get_weight_decay_parameters
from lightly.utils.debug import std_of_l2_normalized
from lightly.transforms.simclr_transform import SimCLRViewTransform
from lightly.loss.ntx_ent_loss import NTXentLoss

from transformers import CLIPTextModel, CLIPTokenizer

class WildlifeDataset(Dataset):
    def __init__(self, df, transform):
        df = df.copy()
        self.transform = transform

        self.data = []
        for _, df_img in df.groupby("file_name", sort=False):
            self.data.append(df_img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        df_sample = self.data[idx].sample(10, replace=True) # Sample up to 10 detections per image, with replacement if less than 10
        bboxes = df_sample[["x", "y", "width", "height"]].values.astype(float)
        bboxes = box_convert(torch.tensor(bboxes), in_fmt="xywh", out_fmt="xyxy").numpy()
        paths = df_sample["file_name"].values
        image = Image.open(paths[0]).convert("RGB")
        width, height = image.size
        
        label = df_sample.iloc[0]["labels"]
         
        crops, final_bboxes = [], []
        for i in range(len(bboxes)):
            left = bboxes[i][0] * width
            top = bboxes[i][1] * height
            right = bboxes[i][2] * width
            bottom = bboxes[i][3] * height
            bbox = [left, top, right, bottom]
            crops.append(image.crop(bbox))
            final_bboxes.append(torch.tensor(bboxes[i], dtype=torch.float32))

        crops = torch.stack([self.transform(crop) for crop in crops])
        final_bboxes = torch.stack(final_bboxes)
        return crops, final_bboxes, label

class Herd(LightningModule):
    def __init__(self, text_labels, text_to_label):
        super().__init__()
        self.vision_model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        self.vision_model = self.vision_model.visual
        deactivate_requires_grad(self.vision_model)
        
        input_dim = 768
        text_dim = 512
        
        self.position_embedding = nn.Linear(4, 768) # For bounding box coordinates
        
        self.mha = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(768)
        self.pool_v = nn.Linear(input_dim, 256)
        self.pool_u = nn.Linear(input_dim, 256)
        self.pool_w = nn.Linear(256, 1)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
        # 1. Get Text Embedding (e.g., from CLIP)
        text_model_name = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(text_model_name)
        text_encoder = CLIPTextModel.from_pretrained(text_model_name)
        deactivate_requires_grad(text_encoder)
        with torch.no_grad():
            inputs = tokenizer(text_labels, padding=True, return_tensors="pt")
            outputs = text_encoder(**inputs)
            text_embeddings = outputs.pooler_output # Shape: [num_labels, 512]
            
        self.text_embeddings = {}
        for idx, label in enumerate(text_labels):
            self.text_embeddings[text_to_label[label]] = text_embeddings[idx]
        
        self.criterion = NTXentLoss(temperature=0.5)
        
    def forward(self, images, bboxes):
        # Extract instance features
        with torch.no_grad():
            instance_features = self.vision_model(images) # Shape: [num_animals, 768]
        
        # Add positional embeddings
        pos_embeds = self.position_embedding(bboxes) # Shape: [num_animals, 768]
        x = instance_features + pos_embeds
        
        # MIL Attention Layer
        x_seq = x.unsqueeze(0) # [1, N, 768]
        attn_out, _ = self.mha(x_seq, x_seq, x_seq) # Self-attention across instances [1, N, 768]
        x = self.layer_norm(x + attn_out.squeeze(0)) # Residual connection [N, 768]
        
        # Gated MIL Pooling
        # This decides which refined animal features represent the species best
        v = torch.tanh(self.pool_v(x))
        u = torch.sigmoid(self.pool_u(x))
        weights = torch.softmax(self.pool_w(v * u), dim=0) # [N, 1]
        
        bag_latent = (x * weights).sum(dim=0) # [768]
        
        # Project to match CLIP Text dimension
        return self.projection(bag_latent.unsqueeze(0)).squeeze(0) # [512]
    
    def attention_net(self, instance_latents):
        # Replace your attention_net call in forward:
        # Reshape to [1, N, 768] for MultiheadAttention
        contextual_features, _ = self.self_attn(
            instance_latents.unsqueeze(0), 
            instance_latents.unsqueeze(0), 
            instance_latents.unsqueeze(0)
        )
        instance_latents = self.layer_norm(instance_latents + contextual_features.squeeze(0))
        return self.bag_projection(instance_latents)
    
    def training_step(self, batch, batch_index):
        images, bboxes, labels = batch[0], batch[1], batch[2]
        batch_size = images.shape[0]
        
        bag_latent = []
        for img, bb in zip(images, bboxes):
            output = self.forward(img, bb)
            bag_latent.append(output)
        bag_latent = torch.stack(bag_latent)
        
        text_embeddings = torch.stack([self.text_embeddings[label.item()] for label in labels]).to(bag_latent.device)
        
        representation_std = std_of_l2_normalized(bag_latent)
        
        loss = self.criterion(bag_latent, text_embeddings)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
    
    def get_params(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.position_embedding, self.mha, self.pool_v, self.pool_u, self.pool_w, self.projection]
        )
        return params, params_no_weight_decay
    
    def configure_optimizers(self):
        params, params_no_weight_decay = self.get_params()
        optimizer = torch.optim.AdamW(
            [
                {"params": params, "weight_decay": 1e-4},
                {"params": params_no_weight_decay, "weight_decay": 0.0},
            ],
            lr=1e-4,
        )
        return optimizer
            
df = pd.read_csv("data/serengeti/serengeti_herd.csv")
df.dropna(inplace=True)
df = df[df["confidence"] > 0.5]
df["labels"], text_labels = pd.factorize(df["scientific_name"])
text_to_label = {text: label for text, label in zip(text_labels, range(len(text_labels)))}
df["file_name"] = "data/serengeti/images/"+df["file_name"]

# Downsample Wildebeest
df_wildebeest = df[df["scientific_name"] == "connochaetes taurinus"]
paths = df_wildebeest["file_name"].unique()
paths = np.random.choice(paths, size=40000, replace=False)
df_wildebeest = df_wildebeest[df_wildebeest["file_name"].isin(paths)]

df = pd.concat([df[df["scientific_name"] != "connochaetes taurinus"], df_wildebeest], ignore_index=True)

df.reset_index(drop=True, inplace=True)

TRAIN_TRANSFORM = SimCLRViewTransform()
train_dataset = WildlifeDataset(df, TRAIN_TRANSFORM)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

text_labels = df["scientific_name"].unique().tolist()
model = Herd(text_labels, text_to_label)

checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename="herd")
callbacks = [checkpoint_callback]

trainer = pl.Trainer(
    max_epochs=100, 
    accelerator="gpu",
    devices=[0],
    strategy="auto",
    precision="16-mixed",
    sync_batchnorm=True,
    callbacks=callbacks
)

trainer.fit(model, train_dataloaders=train_dataloader)