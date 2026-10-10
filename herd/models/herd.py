import torch
import torch.nn as nn

import open_clip

from pytorch_lightning import LightningModule

from lightly.models.utils import deactivate_requires_grad, get_weight_decay_parameters
from lightly.utils.debug import std_of_l2_normalized

from transformers import CLIPTextModel, CLIPTokenizer

class Herd(LightningModule):
    def __init__(self,):
        super().__init__()
        
        # Species Level Embedding
        self.species_encoder, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        self.species_encoder = self.species_encoder.visual
        deactivate_requires_grad(self.species_encoder)
        
        # Position Level Embedding
        self.position_embedding = nn.Linear(4, 768)
        
        # Bounding Box Level Embedding
        self.bbox_encoder = None
        
        # Text Label Level Embedding
        text_model_name = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(text_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_model_name)
        deactivate_requires_grad(self.text_encoder)
        
        # Fusion Layer
        self.mha = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(768)
        self.pool_v = nn.Linear(input_dim, 256)
        self.pool_u = nn.Linear(input_dim, 256)
        self.pool_w = nn.Linear(256, 1)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
    def forward(self, images, positions):
        # Extract instance features
        with torch.no_grad():
            species_features = self.species_encoder(images) # Shape: [num_animals, 768]
        
        # Add positional embeddings
        pos_embeds = self.position_embedding(positions) # Shape: [num_animals, 768]
        x = species_features + pos_embeds
        
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
    
    def training_step(self, batch, batch_index):
        images, positions, labels = batch[0], batch[1], batch[2]
        batch_size = images.shape[0]
        
        bag_latent = []
        for img, pos in zip(images, positions):
            output = self.forward(img, pos)
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

                
        