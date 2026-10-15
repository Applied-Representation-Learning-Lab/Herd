import torch
import torch.nn as nn
import numpy as np
import open_clip

from torchvision.ops import box_convert
from pytorch_lightning import LightningModule

from lightly.models.utils import deactivate_requires_grad, get_weight_decay_parameters
from lightly.utils.debug import std_of_l2_normalized
from lightly.loss.ntx_ent_loss import NTXentLoss

from transformers import CLIPTextModel, CLIPTokenizer

from motrackers import CentroidKF_Tracker

class Herd(LightningModule):
    def __init__(self, labels, transform):
        super().__init__()
        
        self.transform = transform
        self.labels = labels
        input_dim = 768
        text_dim = 512
        
        # Species Level Embedding
        self.species_encoder, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        self.species_encoder = self.species_encoder.visual
        deactivate_requires_grad(self.species_encoder)
        
        # Position Level Embedding
        self.position_embedding = nn.Linear(4, input_dim)
        
        # Bounding Box Level Embedding
        self.bbox_encoder = None
        
        # Text Label Level Embedding
        text_model_name = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(text_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_model_name)
        deactivate_requires_grad(self.text_encoder)
        with torch.no_grad():
            inputs = tokenizer(labels, padding=True, return_tensors="pt")
            outputs = self.text_encoder(**inputs)
            text_embeddings = outputs.pooler_output # Shape: [num_labels, 512]
        self.text_embeddings = {}
        for idx, label in enumerate(labels):
            self.text_embeddings[label] = text_embeddings[idx]
                    
        
        # Fusion Layer
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.pool_v = nn.Linear(input_dim, 256)
        self.pool_u = nn.Linear(input_dim, 256)
        self.pool_w = nn.Linear(256, 1)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, text_dim),
            nn.LayerNorm(text_dim)
        )
        
        self.criterion = NTXentLoss(temperature=0.5)
        
    def forward(self, features):
        # MIL Attention Layer
        x_seq = features.unsqueeze(0) # [1, N, 768]
        attn_out, _ = self.mha(x_seq, x_seq, x_seq) # Self-attention across instances [1, N, 768]
        x = self.layer_norm(features + attn_out.squeeze(0)) # Residual connection [N, 768]
        
        # Gated MIL Pooling
        # This decides which refined animal features represent the species best
        v = torch.tanh(self.pool_v(x))
        u = torch.sigmoid(self.pool_u(x))
        weights = torch.softmax(self.pool_w(v * u), dim=0) # [N, 1]
        
        bag_latent = (x * weights).sum(dim=0) # [768]
        
        # Project to match CLIP Text dimension
        return self.projection(bag_latent.unsqueeze(0)).squeeze(0) # [512]
    
    def training_step(self, batch, batch_index):
        frames, bboxes, confidence, labels = batch[0], batch[1], batch[2], batch[3]
        # frames.shape # [Batch, Frames, Channels, Height, Width]
        # bboxes.shape # [Batch, Frames, Crops, 4]
        # labels # [Batch, Frames, Labels]
        
        batch_size = len(frames)
        
        visual_features, text_features = [], []
        for i in range(len(frames)):
            tracer = self.tracker(frames[i], bboxes[i], confidence[i], labels[i])
            visual_feats, text_feats = self.feature_fusion(tracer)
            
            bag_latent = self.forward(visual_feats)
            
            visual_features.append(bag_latent.repeat(len(visual_feats), 1))
            text_features.append(text_feats)
            
        loss = torch.tensor(0.0, device=bag_latent.device)
        for visual_feats, text_feats in zip(visual_features, text_features):
            loss += self.criterion(visual_feats, text_feats)
        loss = loss / len(visual_features) 
        
        #representation_std = std_of_l2_normalized(bag_latent)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        #self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
    
    def crop_image(self, frame, bboxe):
        width, height = frame.size
        bbox = box_convert(torch.tensor([bboxe]), in_fmt="xywh", out_fmt="xyxy").numpy()[0]
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height
        return self.transform(frame.crop(bbox)).to(self.device)
    
    def tracker(self, frames, bboxes, confidence, labels):
        tracker = CentroidKF_Tracker()
        
        tracer = {}
        for frame, frame_bboxes, frame_confidence, frame_labels in zip(frames, bboxes, confidence, labels):
            tracks = tracker.update(
                bboxes=frame_bboxes, 
                detection_scores=frame_confidence,
                class_ids=[self.labels.index(label) for label in frame_labels]
            )
            
            id_to_class_map = {t: tracker.tracks[t].class_id for t in tracker.tracks}

            for t in tracks:
                # t format: [frame_idx_1based, track_id, xmin, ymin, width, height, confidence]
                track_id = int(t[1])
                if track_id not in tracer:
                    tracer[track_id] = {
                        "crops": [],
                        "bboxes": [],
                        "labels": []
                    }
                
                xmin, ymin, width, height = t[2:6]
                if width <= 0 or height <= 0:
                    continue
                class_id = id_to_class_map[t[1]]
                crop = self.crop_image(frame, [xmin, ymin, width, height])
                tracer[track_id]["crops"].append(crop)
                bbox = box_convert(torch.tensor([xmin, ymin, width, height], dtype=torch.float32), in_fmt="xywh", out_fmt="cxcywh")
                tracer[track_id]["bboxes"].append(bbox)
                tracer[track_id]["labels"].append(self.labels[class_id])
                tracer[track_id]["labels"] = np.unique(tracer[track_id]["labels"]).tolist()
                
        return tracer
    
    def feature_fusion(self, tracer):
        features, text_features = [], []
        for track_id, track_data in tracer.items():
            crops = torch.stack(track_data["crops"])
            bboxes = torch.stack(track_data["bboxes"]).to(self.device)
            # Example fusion: average the features of all crops for this track
            species_features = self.species_encoder(crops).mean(dim=0)
            # position_features = self.position_embedding(bboxes).mean(dim=0)
            # feats = torch.concat([species_features, position_features])
            features.append(species_features)
            
            text_embeddings = torch.stack([self.text_embeddings[label] for label in track_data["labels"]]).to(self.device)
            text_embeddings = text_embeddings.mean(dim=0)
            text_features.append(text_embeddings)
        features = torch.stack(features)
        text_features = torch.stack(text_features)
        
        return features, text_features
    
    
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

                
        