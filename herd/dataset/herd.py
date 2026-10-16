import os
import torch
import torch.nn as nn

import numpy as np

from PIL import Image
from torchvision.ops import box_convert

from torch.utils.data import Dataset

class HerdDataset(Dataset):
    def __init__(self, root, df, transform, min_window):
        self.transform = transform
        self.min_window = min_window
        self.root = root
        
        self.data = [group for _, group in df.groupby(["video_id", "segment_id"])]
        print("Total number of video clips {}".format(len(self.data)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Loop until we successfully load a full valid sequence
        while True:
            try:
                df = self.data[idx]
                
                # Sample a window
                window_size = min(self.min_window, len(df))
                start_frame = np.random.randint(0, window_size)
                frames = df["frame_num"].values
                frames = frames[start_frame:start_frame + window_size]
                df = df[df["frame_num"].isin(frames)]
                
                all_frames, all_bboxes, all_labels, all_confidence = [], [], [], []
                
                for frame_num, frame_group in df.groupby(["frame_num"]):
                    tmp = frame_group.drop_duplicates(subset=["x", "y", "width", "height"])
                    
                    path = tmp["path"].values[0]
                    full_path = os.path.join(self.root, path)
                    
                    with Image.open(full_path) as img:
                        img.verify() 
                    
                    frame = Image.open(full_path).convert("RGB")
                    
                    bboxes = tmp[["x", "y", "width", "height"]].values.astype(float)
                    
                    all_frames.append(frame)
                    all_bboxes.append(np.array(bboxes))
                    all_labels.append(frame_group["label"].values)
                    all_confidence.append(frame_group["confidence"].values)

                return all_frames, all_bboxes, all_confidence, all_labels
            
            except (IOError, SyntaxError, OSError) as e:
                print(f"Warning: Corrupted data at index {idx}, resampling. Error: {e}")
                idx = np.random.randint(0, len(self.data))
                
        
def pretrain_collate_fn(batch):
    """
    Custom collate function to handle variable numbers of crops and bounding boxes.
    
    Args:
        batch: A list of tuples (images, final_bbox, is_present) returned by __getitem__
    """
    batched_crops = []
    batched_bboxes = []
    batched_confidence = []
    batched_labels = []

    for crops, bboxes, confidence, labels in batch:
        batched_crops.append(crops)
        batched_bboxes.append(bboxes)
        batched_confidence.append(confidence)
        batched_labels.append(labels)

    return batched_crops, batched_bboxes, batched_confidence, batched_labels