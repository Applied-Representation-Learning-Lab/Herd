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
            frame_bboxes, frame_labels, frame_confidence = [], [], []
            path = tmp["path"].values[0]
            frame = Image.open(os.path.join(self.root, path)).convert("RGB")
            bboxes = tmp[["x", "y", "width", "height"]].values.astype(float)
            # dims = [img.size for img in frames]
            # bboxes = box_convert(torch.tensor(bboxes), in_fmt="xywh", out_fmt="xyxy").numpy()
            # for i in range(len(bboxes)):
            #     width, height = dims[i]
            #     bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3] = bboxes[i][0]*width, bboxes[i][1]*height, bboxes[i][2]*width, bboxes[i][3]*height
            #     bboxes[i] = bboxes[i]
            
            all_frames.append(frame)
            all_bboxes.append(np.array(bboxes))
            all_labels.append(frame_group["label"].values)
            all_confidence.append(frame_group["confidence"].values)

        return all_frames, all_bboxes, all_confidence, all_labels

    '''
    def __getitem__(self, idx):
        df = self.data[idx]
        
        # Sample a window
        window_size = min(self.min_window, len(df))
        start_frame = np.random.randint(0, window_size)
        frames = df["frame_num"].values
        frames = frames[start_frame:start_frame + window_size]
        df = df[df["frame_num"].isin(frames)]
        
        all_crops, all_bboxes, all_labels = [], [], []
        for frame_num, frame_group in df.groupby(["frame_num"]):
            tmp = frame_group.drop_duplicates(subset=["x", "y", "width", "height"])
            frame_crops, frame_bboxes, frame_labels = [], [], []
            images = [Image.open(os.path.join(self.root, path)).convert("RGB") for path in tmp["path"].values]
            bboxes = tmp[["x", "y", "width", "height"]].values.astype(float)
            dims = [img.size for img in images]
            bboxes = box_convert(torch.tensor(bboxes), in_fmt="xywh", out_fmt="xyxy").numpy()
            for i, bbox in enumerate(bboxes):
                width, height = dims[i]
                bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height
                frame_crops.append(self.transform(images[i].crop(bbox)))
                frame_bboxes.append(torch.tensor(bbox, dtype=torch.float32))
            frame_labels.append(frame_group["label"].values)
            
            all_crops.append(torch.stack(frame_crops))
            all_bboxes.append(torch.stack(frame_bboxes))
            all_labels.append(frame_labels)

        return all_crops, all_bboxes, all_labels
    '''
                
        
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