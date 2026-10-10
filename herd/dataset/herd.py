import torch
import torch.nn as nn

from PIL import Image
from torchvision.ops import box_convert

from torch.utils.data import Dataset

class HerdDataset(Dataset):
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