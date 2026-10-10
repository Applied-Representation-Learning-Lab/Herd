import os, json, threading
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from PIL import Image
from PIL.ExifTags import TAGS
from io import BytesIO

from exif import Image as ExifImage

def get_images(json_path, save_folder):

    with open(f"metadata/{json_path}", 'r') as file:
        data = json.load(file)
        
    print(data.keys())
    print(data["info"])
    print("")
    print("Images: ", len(data["images"]))
    print(data["images"][0])
    print("")
    print("Annotations: ", len(data["annotations"]))
    print(data["annotations"][0])
    print("Categories: ", len(data["categories"]))
    print(data["categories"][0])

    df_images = pd.DataFrame(
        [
            {
                "id": img.get("id"),
                "path": img.get("file_name"),
                "location": img.get("location"),
                "datetime": img.get("datetime"),
                "frame_num": img.get("frame_num"),
                "video_id": img.get("seq_id"),
            }
            for img in data["images"]
        ]
    )

    df_annotations = pd.DataFrame(
        [
            {
                "id": img.get("image_id"),
                "label": img.get("category_id"),
            }
            for img in data["annotations"]
        ]
    )

    df_category = pd.DataFrame(
        [
            {
                "label": img.get("name"),
                "id": img.get("id"),
            }
            for img in data["categories"]
        ]
    )

    df_annotations["label"] = df_annotations["label"].map(df_category.set_index("id")["label"])
    df = pd.merge(df_images, df_annotations, on="id", how="inner")
    df.drop(columns=["id"], inplace=True)

    # df.to_csv(f"data/ammonitor/{save_folder}", index=False)
    
    return df

def get_detections(json_path):

    with open(f"metadata/{json_path}", 'r') as file:
            data = json.load(file)
            
    print(data.keys())
    print(data["info"])
    print("")
    print("Images: ", len(data["images"]))
    print(data["images"][0])
    print("")
    print("Categories: ", len(data["detection_categories"]))
    print(data["detection_categories"])

    df_crops = pd.DataFrame(
        [
            {
                "path": img.get("file"),
                "category": detections.get("category"),
                "confidence": detections.get("conf"),
                "x": detections.get("bbox")[0],
                "y": detections.get("bbox")[1],
                "width": detections.get("bbox")[2],
                "height": detections.get("bbox")[3],
            }
            for img in data["images"]
            for detections in (img.get("detections") or [])
        ]
    )

    df_crops["category"] = df_crops["category"].map(data["detection_categories"])

    return df_crops

df_images = get_images(json_path="ammonitor-camera-traps.json", save_folder="metadata.csv")
df_detections = get_detections(json_path="ammonitor-camera-traps-2026-06-12-v1000.0.0-redwood_detections.filtered.json")

def mapper(row):
    tmp1 = df_detections[df_detections["path"].isin([row["path"]]).copy(deep=True)].reset_index(drop=True)

    tmp2 = pd.DataFrame(np.tile(row.to_numpy(), (len(tmp1), 1)), columns=row.index)
    tmp1.drop(columns=["path"], inplace=True)
    tmp3 = pd.concat([tmp2, tmp1], axis=1)
    
    return tmp3

df = pd.merge(
    df_images, 
    df_detections, 
    on="path", 
    how="left", 
    suffixes=("", "_detection")
)

df = df[df["label"]!="unlabeled"]
df.to_csv(f"data/ammonitor/metadata.csv", index=False)
print(df)
    