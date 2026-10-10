import os, json, threading
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from PIL import Image
from PIL.ExifTags import TAGS
from io import BytesIO

from exif import Image as ExifImage

def clean_labels(df):
    df["label"] = df["label"].str.replace(" sp.", "")
    return df

def get_images(json_path):

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
    
    df = clean_labels(df)
    
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

def mapper(row):
    tmp1 = df_detections[df_detections["path"].isin([row["path"]]).copy(deep=True)].reset_index(drop=True)

    tmp2 = pd.DataFrame(np.tile(row.to_numpy(), (len(tmp1), 1)), columns=row.index)
    tmp1.drop(columns=["path"], inplace=True)
    tmp3 = pd.concat([tmp2, tmp1], axis=1)
    
    return tmp3

def verion_zero_dataset(df):
    # TEMPORARY
    df = df[df["confidence"].notna() & (df["confidence"] >= 0.2)]

    # Extract video segments based on a reasonable time interval between frames
    MAX_TIME_INTERVAL = 3 # seconds
    # Convert datetime column to pandas datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    # Sort by video_id and datetime
    df = df.sort_values(["video_id", "datetime"]).reset_index(drop=True)
    # Calculate time difference between consecutive frames within the same video
    df["time_diff"] = df.groupby("video_id")["datetime"].diff().dt.total_seconds()
    # Identify the start of new segments based on the MAX_TIME_INTERVAL
    df["new_segment"] = (df["time_diff"].isnull()) | (df["time_diff"] > MAX_TIME_INTERVAL)
    # Assign segment IDs based on the new segments
    df["segment_id"] = df.groupby("video_id")["new_segment"].cumsum()

    # Keep video segments with atleast four frames
    df = df.groupby(["video_id", "segment_id"]).filter(lambda x: len(x) >= 4)
    return df

def helper_view_average_herd_size(df):
    frame_counts = (
        df.groupby(["video_id", "frame_num", "label"])
        .size()
        .reset_index(name="animal_count")
    )

    # 3. Calculate the average group/herd size per animal when present
    avg_herd_size = (
        frame_counts.groupby("label")["animal_count"]
        .mean()
        .reset_index(name="avg_herd_size")
    )

    # 4. Sort to see which animals tend to travel in larger groups
    print(avg_herd_size.sort_values(by="avg_herd_size", ascending=False))
    
    
DATASETS = {
    "ammonitor": {
        "images": "ammonitor-camera-traps.json",
        "detections": "ammonitor-camera-traps-2026-06-12-v1000.0.0-redwood_detections.filtered.json"
    },
    "WSULynx": {
        "images": "wsu-lynx.json",
        "detections": "wsu-lynx-2026-02-18-v1000.0.0-redwood_detections.filtered.json"
    },
    "SnapshotSafariExpansion": {
        "images": "snapshot_safari_2024_metadata.json",
        "detections": [
            "snapshot_safari_2024_lila_mdv5a-rde_0.100_0.850_15_0.200.json",
            "snapshot-safari-2024-expansion-SER-subset-v1000.0.0-redwood_detections.threshold.filtered.json"
            ]
    },
    "SeattleishCameraTraps": {
        "images": "seattleish_camera_traps.json",
        "detections": "seattleish-camera-traps-2025-03-23-v1000.0.0-redwood_detections.filtered.json"
    },
    "IdahoCameraTraps": {
        "images": "idaho-camera-traps.json",
        "detections": "idaho-camera-traps_mdv5a.0.0_results.filtered_rde_0.150_0.850_30_0.200.json"
    },
    "ChannelIslandsCameraTraps": {
        "images": "channel_islands_camera_traps.json",
    }
    
}

dataset_name = "SnapshotSafariExpansion"

df_images = get_images(json_path=DATASETS[dataset_name]["images"])
if isinstance(DATASETS[dataset_name]["detections"], list):
    df_detections = pd.concat([get_detections(json_path=path) for path in DATASETS[dataset_name]["detections"]], ignore_index=True)
else:
    df_detections = get_detections(json_path=DATASETS[dataset_name]["detections"])

df = pd.merge(
    df_images, 
    df_detections, 
    on="path", 
    how="left", 
    suffixes=("", "_detection")
)

# Make the videos have frames that are 1 second apart
df["datetime"] = pd.to_datetime(df["datetime"])
# 1. Ensure datetime format and sort chronologically
df = df.sort_values(["video_id", "datetime"]).reset_index(drop=True)
# 2. Get the representative datetime for each unique frame_num within each video_id
# (Equivalent to taking the first datetime of each group)
frame_times = df.groupby(["video_id", "frame_num"], sort=False)["datetime"].first().reset_index()
# 3. Calculate time difference from the previous kept frame within the same video_id
frame_times["time_diff"] = frame_times.groupby("video_id")["datetime"].diff().dt.total_seconds()
# 4. Filter to keep the first frame of every video, and any frame >= 1 second apart from the previous kept frame
# (Fill NaN with 1 so the very first frame of each video is always kept)
frame_times["keep"] = (frame_times["time_diff"].isnull()) | (frame_times["time_diff"] >= 1)
# 5. Filter the original dataframe to only include the kept frames
valid_frames = frame_times[frame_times["keep"]][["video_id", "frame_num"]]
df = df.merge(valid_frames, on=["video_id", "frame_num"], how="inner")

 # Drop unlablled frames
df = df[df["label"]!="unlabeled"]
df = df[df["label"]!="unknown"]

df = verion_zero_dataset(df)
print("Average Number of Animals per Herd")
helper_view_average_herd_size(df)

df.to_csv(f"data/{dataset_name}/metadata.csv", index=False)
print(df)
    