import os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import torch

from herd.models.herd import Herd
from torchvision.ops import box_convert
import torchvision.transforms as T

from herd.utils.bounding_box import BoundingBox
from herd.utils.video import video


@torch.no_grad()
def inference_with_visualization(df, labels, transform, root):
    model = Herd(labels=labels, transform=transform)
    checkpoint = torch.load("checkpoints/herd-epoch=08.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    # sample a clip
    # first look for a video with herds in them
    # 1. Count the number of bounding boxes (animals) per frame for each video segment
    frame_counts = df.groupby(["video_id", "segment_id", "frame_num"]).size().reset_index(name="animal_count")
    # 2. Filter for frames that contain a herd (e.g., more than 3 animals)
    video_ids = frame_counts[frame_counts["animal_count"] > 2]["video_id"].values
    df = df[df["video_id"].isin(video_ids)]

    tmp = df.sample(1)
    video_id, segment_id = tmp.iloc[0]["video_id"], tmp.iloc[0]["segment_id"]
    df_clip = df[(df["video_id"] == video_id) & (df["segment_id"] == segment_id)].sort_values("frame_num")

    frames, bboxes, confidences, dummy_labels = [], [], [], []
    for frame_num, df_frame in df_clip.groupby("frame_num"):
        image = Image.open(os.path.join(root, df_frame["path"].values[0])).convert("RGB")
        frames.append(image)

        frame_bboxes = df_frame[["x", "y", "width", "height"]].values.astype(float)
        bboxes.append(frame_bboxes)

        conf = df_frame["confidence"].values
        confidences.append(conf)
        
        dummy_labels.append(np.repeat("moose", len(df_frame))) 

    tracer = model.tracker(frames, bboxes, confidences, dummy_labels)
    
    track_ids = list(tracer.keys()) 
    
    visual_feats, _ = model.feature_fusion(tracer)
    herd_latent, instance_latents, attn_weights = model(visual_feats)
    
    text_embeddings = torch.stack(list(model.text_embeddings.values())).to(model.device)
    
    similarities = torch.nn.functional.cosine_similarity(instance_latents.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)
    track_preds = torch.argmax(similarities, dim=1)
    max_similarities, _ = torch.max(similarities, dim=1)
    
    for idx, track_id in enumerate(tracer):
        species = labels[track_preds[idx].item()]
        weight = max_similarities[idx].item()
        tracer[track_id]["labels"] = species
        tracer[track_id]["weight"] = weight
        print(f"Track {track_id}: {species} (MIL Weight: {weight:.4f})")
        
    # Visualization
    bounding_box = BoundingBox()
    video_frames = []
    for frame_idx, frame in enumerate(frames):
        image = np.array(frame)
        for track_id in tracer:
            for i, frame_num in enumerate(tracer[track_id]["frame_idx"]):
                if frame_num == frame_idx+1:
                    track_data = tracer[track_id]
                    bbox = track_data["bboxes"][i]
                    
                    width, height = frame.size
                    bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height
                    bbox = box_convert(torch.tensor(bbox, dtype=torch.float32), in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy()
                    label_str = f"Track {track_id}: {track_data['labels']} | {species} (w:{track_data['weight']:.2f})"
                    bounding_box.add(image, *bbox, label_str, "olive")
        image = Image.fromarray(image)
        #image.save(f"results/frame_{frame_idx}.png")
        video_frames.append(image)
    
    # Save video
    video(video_frames, "results")


root = "data/ammonitor/images/"

# Data preparation
df = pd.read_csv("data/ammonitor/metadata.csv")
paths = [os.path.join(root.split(os.sep)[-1], fname) for root, _, fnames in os.walk(root) for fname in fnames]
df = df[df["path"].isin(paths)]
df = df[df["label"]!="empty"]
labels = df["label"].unique().tolist()


# Setup your VAL_TRANSFORM here (e.g., Resize, ToTensor, Normalize)
VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Run inference
inference_with_visualization(df, labels, transform=VAL_TRANSFORM, root=root) # Pass actual transform here






@torch.no_grad()
def inference_with_visualization_tmp(df, labels, transform, root):
    model = Herd(labels=labels, transform=transform)
    checkpoint = torch.load("checkpoints/herd-epoch=07.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    
    # 1. Isolate a sequence
    if "video_id" in df.columns:
        seq_id = df.sample(1).iloc[0]["video_id"]
        seq_df = df[df["video_id"] == seq_id].sort_values("frame_num")
    else:
        img_path = df.sample(1).iloc[0]["path"]
        seq_df = df[df["path"] == img_path]

    frames, all_bboxes, all_confidences, all_labels = [], [], [], []
    
    # 2. Prepare the sequential data arrays
    for file_name, frame_group in seq_df.groupby("path"):
        image = Image.open(os.path.join(root, file_name)).convert("RGB")
        frames.append(image)
        
        bboxes = frame_group[["x", "y", "width", "height"]].values.astype(float)
        all_bboxes.append(bboxes)
        
        confidences = frame_group["confidence"].values if "confidence" in frame_group else np.ones(len(bboxes))
        all_confidences.append(confidences)
        
        frame_labels = frame_group["label"].values
        all_labels.append(frame_labels)

    # 3. Track animals across the frames
    tracer = model.tracker(frames, all_bboxes, all_confidences, all_labels)
    if not tracer:
        print("No valid tracks found in this sequence.")
        return

    # Keep track of IDs to map predictions back to the correct boxes
    track_ids = list(tracer.keys()) 
    visual_feats, _ = model.feature_fusion(tracer)
    
    # 4. MIL Forward Pass (Unpack instance latents and weights)
    herd_latent, instance_latents, attn_weights = model(visual_feats)
    
    # 5. Compare INDIVIDUAL track representations against text embeddings
    text_embeddings = torch.stack(list(model.text_embeddings.values())).to(model.device)
    
    # Compute similarity for each instance: [num_tracks, num_classes]
    similarities = torch.nn.functional.cosine_similarity(instance_latents.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)
    track_preds = torch.argmax(similarities, dim=1)
    
    # 6. Map results to tracks
    track_results = {}
    for i, track_id in enumerate(track_ids):
        species = labels[track_preds[i].item()]
        weight = attn_weights[i].item()
        track_results[track_id] = (species, weight)
        print(f"Track {track_id}: {species} (MIL Weight: {weight:.4f})")
    
    # 7. Visualization
    final_image = np.array(frames[-1])
    bounding_box = BoundingBox()
    
    # Iterate through tracks to draw their final known bounding box
    for track_id, data in tracer.items():
        species, weight = track_results[track_id]
        
        # Get the last bounding box recorded for this track
        # Note: tracer stores boxes as cxcywh tensors, convert back to xywh for drawing
        bbox_cxcywh = data["bboxes"][-1].unsqueeze(0)
        bbox_xywh = box_convert(bbox_cxcywh, in_fmt="cxcywh", out_fmt="xywh").squeeze(0).cpu().numpy()
        
        
        label_str = f"{species} (w:{weight:.2f})"
        bounding_box.add(final_image, *bbox_xywh, label_str, "olive")

    final_image_pil = Image.fromarray(final_image)
    final_image_pil.save("results/attention_visualization.png")
    