import os, json, threading
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

def chunk_dataframe(df, num_chunks):
  chunk_size = len(df) // num_chunks
  for i in range(0, len(df), chunk_size):
    yield df.iloc[i : i + chunk_size]

def download_images(df_chunk, url_root, dataset_name):
    for index, row in tqdm(df_chunk.iterrows(), desc=f"Downloading images for {dataset_name}", total=len(df_chunk)):
        image_url = os.path.join(url_root, row['path'])
        image_id = row['path']
        
        try:
            # Download the image
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                path = os.path.join(*image_id.split("/")[:-1])
                os.makedirs(os.path.join(f"data/{dataset_name}/images/", path), exist_ok=True)
                if os.path.exists(os.path.join(f"data/{dataset_name}/images/", image_id)):
                    print(f"Image {image_id} already exists. Skipping download.")
                    continue
                with open(os.path.join(f"data/{dataset_name}/images/", image_id), 'wb') as f:
                    for chunk in response.iter_content(2048):
                        f.write(chunk)
            else:
                print(f"Failed to download image {image_id}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading image {image_id}: {e}")
            
df = pd.read_csv("data/ammonitor/metadata.csv")
df.drop_duplicates("path", inplace=True)

chunks = list(chunk_dataframe(df, num_chunks=6)) 
threads = []
url_root = "https://lilawildlife.blob.core.windows.net/lila-wildlife/ammonitor-camera-traps/"  # Replace with the actual URL root
dataset_name = "ammonitor"

for chunk in chunks:
    thread = threading.Thread(target=download_images, args=(chunk, url_root, dataset_name))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()