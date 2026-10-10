import pandas as pd





df = pd.read_csv("data/ammonitor/metadata.csv")
print(df["video_id"].value_counts())