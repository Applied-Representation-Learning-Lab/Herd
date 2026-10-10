import pandas as pd





df = pd.read_csv("data/ammonitor/metadata.csv")
data = df.groupby(["video_id", "segment_id"])
print("Total number of video clips {}".format(len(data)))

for id, group in data:
    print(group)
    sdjhs


