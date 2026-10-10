import json



with open(f"snapshot-safari-2024-expansion-SER-subset-v1000.0.0-redwood_detections.threshold.filtered.json", 'r') as file:
    data = json.load(file)
    
print(data.keys())
print(data["info"])
print("")
print("Images: ", len(data["images"]))
print(data["images"][0])
print("")
print("Categories: ", len(data["detection_categories"]))
print(data["detection_categories"])