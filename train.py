from ultralytics import YOLOv10, settings
import os
import torch

if os.getcwd().split('/')[-1] != "coinprediction":
    os.chdir("./coinprediction")
if not os.path.exists("./model"):
    os.makedirs("./model")


settings.update({"runs_dir": "./model/runs"})
settings.update({"datasets_dir": "./"})
settings.update({"weights_dir": "./model/weights"})

model = YOLOv10("yolov10n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
results = model.train(data=f"./data.yaml", epochs=1)