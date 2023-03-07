from src import generate_cv_batch

import re
import json
import shutil

import torch
import pandas as pd
import numpy as np

from ultralytics import YOLO
from roboflow import Roboflow

from src.confident_learning import *
from src.utils import *
from src.train_wrapper import *

root_path = "/visible_skin_concern/"
dataset_path = "/visible_skin_concern/datasets/"
folder_name = "Pore-detection-14"
cv_path = "/visible_skin_concern/datasets/Pore-detection"

# model = YOLO("yolov8s.pt")
# model = YOLO("H:/visible_skin_concern/runs/detect/train/weights/best.pt")
# model = YOLO(r"H:\visible_skin_concern\runs\detect\train3\weights\best.pt")
# model.val(r"H:\visible_skin_concern\datasets\Pore-detection-14-0\data.yaml")
# model.train(data="/visible_skin_concern/datasets/Pore-detection-14/data.yaml", epochs=10, imgsz=320, lr0=1e-3)
# generate_cv_batch(dataset_path, dataset_path, folder_name)
confident_learning(root_path, dataset_path)
# model = YOLO("yolov8s.pt")
# train(model, data="/visible_skin_concern/datasets/Pore-detection-15/data.yaml", epochs=50, imgsz=320, lr0=1e-3)

# model.val(data="/visible_skin_concern/datasets/Pore-detection-14/data.yaml", save_json=True)

