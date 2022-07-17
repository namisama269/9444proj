import os
import gc
import psutil
import glob
import random
import tarfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle

# https://www.kaggle.com/datasets/atulanandjha/lfwpeople?resource=download

from count_data import count_data

path = "lfw_funneled"

# make data dir if not exist
if not os.path.exists(path):
    with tarfile.open("lfw-funneled.tgz") as tar:
        tar.extractall()

img_size = (125, 125, 3)

# number of unique people and total images
m, n = 5749, 13233
# m, n = count_data("lfw_funneled")

# initialize label dicts
labels = {}
inv_labels = {}

X = np.zeros([n] + [_ for _ in img_size], dtype='float32')
y = np.zeros(n, dtype='int16')

i = 0
j = 0
for name_dir in os.listdir("lfw_funneled"):
    # skip non directory files
    if not os.path.isdir(os.path.join(path, name_dir)):
        continue

    name = name_dir.replace('_', ' ')
    labels[i] = name
    inv_labels[name] = i

    for s in os.listdir(os.path.join(path, name_dir)):
        img_path = os.path.join(path, name_dir, s)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1] 
        X[j, :, :, :] = cv2.resize(img, img_size[:2], interpolation=cv2.INTER_AREA) / 255
        y[j] = i
        j += 1

    i += 1

# save data
if not os.path.exists("data"):
    os.makedirs("data")

# save labels
with open("data/labels.pkl", "wb") as f:
    pickle.dump(labels, f)
with open("data/inv_labels.pkl", "wb") as f:
    pickle.dump(inv_labels, f)

# save X and y
with open("data/X.npy", "wb") as f:
    np.save(f, X)
with open("data/y.npy", "wb") as f:
    np.save(f, y)
