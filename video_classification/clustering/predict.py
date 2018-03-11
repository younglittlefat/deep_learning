import sys
import os
import pickle


import numpy as np
from sklearn.cluster import MiniBatchKMeans

def predict():
    file_path = "video_embedding"
    model_path = "clusters_50"

    with open(model_path, "rb") as f:
        kmeans = pickle.load(f)

    with open(file_path, "r") as f:
        for line in f:
            temp = line.strip().split(" ")
            video_name = temp[0]
            feat = np.array([map(float, temp[1:])])
            idx = kmeans.predict(feat)
            print "%s %d" % (video_name, idx)


predict()
