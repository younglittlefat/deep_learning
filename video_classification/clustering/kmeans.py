import sys
import os
import pickle

import numpy as np
from sklearn.cluster import MiniBatchKMeans

def import_data():
    file_path = "video_embedding"
    with open(file_path, "r") as f:
        feat_list = [map(float, temp[1:]) for temp in map(lambda x:x.strip().split(" "), f.readlines())]

    feat_arr = np.array(feat_list)
    print feat_arr.shape
    return feat_arr


def clustering(feat_arr):
    n_clusters = 50
    kmeans = MiniBatchKMeans(n_clusters = n_clusters)
    kmeans.fit(feat_arr)
    with open("clusters_" + str(n_clusters), "wb") as f:
        pickle.dump(kmeans, f)


feat_arr = import_data()
clustering(feat_arr)
