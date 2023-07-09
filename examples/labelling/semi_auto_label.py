"""
PPG Semi-Automatic Labelling
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

# from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering  # , DBSCAN

# load the preprocessed PPG-SQI data
n_sqi = 7
sqi_saved = pd.read_csv("D:/IC Final Year Project/fyp2021-jw5920/saved_data/sqi2001_w30s_t5_pifix_inf100.csv")
sqi = sqi_saved.iloc[:, 3:3 + n_sqi]


# %%
def KMeans_labelling(data, n_cluster):
    """
    cluster the windows by KMeans based on similarity in high dimension
    :param data: windowed PPG-SQI
    :param n_cluster: number of clusters (signal qualities)
    :return: labels of each window
    """
    return KMeans(n_clusters=n_cluster,
                  init="k-means++",
                  n_init=100,
                  max_iter=400,
                  tol=1e-5,
                  algorithm="full"
                  ).fit_predict(data)


def MiniBatchKMeans_labelling(data, n_cluster):
    return MiniBatchKMeans(n_clusters=n_cluster,
                           max_iter=400,
                           batch_size=256*8,
                           random_state=42,
                           n_init=3,
                           ).fit_predict(data)


# def DBSCAN_labelling(data, n_cluster):
#     return DBSCAN(eps=0.4,
#                   min_samples=5
#                   ).fit_predict(data)


def SpectralClustering_labelling(data, n_cluster):
    return SpectralClustering(n_clusters=n_cluster,
                              random_state=100,
                              assign_labels='discretize',
                              ).fit_predict(data)


def AgglomerativeClustering_labelling(data, n_cluster):
    return AgglomerativeClustering(n_clusters=n_cluster,
                                   ).fit_predict(data)


# for observation, add clustering labels to PPG-SQI
# n_clusters = 3
# labels_kmeans = KMeans_labelling(sqi, n_clusters)  # 0: 1695, 1: 154, 2: 7

# labels_minibatch_kmeans = MiniBatchKMeans_labelling(sqi, n_clusters)  # 2: 1455, 0: 247, 1: 154

# labels_dbscan = DBSCAN_labelling(sqi, n_clusters)  # 0: 1661, 1: 153, -1: 42

# labels_spectral = SpectralClustering_labelling(sqi, n_clusters)

# labels_agglomerative = AgglomerativeClustering_labelling(sqi, n_clusters)

# count number of each label
# count_kmeans = Counter(labels_kmeans)

# count_minibatch = Counter(labels_minibatch_kmeans)

# count_dbscan = Counter(labels_dbscan)

# count_spectral = Counter(labels_spectral)

# count_agglomerative = Counter(labels_agglomerative)

# add to sqi
# sqi['cluster'] = pd.DataFrame(data=labels_minibatch_kmeans, columns=['cluster'])

# print("------------complete!------------")
# print(count_kmeans)
# print(count_minibatch)
# print(count_dbscan)
# print(count_spectral)
# print(count_agglomerative)
