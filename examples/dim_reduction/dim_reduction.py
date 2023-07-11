"""
Dimensionality Reduction and Visualization for PPG-SQI by Autoencoder & PCA
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

from collections import Counter
import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans  # , KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from pkgname.core.AE.ae_model_builder import AE_model_builder

# read from the csv file
n_sqi = 7
# sqi_saved = pd.read_csv("saved_data/sqi_norm_2001_w30s_t5_pifix_inf100.csv")
saved_paths = [
    "saved_data/sqi_norm_2001_w10s_t5_pifix_inf100.csv",
    "saved_data/sqi_norm_2012_w10s_t5_pifix_inf100.csv",
    "saved_data/sqi_norm_2103_w10s_t5_pifix_inf100.csv",
    "saved_data/sqi_norm_2104_w10s_t5_pifix_inf100.csv"
]

sqi2001 = pd.read_csv(saved_paths[0])
sqi2012 = pd.read_csv(saved_paths[1])
sqi2103 = pd.read_csv(saved_paths[2])
sqi2104 = pd.read_csv(saved_paths[3])

sqi = pd.concat([sqi2001, sqi2012, sqi2103, sqi2104], ignore_index=True)
sqi = sqi.iloc[:, 3:3 + n_sqi]

# standardise the 7-D SQI, NOTE that this does not necessarily lead to better clustering results
# std_scaler = StandardScaler()
# sqi = std_scaler.fit_transform(sqi)

# Compute pairwise correlation of 7 SQIs
corr_pearson = sqi.corr(method='pearson')
corr_kendall = sqi.corr(method='kendall')
corr_spearman = sqi.corr(method='spearman')

# %%
# NOW let's play with the model and hyperparameters
# parameter tuning
trial1 = [[20, 14, 8, 5, 2], "relu", 1e-3, 20, 128]
ae_model = AE_model_builder(encoder_layer=trial1[0],
                            activation_func=trial1[1],
                            input_dim=n_sqi)

ae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial1[2]),
                 loss='mean_squared_error',
                 metrics=['accuracy'])

History = ae_model.fit(x=sqi,
                       y=sqi,
                       batch_size=trial1[4],
                       epochs=trial1[3],
                       validation_split=0.1)

epochs_range = range(trial1[3])

# plot accuracy and loss curve
plt.figure(1)
plt.subplot(121)
plt.plot(epochs_range, History.history['accuracy'])
plt.plot(epochs_range, History.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.grid(True)

plt.subplot(122)
plt.plot(epochs_range, History.history['loss'])
plt.plot(epochs_range, History.history['val_loss'])
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.grid(True)
plt.show()

# plot and save the model
tf.keras.utils.plot_model(ae_model,
                          to_file="saved_models/ae_model_2.png",
                          show_shapes=True,
                          show_layer_names=False,
                          rankdir='TB',
                          dpi=300,)

# %%
# dimensionality reduction by the Encoder
dr_layers = ae_model.layers[:len(trial1[0])]
dr_model = tf.keras.Sequential(dr_layers)
dr_model.summary()
dr_ae = dr_model.predict(sqi)

plt.figure(2)
plt.subplot(121)
plt.scatter(dr_ae[:, 0], dr_ae[:, 1])
plt.title('Dimensionality reduction by Autoencoder')
plt.grid(True)
# plt.show()

# 2-D clustering by KMeans
n_clusters = 3
# ae_kmeans = KMeans(n_clusters=n_clusters, n_init=100).fit_predict(predictions)
ae_kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                            max_iter=400,
                            batch_size=256 * 8,
                            random_state=42,
                            n_init=3,
                            ).fit_predict(dr_ae)

count_ae = Counter(ae_kmeans)
print('K-Means labels with number of clusters = ', n_clusters)
print(count_ae)

plt.subplot(122)
plt.scatter(dr_ae[:, 0], dr_ae[:, 1], c=ae_kmeans.astype(float), edgecolor='k')
plt.title('Clustering by KMeans, n_clusters = ' + str(n_clusters))
plt.grid(True)
plt.show()

# %%
# grid search for parameter setting - Optional
en_layer = [[6, 4, 2], [6, 4, 3, 2], [5, 3, 2]]
# act_func = ["sigmoid", "relu"]
learn_rate = [5e-4, 1e-3]
epoch_n = [20, 30]
bat_size = [64, 128]

for el in en_layer:
    for lr in learn_rate:
        for ep in epoch_n:
            for bs in bat_size:
                ae_model = AE_model_builder(encoder_layer=el, activation_func="relu", input_dim=n_sqi)
                ae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                 loss='mean_squared_error', metrics=['mse'])
                ae_model.fit(x=sqi, y=sqi, batch_size=bs, epochs=ep, validation_split=0.1)

# %%
# PCA dimensionality reduction
pca = PCA(n_components=2)
dr_pca = pca.fit_transform(sqi)

plt.figure(3)
plt.subplot(121)
plt.scatter(dr_pca[:, 0], dr_pca[:, 1], edgecolor='k')
plt.title('Dimensionality reduction by PCA')
plt.grid(True)

# PCA 2-D clustering by KMeans
n_clusters = 3
# pca_kmeans = KMeans(n_clusters=n_clusters, n_init=100).fit_predict(dr_pca)
pca_kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             max_iter=400,
                             batch_size=256 * 8,
                             random_state=42,
                             n_init=3,
                             ).fit_predict(dr_pca)

count_pca = Counter(pca_kmeans)
print('K-Means labels with number of clusters = ', n_clusters)
print(count_pca)

# plt.figure()
plt.subplot(122)
plt.scatter(dr_pca[:, 0], dr_pca[:, 1], c=pca_kmeans.astype(float), edgecolor='k')
plt.title('Clustering by KMeans, n_clusters = ' + str(n_clusters))
plt.grid(True)
plt.show()

# %%
# Compute stats in each cluster - separated tables
windowed_ppg_ae = pd.DataFrame(data=sqi)
windowed_ppg_pca = pd.DataFrame(data=sqi)
# Add AE projections
windowed_ppg_ae['ae_x'] = dr_ae[:, 0]
windowed_ppg_ae['ae_y'] = dr_ae[:, 1]
# Add PCA projections
windowed_ppg_pca['pca_x'] = dr_pca[:, 0]
windowed_ppg_pca['pca_y'] = dr_pca[:, 1]
# Add AE cluster ID
windowed_ppg_ae['ae_cluster'] = ae_kmeans  # .labels_
# Add PCA cluster ID
windowed_ppg_pca['pca_cluster'] = pca_kmeans  # .labels_
# Compute stats
ae_table_2 = windowed_ppg_ae.groupby('ae_cluster').agg(['mean', 'std'])
pca_table_2 = windowed_ppg_pca.groupby('pca_cluster').agg(['mean', 'std'])
