"""
Raw PPG Signal Classification
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

# import os
from collections import Counter
# import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from main.examples.labelling.semi_auto_label import MiniBatchKMeans_labelling

# Data path
saved_path = "saved_data/ppg_norm_2001_2012_2103_2104_w10s_t5.csv"
raw_signal = pd.read_csv(saved_path)

fs = 100
window = 10 * fs

print("\nwindowed raw signal length:")
print(len(raw_signal))

# Reshape the raw signal: each row as a window
raw_signal = raw_signal.iloc[:, 1].values.reshape(-1, window)
print("\nafter reshaping:")
print(raw_signal.shape)

# load the preprocessed PPG-SQI data
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

# semi-automatic labelling with KMeans
n_clusters = 3
labels = MiniBatchKMeans_labelling(sqi, n_clusters)

# count number of each label
label_count = Counter(labels)

# %%
n_epochs = 10
proba_decision_tree = 0
proba_random_forest = 0
proba_adaboost = 0

for _ in range(n_epochs):
    # preprocess dataset, randomly split into training and test part
    X_train, X_test, y_train, y_test = train_test_split(raw_signal, labels, test_size=0.25)

    # Classifications of raw PPG signal
    # Decision tree classifier
    decision_tree = DecisionTreeClassifier()
    fitted_tree = decision_tree.fit(X_train, y_train)
    # tree.plot_tree(decision_tree)  # plot tree

    # y_predicted = decision_tree.predict(X_test)  # fitted_tree
    proba_decision_tree += decision_tree.score(X_test, y_test)  # fitted_tree
    # sum_decision_tree = np.sum(np.diff([y_test, y_predicted]))

    # Random forest classifier
    random_forest = RandomForestClassifier()  # max_depth=5, n_estimators=10, max_features=1)
    fitted_forest = random_forest.fit(X_train, y_train)
    # y_predicted = random_forest.predict(X_test)  # fitted_forest
    proba_random_forest += random_forest.score(X_test, y_test)  # fitted_forest
    # sum_random_forest = np.sum(np.diff([y_test, y_predicted]))

    # AdaBoost classifier
    adaboost = AdaBoostClassifier()
    fitted_adaboost = adaboost.fit(X_train, y_train)
    # y_predicted = adaboost.predict(X_test)  # fitted_adaboost
    proba_adaboost += adaboost.score(X_test, y_test)  # fitted_adaboost
    # sum_adaboost = np.sum(np.diff([y_test, y_predicted]))

acc_decision_tree = proba_decision_tree / n_epochs
acc_random_forest = proba_random_forest / n_epochs
acc_adaboost = proba_adaboost / n_epochs

print("------------Complete!------------")
