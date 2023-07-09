"""
PPG-Spectrogram Classification
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

# import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from main.examples.labelling.semi_auto_label import MiniBatchKMeans_labelling

# Data path
saved_path = "saved_data/ppg_norm_2001_2012_2103_2104_w10s_t5.csv"
raw_signal = pd.read_csv(saved_path)

fs = 100  # sampling frequency (Hz)
window = 10 * fs

print("\nwindowed raw signal length:")
print(raw_signal.shape)

# Reshape the raw signal: each row as a window
raw_signal = raw_signal.iloc[:, 1].values.reshape(-1, window)
print("\nafter reshaping:")
print(raw_signal.shape)
num_win = raw_signal.shape[0]

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

# compute Spectrograms for each raw-PPG window
images = []
for i in range(num_win):
    f, t, Sxx = spectrogram(raw_signal[i, :], fs)

    Sxx_trimmed = Sxx[:20, :]
    images.append(Sxx_trimmed)

images = np.stack(images)

# %%
# Build a CNN model
n_classes = 3
n_epochs = 100
batch_size = 256
input_shape = images.shape

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 3, strides=1, padding='same', activation='relu', input_shape=(20, 4, 1)),
    tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', activation='relu', input_shape=(20, 4, 8)),
    tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu', input_shape=(20, 4, 16)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
    ], name="cnn_clf")

cnn_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

cnn_model.summary()

# Train the CNN model and plot results
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
print("\nX_train, X_test, y_train, y_test shapes:")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

history_cnn = cnn_model.fit(X_train,
                            y_train,
                            epochs=n_epochs,
                            batch_size=batch_size,
                            validation_split=0.15)

epochs_range = range(n_epochs)

# %%
# plot accuracy and loss curve
plt.figure()
plt.subplot(121)
plt.plot(epochs_range, history_cnn.history['accuracy'], label='Training')
plt.plot(epochs_range, history_cnn.history['val_accuracy'], label='Validation')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()

plt.subplot(122)
plt.plot(epochs_range, history_cnn.history['loss'], label='Training')
plt.plot(epochs_range, history_cnn.history['val_loss'], label='Validation')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.show()

# plot and save the model
tf.keras.utils.plot_model(cnn_model,
                          to_file="saved_models/cnn_model_5.png",
                          show_shapes=True,
                          show_layer_names=False,
                          rankdir='TB',
                          dpi=300,)

# Test the CNN model
# evaluate - Returns the loss value & metrics values for the model in test mode
test_result = cnn_model.evaluate(X_test, y_test, batch_size=32)
print("\ntest [loss, accuracy]:")
print(test_result)

predictions = cnn_model.predict(X_test, batch_size=32)
