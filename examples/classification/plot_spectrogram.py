"""
Plot the Spectrograms
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

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

# %%
# plot one spectrogram
f, t, Sxx = spectrogram(raw_signal[8206, :], fs)  # NOT yet tuned!

Sxx_trimmed = Sxx[:20, :]

plt.figure()
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.show()

print(f.shape)  # (129,)
print(t.shape)  # (13,)
print(Sxx.shape)  # (129, 13)
