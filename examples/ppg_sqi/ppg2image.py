# Generic
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# vitalSQI
from vital_sqi.data.signal_io import PPG_reader
import vital_sqi.highlevel_functions.highlevel as sqi_hl
import vital_sqi.data.segment_split as sqi_sg
from vital_sqi.common.rpeak_detection import PeakDetector

# ----------------------------
# Load data
# ----------------------------
# Filepath
filepath = 'D:/IC Final Year Project/01NVa-003-2001/PPG'
filename = '01NVa-003-2001 Smartcare.csv'

# Prepare variables for playbook
sampling_rate = 100  # Hz
hp_filt_params = (1, 1)  # (Hz, order)
lp_filt_params = (20, 4)  # (Hz, order)
filter_type = 'butter'
trim_amount = 20  # s
segment_length = 30  # s
# 3000 entries for each segment
width = 60  # image width
height = segment_length * sampling_rate // width  # image height

ppg_data = PPG_reader(os.path.join(filepath, filename),
                      signal_idx=['PLETH'],
                      timestamp_idx=['TIMESTAMP_MS'],
                      info_idx=['SPO2_PCT', 'PULSE_BPM', 'PERFUSION_INDEX'],
                      timestamp_unit='ms', sampling_rate=sampling_rate, start_datetime=None)
raw_signal = np.copy(
    ppg_data.signals.T[trim_amount * sampling_rate:-trim_amount * sampling_rate].T)  # save the raw signal

# ----------------------------
# Transform a continuous PPG signal (one window) into an image
# ----------------------------
# 1. Directly convert the raw signal into an image img1
ppg_len = raw_signal.shape[1]
print('The length of PPG data =', ppg_len)
ppg_cropped_len = ppg_len - ppg_len % (width * height)
num_img = ppg_cropped_len // (width * height)
img1 = np.reshape(raw_signal[0, 0:ppg_cropped_len], (num_img, height, width))
print('The length of img1 =', ppg_cropped_len)
print('The shape of img1 =', img1.shape)

# 2. Compute the spectrogram
# no no! Calculate spectrogram on each images (segments)
f, t, Sxx = signal.spectrogram(raw_signal, sampling_rate)
img2 = Sxx
plt.pcolormesh(t, f, img2, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# 3. Extract various features (fourier, sqi_agg, shape, others) and create an image reshaping those values
# 3.1 FFT
# 2-D fft to each image computed in method 1

# 3.2 SQIs
# one image for each segment, and different images have the same SQI in corresponding positions. Different SQIs
# take different areas (large or small).

