# Generic
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scipy
from scipy.stats import skew
from scipy.stats import kurtosis

# Heartpy
import heartpy as hp

# vitalSQI
from vital_sqi.data.signal_io import PPG_reader

# ----------------------------
# Constant
# ----------------------------
# Constant to define whether we are using the terminal in
# order to plot the dataframes. Keep to false by default
# specially when building the documentation.
TERMINAL = True

# ----------------------------
# Load data
# ----------------------------
# Filepath
filepath = 'D:/IC Final Year Project/01NVa-003-2001/PPG'
filename = '01NVa-003-2001 Smartcare.csv'

# # Load
# data = PPG_reader(os.path.join(filepath, filename), signal_idx=['PLETH'], timestamp_idx=['TIMESTAMP_MS'],
#                   info_idx=['COUNTER','DEVICE_ID','PULSE_BPM','SPO2_PCT','SPO2_STATUS','BATTERY_PCT','RED_ADC','IR_ADC','PERFUSION_INDEX'],  # , 2, 3, 4, 5, 7, 8, 9, 10
#                   timestamp_unit='ms', sampling_rate=None, start_datetime=None)
#
# # The attributes!
# print(data)
# print(data.signals)
# print(data.sampling_rate)
# print(data.start_datetime)
# print(data.wave_type)
# # print(data.sqi_indexes)
# # print(data.info)
#
# fs = data.sampling_rate

# Prepare variables for playbook
sampling_rate = 100  # Hz
hp_filt_params = (1, 1)  # (Hz, order)
lp_filt_params = (20, 4)  # (Hz, order)
filter_type = 'butter'
trim_amount = 20  # s
segment_length = 30  # s

# file_name = "ppg_smartcare.csv"
ppg_data = PPG_reader(os.path.join(filepath, filename),
                      signal_idx=['PLETH', 'IR_ADC'],
                      timestamp_idx=['TIMESTAMP_MS'],
                      info_idx=['SPO2_PCT', 'PULSE_BPM', 'PERFUSION_INDEX'],
                      timestamp_unit='ms', sampling_rate=sampling_rate, start_datetime=None)

# We have loaded a single data column, therefore we only have 1D timeseries
print(ppg_data.signals.shape)
# Plot a random 10s segment of the signal
s = np.arange(0, 1000, 1)
fig, ax = plt.subplots()
ax.plot(s, ppg_data.signals[0][10000:11000])
plt.show()
