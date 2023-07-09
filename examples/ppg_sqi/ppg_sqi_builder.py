"""
PPG-SQI Calculation and Storage
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from pkgname.core.sqi.sqi import sqi_all


plt.close('all')  # close all the plots

# Time specification
start_datetime = '2021-01-01 17:00:00'
trimmed_min = 5
window_len = '10s'

# Data path
patient = '2104'
filepath = 'D:/IC Final Year Project/OUCRU/adults/01NVa-003-' + patient + '/PPG'
filename = '20200918T152608.306+0700.csv'
# TIMESTAMP_MS, COUNTER, DEVICE_ID, PULSE_BPM, SPO2_PCT, SPO2_STATUS,
# PLETH, BATTERY_PCT, RED_ADC, IR_ADC, PERFUSION_INDEX

# Load patient data
# Read a comma-separated values (csv) file into DataFrame - return a 2D data structure with labeled axes - 11 columns
signals = pd.read_csv(os.path.join(filepath, filename))  # , nrows=1e6)  # include the number of rows to load

# Normalization (min-max to [0, 1]) on PLETH and put back
ppg_np = signals['PLETH'].to_numpy()
min_max_scaler = MinMaxScaler()
ppg_np_nml = min_max_scaler.fit_transform(ppg_np.reshape(-1, 1))
signals['PLETH'] = pd.DataFrame(ppg_np_nml)

# Calculate sampling rate
fs = 1 / (signals.TIMESTAMP_MS.diff().median() * 0.001)

# Show original signals - 11 columns
# print("\nLoaded signals:")
# signals
# print(signals)

# Format the data
# Display (shows timedelta aligned)
pd.Timedelta.__str__ = lambda x: x._repr_base('all')  # w: access to a protected member of a class

# Include column with index - 12 columns
signals = signals.reset_index()
# Create timedelta
signals['timedelta'] = pd.to_timedelta(signals.TIMESTAMP_MS, unit='ms')
# Create date
signals['date'] = pd.to_datetime(start_datetime)
signals['date'] += pd.to_timedelta(signals.timedelta)

# Set the timedelta index (keep numeric index too)
signals = signals.set_index('timedelta')

# Rename column index to avoid confusion
signals = signals.rename(columns={'index': 'idx'})

# Show - 13 columns
# print("\nSignals:")
# signals
# print(signals)

# Trim first/last 'trimmed_min' minutes
# Offset
offset = pd.Timedelta(minutes=trimmed_min)

# Indexes
idxs = (signals.index >= offset) & (signals.index <= signals.index[-1] - offset)

# Filter
signals = signals[idxs]


def bbpf_rep(x):
    """Butter Band Pass Filter Scipy
    """
    # Configure high/low filters
    bh, ah = signal.butter(1, 1, btype='highpass', analog=False, output='ba', fs=fs)
    bl, al = signal.butter(4, 20, btype='lowpass', analog=False, output='ba', fs=fs)
    # Apply filters
    aux = signal.filtfilt(bh, ah, x)
    aux = signal.lfilter(bl, al, aux)
    # Return
    return aux


# Add BPF
signals['PLETH_BPF'] = bbpf_rep(signals.PLETH)

# Show - 14 columns, added: idx, date, PLETH_BPF
print("\nOriginal signals prepared (14 columns)")

# Compute SQIs
# 1. use the method 'agg'
# Group by 30s windows/aggregate, must be in time
# sqi_agg = signals \
#     .groupby(pd.Grouper(freq=groupby_freq)) \
#     .agg({'idx': ['first', 'last'],
#           'PLETH': [skew, kurtosis, snr, mcr],
#           'IR_ADC': [skew, kurtosis, snr, mcr],
#           'PLETH_BPF': [zcr, msq, correlogram]  # dtw]
#           })
#
# # Add window id (if needed)
# sqi_agg['w'] = np.arange(sqi_agg.shape[0])
#
# # Show - 14 columns
# print("\nSQIs (agg):")
# # sqi_agg
#
# if TERMINAL:
#     print(sqi_agg)

# 2. use 'apply' to compare
# Group by 30s windows/apply
sqi_apply = signals \
    .groupby(pd.Grouper(freq=window_len)) \
    .apply(sqi_all)

# Show - 10 columns
print("\nsqi_apply (10 columns)")

# slicing - 7 columns
# sqi = sqi_apply.iloc[:, 2:9]
# sqi['perfusion'][np.isinf(sqi['perfusion'])] = sys.maxsize  # SettingWithCopyWarning

# replace the inf entries in perfusion index with sys.maxsize or a specified number
# sqi = sqi.replace(np.inf, sys.maxsize)
sqi = sqi_apply.replace(np.inf, 100)
print('\nsqi (10 columns)')

# %%
save_path = "D:/IC Final Year Project/fyp2021-jw5920/saved_data/sqi_norm_" + patient + "_w" + \
           window_len + "_t" + str(trimmed_min) + "_pifix_inf100.csv"

# fix bug of number
sqi = sqi[:-1]

# write to a csv file
sqi.to_csv(save_path)

# read from the csv file to check
sqi_saved = pd.read_csv(save_path)
