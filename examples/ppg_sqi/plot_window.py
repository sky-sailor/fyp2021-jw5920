"""
Plot a Window of Specified Position
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Data path
patient = '2001'
filepath = 'D:/IC Final Year Project/OUCRU/adults/01NVa-003-' + patient + '/PPG'
filename = '01NVa-003-2001 Smartcare.csv'
# TIMESTAMP_MS, COUNTER, DEVICE_ID, PULSE_BPM, SPO2_PCT, SPO2_STATUS,
# PLETH, BATTERY_PCT, RED_ADC, IR_ADC, PERFUSION_INDEX

start_datetime = '2021-01-01 17:00:00'

# Load patient data
# Read a comma-separated values (csv) file into DataFrame - return a 2D data structure with labeled axes - 11 columns
signals = pd.read_csv(os.path.join(filepath, filename))  # , nrows=1e6)  # include the number of rows to load

# Normalization (min-max to [0, 1]) on PLETH and put back
ppg_np = signals['PLETH'].to_numpy()
min_max_scaler = MinMaxScaler()
ppg_np_nml = min_max_scaler.fit_transform(ppg_np.reshape(-1, 1))
signals['PLETH'] = pd.DataFrame(ppg_np_nml)

signals = signals.reset_index()  # 12 columns

# Create timedelta
signals['timedelta'] = pd.to_timedelta(signals.TIMESTAMP_MS, unit='ms')

# Create date
signals['date'] = pd.to_datetime(start_datetime)
signals['date'] += pd.to_timedelta(signals.timedelta)  # 14 columns

print(signals.columns)

# print(signals['TIMESTAMP_MS'].head())
# print(signals['COUNTER'].head())
# print(signals['timedelta'].head())
# print(signals['date'].head())

# %%
# use indices to find date and time
# signals['date'] = signals['date'].round(decimals=3)
first = 231598
last = first - 1 + 1000
fig = plt.figure()
# date, timedelta, index are available for x-axis, date is the most intuitive choice
plt.plot(signals.TIMESTAMP_MS[first:last]/1000, signals.PLETH[first:last])
# plt.plot(signals.date, signals.PLETH)  # plot the whole data
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('PPG')
plt.grid(True)
plt.show()
