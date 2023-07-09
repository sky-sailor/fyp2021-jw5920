# Generic
import os
import numpy as np
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

raw_signal = np.copy(
    ppg_data.signals.T[trim_amount * sampling_rate:-trim_amount * sampling_rate].T)  # save the raw signal
ppg_data.update_signal(sqi_hl.signal_preprocess(signal_channel=ppg_data.signals, hp_cutoff_order=hp_filt_params,
                                                lp_cutoff_order=lp_filt_params, trim_amount=trim_amount,
                                                filter_type=filter_type, sampling_rate=sampling_rate))

print(ppg_data.signals.shape)

s = np.arange(0, 1000, 1)
fig, ax = plt.subplots()
ax.plot(s, ppg_data.signals[0][8000:9000])
plt.show()

ppg_data.update_segment_indices(sqi_sg.generate_segment_idx(segment_length=segment_length, sampling_rate=sampling_rate,
                                                            signal_array=ppg_data.signals))
print(ppg_data.segments.shape)
print(ppg_data.segments)

detector = PeakDetector()
peak_list, trough_list = detector.ppg_detector(ppg_data.signals[0][ppg_data.segments[0][0]:ppg_data.segments[0][1]], 7)

# Plot results of peak detection
s = np.arange(0, 3000, 1)
fig, ax = plt.subplots()
ax.plot(s, ppg_data.signals[0][ppg_data.segments[0][0]:ppg_data.segments[0][1]])
if len(peak_list) != 0:
    ax.scatter(peak_list, ppg_data.signals[0][peak_list], color="r", marker="v")
if len(trough_list) != 0:
    ax.scatter(trough_list, ppg_data.signals[0][trough_list], color="b", marker="v")
plt.show()

# Plot a single period
fig, ax = plt.subplots()
ax.plot(ppg_data.signals[0][trough_list[0]:trough_list[1]])
plt.show()

computed_sqi = sqi_hl.compute_all_SQI(signal=ppg_data.signals[0], segments=ppg_data.segments[0],
                                      raw_signal=raw_signal[0], primary_peakdet=7, secondary_peakdet=6, template_type=0)
print(computed_sqi[2])
