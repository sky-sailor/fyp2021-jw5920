"""
PPG Data Trimmer, Concatenation and Storage
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

fs = 100
trim = 5 * 60 * fs
window = 10 * fs

file_paths = [
    "D:/IC Final Year Project/OUCRU/adults/01NVa-003-2001/PPG/01NVa-003-2001 Smartcare.csv",
    "D:/IC Final Year Project/OUCRU/adults/01NVa-003-2012/PPG/20200811T162822.961+0700.csv",
    "D:/IC Final Year Project/OUCRU/adults/01NVa-003-2103/PPG/20200916T114221.687+0700.csv",
    "D:/IC Final Year Project/OUCRU/adults/01NVa-003-2104/PPG/20200918T152608.306+0700.csv"
]

ppg_concat = np.array([[0], [0], [0]])

# %%
for file in file_paths:
    signals = pd.read_csv(file)

    # Normalization (min-max to [0, 1]) on PLETH and put back
    ppg_np = signals['PLETH'].to_numpy()
    min_max_scaler = MinMaxScaler()
    ppg_np_nml = min_max_scaler.fit_transform(ppg_np.reshape(-1, 1))

    # trim
    total_len = len(ppg_np_nml)
    num_win = math.ceil((total_len - 2 * trim) / window)
    ppg_trimmed = ppg_np_nml[trim: (trim + num_win * window)]

    # concatenation
    ppg_concat = np.concatenate((ppg_concat, ppg_trimmed), axis=0)

# np convert to pd
ppg_concat = pd.DataFrame(ppg_concat[3:])

# storage: write to a csv file
save_path = "D:/IC Final Year Project/fyp2021-jw5920/saved_data/ppg_norm_2001_2012_2103_2104_w10s_t5.csv"
ppg_concat.to_csv(save_path)

# %%
# read from the csv file to check
ppg_saved = pd.read_csv(save_path)

print("---------complete!---------")
