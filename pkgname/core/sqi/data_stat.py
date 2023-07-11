"""
Description of the Dataset (size, mean, std)
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

import pandas as pd
import os

filepath = 'D:/IC Final Year Project/OUCRU/adults/01NVa-003-2162/PPG'
filename = '20200720T140507.193+0700.csv'
signals = pd.read_csv(os.path.join(filepath, filename))
print(signals.shape)

# mean and std calculation in 3 ways
mean1 = signals.PLETH.mean()
mean2 = signals['PLETH'].mean()
std1 = signals.PLETH.std()
std2 = signals['PLETH'].std()
signals['PLETH'].describe()
