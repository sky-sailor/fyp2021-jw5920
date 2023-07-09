"""
Description of the Dataset (Size, Mean, Std)
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

import pandas as pd
import os
filepath = 'D:/IC Final Year Project/OUCRU/adults/01NVa-003-2162/PPG'
filename = '20200720T140507.193+0700.csv'
signals = pd.read_csv(os.path.join(filepath, filename))  # include the number of rows to load
# print("\nLoaded signals:")
print(signals.shape)

# Mean and Std calculation in 3 ways
# mean = signals.PLETH.mean()
# mean2 = signals['PLETH'].mean()
# std = signals.PLETH.std()
# std2 = signals['PLETH'].std()
signals['PLETH'].describe()
