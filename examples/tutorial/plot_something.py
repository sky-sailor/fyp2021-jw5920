import numpy as np
import matplotlib.pyplot as plt

path = "D:/IC Final Year Project/01NVa-003-2001/Monitor/"
fileName = "01NVa-003-2001waves.asc"
# 247200
waves = np.loadtxt(path+fileName, delimiter='\t', skiprows=2)
timestamps = waves[:, 0]-1591184022  # the start of time stamps
ecgwaves = waves[:, 1]
ppgwaves = waves[:, 2]
rrwaves = waves[:, 3]

plt.plot(timestamps[1000: 4000], ppgwaves[1000: 4000])
plt.xlabel('Time (s)')
plt.ylabel('PPG')
plt.grid()
plt.show()
