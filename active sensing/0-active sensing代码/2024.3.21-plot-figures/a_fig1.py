import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import numpy as np
import matplotlib.pyplot as plt

# import datetime
# import scipy.io as sio




snrdBvec = np.load('snrdBvec.npy')
# data_performance_active = np.load('new-performance.npy')
# data_performance_fixed = np.load('new-performance_fixed.npy')
# data_performance_benchmark2 =np.load('new-performance_benchmark-2.npy')

# tnum = 6
# data_performance_active = np.load('2-performance.npy')
# data_performance_fixed = np.load('2-performance_fixed.npy')
# data_performance_benchmark2 =np.load('2-performance_benchmark-2.npy')




# tnum = 4,  6000
# data_performance_active = np.load('5-performance.npy')
# data_performance_fixed = np.load('5-performance_fixed.npy')
# data_performance_benchmark2 =np.load('5-performance_benchmark-2.npy')


# data_performance_active = np.load('performance.npy')
# data_performance_fixed = np.load('performance_fixed_old.npy')
# data_performance_benchmark2 =np.load('performance_benchmark-2.npy')

###############################################################

# tnum = 2,  2000,  K = 10
data_performance_active = np.load('5-performance.npy')
data_performance_fixed = np.load('5-performance_fixed.npy')
data_performance_benchmark2 =np.load('5-performance_benchmark-2.npy')

# tnum = 3,  2000
data_performance_active = np.load('6-performance.npy')
data_performance_fixed = np.load('6-performance_fixed.npy')
data_performance_benchmark2 =np.load('6-performance_benchmark-2.npy')

# tnum = 4,  2000,  K = 10
data_performance_active = np.load('4-performance.npy')
data_performance_fixed = np.load('4-performance_fixed.npy')
data_performance_benchmark2 =np.load('4-performance_benchmark-2.npy')

# tnum = 5,  epoch = 2000,  K = 10
data_performance_active = np.load('7-performance.npy')
data_performance_fixed = np.load('7-performance_fixed.npy')
data_performance_benchmark2 =np.load('7-performance_benchmark-2.npy')



# # tnum = 3,  epoch = 2000,  K = 0.5
data_performance_active = np.load('11-performance.npy')
data_performance_fixed = np.load('11-performance_fixed.npy')
data_performance_benchmark2 =np.load('11-performance_benchmark-2.npy')
#
# tnum = 5,  epoch = 2000,  K = 0.5
# data_performance_active = np.load('10-performance.npy')
# data_performance_fixed = np.load('10-performance_fixed.npy')
# data_performance_benchmark2 =np.load('10-performance_benchmark-2.npy')





print(snrdBvec)
print(data_performance_active)
print(data_performance_fixed)
print(data_performance_benchmark2)

angle_degree_active = np.sqrt(data_performance_active)*180/np.pi
angle_degree_fixed = np.sqrt(data_performance_fixed)*180/np.pi
angle_degree_benchmark2 = np.sqrt(data_performance_benchmark2)*180/np.pi

 # Create a figure and axis
fig, ax = plt.subplots()
# Plot data

ax.semilogy(snrdBvec, angle_degree_active, label='Active prediction scheme', marker='o')
ax.semilogy(snrdBvec, angle_degree_fixed, label='Fixed beamforming scheme', marker='*')
ax.semilogy(snrdBvec, angle_degree_benchmark2, label='Adaptive design scheme', marker='>')
# Set labels and title
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('angular deviation (degree)')
ax.set_title('angle degree vs SNR')
# Display legend
ax.legend()
# Show the plot
plt.show()