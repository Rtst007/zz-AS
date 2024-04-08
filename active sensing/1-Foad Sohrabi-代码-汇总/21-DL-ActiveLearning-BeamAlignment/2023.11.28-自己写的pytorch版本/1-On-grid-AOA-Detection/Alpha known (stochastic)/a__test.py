
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from func_codedesign import func_codedesign

N_grid_point_num = 128
batch_size_order = 2

phi_idx_batch_np = np.tile(list(range(0, N_grid_point_num)), batch_size_order)

posteriors_normal = torch.ones((2,1))


snr_normal = 10 * torch.ones( (2, 1), dtype=torch.float32 )


time_idx_normal = 0 * torch.ones( (2, 1), dtype=torch.float32 )

aa = torch.cat([posteriors_normal, snr_normal, time_idx_normal], dim=1)

phi_idx_est_dict = []
for i in range(10):
    phi_idx_est_dict.append(torch.tensor([1]))

a = torch.tensor([[ 0, 1, 2, 3],
        [2, 4, 8 ,1],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])


snr_idx = np.random.randint(low=7, high=9, size=1)

cc = np.sum([[0, 1, 2]], axis=0)


print(cc)

# print(phi_idx_est_dict)
print('-------------')

phi_idx_est_dict = torch.stack(phi_idx_est_dict, dim=1)

print(phi_idx_est_dict)