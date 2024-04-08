import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn.functional as F

from func_codedesign import func_codedesign

# System Information
M_antenna_num = 64                  #Number of BS's antennas
N_grid_point_num = 128              # Number of posterior inputed to DNN，也是 AOA网格划分点的个数， AOA离散值的个数

# M_antenna_num = 8
# N_grid_point_num = 2

S = int(np.log2(N_grid_point_num))  # 波束模式的分级 数
tau = int(2 * S)                    #Pilot length
delta = 1 / N_grid_point_num

# Channel Information
phi_min = -60*(np.pi/180)   #Lower-bound of AoAs
phi_max = 60*(np.pi/180)    #Upper-bound of AoAs
num_SNR = 9                 #Number of considered SNRs
low_SNR_idx = 7             #Index of Lowest SNR for training
high_SNR_idx = 9            ##Index of highest SNR for training + 1
idx_SNR_val = 8             #Index of SNR for validation (saving parameters)

snrdBvec = np.linspace(start=-10,stop=30,num=num_SNR) #Set of SNRs
Pvec = 10**(snrdBvec/10)            #Set of considered TX powers
mean_true_alpha = 0.0 + 0.0j        #Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)    #STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5)     #STD of the Gaussian noise per real dim.
#####################################################
#####################################################
#####################################################
# Learning Parameters
initial_run = 1     #0: Continue training 从保存的网络中跑; 1: Starts from the scratch 从头开始跑

n_epochs = 1000        #Num of epochs

learning_rate = 0.0001  #Learning rate
batch_per_epoch = 10    #Number of mini batches per epoch，表示mini batches的个数
batch_size_order = 32   #Mini_batch_size = batch_size_order * N_grid_point_num
val_size_order = 782    #Validation_set_size = val_size_order * N_grid_point_num
scale_factor = 1        #Scaling the number of tests
test_size_order = 782   #Test_set_size = test_size_order * N_grid_point_num * scale_factor
######################################################  构造 阵列响应
#Constructing the array responses for AoA candidates， 产生 实际的天线阵列响应矩阵  和  AOA 角度
A_BS, phi_set = func_codedesign(N_grid_point_num, phi_min, phi_max, M_antenna_num)

A_BS = torch.tensor(A_BS, dtype=torch.complex64)
phi_set = torch.tensor(phi_set, dtype=torch.float32)

########################################################
##################### NETWORK  PyTorch Model
class My_4layer_DNN(nn.Module):
    def __init__(self, input_size, M_antenna_num):
        super(My_4layer_DNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 2*M_antenna_num)

        self.M_antenna_num = M_antenna_num

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.fc4(x)

        w_her = x
        w_norm = torch.norm(w_her, dim=1).view(-1, 1)
        w_her = w_her / w_norm

        w_her_real = w_her[:, :self.M_antenna_num]  # Extract the real part
        w_her_imag = w_her[:, self.M_antenna_num:]  # Extract the imaginary part
        w_her_complex = w_her_real + 1j * w_her_imag

        return w_her_complex



# class Training_Model(nn.Module):
#     def __init__(self, input_size, M_antenna_num):
#         super(Training_Model, self).__init__()
#         self.dnn = My_4layer_DNN(input_size, M_antenna_num)
#
#     def forward(self, alpha_input, idx_input, phi_input, \
#                 M_antenna_num, N_grid_point_num, \
#                 lay_P, noiseSTD_per_dim, tau):
#
#         from0toM = torch.arange(0, M_antenna_num, 1, dtype=torch.float32)
#
#         phi = phi_input.reshape(-1, 1)  # Reshape phi_input to get a single value
#         h_act = {0: 0}
#         hR_act = {0: 0}
#         hI_act = {0: 0}
#
#         phi_expanded = phi.expand(-1, M_antenna_num)  # Expand phi along the second dimension
#         a_phi = torch.exp(1j * np.pi * torch.multiply(torch.sin(phi_expanded), from0toM)).to(dtype=torch.complex64)  # torch.mul按位相乘，输出是一个 复数
#
#         ##################################################################################################
#         posteriors = (1/N_grid_point_num) * torch.ones((phi_input.shape[0], N_grid_point_num), dtype=torch.float32)
#
#         w_dict = []
#         posterior_dict = []
#         idx_est_dict = []
#
#         for t in range(tau):
#             snr_linear = lay_P * torch.ones((phi_input.shape[0], 1), dtype=torch.float32)
#             snr_dB = torch.log10(snr_linear)  # Log base 10
#             snr_normal = (snr_dB-1)/torch.sqrt(torch.tensor(1.6666, dtype=torch.float32))
#
#             iter_idx = t * torch.ones((phi_input.shape[0], 1), dtype=torch.float32)
#             iter_idx_normal = (iter_idx-6.5)/torch.sqrt(torch.tensor(16.25, dtype=torch.float32))
#
#             posteriors_normal = (posteriors-0.5)/torch.sqrt(torch.tensor(1.0/12.0, dtype=torch.float32))
#
#             'DNN designs the next sensing direction'
#             dnn_input_x = torch.cat([posteriors_normal, snr_normal, iter_idx_normal], dim=1)
#
#             input_size = N_grid_point_num+2
#             my_4layer_dnn = My_4layer_DNN(input_size, M_antenna_num)
#
#             w_her_complex = my_4layer_dnn(dnn_input_x)
#             w_dict.append(w_her_complex)
#             W_her = torch.stack(w_dict, dim=1)
#
#             'BS observes the next measurement'
#             y_noiseless_complex = torch.sum(torch.mul(w_her_complex, a_phi), dim=1, keepdim=True)
#             noise_complex = torch.normal(mean=0.0, std=noiseSTD_per_dim, size=y_noiseless_complex.size()) \
#                             + 1j * torch.normal(mean=0.0, std=noiseSTD_per_dim, size=y_noiseless_complex.size())
#
#             y_complex = torch.sqrt(lay_P).to(dtype=torch.complex64) * alpha_input + noise_complex  # multiply 点乘，元素对元素的乘法
#             y_complex = y_complex.repeat(1, N_grid_point_num)
#
#             'BS updates the posterior distribution'
#             mean_complex = torch.sqrt(lay_P).to(dtype=torch.complex64) * alpha_input * torch.matmul(w_her_complex, A_BS)
#             unnorm_post = torch.exp(-torch.pow(torch.abs(y_complex - mean_complex), 2))
#             posteriors_temp = torch.multiply(unnorm_post, posteriors)
#
#             sum_posteriors_temp = torch.sum(posteriors_temp, dim=1).view(-1, 1) + 0.00000001 #added to avoid numerical errors；  axis=1 是 N_grid_num 维度叠加，对应公式（10）和（16）对phi的连续时积分或离散时累加
#             posteriors = torch.div(posteriors_temp,sum_posteriors_temp)  # torch.div 元素级的除法
#
#             posterior_dict.append(posteriors)                      # 数据格式为  torch.float32
#             idx_est_dict.append(torch.argmax(posteriors, dim=1))  # 数据格式为  torch.int64
#
#         posterior_dict = torch.stack(posterior_dict, dim=1)  # 数据格式为  torch.float32
#         idx_est_dict = torch.stack(idx_est_dict, dim=1)  # 数据格式为  torch.int64
#         idx_est = torch.argmax(posteriors, dim=1)    # 数据格式为  torch.int64
#
#         # Loss Function
#         logits_phi = torch.log(posteriors + 0.00000001)
#         logits_phi = logits_phi.reshape(1,-1)
#         # idx_input = idx_input.repeat(1, phi_input.shape[0])
#         print(idx_input.shape)
#         loss = nn.functional.cross_entropy(input=logits_phi, target=idx_input, reduction='mean')
#         loss = nn.functional.cross_entropy()
#
#
#         # Optimizer
#         optimizer = optim.Adam(params=my_4layer_dnn.parameters(),lr=learning_rate)
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新模型参数
#         optimizer.zero_grad()  # 手动清零梯度
#
#         # return loss, posterior_dict, idx_est_dict, idx_est
#         return posteriors, posterior_dict, idx_est_dict, idx_est


# M_antenna_num=8
# N_grid_point_num = 2
# input_size = N_grid_point_num + 2
#
# training_model = Training_Model(input_size, M_antenna_num)
#
# aa, bb, cc, dd = training_model(
#     alpha_input = torch.arange(N_grid_point_num, dtype=torch.float32).reshape(-1,1),  # 类型为 complex64
#     idx_input = torch.arange(2*N_grid_point_num, dtype=torch.float32).reshape(-1,2),  # 类型为 float32
#     phi_input = torch.arange(N_grid_point_num, dtype=torch.float32),     # 类型为 float32
#     M_antenna_num = M_antenna_num,
#     N_grid_point_num = N_grid_point_num,
#     lay_P = torch.tensor(1),
#     noiseSTD_per_dim = 0.5,
#     tau = 10
# )
#
# print(aa.shape)
# print('-----------------')
# print(bb)
# print('-----------------')
# print(cc)
# print('-----------------')
# print(dd)


# 定义模型
#########################################################################
###########  Part of Batch Set    训练集
idx_batch_np = np.tile(list(range(0, N_grid_point_num)), batch_size_order)  # list() 表示 创建列表
phi_batch = phi_set[idx_batch_np]
phi_batch = phi_batch.to(dtype=torch.float32)
idx_batch = torch.tensor(idx_batch_np, dtype=torch.int64)


print(torch.cuda.is_available())

for epoch in range(n_epochs):
    batch_iter = 0

    for rnd_indices in range(batch_per_epoch):
        idx_temp = np.random.randint(low=low_SNR_idx, high=num_SNR, size=1)
        snr_temp = snrdBvec[idx_temp[0]]
        P_temp = 10**(snr_temp/10)
        P_temp = torch.tensor(P_temp, dtype=torch.float32)

        # Known alpha situation
        alpha_batch_real = torch.normal(mean=mean_true_alpha.real, std=std_per_dim_alpha, size=(batch_size_order * N_grid_point_num, 1))
        alpha_batch_imag = torch.normal(mean=mean_true_alpha.imag, std=std_per_dim_alpha, size=(batch_size_order * N_grid_point_num, 1))
        alpha_batch = alpha_batch_real + 1j*alpha_batch_imag

        alpha_input = alpha_batch.view(-1,1)
        idx_input = idx_batch
        phi_input = phi_batch.view(-1,1)
        lay_P = P_temp


        M_antenna_num = M_antenna_num
        N_grid_point_num = N_grid_point_num
        noiseSTD_per_dim = noiseSTD_per_dim
        tau = 1

        ###################################################################


        from0toM = torch.arange(0, M_antenna_num, 1, dtype=torch.float32)
        # phi = phi_input  # Reshape phi_input to get a single value

        h_act = {0: 0}
        hR_act = {0: 0}
        hI_act = {0: 0}

        phi_expanded = phi_input.repeat(1, M_antenna_num)  # Expand phi along the second dimension
        a_phi = torch.exp(1j * np.pi * torch.multiply(torch.sin(phi_expanded), from0toM)).to(dtype=torch.complex64)  # torch.mul按位相乘，输出是一个 复数

        ##################################################################################################
        posteriors = (1/N_grid_point_num) * torch.ones((phi_input.shape[0], N_grid_point_num), dtype=torch.float32)


        w_dict = []
        posterior_dict = []
        idx_est_dict = []

        for t in range(tau):
            snr_linear = lay_P * torch.ones((phi_input.shape[0], 1), dtype=torch.float32)
            snr_dB = torch.log10(snr_linear)  # Log base 10
            snr_normal = (snr_dB-1)/torch.sqrt(torch.tensor(1.6666, dtype=torch.float32))

            iter_idx = t * torch.ones((phi_input.shape[0], 1), dtype=torch.float32)
            iter_idx_normal = (iter_idx-6.5)/torch.sqrt(torch.tensor(16.25, dtype=torch.float32))

            posteriors_normal = (posteriors-0.5)/torch.sqrt(torch.tensor(1.0/12.0, dtype=torch.float32))

            'DNN designs the next sensing direction'
            dnn_input_x = torch.cat([posteriors_normal, snr_normal, iter_idx_normal], dim=1)

            input_size = N_grid_point_num+2
            my_4layer_dnn = My_4layer_DNN(input_size, M_antenna_num)

            w_her_complex = my_4layer_dnn(dnn_input_x)
            w_dict.append(w_her_complex)
            W_her = torch.stack(w_dict, dim=1)

            'BS observes the next measurement'
            y_noiseless_complex = torch.sum(torch.mul(w_her_complex, a_phi), dim=1, keepdim=True)
            noise_complex = torch.normal(mean=0.0, std=noiseSTD_per_dim, size=y_noiseless_complex.size()) \
                            + 1j * torch.normal(mean=0.0, std=noiseSTD_per_dim, size=y_noiseless_complex.size())

            y_complex = torch.sqrt(lay_P).to(dtype=torch.complex64) * alpha_input + noise_complex  # multiply 点乘，元素对元素的乘法
            y_complex = y_complex.repeat(1, N_grid_point_num)

            'BS updates the posterior distribution'
            mean_complex = torch.sqrt(lay_P).to(dtype=torch.complex64) * alpha_input * torch.matmul(w_her_complex, A_BS)
            unnorm_post = torch.exp(-torch.pow(torch.abs(y_complex - mean_complex), 2))
            posteriors_temp = torch.multiply(unnorm_post, posteriors)

            sum_posteriors_temp = torch.sum(posteriors_temp, dim=1).view(-1, 1) + 0.00000001 #added to avoid numerical errors；  axis=1 是 N_grid_num 维度叠加，对应公式（10）和（16）对phi的连续时积分或离散时累加
            posteriors = torch.div(posteriors_temp,sum_posteriors_temp)  # torch.div 元素级的除法

            posterior_dict.append(posteriors)                      # 数据格式为  torch.float32
            idx_est_dict.append(torch.argmax(posteriors, dim=1))  # 数据格式为  torch.int64

        posterior_dict = torch.stack(posterior_dict, dim=1)  # 数据格式为  torch.float32
        idx_est_dict = torch.stack(idx_est_dict, dim=1)  # 数据格式为  torch.int64
        idx_est = torch.argmax(posteriors, dim=1)    # 数据格式为  torch.int64

        # Loss Function
        logits_phi = torch.log(posteriors + 0.00000001)
        loss = nn.functional.cross_entropy(input=logits_phi, target=idx_input, reduction='mean')
        # Optimizer
        optimizer = optim.Adam(params=my_4layer_dnn.parameters(),lr=learning_rate)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 手动清零梯度

        batch_iter += 1

    print('epoch is', epoch)
    print('loss is', loss.item())
    # if epoch % 20 == 0:
    #     print('epoch is', epoch)
    #     print('loss is', loss.item())


    # if epoch % 10 == 9:
    #     loss_val, idx_est_val = loss(feed_dict_val), idx_est(feed_dict_val)
    #     if loss_val < best_loss:
    #         torch.save(model.state_dict(), 'params.pth')
    #         best_loss = loss_val.item()
    #
    #     print(f'epoch {epoch}, 1 - {torch.sum(idx_est_val == idx_val).item() / len(idx_val)}')
    #     print(f'         loss_val: {loss_val:.5f},  best_test: {best_loss:.5f}')




print('-------------------------------')
print(loss)
# print(idx_input.dtype)






















