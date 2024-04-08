import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from func_codedesign import func_codedesign

'设置随机种子'
# seed = 116
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(seed)


'System Information'
M_antanna_num = 64              #Number of BS's antennas
N_grid_point_num = 128          # Number of posterior inputed to DNN，也是 AOA网格划分点的个数， AOA离散值的个数
S = np.log2(N_grid_point_num)   # 波束模式的分级 数
tau = int(2*S)                  #Pilot length
delta = 1 / N_grid_point_num

# M_antenna_num = 8
# N_grid_point_num = 2

'Channel Information'
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
'Learning Parameters'
initial_run = 1 #0: Continue training 从保存的网络中跑; 1: Starts from the scratch 从头开始跑

n_epochs = 10000        #Num of epochs
learning_rate = 0.0001  #Learning rate
batch_num_per_epoch = 10    #Number of mini batches per epoch，表示mini batches的个数
batch_size_order = 32   #Mini_batch_size = batch_size_order * N_grid_point_num
batch_size = batch_size_order * N_grid_point_num
# 假设输入的 X 为 32行， 10列的数， DNN中的训练参数 W 为 10x1024, b为 1x1024
# 因此 X的 batch size 为32， 不管 列数是10 还是 100， batch size 一直是 32， batch size 本质含义是 一次性 送进去 多少数据运行
val_size_order = 782    #Validation_set_size = val_size_order * N_grid_point_num
scale_factor = 1        #Scaling the number of tests
test_size_order = 782   #Test_set_size = test_size_order * N_grid_point_num * scale_factor


######################################################  构造 阵列响应
'Constructing the real array responses for AoA candidates'  # 产生 实际真实的天线阵列响应矩阵  和  AOA 角度
A_BS, phi_set = func_codedesign(N_grid_point_num, phi_min, phi_max, M_antanna_num)
A_BS = torch.tensor(A_BS, dtype=torch.complex64)
phi_set = torch.tensor(phi_set, dtype=torch.float32)

########################################################
##################### NETWORK  PyTorch Model
class My_4layer_DNN(nn.Module):
    def __init__(self, N_grid_point_num, M_antenna_num):
        super(My_4layer_DNN, self).__init__()

        self.fc1 = nn.Linear(N_grid_point_num+2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)   # 添加 Batch Normalization
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 2*M_antenna_num)

        self.M_antenna_num = M_antenna_num

    def forward(self, x):
        M_antenna_num = self.M_antenna_num
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.fc4(x)

        w_her = x
        w_norm = torch.norm(w_her, dim=1).view(-1, 1)
        w_her = w_her / w_norm

        w_her_real = w_her[:, 0:M_antenna_num]  # Extract the real part
        w_her_imag = w_her[:, M_antenna_num:2*M_antenna_num]  # Extract the imaginary part
        # w_her_complex = w_her_real + 1j * w_her_imag
        w_her_complex = torch.complex(w_her_real, w_her_imag)
        return w_her_complex

'Initializes the model architecture --> instantiation '
my_4layer_dnn = My_4layer_DNN(N_grid_point_num=N_grid_point_num, M_antenna_num=M_antanna_num)
'Define optimizer'
optimizer = optim.Adam(params=my_4layer_dnn.parameters(), lr=learning_rate)

#########################################################################
###########  Part of Batch Set    训练集
##  训练标签
idx_batch_np = np.tile(list(range(0, N_grid_point_num)), batch_size_order)  # list() 表示 创建列表
idx_batch = torch.tensor(idx_batch_np, dtype=torch.int64)   # 数据格式为：torch.int64
##  训练数据
phi_batch = phi_set[idx_batch_np]
phi_batch = phi_batch.reshape(-1,1)

'Array_response_construction 产生a_phi'
phi_expanded = phi_batch.repeat(1, M_antanna_num)  # Expand phi along the second dimension
from0toM = torch.arange(0, M_antanna_num, 1, dtype=torch.float32)
a_phi = torch.exp(1j * np.pi * torch.multiply(torch.sin(phi_expanded), from0toM)).to(dtype=torch.complex64)  # torch.mul按位相乘，输出是一个 复数

##################################################################################################
print(torch.cuda.is_available())   #Prints whether or not GPU is on

'Begin training'
for epoch in range(n_epochs):
    my_4layer_dnn.train()  # 设置为训练模式
    batch_iter = 0

    for rnd_indices in range(batch_num_per_epoch):   #  batch 级别需要 运行batch_per_epoch， 也就是说 有 batch_per_epoch 个 batch

        P_temp = torch.tensor(1.0, dtype=torch.float32)
        posteriors = (1/N_grid_point_num) * torch.ones((batch_size, N_grid_point_num), dtype=torch.float32)

        'Known alpha situation'  # 总共有 batch_size_order 个 batch，每个 batch中包含  N_grid_point_num 个值
        alpha_batch_real = torch.normal(mean=mean_true_alpha.real, std=std_per_dim_alpha, size=(batch_size, 1))
        alpha_batch_imag = torch.normal(mean=mean_true_alpha.imag, std=std_per_dim_alpha, size=(batch_size, 1))
        alpha_batch = torch.complex(alpha_batch_real, alpha_batch_imag)

        for t in range(tau):
            # when t==0, initial posteriors is part of DNN input.
            posteriors_normal = (posteriors-0.5) / torch.sqrt(torch.tensor(1.0/12.0, dtype=torch.float32))

            snr_linear = P_temp * torch.ones( (batch_size, 1), dtype=torch.float32 )
            snr_dB = torch.log10(snr_linear)  # Log base 10
            snr_normal = (snr_dB-1.0)/torch.sqrt(torch.tensor(1.6666, dtype=torch.float32))

            time_idx = t * torch.ones((batch_size, 1), dtype=torch.float32)
            time_idx_normal = (time_idx-6.5)/torch.sqrt(torch.tensor(16.25, dtype=torch.float32))

            'DNN designs the next sensing direction'
            dnn_input_x = torch.cat([posteriors_normal, snr_normal, time_idx_normal], dim=1)
            w_her_complex = my_4layer_dnn(dnn_input_x)   # 送入网络

            'BS observes the next measurement'
            y_noiseless_complex = torch.sum(torch.multiply(w_her_complex, a_phi), dim=1, keepdim=True)
            noise_real = torch.normal(mean=0.0, std=noiseSTD_per_dim, size=y_noiseless_complex.size())
            noise_imag = torch.normal(mean=0.0, std=noiseSTD_per_dim, size=y_noiseless_complex.size())
            noise_complex = torch.complex(noise_real, noise_imag)

            y_complex = (torch.sqrt(P_temp)+1j*0.0) * torch.multiply(y_noiseless_complex,alpha_batch) + noise_complex  # multiply 点乘，元素对元素的乘法
            y_complex = y_complex.repeat(1, N_grid_point_num)

            'BS updates the posterior distribution'
            mean_complex = (torch.sqrt(P_temp)+1j*0.0) * alpha_batch * torch.matmul(w_her_complex, A_BS)
            unnorm_post = torch.exp(-torch.pow(torch.abs(y_complex - mean_complex), 2))
            posteriors_temp = torch.multiply(posteriors, unnorm_post)

            sum_posteriors_temp = torch.sum(posteriors_temp, dim=1).view(-1, 1) + 0.00000001 #added to avoid numerical errors；  axis=1 是 N_grid_num 维度叠加，对应公式（10）和（16）对phi的连续时积分或离散时累加
            posteriors = torch.div(posteriors_temp,sum_posteriors_temp)  # torch.div 元素级的除法

            ##########################################
            # snr_idx = np.random.randint(low=low_SNR_idx, high=num_SNR, size=1)
            # snr_temp = snrdBvec[snr_idx[0]]
            # P_temp = 10 ** (snr_temp / 10)
            # P_temp = torch.tensor(P_temp, dtype=torch.float32)  # 每一个 batch中的 功率 都不一样， 总共有

        # Loss Function
        logits_phi = torch.log(posteriors + 0.00000001)
        # print(logits_phi.dtype)
        # a
        loss = nn.functional.cross_entropy(input=logits_phi, target=idx_batch, reduction='mean')

        print(loss)
        a = 1

        # Optimizer
        optimizer.zero_grad()  # 手动清空过往梯度,  在每次的 loss.backward() 之前使用
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        # optimizer.zero_grad()  # 手动清零梯度

        batch_iter += 1

    print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item()}')

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item()}')




