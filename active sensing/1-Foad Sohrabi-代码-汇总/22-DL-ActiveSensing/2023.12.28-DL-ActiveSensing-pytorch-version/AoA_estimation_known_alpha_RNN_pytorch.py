import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from func_codedesign_cont import func_codedesign_cont

'设置随机种子'
# seed = 116
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(seed)

'System Information'
M_antanna_num = 64      # Number of BS's antennas, 对应论文中的 Mr 基站的天线个数
N_grid_point_num = 128  # Number of posterior intervals inputed to DNN
N_grid_point_num = 32  # Number of posterior intervals inputed to DNN
S = np.log2(N_grid_point_num)   # 波束模式的分级 数
tau = int(2*S)  # Pilot length
delta = 1/N_grid_point_num

OS_rate = 20  # Over sampling rate in each AoA interval (N_s in the paper)
delta_inv_OS = OS_rate * N_grid_point_num  # Total number of AoAs for posterior computation
delta_OS = 1/delta_inv_OS

'Channel Information'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)  # Upper-bound of AoAs
num_SNR = 8  # Number of considered SNRs
low_SNR_idx = 0  # Index of Lowest SNR for training
high_SNR_idx = 8  # Index of highest SNR for training + 1
idx_SNR_val = 7  # Index of SNR for validation (saving parameters)

snrdBvec = np.linspace(start=-10, stop=25, num=num_SNR)  # Set of SNRs
Pvec = 10 ** (snrdBvec / 10)  # Set of considered TX powers
mean_true_alpha = 0.0 + 0.0j  # Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)  # STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5)  # STD of the Gaussian noise per real dim.
#####################################################
#####################################################
#####################################################
'Learning Parameters'
initial_run = 1  # 0: Continue training; 1: Starts from the scratch

n_epochs = 10000  # Num of epochs
n_epochs = 20  # Num of epochs
learning_rate = 0.0001  # Learning rate
batch_per_epoch = 10  # Number of mini batches per epoch
batch_size_order = 8  # Mini_batch_size = batch_size_order*delta_inv
batch_size = batch_size_order * N_grid_point_num

val_size_order = 782  # Validation_set_size = val_size_order*delta_inv
scale_factor = 1  # Scaling the number of tests
test_size_order = 782  # Test_set_size = test_size_order*delta_inv*scale_factor

######################################################  构造 阵列响应
'Constructing the array responses for AoA samples' # 产生 实际的天线阵列响应矩阵  和  AOA 角度
A_BS, phi_set, A_BS_OS, phi_OS_set = func_codedesign_cont(N_grid_point_num, delta_inv_OS, phi_min, phi_max,
                                                          M_antanna_num)
A_BS = torch.tensor(A_BS, dtype=torch.complex64)
phi_set = torch.tensor(phi_set, dtype=torch.float32)

########################################################
##################### NETWORK  PyTorch Model
'Channel Sensing'

class My_RNN(nn.Module):
    def __init__(self, M_antenna_num):
        super(My_RNN, self).__init__()

        self.hidden_size = 512
        hidden_size = self.hidden_size
        input_size = 3

        self.fc1 = nn.Linear(hidden_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)   # 添加 Batch Normalization
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 2*M_antenna_num)
        '#########################################'
        self.layer_Af = nn.Linear(input_size, hidden_size)
        self.layer_Uf = nn.Linear(hidden_size, hidden_size)

        self.layer_Ai = nn.Linear(input_size, hidden_size)
        self.layer_Ui = nn.Linear(hidden_size, hidden_size)

        self.layer_Ao = nn.Linear(input_size, hidden_size)
        self.layer_Uo = nn.Linear(hidden_size, hidden_size)

        self.layer_Ac = nn.Linear(input_size, hidden_size)
        self.layer_Uc = nn.Linear(hidden_size, hidden_size)
        '#########################################'
    def RNN(self, input_x, h_old, c_old):
        # input_x 对应的是  接收的  上行  导频信号,对应图 1中的 y_t
        # h_old  对应的是  新输入的 当前状态信息矢量， 对应图 1中的 s_{t-1}
        # c_old 对应的是  上一个时刻的 cell state vector,对应图 1中的 c_{t-1}
        f_t = torch.sigmoid(self.layer_Af(input_x) + self.layer_Uf(h_old))
        i_t = torch.sigmoid(self.layer_Ai(input_x) + self.layer_Ui(h_old))
        o_t = torch.sigmoid(self.layer_Ao(input_x) + self.layer_Uo(h_old))

        c_tanh = torch.tanh(self.layer_Ac(input_x) + self.layer_Uc(h_old))
        c_t = f_t * c_old + i_t * c_tanh
        h_new = o_t * torch.tanh(c_t)
        return h_new, c_t
        # 输出的    h_new  为 s_t


    def forward(self, M_antenna_num, N_grid_point_num, tau, noiseSTD_per_dim,
                batch_size, alpha_batch, phi_batch, P_temp):

        hidden_size = self.hidden_size

        # Array_response_construction 产生a_phi
        phi = torch.reshape(torch.tensor(phi_batch, dtype=torch.float32), [-1, 1])
        phi_expanded = phi.repeat(1, M_antanna_num)  # Expand phi along the second dimension
        from0toM = torch.arange(0, M_antanna_num, 1, dtype=torch.float32)
        a_phi = torch.exp(1j * np.pi * torch.multiply(torch.sin(phi_expanded), from0toM)).to(
            dtype=torch.complex64)  # torch.mul按位相乘，输出是一个 复数

        snr_linear = P_temp * torch.ones((batch_size, 1), dtype=torch.float32)
        snr_dB = torch.log10(snr_linear)  # Log base 10
        snr_normal = (snr_dB - 1.0) / torch.sqrt(torch.tensor(1.6666, dtype=torch.float32))

        W_dict = []
        posterior_dict = []
        idx_est_dict = []
        '进行后面的操作'
        for t in range(tau):# 表示 RNN 网络 有 tau 个 LSTM 个 cell
            'DNN designs the next sensing direction'
            if t == 0:
                y_real = torch.ones([batch_size, 2])  # 对应的是  接收的  上行  导频信号,对应图1中的 y_t
                h_old = torch.zeros([batch_size, hidden_size])  # 新输入的 当前状态信息矢量， 对应图1中的 s_{t-1}
                c_old = torch.zeros([batch_size, hidden_size])  # 上一个时刻的 cell state vector,对应图1中的 c_{t-1}

            rnn_input = torch.cat([y_real, snr_normal], dim=1)
            h_old, c_old = self.RNN(rnn_input, h_old, c_old)

            'DNN designs the next sensing direction'
            x1 = self.fc1(h_old)
            x1 = F.relu(x1)
            x1 = self.bn1(x1)

            x2 = self.fc2(x1)
            x2 = F.relu(x2)
            x2 = self.bn2(x2)

            x3 = self.fc3(x2)
            x3 = F.relu(x3)
            x3 = self.bn3(x3)

            x4 = self.fc4(x3)
            # x = torch.matmul(  x, torch.ones( (N_grid_point_num+2, M_antenna_num*2), dtype=torch.float32)  )

            w_her = x4
            w_norm = torch.norm(w_her, dim=1).view(-1, 1)
            w_her = w_her / w_norm

            w_her_real = w_her[:, 0:M_antenna_num]  # Extract the real part
            w_her_imag = w_her[:, M_antenna_num:2*M_antenna_num]  # Extract the imaginary part
            # w_her_complex = w_her_real + 1j * w_her_imag
            w_her_complex = torch.complex(w_her_real, w_her_imag)

            'BS observes the next measurement'
            y_noiseless_complex = torch.sum(torch.multiply(w_her_complex, a_phi), dim=1, keepdim=True)
            noise_real = torch.normal(mean=0.0, std=noiseSTD_per_dim, size=y_noiseless_complex.size())
            noise_imag = torch.normal(mean=0.0, std=noiseSTD_per_dim, size=y_noiseless_complex.size())
            noise_complex = torch.complex(noise_real, noise_imag)
            # noise_real = torch.ones(y_noiseless_complex.size(),dtype=torch.float32)
            # noise_imag = torch.ones(y_noiseless_complex.size(),dtype=torch.float32)
            # noise_complex = torch.complex(noise_real, noise_imag)

            y_complex = (torch.sqrt(P_temp)+1j*0.0) * torch.multiply(y_noiseless_complex,alpha_batch) + noise_complex  # multiply 点乘，元素对元素的乘法
            y_real = torch.cat([torch.real(y_complex), torch.imag(y_complex)], dim=1) / torch.sqrt(P_temp)  # 输入新的 y

        h_old, c_old = self.RNN(torch.cat([y_real, snr_normal], dim=1), h_old, c_old)  # 表示 最后一个 LSTM cell

        phi_hat_layer = nn.Linear(hidden_size, 1)
        phi_hat = phi_hat_layer(c_old)   #  估计得到的  环境参数

        phi_batch_new = torch.reshape(torch.tensor(phi_batch, dtype=torch.float32), [-1, 1])
        'Loss Function'
        criterion = nn.MSELoss()
        loss = criterion(phi_batch_new[:,0], phi_hat[:,0])  # 或者使用  F.mse_loss

        return loss

'Initializes the model architecture --> instantiation '
my_Rnn = My_RNN(M_antenna_num=M_antanna_num)
'Define optimizer'
optimizer = optim.Adam(params=my_Rnn.parameters(), lr=learning_rate)

#########################################################################
###########  Validation Set  # 验证集 输入的数据
batch_size_val = val_size_order * N_grid_point_num
alpha_batch_val = torch.complex(torch.normal(mean=mean_true_alpha.real, std=std_per_dim_alpha, size=(batch_size_val, 1)),
                          torch.normal(mean=mean_true_alpha.imag, std=std_per_dim_alpha, size=(batch_size_val, 1)))
phi_batch_val = np.random.uniform(low=phi_min, high=phi_max, size=N_grid_point_num * val_size_order)


#########################################################################
###########  开始   训练
##################################################################################################
best_loss = 100
print('Initial best loss is:', best_loss)
print(torch.cuda.is_available())   #Prints whether or not GPU is on

'Begin training'
train_loss_dir = []
for epoch in range(n_epochs):
    my_Rnn.train()  # 设置为训练模式

    batch_iter = 0
    for rnd_indices in range(batch_per_epoch):   #  batch 级别需要 运行batch_per_epoch， 也就是说 有 batch_per_epoch 个 batch

        snr_idx = np.random.randint(low=low_SNR_idx, high=num_SNR, size=1)
        snr_temp = snrdBvec[snr_idx[0]]
        P_temp = 10 ** (snr_temp / 10)
        P_temp = torch.tensor(P_temp, dtype=torch.float32)  # 每一个 batch中的 功率 都不一样， 总共有

        'Known alpha situation'  # 总共有 batch_size_order 个 batch，每个 batch中包含  N_grid_point_num 个值
        alpha_batch_real = torch.normal(mean=mean_true_alpha.real, std=std_per_dim_alpha, size=(batch_size, 1))
        alpha_batch_imag = torch.normal(mean=mean_true_alpha.imag, std=std_per_dim_alpha, size=(batch_size, 1))
        alpha_batch = torch.complex(alpha_batch_real, alpha_batch_imag)

        phi_batch = np.random.uniform(low=phi_min, high=phi_max, size=batch_size)

        # print(phi_batch.shape)
        # a
        loss = my_Rnn(M_antenna_num=M_antanna_num,
                      N_grid_point_num=N_grid_point_num,
                      tau=tau,
                      noiseSTD_per_dim=noiseSTD_per_dim,
                      batch_size=batch_size,
                      alpha_batch=alpha_batch,
                      phi_batch=phi_batch,
                      P_temp=P_temp)
        'Optimizer'
        optimizer.zero_grad()   # 手动清空过往梯度,  在每次的 loss.backward() 之前使用
        loss.backward()         # 反向传播
        optimizer.step()        # 更新模型参数

        batch_iter += 1

    loss_val = my_Rnn(M_antenna_num=M_antanna_num,
                      N_grid_point_num=N_grid_point_num,
                      tau=tau,
                      noiseSTD_per_dim=noiseSTD_per_dim,
                      batch_size=batch_size_val,
                      alpha_batch=alpha_batch_val,
                      phi_batch=phi_batch_val,
                      P_temp=torch.tensor(Pvec[idx_SNR_val], dtype=torch.float32) )
    print('epoch', epoch, '  loss_test:%2.5f' % loss_val, '  best_test:%2.5f' % best_loss)

    # if epoch % 10 == 9: #Every 10 iterations it checks if the validation performace is improved, then saves parameters
    if loss_val < best_loss:
        best_loss = loss_val
######################################################################
######################################################################
    '##################  自己写的部分：  记录 训练集的loss  ################'
    train_loss_dir.append(loss.item())  # 损失加入到列表中
with open("./train_loss.txt", 'w') as op_func:
    op_func.write(str(train_loss_dir))
######################################################################
######################################################################
'###############  测试集  ###############'
###########  Part of Final Test Set
batch_size_test = test_size_order * N_grid_point_num

'###########  Final Test  ###########'
performance = np.zeros([len(snrdBvec), scale_factor])
'***********************************************************************************************************************'
'***********************************************************************************************************************'
'##########   将模型设置为评估模式   ###########'

my_Rnn.eval()  # 将模型设置为评估模式
with torch.no_grad():  # 在评估模式下不需要计算梯度
    for j in range(scale_factor):
        print(j)
        'Known alpha situation'  # 总共有 batch_size_order 个 batch，每个 batch中包含  N_grid_point_num 个值
        alpha_test_real = torch.normal(mean=mean_true_alpha.real, std=std_per_dim_alpha, size=(batch_size_test, 1))
        alpha_test_imag = torch.normal(mean=mean_true_alpha.imag, std=std_per_dim_alpha, size=(batch_size_test, 1))

        alpha_test = torch.complex(alpha_test_real, alpha_test_imag)

        phi_test = np.random.uniform(low=phi_min, high=phi_max, size=batch_size_test)
        for i in range(len(snrdBvec)):
            mse_loss = my_Rnn(M_antenna_num=M_antanna_num,
                              N_grid_point_num = N_grid_point_num,
                              tau = tau,
                              noiseSTD_per_dim = noiseSTD_per_dim,
                              batch_size = batch_size_test,
                              alpha_batch = alpha_test,
                              phi_batch = phi_test,
                              P_temp = torch.tensor(Pvec[i], dtype=torch.float32))

            performance[i, j] = mse_loss

        performance = np.mean(performance, axis=1)

'######### Plot the test result  #########'
plt.semilogy(snrdBvec, performance)

plt.xlabel('SNR (dB)')
plt.ylabel('Average MSE')
# plt.legend()

plt.grid()
plt.show()

sio.savemat('data_RNN_coherent.mat',
            dict(performance=performance,
                 snrdBvec=snrdBvec,
                 N=M_antanna_num,
                 delta_inv=N_grid_point_num,
                 mean_true_alpha=mean_true_alpha,
                 std_per_dim_alpha=std_per_dim_alpha,
                 noiseSTD_per_dim=noiseSTD_per_dim,
                 tau=tau))




