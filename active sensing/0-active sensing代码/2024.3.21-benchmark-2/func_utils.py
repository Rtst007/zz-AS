# -*- coding: utf-8 -*-
# 
# Author: Xiangming Cai
# Date:   2024.1.19
# 
# Last Modified time: 2024.1.19


import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def data_generator(args):
    snr_idx = np.random.randint(low=args.low_SNR_idx, high=args.high_SNR_idx, size=1)
    snr_temp = args.snrdBvec[snr_idx[0]]
    P_temp_train = 10 ** (snr_temp / 10)

    P_temp_train = torch.tensor(P_temp_train, dtype=torch.float32)  # 每一个 batch中的 功率 都不一样， 总共有

    'Known alpha situation'  # 总共有 batch_size_order 个 batch，每个 batch中包含  N_grid_point_num 个值
    alpha_batch_train_real = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(args.batch_size_train, 1))
    alpha_batch_train_imag = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(args.batch_size_train, 1))
    alpha_batch_train = torch.complex(alpha_batch_train_real, alpha_batch_train_imag)

    phi_batch_train = np.random.uniform(low=args.phi_min, high=args.phi_max, size=args.batch_size_train)
    phi_batch_train = torch.tensor(phi_batch_train, dtype=torch.float32).view(-1, 1)   

    return P_temp_train.to(device), alpha_batch_train.to(device), phi_batch_train.to(device)


def train_model(model, criterion, optimizer, args):
    model.train()
    loss_ave = 0.0
    for _ in range(args.batch_per_epoch):  # batch 级别需要 运行batch_per_epoch， 也就是说 有 batch_per_epoch 个 batch

        P_temp_train, alpha_batch_train, phi_batch_train = data_generator(args)
        optimizer.zero_grad()  # 手动清空过往梯度, 清空梯度缓存, 在每次的 loss.backward() 之前使用
        loss = model(alpha_batch=alpha_batch_train,
                     phi_batch=phi_batch_train,
                     P_temp=P_temp_train,
                     args=args,
                     criterion=criterion)
        'Loss Function'
        loss_ave += loss.item()  # 或者使用  F.mse_loss
        'Optimizer'
        loss.backward()  # 反向传播  Backpropagation
        optimizer.step()  # 更新模型参数，更新权重

    loss_ave /= args.batch_per_epoch

    return loss_ave

def validate_model(model, criterion, args):
    model.eval()
    validate_loss = 0.0
    with torch.no_grad():
        for _ in range(args.batch_per_epoch):  # batch 级别需要 运行batch_per_epoch， 也就是说 有 batch_per_epoch 个 batch

            alpha_val_real = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(args.batch_size_val, 1))
            alpha_val_imag = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(args.batch_size_val, 1))
            alpha_val = torch.complex(alpha_val_real, alpha_val_imag)

            phi_val = np.random.uniform(low=args.phi_min, high=args.phi_max, size=args.batch_size_val)
            phi_val = torch.tensor(phi_val, dtype=torch.float32).view(-1, 1)

            P_temp_val = torch.tensor(args.Pvec[args.idx_SNR_val], dtype=torch.float32)

            #########
            P_temp_val, alpha_val, phi_val = P_temp_val.to(device), alpha_val.to(device), phi_val.to(device)

            loss_val = model(alpha_batch=alpha_val,
                             phi_batch=phi_val,
                             P_temp=P_temp_val,
                             args=args,
                             criterion = criterion)

            'Loss Function'
            validate_loss += loss_val.item()  # 或者使用  F.mse_loss

        validate_loss /= args.batch_per_epoch

    return loss_val

###############################################################################################################
def test_model(model, criterion, args):

    model.eval() 
    scale_factor = 1
    with torch.no_grad():  
        performance = np.zeros([len(args.snrdBvec), scale_factor])
        for j in range(scale_factor):
            'Known alpha situation'  # 总共有 batch_size_order 个 batch，每个 batch中包含  N_grid_point_num 个值
            alpha_test_real = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(args.batch_size_test, 1))
            alpha_test_imag = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(args.batch_size_test, 1))
            alpha_test = torch.complex(alpha_test_real, alpha_test_imag)

            phi_test = np.random.uniform(low=args.phi_min, high=args.phi_max, size=args.batch_size_test)
            phi_test = torch.tensor(phi_test, dtype=torch.float32).view(-1, 1)

            for i in range(len(args.snrdBvec)):
                P_temp_test = torch.tensor(args.Pvec[i], dtype=torch.float32)

                #########
                P_temp_test, alpha_test, phi_test = P_temp_test.to(device), alpha_test.to(device), phi_test.to(device)

                test_loss = model(alpha_batch = alpha_test,
                                phi_batch = phi_test,
                                P_temp = P_temp_test,
                                args = args,
                                criterion = criterion)
                
                test_loss = test_loss.item()  # 或者使用  F.mse_loss

                performance[i, j] = test_loss

            performance = np.mean(performance, axis=1)

    return args.snrdBvec, performance

if __name__ == "__main__":

    P_temp_train, alpha_batch_train, phi_batch_train = data_generator()
    # print(alpha_batch_train)
    # print(alpha_batch_train.size())
    # print(phi_batch_train.size())
    # print(P_temp_train)




