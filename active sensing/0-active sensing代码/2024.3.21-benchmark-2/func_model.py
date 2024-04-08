# -*- coding: utf-8 -*-
# 
# Author: Xiangming Cai
# Date:   2024.1.19
# 
# Last Modified time: 2024.1.19

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class My_RNN_model(nn.Module):
    def __init__(self, args):
        super(My_RNN_model, self).__init__()

        input_size = 3

        self.dnn_network = nn.Sequential(
            nn.Linear(args.hidden_size, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),  # 添加 Batch Normalization
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, 2*args.M_antenna_num)
        )

        self.dnn_network_22 = nn.Sequential(
            nn.Linear(args.hidden_size, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),  # 添加 Batch Normalization
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, 2*args.M_antenna_num)
        )

        # self.lstm = nn.LSTM(input_size=3, hidden_size=512, num_layers=1)

        self.layer_Ui = nn.Linear(input_size, args.hidden_size)
        self.layer_Wi = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_Uf = nn.Linear(input_size, args.hidden_size)
        self.layer_Wf = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_Uo = nn.Linear(input_size, args.hidden_size)
        self.layer_Wo = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_Uc = nn.Linear(input_size, args.hidden_size)
        self.layer_Wc = nn.Linear(args.hidden_size, args.hidden_size)

        self.fc_final = nn.Linear(args.hidden_size, 1)

    def name(self):
        return 'LSTM_model'

    def RNN(self, input_x, h_old, c_old):
        i_t = torch.sigmoid(self.layer_Ui(input_x) + self.layer_Wi(h_old))
        f_t = torch.sigmoid(self.layer_Uf(input_x) + self.layer_Wf(h_old))
        o_t = torch.sigmoid(self.layer_Uo(input_x) + self.layer_Wo(h_old))
        c_t = torch.tanh(self.layer_Uc(input_x) + self.layer_Wc(h_old))
        c = i_t * c_t + f_t * c_old
        h_new = o_t * torch.tanh(c)
        return h_new, c

    def forward(self, alpha_batch, phi_batch, P_temp, args, criterion):

        batch_size = phi_batch.shape[0] # 或者 phi_batch.size(0)

        snr_linear = P_temp * torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        snr_dB = torch.log10(snr_linear)  # Log base 10
        snr_normal = (snr_dB - 1.0) / np.sqrt(1.6666)

        phi_dir = []
        phi_hat_dir = []
        
        phi_batch_00 = phi_batch

        tnum = 3
        for id in range(tnum):
            if id == 0:
                phi_batch = phi_batch_00
            else:
                v0 = 0.0
                acc_speed = 20.5
                phi_batch = phi_batch_00 + 0.5 * acc_speed* id * id* (np.pi/180) * torch.ones((batch_size, 1), dtype=torch.float32, device=device)

                # phi_batch = phi_batch_00 + 5 * id * (np.pi/180) * torch.ones((batch_size, 1), dtype=torch.float32, device=device)

            tau = args.tau
            tau = 1
            for t in range(tau):  # 表示 RNN 网络 有 tau 个 LSTM cell
                'DNN designs the next sensing direction'
                if id == 0 and t == 0:  # 参数 初始化
                    y_real = torch.ones([batch_size, 2], device=device)  # 对应的是 接收的 上行 导频信号,对应图1中的 y_t
                    h_old = torch.zeros([batch_size, args.hidden_size], device=device)  # 新输入的 当前状态信息矢量，对应图1中的 s_{t-1}
                    c_old = torch.zeros([batch_size, args.hidden_size], device=device)  # 上一个时刻的cell state vector,对应图1中的 c_{t-1}
                
                    rnn_input = torch.cat([y_real, snr_normal], dim=1)
                    h_old, c_old = self.RNN(rnn_input, h_old, c_old)

                    'DNN designs the next sensing direction'
                    w_her = self.dnn_network(h_old)
                    w_norm = torch.norm(w_her, dim=1).view(-1, 1)
                    w_her = torch.divide(w_her, w_norm)  # 按 元素 逐位除
                    w_her_real, w_her_imag = w_her[:, 0:args.M_antenna_num], w_her[:, args.M_antenna_num:2*args.M_antenna_num]  
                    w_her_complex = torch.complex(real=w_her_real, imag=w_her_imag)

                    w_her_22 = self.dnn_network_22(h_old)
                    w_norm_22 = torch.norm(w_her_22, dim=1).view(-1, 1)
                    w_her_22 = torch.divide(w_her_22, w_norm_22)  # 按 元素 逐位除
                    w_her_real_22, w_her_imag_22 = w_her_22[:, 0:args.M_antenna_num], w_her_22[:, args.M_antenna_num:2*args.M_antenna_num]  
                    w_her_complex_22 = torch.complex(real=w_her_real_22, imag=w_her_imag_22)
                else:

                    phi_hat_expanded = phi_hat.repeat(1, args.M_antenna_num)  # Expand phi along the second dimension
                    from0toM = torch.arange(0, args.M_antenna_num, 1, dtype=torch.float32, device=device)
                    a_hat_phi = torch.exp(1j * np.pi * torch.multiply(torch.sin(phi_hat_expanded), from0toM)).to(dtype=torch.complex64)

                    w_her_complex = a_hat_phi
                    w_her_complex_22 = a_hat_phi

                # Array_response_construction 产生a_phi
                phi_expanded = phi_batch.repeat(1, args.M_antenna_num)  # Expand phi along the second dimension
                from0toM = torch.arange(0, args.M_antenna_num, 1, dtype=torch.float32, device=device)
                a_phi = torch.exp(1j * np.pi * torch.multiply(torch.sin(phi_expanded), from0toM)).to(dtype=torch.complex64)
               

                # Rician channel
                h_Los = a_phi
                h_NLos = torch.complex(real=torch.normal(mean=0.0, std=np.sqrt(0.5), size=a_phi.size(), device=device),
                                       imag=torch.normal(mean=0.0, std=np.sqrt(0.5), size=a_phi.size(), device=device))
                epslon = 0.5
                h_Rician = np.sqrt(epslon/(epslon+1.0))*h_Los  + np.sqrt(1.0/(epslon+1.0))*h_NLos


                h_Los_22 = a_phi
                h_NLos_22 = torch.complex(real=torch.normal(mean=0.0, std=np.sqrt(0.5), size=a_phi.size(), device=device),
                                          imag=torch.normal(mean=0.0, std=np.sqrt(0.5), size=a_phi.size(), device=device))
                h_Rician_22 = np.sqrt(epslon/(epslon+1.0))*h_Los_22  + np.sqrt(1.0/(epslon+1.0))*h_NLos_22

                # h_Rician = h_Los
                # h_Rician_22 = h_Los_22


                'BS observes the next measurement'
                wa_product = torch.sum(torch.multiply(torch.conj(h_Rician), w_her_complex), dim=1, keepdim=True)
                wa_product_22 = torch.sum(torch.multiply(torch.conj(w_her_complex_22), h_Rician_22), dim=1, keepdim=True)
                y_noiseless = torch.multiply(wa_product, wa_product_22)
                # y_noiseless = wa_product

                noise_real = torch.normal(mean=0.0, std=np.sqrt(0.5), size=y_noiseless.size(), device=device)
                noise_imag = torch.normal(mean=0.0, std=np.sqrt(0.5), size=y_noiseless.size(), device=device)
                noise_complex = torch.complex(real=noise_real, imag=noise_imag)

                # y_complex = (torch.sqrt(P_temp) + 1j * 0.0) * torch.multiply(y_noiseless, alpha_batch) + noise_complex

                y_complex = (torch.sqrt(P_temp) + 1j * 0.0) * y_noiseless + noise_complex
                y_real = torch.cat([torch.real(y_complex), torch.imag(y_complex)], dim=1) / torch.sqrt(P_temp)

            # 经过最后一个 LSTM cell
            h_old, c_old = self.RNN(torch.cat([y_real, snr_normal], dim=1), h_old, c_old)

            phi_hat = self.fc_final(c_old)

            phi_dir.append(phi_batch)
            phi_hat_dir.append(phi_hat)
        

        phi_all = torch.stack(phi_dir, axis=1)
        phi_all = torch.squeeze(phi_all, dim=-1)   
        
        phi_hat_all = torch.stack(phi_hat_dir, axis=1)
        phi_hat_all = torch.squeeze(phi_hat_all, dim=-1)

        # loss = 0.0
        # for id in range(tnum):
        #     if id <= (tnum-2):
        #         loss = loss + criterion(phi_all[:, id+1], phi_hat_all[:, id])  # 或者使用  F.mse_loss
        # loss /= (tnum-1)

        
        loss = criterion(phi_all[:, tnum-1], phi_hat_all[:, tnum-2])  # 或者使用  F.mse_loss


        return loss
    











