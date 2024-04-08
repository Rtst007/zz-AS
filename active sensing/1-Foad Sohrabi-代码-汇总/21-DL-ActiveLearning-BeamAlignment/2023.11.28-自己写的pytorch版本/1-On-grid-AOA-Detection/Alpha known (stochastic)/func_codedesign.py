
#Constructing the array responses for AoA candidates

import numpy as np

"""
************** Input
phi_min: Lower-bound of AoAs
phi_max: Upper-bound of AoAs
N: # Antennas
delta_inv: # AoA Candidates 
************** Ouput 
phi: AoA Candidates   
A_BS: Collection of array responses for AoA Candidates
"""
def func_codedesign(N_grid_point_num, phi_min, phi_max, M_antanna_num):
    phi_set = np.linspace(start=phi_min, stop=phi_max, num=N_grid_point_num)
    from0toM = np.float32(list(range(0, M_antanna_num)))
    A_BS = np.zeros([M_antanna_num, N_grid_point_num], dtype=np.complex64)
    # 这里A_BS的每一列，表示一个在该角度下的 所有M个天线的阵列响应
    for i in range(N_grid_point_num):
        a_phi = np.exp( 1j*np.pi*from0toM*np.sin(phi_set[i]) )
        A_BS[:,i] = np.transpose(a_phi)      
        
    return A_BS, phi_set
# 这里A_BS的每一列，表示一个在该角度下的 所有M个天线的阵列响应
# phi_set 表示 AOA 角度的 集合， 总共有 N_grid_point_num 个AOA值，也就是N_grid_point_num个phi值