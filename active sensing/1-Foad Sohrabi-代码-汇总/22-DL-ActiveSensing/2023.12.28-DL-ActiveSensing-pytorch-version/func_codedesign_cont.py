

import numpy as np
#Constructing the array responses for AoA candidates
"""
************** Input
phi_min: Lower-bound of AoAs
phi_max: Upper-bound of AoAs
N: # Antennas
delta_inv: # Coarse AoA Candidates (Intervals)
delta_inv_OS: # AoA Candidates (After over-sampling)
************** Ouput 
phi: Coarse AoA Candidates   
A_BS: Collection of array responses for Coarse AoA Candidates
phi: AoA Candidates   
A_BS: Collection of array responses for AoA Candidates
"""
def func_codedesign_cont( N_grid_point_num, delta_inv_OS, phi_min, phi_max, M_antanna_num ):
    from0toN = np.float32(list(range(0, M_antanna_num)))

    delta_theta = (phi_max-phi_min) / N_grid_point_num
    phi = np.linspace(start=phi_min+delta_theta/2, stop=phi_max-delta_theta/2, num=N_grid_point_num)
    A_BS = np.zeros([M_antanna_num, N_grid_point_num], dtype=np.complex64)
    for i in range(N_grid_point_num):
        a_phi = np.exp( 1j*np.pi*from0toN*np.sin(phi[i]) )
        A_BS[:,i] = np.transpose(a_phi)
########################################################################################

    delta_theta = (phi_max-phi_min)/delta_inv_OS
    phi_OS = np.linspace(start=phi_min+delta_theta/2,stop=phi_max-delta_theta/2,num=delta_inv_OS)  
    A_BS_OS = np.zeros([M_antanna_num, delta_inv_OS], dtype=np.complex64)
    for i in range(delta_inv_OS):
        a_phi = np.exp( 1j*np.pi*from0toN*np.sin(phi_OS[i]) )
        A_BS_OS[:,i] = np.transpose(a_phi)
        
    return A_BS, phi, A_BS_OS, phi_OS   