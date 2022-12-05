# -*- coding: utf-8 -*-


import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np
from mathlib import math, parameter

from SCF_calc import A_size, hdiag, delta_hdiag2, TDDFT_matrix_vector, show_memory_info
from dump_yaml import fill_dictionary


def TDDFT_solver(N_states, initial_guess, preconditioner,
                matrix_vector_product = TDDFT_matrix_vector,
                               A_size = A_size,
                                hidag = delta_hdiag2,
                                  max = 35,
                             conv_tol = 1e-5,
                         extrainitial = 8):
    '''
    [A  B] X - [1  0] Y Ω = 0
    [B  A] Y   [0 -1] X   = 0
    '''
    print('====== TDDFT Calculation Starts ======')
    Davidson_dict = {}
    Davidson_dict['iteration'] = []
    iteration_list = Davidson_dict['iteration']

    TDDFT_start = time.time()

    size_old = 0

    size_new = min([N_states+extrainitial, 2*N_states, A_size])

    max_N_mv = max * N_states + size_new
    V_holder = np.zeros((A_size, max_N_mv))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    VU1_holder = np.zeros((max_N_mv,max_N_mv))
    VU2_holder = np.zeros_like(VU1_holder)
    WU1_holder = np.zeros_like(VU1_holder)
    WU2_holder = np.zeros_like(VU1_holder)

    VV_holder = np.zeros_like(VU1_holder)
    VW_holder = np.zeros_like(VU1_holder)
    WW_holder = np.zeros_like(VU1_holder)

    init_start = time.time()

    (V_holder,
    W_holder,
    size_new,
    initial_energies,
    X_ig,
    Y_ig) = initial_guess(V_holder = V_holder,
                          W_holder = W_holder,
                          N_states = size_new)

    init_end = time.time()
    init_time = init_end - init_start

    initial_energies = initial_energies[:N_states]

    print('size_new =', size_new)
    print('initial_energies')
    print(initial_energies)
    print('initial guess done')

    Pcost = 0
    for ii in range(max):
        print()
        print('Iteration', ii+1)
        show_memory_info('beginning of step '+ str(ii))

        V = V_holder[:,:size_new]
        W = W_holder[:,:size_new]

        '''
        U1 = AV + BW
        U2 = AW + BV
        '''
        U1_holder[:, size_old:size_new], U2_holder[:, size_old:size_new] = TDDFT_matrix_vector(
                                                                V[:, size_old:size_new],
                                                                W[:, size_old:size_new])

        U1 = U1_holder[:,:size_new]
        U2 = U2_holder[:,:size_new]

        (sub_A, sub_B, sigma, pi,
        VU1_holder, WU2_holder, VU2_holder, WU1_holder,
        VV_holder, WW_holder, VW_holder) = math.gen_sub_ab(
                      V_holder, W_holder, U1_holder, U2_holder,
                      VU1_holder, WU2_holder, VU2_holder, WU1_holder,
                      VV_holder, WW_holder, VW_holder,
                      size_old, size_new)

        print('subspace size:', sub_A.shape[0])

        omega, x, y = math.TDDFT_subspace_eigen_solver(a = sub_A,
                                                       b = sub_B,
                                                   sigma = sigma,
                                                      pi = pi,
                                                       k = N_states)

        '''
        compute the residual
        R_x = U1x + U2y - X_full*omega
        R_y = U2x + U1y + Y_full*omega
        X_full = Vx + Wy
        Y_full = Wx + Vy
        '''

        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega

        residual = np.vstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        max_norm = np.max(r_norms)
        print('Maximum residual norm: {:.2e}'.format(max_norm))
        if max_norm < conv_tol or ii == (max -1):
            print('TDDFT precedure Done')
            break
        # index = [r_norms.index(i) for i in r_norms if i > conv_tol]
        index = [i for i,R in enumerate(r_norms) if R > conv_tol]
        print('unconverged states', index)

        P_start = time.time()
        X_new, Y_new = preconditioner(R_x[:,index],
                                      R_y[:,index],
                                      omega = omega[index])
        P_end = time.time()
        Pcost += P_end - P_start

        size_old = size_new
        V_holder, W_holder, size_new = math.VW_Gram_Schmidt_fill_holder(
                                        V_holder = V_holder,
                                        W_holder = W_holder,
                                               m = size_old,
                                           X_new = X_new,
                                           Y_new = Y_new)

        if size_new == size_old:
            print('All new guesses kicked out during GS orthonormalization')
            break

    omega = omega*parameter.Hartree_to_eV


    energy_diff = float(np.average((initial_energies - omega)**2)**0.5)

    XY_overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full))
                    + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    TDDFT_end = time.time()
    TDDFT_cost = TDDFT_end - TDDFT_start

    Davidson_dict = fill_dictionary(Davidson_dict,
                           N_itr = ii+1,
                           pcost = Pcost,
                           icost = init_time,
                            N_mv = sub_A.shape[0],
                       wall_time = TDDFT_cost,
                  final_solution = omega.tolist(),
                initial_solution = initial_energies.tolist(),
                      difference = energy_diff,
                         overlap = XY_overlap)
    if ii == (max -1):
        print('===== TDDFT Failed Due to Iteration Limit============')
        print('current residual norms', r_norms)
    else:
        print('============= TDDFT Calculation Done ==============')
    print('energies:')
    print(omega)
    print('Maximum residual norm = {:.2e}'.format(max_norm))
    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, TDDFT_cost))
    print('Final subspace size = {:d}'.format(sub_A.shape[0]))
    print('Initial guess time : {:.2f} seconds, {:.2%}'.format(init_time, init_time/TDDFT_cost))
    print('Total precondition time: {:.2f} seconds, {:.2%}'.format(Pcost, Pcost/TDDFT_cost))

    show_memory_info('after TDDFT done')
    return omega, X_full, Y_full, Davidson_dict
