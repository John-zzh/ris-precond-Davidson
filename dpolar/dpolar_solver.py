# -*- coding: utf-8 -*-

'''
absolute import
'''
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np
from SCF_calc import (TDDFT_matrix_vector, A_size, show_memory_info,
                    hdiag, delta_hdiag2, n_occ, n_vir, trunced_vir, gen_P)
from mathlib import math, parameter
from dump_yaml import fill_dictionary


def gen_alpha_tensor(X, Y, pqnorm, k, P_origin):
    alpha_omega = []
    X_p_Y = X + Y
    X_p_Y *= np.tile(pqnorm, k)
    for jj in range(k):
        '''*-1 from the definition of dipole moment. *2 for double occupancy'''
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        alpha_omega.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)
    return alpha_omega

def dpolar_solver(initial_guess, preconditioner, dpolar_omega,
                                    max = 30,
                               conv_tol = 1e-5,
                            initial_TOL = 1e-3,
                            precond_TOL = 1e-2,
                  matrix_vector_product = TDDFT_matrix_vector,
                                 double = False):
    '''
    [ A  B ] - [1  0]X  w = -P
    [ B  A ]   [0 -1]Y    = -Q
    pretty same structure as TDDFT iterative precodnitioner
    (with |P,Q> as the righht hand side residuals)
    '''
    print('====== Dynamic Polarizability Calculation Starts ======')
    dp_start = time.time()

    Davidson_dict = {}
    Davidson_dict['iteration'] = []
    iteration_list = Davidson_dict['iteration']

    '''
    k is the amount of external peturbations (omegas)
    '''
    print('Wavelength we look at', dpolar_omega)
    k = len(dpolar_omega)
    omega =  np.zeros([3*k])
    for jj in range(k):
        '''
        if have 3 ω, [ω1 ω1 ω1, ω2 ω2 ω2, ω3 ω3 ω3]
        convert nm to Hartree
        '''
        omega[3*jj:3*(jj+1)] = 45.56337117/dpolar_omega[jj]

    P = gen_P()
    P = P.reshape(-1,3)

    P_origin = np.copy(P)

    '''
    normalize the right hand side
    '''

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    pqnorm = pnorm * (2**0.5)

    P /= pqnorm

    '''
    repeat k times
    '''
    P = np.tile(P,k)
    Q = np.copy(P)

    size_old = 0
    max_N_mv = (max+1)*k*3
    V_holder = np.zeros((A_size, max_N_mv))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    VU1_holder = np.zeros((max_N_mv, max_N_mv))
    VU2_holder = np.zeros_like(VU1_holder)
    WU1_holder = np.zeros_like(VU1_holder)
    WU2_holder = np.zeros_like(VU1_holder)

    VV_holder = np.zeros_like(VU1_holder)
    VW_holder = np.zeros_like(VU1_holder)
    WW_holder = np.zeros_like(VU1_holder)

    VP_holder = np.zeros((max_N_mv, P.shape[1]))
    VQ_holder = np.zeros_like(VP_holder)
    WP_holder = np.zeros_like(VP_holder)
    WQ_holder = np.zeros_like(VP_holder)
    init_start = time.time()
    X_ig, Y_ig = initial_guess(-P, -Q,
                            omega = omega,
                         conv_tol = initial_TOL)


    alpha_omega_ig = gen_alpha_tensor(X = X_ig,
                                      Y = Y_ig,
                                 pqnorm = pqnorm,
                                      k = k,
                               P_origin = P_origin)

    print('initial guess of tensor alpha')
    for i in range(k):
        print(dpolar_omega[i],'nm')
        print(alpha_omega_ig[i])

    V_holder, W_holder, size_new = math.VW_Gram_Schmidt_fill_holder(
                                            V_holder = V_holder,
                                            W_holder = W_holder,
                                                   m = 0,
                                               X_new = X_ig,
                                               Y_new = Y_ig)
    init_end = time.time()
    initial_cost = init_end - init_start
    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        print('Iteration', ii+1)
        MV_start = time.time()
        U1_holder[:,size_old:size_new], U2_holder[:,size_old:size_new] = matrix_vector_product(
                                                            X=V_holder[:,size_old:size_new],
                                                            Y=W_holder[:,size_old:size_new])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        subgenstart = time.time()
        (sub_A, sub_B, sigma, pi,
        VU1_holder, WU2_holder, VU2_holder, WU1_holder,
        VV_holder, WW_holder, VW_holder) = math.gen_sub_ab(
                      V_holder, W_holder, U1_holder, U2_holder,
                      VU1_holder, WU2_holder, VU2_holder, WU1_holder,
                      VV_holder, WW_holder, VW_holder,
                      size_old, size_new)
        p, q, VP_holder, WQ_holder, WP_holder, VQ_holder = math.gen_sub_pq(
                                V_holder, W_holder, P, Q,
                                VP_holder, WQ_holder, WP_holder, VQ_holder,
                                size_old, size_new)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        print('sub_A.shape', sub_A.shape)

        subcost_start = time.time()
        x, y = math.TDDFT_subspace_liear_solver(sub_A, sub_B, sigma, pi, -p, -q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R_x = U1x + U2y - X_full*omega + P
           R_y = U2x + U1y + Y_full*omega + Q
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''
        V = V_holder[:,:size_new]
        W = W_holder[:,:size_new]
        U1 = U1_holder[:,:size_new]
        U2 = U2_holder[:,:size_new]

        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega
        R_x += P

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega
        R_y += Q

        residual = np.vstack((R_x,R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        max_norm = np.max(r_norms)
        print('maximum residual norm: {:.2e}'.format(max_norm))
        if max_norm < conv_tol or ii == (max -1):
            break
        index = [r_norms.index(i) for i in r_norms if i > conv_tol]

        Pstart = time.time()
        X_new, Y_new = preconditioner(R_x[:,index],
                                      R_y[:,index],
                                   omega = omega[index],
                                conv_tol = precond_TOL)
        Pend = time.time()
        Pcost += Pend - Pstart

        size_old = size_new
        GS_start = time.time()
        V_holder, W_holder, size_new = math.VW_Gram_Schmidt_fill_holder(
                                            V_holder = V_holder,
                                            W_holder = W_holder,
                                               X_new = X_new,
                                               Y_new = Y_new,
                                                   m = size_old,
                                              double = double)
        GS_end = time.time()
        GScost += GS_end - GS_start

        if size_new == size_old:
            print('All new guesses kicked out during GS orthonormalization')
            break

    dp_end = time.time()
    dp_cost = dp_end - dp_start

    if ii == (max -1):
        print('=== Dynamic polarizability Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)

    else:
        print('Dynamic polarizability Done')

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, dp_cost))
    print('final subspace', sub_A.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))
    for enrty in ['initial_cost','MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/dp_cost))




    X_overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full))
                    + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    alpha_omega = gen_alpha_tensor(X = X_full,
                                   Y = Y_full,
                              pqnorm = pqnorm,
                                   k = k,
                            P_origin = P_origin)


    print('tensor alpha')
    for i in range(k):
        print(dpolar_omega[i],'nm')
        print(alpha_omega[i])

    alpha_difference = 0
    for i in range(k):
        alpha_difference += np.mean((alpha_omega_ig[i] - alpha_omega[i])**2)

    alpha_difference = float(alpha_difference)

    show_memory_info('Total Dynamic polarizability')

    Davidson_dict = fill_dictionary(Davidson_dict,
                           N_itr = ii+1,
                           pcost = Pcost,
                           icost = initial_cost,
                            N_mv = sub_A.shape[0],
                       wall_time = dp_cost,
                initial_solution = [i.tolist() for i in alpha_omega_ig],
                  final_solution = [i.tolist() for i in alpha_omega],
                      difference = alpha_difference,
                         overlap = X_overlap)

    return alpha_omega, Davidson_dict
