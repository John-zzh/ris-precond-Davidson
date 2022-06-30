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
from mathlib.diag_ip import TDDFT_diag_initial_guess, TDDFT_diag_preconditioner
from dump_yaml import fill_dictionary
from functools import *

def level_print(a, level='macro'):
    if level=='micro':
        pass
    elif level=='macro':
        print(a)


def TDDFT_linear_equ(Rx, Ry, omega,
                        initial_guess = TDDFT_diag_preconditioner,
                       preconditioner = TDDFT_diag_preconditioner,
                matrix_vector_product = approx_TDDFT_mv,
                             conv_tol = args.precond_TOL,
                                  max = 20,
                                n_occ = n_occ,
                                n_vir = n_vir,
                          trunced_occ = cl_truc_occ,
                          reduced_occ = cl_rest_occ,
                          trunced_vir = cl_truc_vir,
                          reduced_vir = cl_rest_vir,
                                hdiag = hdiag,
                         delta_hdiag1 = delta_hdiag1,
                         delta_hdiag2 = delta_hdiag2,
                         delta_hdiag3 = delta_hdiag3,
                               double = args.GS_double,
                               level = 'macro'):
    '''
    iterative TDDFT semi-empirical preconditioner
    [ A' B' ] - [1  0]X  Ω = P
    [ B' A' ]   [0 -1]Y    = Q
    A' = [ diag1   0        0 ]
         [  0   reduced_A   0 ]
         [  0      0     diag3]
    for TDDFT_as reduced_A == A_size and diag1&diag3 == 0
    P = Rx
    Q = Ry
    '''
    init_start = time.time()
    A_size = n_occ * n_vir
    A_reduced_size = reduced_occ * reduced_vir

    max = 20
    p_start = time.time()
    N_vectors = len(omega)
    size_old = 0
    A_reduced_size = reduced_occ * reduced_vir
    Rx = Rx.reshape(n_occ,n_vir,-1)
    Ry = Ry.reshape(n_occ,n_vir,-1)

    P = Rx[trunced_occ:,:reduced_vir,:].reshape(A_reduced_size,-1)
    Q = Ry[trunced_occ:,:reduced_vir,:].reshape(A_reduced_size,-1)

    P = math.copy_array(P)
    Q = math.copy_array(Q)

    max_N_mv = (max+1)*N_vectors

    V_holder = np.zeros((A_reduced_size, max_N_mv))
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

    VP_holder = np.zeros((max_N_mv,N_vectors))
    VQ_holder = np.zeros_like(VP_holder)
    WP_holder = np.zeros_like(VP_holder)
    WQ_holder = np.zeros_like(VP_holder)

    '''normalzie the RHS'''
    PQ = np.vstack((P,Q))
    pqnorm = np.linalg.norm(PQ, axis=0, keepdims=True)

    P /= pqnorm
    Q /= pqnorm

    X_new, Y_new  = TDDFT_diag_preconditioner(R_x = P,
                                              R_y = Q,
                                            omega = omega,
                                            hdiag = delta_hdiag2)

    V_holder, W_holder, size_new = math.VW_Gram_Schmidt_fill_holder(
                                    V_holder = V_holder,
                                    W_holder = W_holder,
                                           m = 0,
                                       X_new = X_new,
                                       Y_new = Y_new,
                                      double = double)
    init_end = time.time()
    init_cost = init_end - init_start

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    full_cost = 0
    for ii in range(max):


        '''U1 = AV + BW
           U2 = AW + BV'''

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


        '''
        solve the x & y in the subspace
        '''

        subcost_start = time.time()
        x, y = math.TDDFT_subspace_liear_solver(sub_A, sub_B, sigma, pi, p, q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''
        compute the residual
        R_x = U1x + U2y - X_full*omega - P
        R_y = U2x + U1y + Y_full*omega - Q
        X_full = Vx + Wy
        Y_full = Wx + Vy
        '''
        full_cost_start = time.time()

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
        R_x -= P

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega
        R_y -= Q
        full_cost_end = time.time()
        full_cost += full_cost_end - full_cost_start

        residual = np.vstack((R_x,R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < conv_tol or ii == (max -1):
            break
        index = [r_norms.index(i) for i in r_norms if i > conv_tol]

        '''
        preconditioning step
        '''
        Pstart = time.time()
        X_new, Y_new = TDDFT_diag_preconditioner(R_x[:,index],
                                                 R_y[:,index],
                                                 omega = omega[index],
                                                 hdiag = delta_hdiag2)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''
        GS and symmetric orthonormalization
        '''
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

    p_end = time.time()

    P_cost = p_end - p_start

    if ii == (max -1):
        print('=== TDDFT preconditioner Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('TDDFT iterative preconditioner Done')




    X_full = X_full*pqnorm
    Y_full = Y_full*pqnorm

    X = np.zeros((n_occ,n_vir,N_vectors))
    Y = np.zeros((n_occ,n_vir,N_vectors))

    X[trunced_occ:,:reduced_vir,:] = X_full.reshape(reduced_occ,reduced_vir,N_vectors)
    Y[trunced_occ:,:reduced_vir,:] = Y_full.reshape(reduced_occ,reduced_vir,N_vectors)

    if reduced_occ < n_occ:
        P1 = Rx[:trunced_occ,:reduced_vir,:].reshape(trunced_occ*reduced_vir,-1)
        Q1 = Ry[:trunced_occ,:reduced_vir,:].reshape(trunced_occ*reduced_vir,-1)

        X1, Y1 = TDDFT_diag_preconditioner(R_x = P1,
                                           R_y = Q1,
                                         omega = omega,
                                         hdiag = delta_hdiag1)
        X[:trunced_occ,:reduced_vir,:] = X1.reshape(trunced_occ,reduced_vir,-1)
        Y[:trunced_occ,:reduced_vir,:] = Y1.reshape(trunced_occ,reduced_vir,-1)

    if reduced_vir < n_vir:
        P3 = Rx[:,reduced_vir:,:].reshape(n_occ*trunced_vir,-1)
        Q3 = Ry[:,reduced_vir:,:].reshape(n_occ*trunced_vir,-1)

        X3, Y3 = TDDFT_diag_preconditioner(R_x = P3,
                                           R_y = Q3,
                                         omega = omega,
                                         hdiag = delta_hdiag3)
        X[:,reduced_vir:,:] = X3.reshape(n_occ,trunced_vir,-1)
        Y[:,reduced_vir:,:] = Y3.reshape(n_occ,trunced_vir,-1)


    X = X.reshape(A_size,-1)
    Y = Y.reshape(A_size,-1)

    P_cost = p_end - p_start
    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, P_cost))
    print('final subspace', sub_A.shape[0])
    print('max_norsize_old = {:.2e}'.format(max_norm))
    for enrty in ['init_cost', 'MVcost','GScost','subgencost','subcost','full_cost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/P_cost))

    return X, Y


def dpolar_solver(initial_guess, preconditioner, dpolar_omega,
                                max = 30,
                           conv_tol = 1e-5,
                        initial_TOL = 1e-3,
                        precond_TOL = 1e-2,
              matrix_vector_product = TDDFT_matrix_vector):

    P = gen_P()
    P = P.reshape(-1,3)
    P_origin =  math.copy_array(P)

    k = len(dpolar_omega)
    omega =  np.zeros([3*k])
    for jj in range(k):
        '''
        if have 3 ω, [ω1 ω1 ω1, ω2 ω2 ω2, ω3 ω3 ω3]
        convert nm to Hartree
        '''
        omega[3*jj:3*(jj+1)] = 45.56337117/dpolar_omega[jj]
    '''
    repeat k times
    '''
    P = np.tile(P,k)
    Q = math.copy_array(P)
    initial_guess = partial(TDDFT_linear_equ, matrix_vector_product)
    X, Y = TDDFT_linear_equ(Rx=P, Ry=Q, omega=omega, matrix_vector_product = TDDFT_matrix_vector)

    alpha_omega, Davidson_dict =

    return alpha_omega, Davidson_dict
