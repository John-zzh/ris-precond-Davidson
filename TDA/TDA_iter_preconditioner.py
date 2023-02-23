# -*- coding: utf-8 -*-

'''
absolute import
'''
import os, sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np

from approx_mv import approx_TDA_mv

from arguments import args
from SCF_calc import (n_occ, n_vir,
                    cl_rest_vir, cl_truc_occ, cl_truc_vir,
                    ex_truc_occ, cl_rest_occ,
                    delta_hdiag2, hdiag)

from mathlib import math
from mathlib.diag_ip import TDA_diag_initial_guess, TDA_diag_preconditioner



def TDA_iter_preconditioner(residual, sub_eigenvalue,
                        matrix_vector_product = approx_TDA_mv,
                                     conv_tol = args.precond_TOL,
                                          max = 20,
                                        hdiag = hdiag,
                                        n_occ = n_occ,
                                        n_vir = n_vir,
                                  trunced_occ = cl_truc_occ,
                                  reduced_occ = cl_rest_occ,
                                  trunced_vir = cl_truc_vir,
                                  reduced_vir = cl_rest_vir,
                                         misc = None):
    '''
    iterative preconditioner
    [ diag1   0        0 ] [0]    [0]     [P1]
    [  0   reduced_A   0 ] [X]  - [X] Ω = [P2]
    [  0      0     diag3] [0]    [0]     [P3]

    (A - Ω*I)^-1 P = X
    AX - X*Ω = P
    reduced_A X2 - X1*Ω = P2
    P is residuals (in macro Davidson's loop) to be preconditioned
    '''
    p_start = time.time()

    '''
    amount of residual vectors to be preconditioned
    '''
    N_vectors = residual.shape[1]
    Residuals = residual.reshape(n_occ, n_vir,-1)

    omega = sub_eigenvalue

    A_size = n_occ * n_vir
    A_reduced_size = reduced_occ * reduced_vir
    P = Residuals[trunced_occ:,:reduced_vir,:].copy()
    P = P.reshape(A_reduced_size,-1)

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P /= pnorm

    start = time.time()


    max_N_mv = (max+1)*N_vectors
    V_holder = np.zeros((A_reduced_size, max_N_mv))
    W_holder = np.zeros_like(V_holder)
    sub_A_holder = np.zeros((max_N_mv,max_N_mv))
    sub_P_holder = np.zeros((max_N_mv,N_vectors))
    size_old = 0

    sub_A_holder = np.zeros((max_N_mv,max_N_mv))


    '''now V_holder and W_holder are empty holders, 0 vectors
       W = sTDA_mv(V)
       size_old is the amount of vectors that already sit in the holder
       in each iteration, V and W will be filled/updated with new guess basis
       which is the preconditioned residuals
    '''

    '''initial guess: DX - XΩ = P
       Dp is the preconditioner
       <t: returns np.sign(D)*t; else: D
    '''
    t = 1e-14
    Dp = np.repeat(hdiag.reshape(-1,1), N_vectors, axis=1) - omega
    Dp = np.where(abs(Dp)<t, np.sign(Dp)*t, Dp)
    Dp = Dp.reshape(n_occ, n_vir, -1)
    # print('Dp.shape', Dp.shape)
    D = Dp[trunced_occ:,:reduced_vir,:]
    # print('D.shape', Dp.shape)
    # print('A_reduced_size', A_reduced_size)
    D = D.reshape(A_reduced_size,-1)
    inv_D = 1/D

    '''generate initial guess'''
    Xig = P*inv_D
    size_old = 0
    V_holder, size_new = math.Gram_Schmidt_fill_holder(V_holder, size_old, Xig, double = False)

    mvcost = 0
    GScost = 0
    subcost = 0
    subgencost = 0

    for ii in range(max):
        # print('step', ii+1)
        # print('size_old', size_old)
        # print('size_new', size_new)
        '''
        project A matrix and vector P into subspace
        '''
        mvstart = time.time()
        W_holder[:, size_old:size_new] = matrix_vector_product(V_holder[:, size_old:size_new])
        mvend = time.time()
        mvcost += mvend - mvstart

        substart = time.time()

        sub_A_holder = math.gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new)
        sub_P_holder = math.gen_VP(sub_P_holder, V_holder, P, size_old, size_new)

        sub_A = sub_A_holder[:size_new,:size_new]
        sub_P = sub_P_holder[:size_new,:]

        subend = time.time()
        subgencost += subend - substart

        substart = time.time()
        sub_guess = math.solve_AX_Xla_B(sub_A, omega, sub_P)
        subend = time.time()
        subcost += subend - substart

        full_guess = np.dot(V_holder[:,:size_new], sub_guess)
        residual = np.dot(W_holder[:,:size_new], sub_guess) - full_guess*omega - P

        r_norms = np.linalg.norm(residual, axis=0).tolist()

        max_norm = np.max(r_norms)
        if max_norm < conv_tol or ii == (max-1):
            break

        '''
        index of unconverged states
        '''
        index = [r_norms.index(i) for i in r_norms if i > conv_tol]

        '''
        precondition the unconverged residuals
        '''
        new_guess = residual[:,index]*inv_D[:,index]

        GSstart = time.time()
        size_old = size_new
        V_holder, size_new = math.Gram_Schmidt_fill_holder(V_holder, size_old, new_guess, double = False)
        GSend = time.time()
        GScost += GSend - GSstart

    p_end = time.time()
    p_cost = p_end - p_start

    if ii == (max-1):
        print('=== TDA Preconditioner Hit Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('TDA iterative preconditioner Done')

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, p_cost))
    print('final subspace', sub_A.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))

    for enrty in ['mvcost', 'GScost', 'subgencost', 'subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/p_cost))
    full_guess = full_guess*pnorm

    '''
    return the untruncated vectors
    '''
    U = np.zeros((n_occ,n_vir,N_vectors))
    U[trunced_occ:,:reduced_vir,:] = full_guess.reshape(reduced_occ,reduced_vir,-1)

    if reduced_occ < n_occ:
        ''' D1*X1 - X1*Ω = P1'''
        P1 = Residuals[:trunced_occ,:reduced_vir,:]
        P1 = P1.reshape(trunced_occ*reduced_vir,-1)

        D1 = Dp[:trunced_occ,:reduced_vir,:]
        D1 = D1.reshape(trunced_occ*reduced_vir,-1)
        X1 = (P1/D1).reshape(trunced_occ,reduced_vir,-1)
        U[:trunced_occ,:reduced_vir,:] = X1

    if reduced_vir < n_vir:
        ''' D3*X3 - X3*Ω = P3'''
        P3 = Residuals[:,reduced_vir:,:]
        P3 = P3.reshape(n_occ*trunced_vir,-1)

        D3 = Dp[:,reduced_vir:,:]
        D3 = D3.reshape(n_occ*trunced_vir,-1)
        X3 = (P3/D3).reshape(n_occ, trunced_vir, -1)
        U[:,reduced_vir:,:] = X3


    U = U.reshape(A_size, -1)

    '''if we want to know more about the preconditioning process,
        return the current_dic, rather than origin_dic'''
    return U
