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
from SCF_calc import hdiag, max_vir_hdiag, n_occ, n_vir, max_vir

from mathlib import math
from mathlib.diag_ip import TDA_diag_initial_guess, TDA_diag_preconditioner



def TDA_iter_preconditioner(residual, sub_eigenvalue,
                        matrix_vector_product = approx_TDA_mv,
                                     conv_tol = args.precond_TOL,
                                          max = 30,
                                          hdiag = hdiag,
                                        n_occ = n_occ,
                                        n_vir = n_vir,
                                      max_vir = max_vir):
    '''
    iterative TDA semi-empirical preconditioner
    (A - Ω*I)^-1 P = X
    AX - X*Ω = P
    A_truunc X1 - X1*Ω = P1
    P is residuals (in macro Davidson's loop) to be preconditioned
    '''
    p_start = time.time()

    '''
    amount of residual vectors to be preconditioned
    '''
    N_vectors = residual.shape[1]
    Residuals = residual.reshape(n_occ, n_vir,-1)

    omega = sub_eigenvalue
    A_reduced_size = n_occ * max_vir
    P = Residuals[:,:max_vir,:]
    P = P.reshape(A_reduced_size,-1)

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P = P/pnorm

    start = time.time()

    V = np.zeros((A_reduced_size, (max+1)*N_vectors))
    W = np.zeros((A_reduced_size, (max+1)*N_vectors))
    count = 0

    '''now V and W are empty holders, 0 vectors
       W = sTDA_mv(V)
       count is the amount of vectors that already sit in the holder
       in each iteration, V and W will be filled/updated with new guess basis
       which is the preconditioned residuals
    '''

    '''initial guess: DX - XΩ = P
       Dp is the preconditioner
       <t: returns np.sign(D)*t; else: D
    '''
    t = 1e-10
    Dp = np.repeat(hdiag.reshape(-1,1), N_vectors, axis=1) - omega
    Dp = np.where(abs(Dp)<t, np.sign(Dp)*t, Dp)
    Dp = Dp.reshape(n_occ, n_vir, -1)
    D = Dp[:,:max_vir,:].reshape(A_reduced_size,-1)
    inv_D = 1/D

    '''generate initial guess'''
    Xig = P*inv_D
    count = 0
    V, new_count = math.Gram_Schmidt_fill_holder(V, count, Xig)

    mvcost = 0
    GScost = 0
    subcost = 0
    subgencost = 0

    for ii in range(max):

        '''
        project A matrix and vector P into subspace
        '''
        mvstart = time.time()
        W[:, count:new_count] = matrix_vector_product(V[:, count:new_count])
        mvend = time.time()
        mvcost += mvend - mvstart

        substart = time.time()
        sub_P= np.dot(V[:,:new_count].T, P)
        sub_A = np.dot(V[:,:new_count].T, W[:,:new_count])
        subend = time.time()
        subgencost += subend - substart

        sub_A = math.symmetrize(sub_A)
        m = np.shape(sub_A)[0]

        substart = time.time()
        sub_guess = math.solve_AX_Xla_B(sub_A, omega, sub_P)
        subend = time.time()
        subcost += subend - substart

        full_guess = np.dot(V[:,:new_count], sub_guess)
        residual = np.dot(W[:,:new_count], sub_guess) - full_guess*omega - P

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
        count = new_count
        V, new_count = math.Gram_Schmidt_fill_holder(V, count, new_guess)
        GSend = time.time()
        GScost += GSend - GSstart

    p_end = time.time()
    p_cost = p_end - p_start

    if ii == (max-1):
        print('=== TDA Preconditioner Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('TDA iterative preconditioner Done')

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, p_cost))
    print('final subspace', sub_A.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))

    for enrty in ['subgencost', 'mvcost', 'GScost', 'subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/p_cost))
    full_guess = full_guess*pnorm

    '''
    return the untruncated vectors
    '''
    U = np.zeros((n_occ,n_vir,N_vectors))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,-1)

    if max_vir < n_vir:
        ''' DX2 - X2*Ω = P2'''
        P2 = Residuals[:,max_vir:,:]
        P2 = P2.reshape(n_occ*(n_vir-max_vir),-1)

        D2 = Dp[:,max_vir:,:]
        D2 = D2.reshape(n_occ*(n_vir-max_vir),-1)
        X2 = (P2/D2).reshape(n_occ,n_vir-max_vir,-1)
        U[:,max_vir:,:] = X2

    U = U.reshape(n_occ*n_vir, -1)

    '''if we want to know more about the preconditioning process,
        return the current_dic, rather than origin_dic'''
    return U
