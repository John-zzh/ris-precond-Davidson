# -*- coding: utf-8 -*-


import os, sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np

from approx_mv import approx_TDDFT_mv

from arguments import args
from SCF_calc import hdiag, max_vir_hdiag, n_occ, n_vir, max_vir, rst_vir_hdiag

from mathlib import math
from mathlib.diag_ip import TDDFT_diag_initial_guess, TDDFT_diag_preconditioner

def TDDFT_iter_preconditioner(Rx, Ry, omega,
                matrix_vector_product = approx_TDDFT_mv,
                             conv_tol = args.precond_TOL,
                                  max = 30,
                                n_occ = n_occ,
                                n_vir = n_vir,
                              max_vir = max_vir,
                                hdiag = hdiag,
                        max_vir_hdiag = max_vir_hdiag,
                        rst_vir_hdiag = rst_vir_hdiag):
    '''
    iterative TDDFT semi-empirical preconditioner
    [ A' B' ] - [1  0]X  Ω = P
    [ B' A' ]   [0 -1]Y    = Q
    A' = [A_trunc   0  ]
         [   0     diag]
    P = Rx
    Q = Ry
    '''

    max = 30
    p_start = time.time()
    k = len(omega)
    m = 0

    A_size = n_occ * n_vir
    A_reduced_size = n_occ * max_vir

    Rx = Rx.reshape(n_occ,n_vir,-1)
    Ry = Ry.reshape(n_occ,n_vir,-1)

    P = Rx[:,:max_vir,:].reshape(A_reduced_size,-1)
    Q = Ry[:,:max_vir,:].reshape(A_reduced_size,-1)

    initial_start = time.time()
    V_holder = np.zeros((A_reduced_size, (max+1)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    '''normalzie the RHS'''
    PQ = np.vstack((P,Q))
    pqnorm = np.linalg.norm(PQ, axis=0, keepdims=True)

    P /= pqnorm
    Q /= pqnorm

    X_new, Y_new  = TDDFT_diag_preconditioner(R_x = P,
                                              R_y = Q,
                                            omega = omega,
                                            hdiag = max_vir_hdiag)

    V_holder, W_holder, new_m = math.VW_Gram_Schmidt_fill_holder(
                                    V_holder = V_holder,
                                    W_holder = W_holder,
                                           m = 0,
                                       X_new = X_new,
                                       Y_new = Y_new)

    initial_end = time.time()
    initial_cost = initial_end - initial_start

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]

        '''U1 = AV + BW
           U2 = AW + BV'''

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = matrix_vector_product(
                                                            X=V[:, m:new_m],
                                                            Y=W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)

        '''
        p = VP + WQ
        q = WP + VQ
        '''

        p = np.dot(V.T, P)
        p += np.dot(W.T, Q)

        q = np.dot(W.T, P)
        q += np.dot(V.T, Q)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        a = math.symmetrize(a)
        b = math.symmetrize(b)
        sigma = math.symmetrize(sigma)
        pi = math.anti_symmetrize(pi)

        '''
        solve the x & y in the subspace
        '''

        subcost_start = time.time()
        x, y = math.TDDFT_subspace_liear_solver(a, b, sigma, pi, p, q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''
        compute the residual
        R_x = U1x + U2y - X_full*omega - P
        R_y = U2x + U1y + Y_full*omega - Q
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
        R_x -= P

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega
        R_y -= Q

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
                                                 omega[index],
                                               hdiag = max_vir_hdiag)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''
        GS and symmetric orthonormalization
        '''
        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = math.VW_Gram_Schmidt_fill_holder(
                                            V_holder = V_holder,
                                            W_holder = W_holder,
                                               X_new = X_new,
                                               Y_new = Y_new,
                                                   m = m)
        GS_end = time.time()
        GScost += GS_end - GS_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    p_end = time.time()

    P_cost = p_end - p_start

    if ii == (max -1):
        print('=== TDDFT preconditioner Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('TDDFT iterative preconditioner Done')

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, P_cost))
    print('final subspace', a.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))
    for enrty in ['initial_cost','MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/P_cost))

    X_full = X_full*pqnorm
    Y_full = Y_full*pqnorm

    X = np.zeros((n_occ,n_vir,k))
    Y = np.zeros((n_occ,n_vir,k))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,k)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,k)

    if max_vir < n_vir:
        P2 = Rx[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)
        Q2 = Ry[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)

        X2, Y2 = TDDFT_diag_preconditioner(R_x = P2,
                                           R_y = Q2,
                                         omega = omega,
                                         hdiag = rst_vir_hdiag)
        X[:,max_vir:,:] = X2.reshape(n_occ,n_vir-max_vir,-1)
        Y[:,max_vir:,:] = Y2.reshape(n_occ,n_vir-max_vir,-1)

    X = X.reshape(A_size,-1)
    Y = Y.reshape(A_size,-1)

    return X, Y
