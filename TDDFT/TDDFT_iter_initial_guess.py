# -*- coding: utf-8 -*-


import os, sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)
import numpy as np

from mathlib import math, parameter
from mathlib.diag_ip import TDDFT_diag_initial_guess, TDDFT_diag_preconditioner

import time
from arguments import args
from SCF_calc import n_occ, n_vir, max_vir, max_vir_hdiag, hdiag
from approx_mv import approx_TDDFT_mv

def TDDFT_iter_initial_guess_solver(N_states,
                matrix_vector_product = approx_TDDFT_mv,
                             conv_tol = args.initial_TOL,
                                hdiag = hdiag,
                                n_occ = n_occ,
                                n_vir = n_vir,
                              max_vir = max_vir):
    '''
    [ A' B' ] X - [1   0] Y Ω = 0
    [ B' A' ] Y   [0  -1] X   = 0
    A' = [A_trunc   0  ]
         [   0     diag]
    '''

    max = 35
    TD_start = time.time()

    A_size = n_occ * n_vir
    A_reduced_size = n_occ * max_vir

    m = 0
    new_m = min([N_states+8, 2*N_states, A_reduced_size])
    V_holder = np.zeros((A_reduced_size, (max+1)*N_states))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    '''
    set up initial guess V W, transformed vectors U1 U2
    '''

    (V_holder,
    W_holder,
    new_m,
    energies,
    Xig,
    Yig) = TDDFT_diag_initial_guess(V_holder = V_holder,
                                    W_holder = W_holder,
                                    N_states = new_m,
                                       hdiag = max_vir_hdiag)

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]

        '''
        U1 = AV + BW
        U2 = AW + BV
        '''

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


        a = math.symmetrize(a)
        b = math.symmetrize(b)
        sigma = math.symmetrize(sigma)
        pi = math.anti_symmetrize(pi)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''
        solve the eigenvalue omega in the subspace
        '''
        subcost_start = time.time()
        omega, x, y = math.TDDFT_subspace_eigen_solver(a, b, sigma, pi, N_states)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

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
        max_norm = np.max(r_norms)
        if max_norm < conv_tol or ii == (max -1):
            break

        index = [r_norms.index(i) for i in r_norms if i > conv_tol]

        '''
        preconditioning step
        '''
        X_new, Y_new = TDDFT_diag_preconditioner(R_x = R_x[:,index],
                                                   R_y = R_y[:,index],
                                                 omega = omega[index],
                                                 hdiag = max_vir_hdiag)

        '''
        GS and symmetric orthonormalization
        '''
        m = new_m
        GScost_start = time.time()
        V_holder, W_holder, new_m = math.VW_Gram_Schmidt_fill_holder(
                                            V_holder = V_holder,
                                            W_holder = W_holder,
                                               X_new = X_new,
                                               Y_new = Y_new,
                                                   m = m)
        GScost_end = time.time()
        GScost += GScost_end - GScost_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    TD_end = time.time()

    TD_cost = TD_end - TD_start

    if ii == (max -1):
        print('=== TDDFT Initial Guess  Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('TDDFT Iterative Initial Guess Done' )

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, TD_cost))
    print('final subspace', a.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))
    for enrty in ['MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/TD_cost))
    X = np.zeros((n_occ,n_vir,N_states))
    Y = np.zeros((n_occ,n_vir,N_states))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,-1)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,-1)

    X = X.reshape(A_size, -1)
    Y = Y.reshape(A_size, -1)

    energies = omega*parameter.Hartree_to_eV
    return energies, X, Y

def TDDFT_iter_initial_guess(V_holder, W_holder, N_states):
    energies, X_initial, Y_initial = TDDFT_iter_initial_guess_solver(
                                                    N_states=N_states)

    V_holder, W_holder, new_m = math.VW_Gram_Schmidt_fill_holder(
                                                    V_holder=V_holder,
                                                    W_holder=W_holder,
                                                    m=0,
                                                    X_new=X_initial,
                                                    Y_new=Y_initial)

    return V_holder, W_holder, new_m, energies, X_initial, Y_initial
