# -*- coding: utf-8 -*-


import os, sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)
import numpy as np

from mathlib import math, parameter
from mathlib.diag_ip import TDDFT_diag_initial_guess, TDDFT_diag_preconditioner

import time
from arguments import args
from SCF_calc import (n_occ, n_vir, A_size,
                    cl_rest_vir, cl_truc_occ,
                    ex_truc_occ, cl_rest_occ,
                    delta_hdiag2, hdiag, A_size)
from approx_mv import approx_TDDFT_mv

def TDDFT_iter_initial_guess_solver(N_states,
                matrix_vector_product = approx_TDDFT_mv,
                             conv_tol = args.initial_TOL,
                                hdiag = hdiag,
                                n_occ = n_occ,
                                n_vir = n_vir,
                          trunced_occ = cl_truc_occ,
                          reduced_occ = cl_rest_occ,
                          reduced_vir = cl_rest_vir,
                               double = args.GS_double):
    '''
    [ A' B' ] X - [1   0] Y Ω = 0
    [ B' A' ] Y   [0  -1] X   = 0

    A'X = [ diag1   0        0 ] [0]
          [  0   reduced_A   0 ] [X]
          [  0      0     diag3] [0]
    '''


    max = 15
    TD_start = time.time()
    A_size = n_occ * n_vir
    A_reduced_size = reduced_occ * reduced_vir
    size_old = 0
    # size_new = min([N_states+args.extrainitial, 2*N_states, A_reduced_size])
    size_new = N_states

    max_N_mv = (max+1)*N_states

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


    '''
    set up initial guess V W, transformed vectors U1 U2
    '''

    (V_holder,
    W_holder,
    size_new,
    energies,
    Xig,
    Yig) = TDDFT_diag_initial_guess(V_holder = V_holder,
                                    W_holder = W_holder,
                                    N_states = size_new,
                                       hdiag = delta_hdiag2)

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    full_cost = 0
    for ii in range(max):
        # print('step ', ii+1)
        V = V_holder[:,:size_new]
        W = W_holder[:,:size_new]

        '''
        U1 = AV + BW
        U2 = AW + BV
        '''
        # print('size_old =', size_old)
        # print('size_new =', size_new)
        MV_start = time.time()
        U1_holder[:, size_old:size_new], U2_holder[:, size_old:size_new] = matrix_vector_product(
                                                            X=V[:, size_old:size_new],
                                                            Y=W[:, size_old:size_new])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:size_new]
        U2 = U2_holder[:,:size_new]

        subgenstart = time.time()

        '''
        [U1] = [A B][V]
        [U2]   [B A][W]

        a = [V.T W.T][A B][V] = [V.T W.T][U1] = VU1 + WU2
                     [B A][W]            [U2]
        '''

        (sub_A, sub_B, sigma, pi,
        VU1_holder, WU2_holder, VU2_holder, WU1_holder,
        VV_holder, WW_holder, VW_holder) = math.gen_sub_ab(
                      V_holder, W_holder, U1_holder, U2_holder,
                      VU1_holder, WU2_holder, VU2_holder, WU1_holder,
                      VV_holder, WW_holder, VW_holder,
                      size_old, size_new)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''
        solve the eigenvalue omega in the subspace
        '''
        subcost_start = time.time()
        omega, x, y = math.TDDFT_subspace_eigen_solver(sub_A, sub_B, sigma, pi, N_states)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''
        compute the residual
        R_x = U1x + U2y - X_full*omega
        R_y = U2x + U1y + Y_full*omega
        X_full = Vx + Wy
        Y_full = Wx + Vy
        '''
        full_cost_start = time.time()
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

        full_cost_end = time.time()
        full_cost += full_cost_end - full_cost_start

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
                                                 hdiag = delta_hdiag2)

        '''
        GS and symmetric orthonormalization
        '''
        size_old = size_new
        GScost_start = time.time()
        V_holder, W_holder, size_new = math.VW_Gram_Schmidt_fill_holder(
                                            V_holder = V_holder,
                                            W_holder = W_holder,
                                               X_new = X_new,
                                               Y_new = Y_new,
                                                   m = size_old,
                                              double = double)
        GScost_end = time.time()
        GScost += GScost_end - GScost_start

        if size_new == size_old:
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
    print('final subspace', sub_A.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))
    for enrty in ['MVcost','GScost','subgencost','subcost','full_cost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/TD_cost))
    X = np.zeros((n_occ,n_vir,N_states))
    Y = np.zeros((n_occ,n_vir,N_states))

    X[trunced_occ:,:reduced_vir,:] = X_full.reshape(reduced_occ,reduced_vir,-1)
    Y[trunced_occ:,:reduced_vir,:] = Y_full.reshape(reduced_occ,reduced_vir,-1)

    X = X.reshape(A_size, -1)
    Y = Y.reshape(A_size, -1)

    energies = omega*parameter.Hartree_to_eV
    return energies, X, Y

def TDDFT_iter_initial_guess(V_holder, W_holder, N_states):
    energies, X_initial, Y_initial = TDDFT_iter_initial_guess_solver(
                                                    N_states=N_states)

    V_holder, W_holder, size_new = math.VW_Gram_Schmidt_fill_holder(
                                                    V_holder = V_holder,
                                                    W_holder = W_holder,
                                                           m = 0,
                                                       X_new = X_initial,
                                                       Y_new = Y_initial,
                                                      double = False)

    return V_holder, W_holder, size_new, energies, X_initial, Y_initial
