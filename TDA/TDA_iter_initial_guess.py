# -*- coding: utf-8 -*-


import os, sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)
import numpy as np

from mathlib import math, parameter
from mathlib.diag_ip import TDA_diag_initial_guess, TDA_diag_preconditioner

import time
from arguments import args
from SCF_calc import (n_occ, n_vir, A_size,
                    cl_rest_vir, cl_truc_occ,
                    ex_truc_occ, cl_rest_occ,
                    delta_hdiag2, hdiag, A_size)


if args.mix_rid_sTDA:
    from TDDFT_as.TDDFT_as_lib import TDDFT_as as approx
    approx_mv = approx()
    approx_TDA_mv, approx_TDDFT_mv, approx_spolar_mv = approx_mv.build()
else:
    from approx_mv import approx_TDA_mv

def TDA_iter_initial_guess(N_states,
        matrix_vector_product = approx_TDA_mv,
                     conv_tol = args.initial_TOL,
                 delta_hdiag2 = delta_hdiag2,
                        hdiag = hdiag,
                        n_occ = n_occ,
                        n_vir = n_vir,
                truncated_occ = cl_truc_occ,
                  reduced_occ = cl_rest_occ,
                  reduced_vir = cl_rest_vir,
                          max = 15):
    '''
    [ diag1   0        0 ] [0]   [0]
    [  0   rest_A   0 ] [X] = [X] Ω
    [  0      0     diag3] [0]   [0]

           [0]
    return [X]
           [0]
    '''

    Davidson_start = time.time()

    A_size = n_occ * n_vir
    A_rest_size = reduced_occ * reduced_vir
    '''size_new is size of subspace'''
    size_old = 0
    # size_new = min([N_states+args.extrainitial, 2*N_states, A_size])
    size_new = N_states
    max_N_mv = max*N_states + size_new
    V_holder = np.zeros((A_rest_size, max_N_mv))
    W_holder = np.zeros_like(V_holder)
    sub_A_holder = np.zeros((max_N_mv,max_N_mv))
    '''
    V_holder is subsapce basis
    W_holder is transformed guess vectors
    '''
    V_holder[:, :size_new], initial_energies = TDA_diag_initial_guess(N_states = size_new,
                                                        hdiag = delta_hdiag2)

    mvcost = 0
    GScost = 0
    subcost = 0
    subgencost = 0
    for ii in range(max):
        '''
        create subspace
        '''
        mv_start = time.time()
        W_holder[:, size_old:size_new] = matrix_vector_product(V_holder[:, size_old:size_new])
        mv_end = time.time()
        mvcost += mv_end - mv_start

        subgen_start = time.time()

        sub_A_holder = math.gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new)
        sub_A = sub_A_holder[:size_new,:size_new]

        subgen_end = time.time()
        subgencost += subgen_end - subgen_start


        '''
        Diagonalize the subspace Hamiltonian, and sorted.
        sub_eigenvalue[:N_states] are smallest N_states eigenvalues
        '''

        sub_start = time.time()
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        sub_end = time.time()
        subcost += sub_end - sub_start

        sub_eigenvalue = sub_eigenvalue[:N_states]
        sub_eigenket = sub_eigenket[:,:N_states]
        full_guess = np.dot(V_holder[:,:size_new], sub_eigenket)

        '''
        residual = AX - XΩ = AVx - XΩ = Wx - XΩ
        '''
        residual = np.dot(W_holder[:,:size_new], sub_eigenket)
        residual -= full_guess*sub_eigenvalue

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < conv_tol or ii == (max-1):
            break

        '''index for unconverged residuals'''
        index = [r_norms.index(i) for i in r_norms if i > conv_tol]
        '''precondition the unconverged residuals'''
        new_guess = TDA_diag_preconditioner(
                        residual = residual[:,index],
                  sub_eigenvalue = sub_eigenvalue[index],
                           hdiag = delta_hdiag2)

        '''orthonormalize the new guess against basis and put into V holder'''

        GS_start = time.time()
        size_old = size_new
        V_holder, size_new = math.Gram_Schmidt_fill_holder(V_holder, size_old, new_guess)
        GS_end = time.time()
        GScost += GS_end - GS_start

    if ii == max-1:
        print('=== Hit Iteration Limit ===')
    omega = sub_eigenvalue*parameter.Hartree_to_eV
    Davidson_end = time.time()
    Davidson_time = Davidson_end - Davidson_start
    print('TDA Iterative Initial Guess Done')
    print('Total steps =', ii+1)
    print('max_norm =', max_norm)
    print('Total wall time: {:.2f} seconds'.format(Davidson_time))
    print('threshold = {:.2e}'.format(conv_tol))

    for enrty in ['mvcost', 'GScost', 'subgencost', 'subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/Davidson_time))

    U = np.zeros((n_occ,n_vir,N_states))
    U[truncated_occ:,:reduced_vir,:] = full_guess.reshape(reduced_occ,reduced_vir,-1)
    U = U.reshape(A_size, N_states)

    return U, omega
