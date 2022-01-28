# -*- coding: utf-8 -*-


import os, sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)
import numpy as np

from mathlib import math, parameter
from mathlib.diag_ip import TDA_diag_initial_guess, TDA_diag_preconditioner

import time
from arguments import args
from SCF_calc import n_occ, n_vir, max_vir, max_vir_hdiag, hdiag
from approx_mv import approx_TDA_mv


def TDA_iter_initial_guess(N_states,
        matrix_vector_product = approx_TDA_mv,
                     conv_tol = args.initial_TOL,
                max_vir_hdiag = max_vir_hdiag,
                        hdiag = hdiag,
                        n_occ = n_occ,
                        n_vir = n_vir,
                      max_vir = max_vir):
    '''
    [A_trunc   0 ] [X] = [X] Ω
    [   0    diag] [0]   [0]

    return [X]
           [0]
    '''

    Davidson_start = time.time()
    max = 35
    A_size = n_occ * n_vir
    A_reduced_size = n_occ * max_vir
    '''m is size of subspace'''
    m = 0
    new_m = min([N_states+8, 2*N_states, A_size])
    V = np.zeros((A_reduced_size, max*N_states + new_m))
    W = np.zeros_like(V)

    '''
    V is subsapce basis
    W is transformed guess vectors
    '''
    V[:, :new_m], initial_energies = TDA_diag_initial_guess(N_states = new_m,
                                                        hdiag = max_vir_hdiag)
    for ii in range(max):
        '''
        create subspace
        '''
        W[:, m:new_m] = matrix_vector_product(V[:, m:new_m])
        sub_A = np.dot(V[:,:new_m].T, W[:,:new_m])
        sub_A = math.symmetrize(sub_A)

        '''
        Diagonalize the subspace Hamiltonian, and sorted.
        sub_eigenvalue[:N_states] are smallest N_states eigenvalues
        '''
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)

        sub_eigenvalue = sub_eigenvalue[:N_states]
        sub_eigenket = sub_eigenket[:,:N_states]
        full_guess = np.dot(V[:,:new_m], sub_eigenket)

        '''
        residual = AX - XΩ = AVx - XΩ = Wx - XΩ
        '''
        residual = np.dot(W[:,:new_m], sub_eigenket)
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
                           hdiag = max_vir_hdiag)

        '''orthonormalize the new guess against basis and put into V holder'''
        m = new_m
        V, new_m = math.Gram_Schmidt_fill_holder(V, m, new_guess)

    omega = sub_eigenvalue*parameter.Hartree_to_eV
    Davidson_end = time.time()
    Davidson_time = Davidson_end - Davidson_start
    print('Total steps =', ii+1)
    print('Total wall time: {:.2f} seconds'.format(Davidson_time))
    print('threshold = {:.2e}'.format(conv_tol))


    U = np.zeros((n_occ,n_vir,N_states))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,N_states)
    U = U.reshape(A_size, N_states)

    return U, omega
