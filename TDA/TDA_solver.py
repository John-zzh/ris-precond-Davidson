# -*- coding: utf-8 -*-


import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np

from mathlib import math, parameter
from SCF_calc import A_size, hdiag, TDA_matrix_vector
from dump_yaml import fill_dictionary

def TDA_solver(N_states, initial_guess, preconditioner,
            matrix_vector_product = TDA_matrix_vector,
                           A_size = A_size,
                            hdiag = hdiag,
                              max = 35,
                         conv_tol = 1e-5,
                     extrainitial = 8):
    '''
    AX = XΩ
    Davidson frame, can use different initial guess and preconditioner
    initial_guess is a function, takes the number of initial guess as input
    preconditioner is a function, takes the residual as the input
    '''
    print('====== TDA Calculation Starts ======')
    Davidson_dict={}
    D_start = time.time()
    Davidson_dict['iteration'] = []
    iteration_list = Davidson_dict['iteration']


    size_old = 0
    size_new = min([N_states + extrainitial, 2*N_states, A_size])

    max_N_mv = max*N_states + size_new
    V_holder = np.zeros((A_size, max_N_mv))
    W_holder = np.zeros_like(V_holder)
    sub_A_holder = np.zeros((max_N_mv,max_N_mv))
    '''
    generate the initial guesss and put into the basis holder V_holder
    '''

    init_start = time.time()
    initial_vectors, initial_energies = initial_guess(N_states=size_new, hdiag=hdiag)
    # print('initial_vectors.shape', initial_vectors.shape)
    initial_energies = initial_energies[:N_states]


    print('excitation energies:')
    print(initial_energies)

    V_holder[:,:size_new] = initial_vectors[:,:]
    init_end = time.time()
    init_cost = init_end - init_start

    print('initial guess time {:.2f} seconds'.format(init_cost))

    Pcost = 0
    MVcost = 0
    for ii in range(max):
        print()
        print('Iteration:', ii+1)
        istart = time.time()

        MV_start = time.time()
        W_holder[:, size_old:size_new] = matrix_vector_product(V_holder[:,size_old:size_new])
        MV_end = time.time()
        iMVcost = MV_end - MV_start
        MVcost += iMVcost

        sub_A_holder = math.gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new)
        sub_A = sub_A_holder[:size_new,:size_new]
        print('subspace size: ', sub_A.shape[0])

        '''
        Diagonalize the subspace Hamiltonian, and sorted.
        sub_eigenvalue[:N_states] are smallest N_states eigenvalues
        '''
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        sub_eigenvalue = sub_eigenvalue[:N_states]
        sub_eigenket = sub_eigenket[:,:N_states]

        full_guess = np.dot(V_holder[:,:size_new], sub_eigenket)
        AV = np.dot(W_holder[:,:size_new], sub_eigenket)
        residual = AV - full_guess * sub_eigenvalue

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)

        iteration_list.append({})
        current_dict = iteration_list[ii]
        current_dict['residual norms'] = r_norms

        print('maximum residual norm {:.2e}'.format(max_norm))
        if max_norm < conv_tol or ii == (max-1):
            iend = time.time()
            icost = iend - istart
            current_dict['iteration total cost'] = icost
            current_dict['iteration MV cost'] = iMVcost
            iteration_list[ii] = current_dict
            print('iteration MV time {:.2f} seconds'.format(iMVcost))
            print('iteration total time {:.2f} seconds'.format(icost))
            print('Davidson Diagonalization Done')
            break

        index = [r_norms.index(i) for i in r_norms if i>conv_tol]

        P_start = time.time()
        new_guess = preconditioner(residual = residual[:,index],
                             sub_eigenvalue = sub_eigenvalue[index],
                                      hdiag = hdiag)
        P_end = time.time()
        Pcost += P_end - P_start

        size_old = size_new
        V_holder, size_new = math.Gram_Schmidt_fill_holder(V_holder, size_old, new_guess)
        print('amount of newly generated guesses:', size_new - size_old)

        iend = time.time()
        icost = iend - istart
        current_dict['iteration cost'] = icost
        current_dict['iteration MV cost'] = iMVcost
        iteration_list[ii] = current_dict
        print('iteration MV time {:.2f} seconds'.format(iMVcost))
        print('iteration total time {:.2f} seconds'.format(icost))

    energies = sub_eigenvalue*parameter.Hartree_to_eV

    D_end = time.time()
    Dcost = D_end - D_start

    energy_diff = float(np.linalg.norm(initial_energies - energies))
    X_overlap = float(np.linalg.norm(np.dot(initial_vectors[:,:N_states].T, full_guess)))

    Davidson_dict = fill_dictionary(Davidson_dict,
                           N_itr = ii+1,
                           pcost = Pcost,
                           icost = init_cost,
                            N_mv = sub_A.shape[0],
                       wall_time = Dcost,
                initial_solution = initial_energies.tolist(),
                  final_solution = energies.tolist(),
                      difference = energy_diff,
                         overlap = X_overlap)

    if ii == max-1:
        print('=== TDA Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('========== TDA Calculation Done==========')
    print('energies:')
    print(energies)
    print('Maximum residual norm = {:.2e}'.format(max_norm))
    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, Dcost))
    print('Final subspace size = {:d}'.format(sub_A.shape[0]))
    print('Total precondition time: {:.2f} seconds, {:.2%}'.format(Pcost, Pcost/Dcost))
    print('Total Matrix-vector product cost {:.2f} seconds'.format(MVcost))

    return energies, full_guess, Davidson_dict, AV
