# -*- coding: utf-8 -*-

'''
absolute import
'''
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np

from mathlib import math, parameter
from SCF_calc import (static_polarizability_matrix_vector, A_size,
                    hdiag, max_vir_hdiag, n_occ, n_vir, max_vir, gen_P)
from dump_yaml import fill_dictionary




def spolar_solver(initial_guess, preconditioner,
            matrix_vector_product = static_polarizability_matrix_vector,
                              max = 35,
                         conv_tol = 1e-5,
                            hdiag = hdiag,
                    max_vir_hdiag = max_vir_hdiag,
                      initial_TOL = 1e-3,
                      precond_TOL = 1e-2):
    '''
    (A+B)X = -P
    residual = (A+B)X + P
    '''
    print('=== Static polarizability Calculation Starts ===')
    sp_start = time.time()

    P = gen_P()
    P = P.reshape(-1,3)

    P_origin = np.zeros_like(P)
    P_origin[:,:] = P[:,:]

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P /= pnorm

    Davidson_dict = {}
    Davidson_dict['iteration'] = []
    iteration_list = Davidson_dict['iteration']

    m = 0

    V_holder = np.zeros((A_size, (max+1)*3))
    U_holder = np.zeros_like(V_holder)

    init_start = time.time()
    X_ig = initial_guess(P, conv_tol = initial_TOL, hdiag = hdiag)

    alpha_init = np.dot((X_ig*pnorm).T, P_origin)*-4
    print('initial guess tensor_alpha')
    print(alpha_init)

    V_holder, new_m = math.Gram_Schmidt_fill_holder(V_holder, 0, X_ig)
    init_end = time.time()
    initial_cost = init_end - init_start

    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        print()
        print('Iteration', ii)
        MV_start = time.time()
        U_holder[:, m:new_m] = matrix_vector_product(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = math.symmetrize(a_p_b)
        subgenend = time.time()

        '''solve the x in the subspace'''
        x = np.linalg.solve(a_p_b, -p)

        '''compute the residual
           R = Ux + P'''
        Ux = np.dot(U,x)
        residual = Ux + P

        r_norms = np.linalg.norm(residual, axis=0).tolist()

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        '''index for unconverged residuals'''
        index = [r_norms.index(i) for i in r_norms if i > conv_tol]
        max_norm = np.max(r_norms)
        print('max_norm = {:.2e}'.format(max_norm))
        if max_norm < conv_tol or ii == (max -1):
            # print('static polarizability precodure aborted\n')
            break

        '''preconditioning step'''
        Pstart = time.time()

        X_new = preconditioner(Pr = -residual[:,index],
                         conv_tol = precond_TOL,
                            hdiag = hdiag)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, new_m = math.Gram_Schmidt_fill_holder(V_holder, m, X_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    X_full = np.dot(V,x)
    X_overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)))

    X_full = X_full*pnorm

    tensor_alpha = np.dot(X_full.T, P_origin)*-4
    sp_end = time.time()
    sp_cost = sp_end - sp_start

    print('tensor_alpha')
    print(tensor_alpha)

    if ii == (max -1):
        print('=== Static polarizability Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('=== Static polarizability Converged ===')

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, sp_cost))
    print('final subspace', a_p_b.shape)
    print('max_norm = {:.2e}'.format(max_norm))
    for enrty in ['initial_cost','MVcost','Pcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/sp_cost))

    tensor_alpha_difference = float(np.linalg.norm(alpha_init - tensor_alpha))

    sp_end = time.time()
    spcost = sp_end - sp_start

    Davidson_dict = fill_dictionary(Davidson_dict,
                           N_itr = ii+1,
                           pcost = Pcost,
                           icost = initial_cost,
                            N_mv = a_p_b.shape[0],
                       wall_time = sp_cost,
                initial_solution = [i.tolist() for i in alpha_init],
                  final_solution = [i.tolist() for i in tensor_alpha],
                      difference = tensor_alpha_difference,
                         overlap = X_overlap)
    return tensor_alpha, Davidson_dict