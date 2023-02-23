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



def projector_preconditioner(full_guess, return_index, W_H, V_H, sub_A_H,
                        residual=None, sub_eigenvalue=None, current_dic=None):
    '''new eigenvalue solver, to diagonalize the H'(an approximation to H)
       use the traditional Davidson to diagonalize the H' matrix
       W_H, V_H, sub_A_H are from the exact H
    '''
    new_ES_start = time.time()
    tol = args.eigensolver_tol
    max = 30

    k = args.nstates
    m = min([k+8, 2*k, A_size])

    V = np.zeros((A_size, max*k + m))
    W = np.zeros_like(V)

    '''sTDA as initial guess'''
    V = sTDA_eigen_solver(m, V)
    W[:,:m] = on_the_fly_Hx(W_H, V_H, sub_A_H, V[:, :m])

    for i in range(max):
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        residual = np.dot(W[:,:m], sub_eigenket[:,:k])
        residual -= np.dot(V[:,:m], sub_eigenket[:,:k] * sub_eigenvalue[:k])

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]

        new_guess = TDA_A_diag_preconditioner(
                                          residual=residual[:,index],
                                    sub_eigenvalue=sub_eigenvalue[:k][index],
                                             hdiag=hdiag)
        V, new_m = mathlib.Gram_Schmidt_fill_holder(V, m, new_guess)
        W[:, m:new_m] = on_the_fly_Hx(W_H, V_H, sub_A_H, V[:, m:new_m])
        m = new_m

    full_guess = np.dot(V[:,:m], sub_eigenket[:,:k])

    new_ES_end = time.time()
    new_ES_cost = new_ES_end - new_ES_start
    print('H_app diagonalization done in',i,'steps; ','%.2f'%new_ES_cost, 's')
    print('threshold =', tol)
    return full_guess[:,return_index], current_dic
