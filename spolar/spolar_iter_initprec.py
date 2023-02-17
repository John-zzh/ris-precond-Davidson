# -*- coding: utf-8 -*-

'''
absolute import
'''
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np

from approx_mv import approx_spolar_mv

from arguments import args

from mathlib import math
from mathlib.diag_ip import spolar_diag_initprec

from SCF_calc import (hdiag, delta_hdiag, delta_hdiag2, A_size,
             n_occ, n_vir, trunced_occ, reduced_occ, trunced_vir, reduced_vir,
                     trunced_vir, A_reduced_size)

def spolar_iter_initprec(RHS, conv_tol,
                matrix_vector_product = approx_spolar_mv,
                                hdiag = hdiag,
                                n_occ = n_occ,
                                n_vir = n_vir,
                          trunced_occ = trunced_occ,
                          reduced_occ = reduced_occ,
                          trunced_vir = trunced_vir,
                          reduced_vir = reduced_vir,
                       A_reduced_size = A_reduced_size,
                               A_size = A_size):

    ''' [ diag1   0        0 ] [0]    [RHS_1]
        [  0   reduced_A   0 ] [X]  = [RHS_2]
        [  0      0     diag3] [0]    [RHS_3]
       (A' + B')X = RHS
       residual = (A' + B')X - RHS
       X_ig = RHS/d
       X_new = residual/D
    '''
    ssp_start = time.time()
    max = 20
    m = 0
    npvec = RHS.shape[1]


    RHS = RHS.reshape(n_occ,n_vir,-1)

    RHS_2 = RHS[trunced_occ:,:reduced_vir,:].copy()
    RHS_2 = RHS_2.reshape(A_reduced_size,-1)
    RHS_2_norm = np.linalg.norm(RHS_2, axis=0, keepdims = True)
    print('iter ip spolar RHS_2_norm =', RHS_2_norm)
    # print('iter ip spolar RHS_2_norm.shape =', RHS_2_norm.shape)
    RHS_2 = RHS_2/RHS_2_norm

    V_holder = np.zeros((A_reduced_size, (max+1)*npvec))
    U_holder = np.zeros_like(V_holder)

    '''setting up initial guess'''
    init_start = time.time()
    X_ig = spolar_diag_initprec(RHS_2, hdiag=delta_hdiag2)
    V_holder, new_m = math.Gram_Schmidt_fill_holder(V_holder, m, X_ig)
    init_end = time.time()
    initial_cost = init_end - init_start

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        '''creating the subspace'''
        MV_start = time.time()
        '''U = AX + BX = (A+B)X'''
        U_holder[:, m:new_m] = matrix_vector_product(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, RHS_2)
        a_p_b = np.dot(V.T,U)
        # a_p_b = math.symmetrize(a_p_b)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''solve the x in the subspace
           (a+b)x = p
        '''
        subcost_start = time.time()
        x = np.linalg.solve(a_p_b, p)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R = Ux - RHS_2'''
        Ux = np.dot(U,x)
        residual = Ux - RHS_2

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        index = [r_norms.index(i) for i in r_norms if i > conv_tol]
        max_norm = np.max(r_norms)
        if max_norm < conv_tol or ii == (max -1):
            print('Static polarizability procedure aborted')
            break

        Pstart = time.time()
        X_new = spolar_diag_initprec(residual[:,index], hdiag=delta_hdiag2)
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
    X_full = X_full*RHS_2_norm
    '''alpha = np.dot(X_full.T, RHS_2)*-4'''

    ssp_end = time.time()
    ssp_cost = ssp_end - ssp_start

    if ii == (max -1):
        print('=== spolar initial guess Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('spolar precond Converged' )

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, ssp_cost))
    print('final subspace:', a_p_b.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))

    for enrty in ['initial_cost','MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/ssp_cost))



    U = np.zeros((n_occ,n_vir,npvec))
    U[trunced_occ:,:reduced_vir,:] = X_full.reshape(reduced_occ,reduced_vir,-1)

    if reduced_occ < n_occ:
        print('reduced_occ < n_occ')
        ''' D1*X1 = RHS_1 '''
        RHS_1 = RHS.reshape(n_occ,n_vir,-1)[:trunced_occ,:reduced_vir,:]
        RHS_1 = RHS_1.reshape(trunced_occ*reduced_vir,-1)

        D1 = delta_hdiag[:trunced_occ,:reduced_vir]
        D1 = D1.reshape(trunced_occ*reduced_vir,-1)
        X1 = (RHS_1/D1).reshape(trunced_occ*reduced_vir,-1)
        U[:trunced_occ,:reduced_vir,:] = X1


    if reduced_vir < n_vir:
        print('reduced_vir < n_vir')
        ''' D3*X3 = RHS_3 '''
        RHS_3 = RHS.reshape(n_occ,n_vir,-1)[:,reduced_vir:,:]
        RHS_3 = RHS_3.reshape(n_occ*trunced_vir,-1)
        D3 = delta_hdiag[:,reduced_vir:]
        D3 = D3.reshape(n_occ*trunced_vir,-1)
        X3 = (RHS_3/D3).reshape(n_occ,trunced_vir,-1)
        U[:,reduced_vir:,:] = X3
    U = U.reshape(A_size, npvec)
    return U
