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

from SCF_calc import (hdiag, delta_hdiag, max_vir_hdiag, A_size,
                    n_occ, n_vir, max_vir, A_reduced_size)

def spolar_iter_initprec(Pr, conv_tol,
                matrix_vector_product = approx_spolar_mv,
                          initial_TOL = 1e-3,
                          precond_TOL = 1e-2,
                                hdiag = hdiag):
    '''(A' + B')X = -P
       note the negative sign of P!
       residual = (A' + B')X + P
       X_ig = -P/d
       X_new = residual/D
    '''
    ssp_start = time.time()
    max = 30
    m = 0
    npvec = Pr.shape[1]

    P = Pr.reshape(n_occ,n_vir,-1)[:,:max_vir,:]
    P = P.reshape(A_reduced_size,-1)
    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P /= pnorm

    V_holder = np.zeros((A_reduced_size, (max+1)*npvec))
    U_holder = np.zeros_like(V_holder)

    '''setting up initial guess'''
    init_start = time.time()
    X_ig = spolar_diag_initprec(P, hdiag=max_vir_hdiag)
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
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = math.symmetrize(a_p_b)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''solve the x in the subspace'''
        subcost_start = time.time()
        x = np.linalg.solve(a_p_b, -p)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R = Ux + P'''
        Ux = np.dot(U,x)
        residual = Ux + P

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        index = [r_norms.index(i) for i in r_norms if i > conv_tol]
        max_norm = np.max(r_norms)
        if max_norm < conv_tol or ii == (max -1):
            print('Static polarizability procedure aborted')
            break

        Pstart = time.time()
        X_new = spolar_diag_initprec(-residual[:,index], hdiag=max_vir_hdiag)
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
    '''alpha = np.dot(X_full.T, P)*-4'''

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

    X_full = X_full*pnorm

    U = np.zeros((n_occ,n_vir,npvec))
    U[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,-1)[:,:,:]

    if max_vir < n_vir:
        ''' DX2 = -P2'''
        P2 = Pr.reshape(n_occ,n_vir,-1)[:,max_vir:,:]
        P2 = P2.reshape(n_occ*(n_vir-max_vir),-1)
        D2 = delta_hdiag[:,max_vir:]
        D2 = D2.reshape(n_occ*(n_vir-max_vir),-1)
        X2 = (-P2/D2).reshape(n_occ,n_vir-max_vir,-1)
        U[:,max_vir:,:] = X2
    U = U.reshape(A_size, npvec)
    return U
