# -*- coding: utf-8 -*-


import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)
import time
import numpy as np
from arguments import args

from TDA.TDA_iter_preconditioner import TDA_iter_preconditioner
from mathlib.diag_ip import TDA_diag_preconditioner



if args.approx_p:
    print('using semiempirical approximation to K')
    K_inv = TDA_iter_preconditioner
else:
    print('using diagonal approximation to K')
    K_inv = TDA_diag_preconditioner

def Jacobi_preconditioner(residual, sub_eigenvalue, hdiag = None, misc=[]):
    '''(1-uu*)(A-Ω*I)t = -r
        u is full guess

        (1-uu*)Kz = y
        Kz - uu*Kz = y
        Kz + αu = y
        z =  K^-1y - αK^-1u
       r is residual, we want to solve t (approximately)
       z approximates t
       z = (A-Ω*I)^(-1)*r - α(A-Ω*I)^(-1)*u
        let K_inv_r = (A-Ω*I)^(-1)*r
        and K_inv_u = (A-Ω*I)^(-1)*u
       z = K_inv_r - α*K_inv_u
       where α = [u*(A-Ω*I)^(-1)y]/[u*(A-Ω*I)^(-1)u]  (using uz = 0)
       first, solve (A-Ω*I)^(-1)y and (A-Ω*I)^(-1)u

       misc = [full_guess, W_H, V_H, sub_A]
    '''

    full_guess = misc[0]
    # print('full_guess norm', np.linalg.norm(full_guess, axis=0))



    K_inv_r = K_inv(residual=residual, sub_eigenvalue=sub_eigenvalue)
    K_inv_u = K_inv(residual=full_guess, sub_eigenvalue=sub_eigenvalue)

    n = np.multiply(full_guess, K_inv_r).sum(axis=0)
    d = np.multiply(full_guess, K_inv_u).sum(axis=0)
    Alpha = n/d
    print('N in Jacobi =', np.average(n))
    print('D in Jacobi =', np.average(d))
    print('Alpha in Jacobi =', np.average(Alpha))

    z = Alpha*K_inv_u - K_inv_r

    print('K_inv_u norm =', np.linalg.norm(K_inv_u, axis=0))
    print('Alpha*K_inv_u norm =', np.linalg.norm(Alpha*K_inv_u, axis=0))
    print('K_inv_r norm =', np.linalg.norm(K_inv_r, axis=0))
    print('z norm =', np.linalg.norm(z, axis=0))

    return z
