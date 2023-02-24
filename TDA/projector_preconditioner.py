# -*- coding: utf-8 -*-

'''
absolute import
'''
import os, sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np

from pyscf import lib

from arguments import args
from TDA.TDA_iter_initial_guess import TDA_iter_initial_guess
from SCF_calc import delta_fly, A_size, hdiag

einsum = lib.einsum

if args.approx_p:
    from approx_mv import approx_TDA_mv
    print('using semiempirical approximation to K')
else:
    print('using diagonal approximation to K')
    def approx_TDA_mv(V):
        '''
        delta_hdiag.shape = (n_occ, n_vir)
        '''
        V = V.reshape(A_size,-1)
        delta_v = V*hdiag.reshape(-1,1)
        # delta_v = delta_v.reshape(n_occ, n_vir, -1)
        # delta_v = einsum("ia,iam->iam", delta_hdiag, V)
        return delta_v

def projector_preconditioner(misc,
                        residual = None,
                  sub_eigenvalue = None,
                           hdiag = None):

    '''new eigenvalue solver, to diagonalize the H'(an approximation to H)
       use the traditional Davidson to diagonalize the H' matrix
       W, V, sub_A are from the exact H
       misc = [full_guess[:,index],
           W_holder[:,:size_new],
           V_holder[:,:size_new],
           sub_A,
           index]
    '''
    W = misc[1]
    V = misc[2]
    sub_A = misc[3]
    return_index = misc[4]



    def on_the_fly_Hx(x, V=V, W=W, sub_A=sub_A):
        def Qx(x, V=V):
            '''Qx = (1 - V*V.T)*x = x - V*V.T*x'''
            VX = np.dot(V.T,x)
            x = x - np.dot(V,VX)
            return x
        '''on-the-fly compute H'x
           H′ ≡ W*V.T + V*W.T − V*a*V.T + Q*K*Q
           K approximates H, here K = A^se
           H′ ≡ W*V.T + V*W.T − V*a*V.T + (1-V*V.T)K(1-V*V.T)
           H′x ≡ a + b − c + d
        '''
        a = einsum('ij, jk, kl -> il', W, V.T, x)
        b = einsum('ij, jk, kl -> il', V, W.T, x)
        c = einsum('ij, jk, kl, lm -> im', V, sub_A, V.T, x)
        d = Qx(approx_TDA_mv(Qx(x,V=V)),V=V)
        Hx = a + b - c + d
        return Hx

    full_guess, current_energy = TDA_iter_initial_guess(N_states=args.nstates,
                                           matrix_vector_product=on_the_fly_Hx,
                                                      conv_tol = args.precond_TOL,
                                                            max = 45)
    print('projector energy', current_energy)
    return full_guess[:,return_index]
