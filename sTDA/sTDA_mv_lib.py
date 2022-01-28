# -*- coding: utf-8 -*-


from pyscf import gto, scf, dft, tddft, data, lib
from opt_einsum import contract as einsum
import numpy as np

import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from mathlib import parameter
from arguments import args
from SCF_calc import (alpha, beta, mol, N_atm, N_bf, n_occ, n_vir, max_vir,
                    R_array, C_matrix, a_x, delta_fly, delta_max_vir_fly,
                    delta_rst_vir_fly)


def gen_eta(mol=mol):
    ''' mol.atom_pure_symbol(atom_id) returns the element symbol
    '''
    if args.eta == True:
        eta = np.loadtxt('SLSQP_eta.txt')
    else:
        '''a list is a list of chemical hardness for all atoms
        '''
        eta = [parameter.HARDNESS[mol.atom_pure_symbol(atom_id)] for atom_id in range(N_atm)]
        eta = np.asarray(eta).reshape(1,-1)

    return eta

eta = gen_eta()

class sTDA(object):

    def __init__(self, eta=eta):
        self.eta = eta

    def gen_gammaJK(self, eta):
        ''' creat GammaK and GammaK matrix
        '''
        eta = (eta + eta.T)/2
        GammaJ = (R_array**beta + (a_x * eta)**(-beta))**(-1/beta)
        GammaK = (R_array**alpha + eta**(-alpha)) **(-1/alpha)

        return GammaJ, GammaK

    def gen_QJK(self, GammaJ, GammaK):
        '''build q_iajb tensor'''


        aoslice = mol.aoslice_by_atom()
        q_tensors = np.zeros([N_atm, N_bf, N_bf])
        for atom_id in range(N_atm):
            shst, shend, atstart, atend = aoslice[atom_id]
            q_tensors[atom_id,:,:] = np.dot(C_matrix[atstart:atend,:].T,
                                            C_matrix[atstart:atend,:])

        ''' pre-calculate and store the Q-Gamma rank 3 tensor
            qia * gamma * qjb -> qia GK_q_jb
        '''
        q_ij = np.zeros((N_atm, n_occ, n_occ))
        q_ij[:,:,:] = q_tensors[:,:n_occ,:n_occ]

        q_ab = np.zeros((N_atm, max_vir, max_vir))
        q_ab[:,:,:] = q_tensors[:,n_occ:n_occ+max_vir,n_occ:n_occ+max_vir]

        q_ia = np.zeros((N_atm, n_occ, max_vir))
        q_ia[:,:,:] = q_tensors[:,:n_occ,n_occ:n_occ+max_vir]

        GK_q_jb = einsum("Bjb,AB->Ajb", q_ia, GammaK)
        GJ_q_ab = einsum("Bab,AB->Aab", q_ab, GammaJ)

        return q_ij, q_ab, q_ia, GK_q_jb, GJ_q_ab

    def gen_iajb_ijab_ibja_delta_fly(self, q_ij, q_ab, q_ia, GK_q_jb, GJ_q_ab):
        '''
        define sTDA on-the-fly two electron intergeral (pq|rs)
        A_iajb * v = delta_ia_ia*v + 2(ia|jb)*v - (ij|ab)*v
        iajb_v = einsum('Aia,Bjb,AB,jbm -> iam', q_ia, q_ia, GammaK, V)
        ijab_v = einsum('Aij,Bab,AB,jbm -> iam', q_ij, q_ab, GammaJ, V)
        '''

        def iajb_fly(V):
            '''
            (ia|jb)
            '''
            GK_q_jb_V = einsum("Ajb,jbm->Am", GK_q_jb, V)
            iajb_V = einsum("Aia,Am->iam", q_ia, GK_q_jb_V)
            return iajb_V

        def ijab_fly(V):
            '''
            (ij|ab)
            '''
            GJ_q_ab_V = einsum("Aab,jbm->Ajam", GJ_q_ab, V)
            ijab_V = einsum("Aij,Ajam->iam", q_ij, GJ_q_ab_V)
            return ijab_V

        def ibja_fly(V):
            '''
            the Forck exchange energy in B matrix
            (ib|ja)
            '''
            q_ib_V = einsum("Aib,jbm->Ajim", q_ia, V)
            ibja_V = einsum("Aja,Ajim->iam", GK_q_jb, q_ib_V)
            return ibja_V

        return iajb_fly, ijab_fly, ibja_fly

    def gen_mv_fly(self, iajb_fly, ijab_fly, ibja_fly):

        def sTDA_mv(V):
            '''
            return AX
            MV =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)
            '''
            V = V.reshape(n_occ, max_vir, -1)
            MV = delta_max_vir_fly(V) + 2*iajb_fly(V) - ijab_fly(V)
            MV = MV.reshape(n_occ*max_vir,-1)
            return MV

        def full_sTDA_mv(V):
            '''
            used in Jacobi preconditioner
            '''
            V = V.reshape(n_occ,n_vir,-1)
            U = np.zeros_like(V)

            V1 = V[:,:max_vir,:]
            V2 = V[:,max_vir:,:]

            U[:,:max_vir,:] = sTDA_mv(V1).reshape(n_occ, max_vir, -1)
            U[:,max_vir:,:] = delta_rst_vir_fly(V2)

            U = U.reshape(n_occ * n_vir,-1)
            return U

        def sTDDFT_mv(X, Y):
            '''
            return AX+BY and AY+BX
            sTDA_A =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)
            sTDDFT_B = 2*iajb_fly(V) - a_x*ibja_fly(V)
            '''
            X = X.reshape(n_occ, max_vir,-1)
            Y = Y.reshape(n_occ, max_vir,-1)

            X_max_vir = X[:,:max_vir,:]
            Y_max_vir = Y[:,:max_vir,:]

            iajb_X = iajb_fly(X_max_vir)
            iajb_Y = iajb_fly(Y_max_vir)

            ijab_X = ijab_fly(X_max_vir)
            ijab_Y = ijab_fly(Y_max_vir)

            ibja_X = ibja_fly(X_max_vir)
            ibja_Y = ibja_fly(Y_max_vir)

            delta_X = delta_max_vir_fly(X_max_vir)
            delta_Y = delta_max_vir_fly(Y_max_vir)

            AX = delta_X + 2*iajb_X - ijab_X
            AY = delta_Y + 2*iajb_Y - ijab_Y

            BX = 2*iajb_X - a_x*ibja_X
            BY = 2*iajb_Y - a_x*ibja_Y

            U1 = np.zeros_like(X)
            U2 = np.zeros_like(X)

            U1[:,:max_vir,:] = AX + BY
            U2[:,:max_vir,:] = AY + BX

            U1 = U1.reshape(n_occ*max_vir,-1)
            U2 = U2.reshape(n_occ*max_vir,-1)

            return U1, U2

        def sTDDFT_spolar_mv(X):
            '''
            (A+B)X = delta_fly(V) + 4*iajb_fly(V)
                    - ijab_fly(V) - a_x*ibja_fly(V)
            '''
            X = X.reshape(n_occ, max_vir, -1)
            U = delta_max_vir_fly(X) + 4*iajb_fly(X) - ijab_fly(X) - a_x*ibja_fly(X)
            U = U.reshape(n_occ*max_vir, -1)

            return U

        return sTDA_mv, full_sTDA_mv, sTDDFT_mv, sTDDFT_spolar_mv


    def build(self):

        (GammaJ,
        GammaK) = self.gen_gammaJK(eta=self.eta)

        (q_ij,
        q_ab,
        q_ia,
        GK_q_jb,
        GJ_q_ab) = self.gen_QJK(
                    GammaJ=GammaJ,
                    GammaK=GammaK)

        (iajb_fly,
        ijab_fly,
        ibja_fly) = self.gen_iajb_ijab_ibja_delta_fly(
                            q_ij=q_ij,
                            q_ab=q_ab,
                            q_ia=q_ia,
                            GK_q_jb=GK_q_jb,
                            GJ_q_ab=GJ_q_ab)

        (sTDA_mv,
        full_sTDA_mv,
        TDDFT_mv,
        sTDDFT_spolar_mv) = self.gen_mv_fly(
                                iajb_fly=iajb_fly,
                                ijab_fly=ijab_fly,
                                ibja_fly=ibja_fly)

        return sTDA_mv, full_sTDA_mv, TDDFT_mv, sTDDFT_spolar_mv
