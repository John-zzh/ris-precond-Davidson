from pyscf import gto, scf, dft, tddft, data, lib

from collections import Counter
from opt_einsum import contract as einsum
import numpy as np

import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from arguments import args
from SCF_calc import (atom_coordinates, mol, n_occ, n_vir, max_vir, N_bf, N_atm,
                    un_ortho_C_matrix, delta_hdiag, a_x, delta_fly,
                    delta_max_vir_fly, delta_rst_vir_fly)
from mathlib import parameter



if args.functional in parameter.RSH_F:
    a_x = 1
else:
    pass


class TDDFT_as(object):
    def __init__(self, Uk = args.Uk, omega=parameter.RSH_omega):
        if args.Uread == True:
            '''
            read Uk from txt
            '''
            file = os.popen('ls *_Uk.txt').readlines()[0].replace('\n', '')
            self.Uk = float(np.loadtxt(file))

        else:
            self.Uk = Uk

        self.omega = omega

    def gen_auxmol(self):
        print('asigning auxiliary basis set')

        Uk = self.Uk

        print('Uk =', Uk)

        auxmol = gto.M(atom=atom_coordinates, parse_arg = False)
        '''
        parse_arg = False turns off PySCF built-in output file
        '''

        atom_count = Counter(auxmol.elements)

        '''
        auxmol_basis_keys = ['C', 'H', 'H^2', 'H^3', 'H^4', 'O'}
        '''
        auxmol_basis_keys = []
        for key in atom_count:
            for i in range(atom_count[key]):
                if i > 0:
                    auxmol_basis_keys.append(key+'^'+str(i+1))
                else:
                    auxmol_basis_keys.append(key)
        print('auxmol_basis_keys', auxmol_basis_keys)
        '''
        aux_basis = {
        'C': [[0, [0.123, 1.0]], [1, [0.123, 1.0]]],
        'H': [[0, [0.123, 1.0]]],
        'H^2': [[0, [0.123, 1.0]]],
        'H^3': [[0, [0.123, 1.0]]],
        'H^4': [[0, [0.123, 1.0]]],
        'O': [[0, [0.123, 1.0]], [1, [0.123, 1.0]]]
        }
        '''
        aux_basis = {}
        for i in range(len(auxmol_basis_keys)):
            atom_index = auxmol_basis_keys[i]
            atom = atom_index.split('^')[0]

            if args.Uconst != 0.0:
                exp = args.Uconst
            else:
                exp = Uk/(parameter.RADII[atom])**0.5

            if atom != 'H' and args.pobt == True:
                aux_basis[atom_index]= [[0, [exp, 1.0]],[1, [exp, 1.0]]]
            else:
                aux_basis[atom_index]= [[0, [exp, 1.0]]]

        # [print(k, v) for k, v in aux_basis.items()]

        auxmol.basis = aux_basis

        auxmol.build()
        [print(k, v) for k, v in auxmol.basis.items()]

        return auxmol


    def gen_electron_int(self, mol, auxmol, omega=0):

        nao = mol.nao_nr()
        naux = auxmol.nao_nr()

        auxmol.set_range_coulomb(omega)
        mol.set_range_coulomb(omega)


        '''
        (pq|rs) = Σ_PQ (pq|P)(P|Q)^-1(Q|rs)
        2 center 2 electron integral (P|Q)
        N_auxbf * N_auxbf
        '''
        eri2c = auxmol.intor('int2c2e_sph')
        # print('eri2c,', eri2c.shape)
        '''3 center 2 electron integral (pq|P)
            N_bf * N_bf * N_auxbf
        '''
        pmol = mol + auxmol

        eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,
                                                mol.nbas,mol.nbas+auxmol.nbas))

        return eri2c, eri3c

    def gen_GAMMA(self, eri2c, eri3c, max_vir=max_vir):

        N_auxbf = eri2c.shape[0]

        eri2c_inv = np.linalg.inv(eri2c)

        Delta = einsum("PQ, uvQ -> uvP", eri2c_inv, eri3c)
        GAMMA = einsum("up, vq, uvP -> pqP", un_ortho_C_matrix, un_ortho_C_matrix, Delta)
        # print('GAMMA shape', GAMMA.shape)

        GAMMA_ij = np.zeros((n_occ, n_occ, N_auxbf))
        # print('GAMMA_ij shape', GAMMA_ij.shape)
        GAMMA_ij[:,:,:] = GAMMA[:n_occ,:n_occ,:]

        GAMMA_ab = np.zeros((max_vir, max_vir, N_auxbf))
        # print('GAMMA_ab shape', GAMMA_ab.shape)
        GAMMA_ab[:,:,:] = GAMMA[n_occ:n_occ+max_vir,n_occ:n_occ+max_vir,:]

        GAMMA_ia = np.zeros((n_occ, max_vir, N_auxbf))
        # print('GAMMA_ia shape', GAMMA_ia.shape)
        GAMMA_ia[:,:,:] = GAMMA[:n_occ,n_occ:n_occ+max_vir,:]

        GAMMA_J_ia = einsum("iaA , AB -> iaB", GAMMA_ia, eri2c)
        GAMMA_J_ij = einsum("ijA , AB -> ijB", GAMMA_ij, eri2c)

        return GAMMA_ij, GAMMA_ab, GAMMA_ia, GAMMA_J_ia, GAMMA_J_ij


    def gen_as_2e_fly(self, GAMMA_ij, GAMMA_ab, GAMMA_ia, GAMMA_J_ia, GAMMA_J_ij):

        def as_iajb_fly(V):
            '''(ia|jb) '''
            GAMMA_jb_V = einsum("iaA, iam -> Am", GAMMA_ia, V)
            as_iajb_V  = einsum("iaA, Am -> iam", GAMMA_J_ia, GAMMA_jb_V)
            return as_iajb_V

        def as_ijab_fly(V):
            '''(ij|ab) '''
            GAMMA_ab_V = einsum("abA, jbm -> jAam", GAMMA_ab, V)
            as_ijab_V  = einsum("ijA, jAam -> iam", GAMMA_J_ij, GAMMA_ab_V)
            return as_ijab_V

        def as_ibja_fly(V):
            '''the Forck exchange energy in B matrix
               (ib|ja)
            '''
            GAMMA_ja_V = einsum("ibA,jbm->Ajim", GAMMA_ia, V)
            as_ibja_V = einsum("jaA,Ajim->iam", GAMMA_J_ia, GAMMA_ja_V)
            return as_ibja_V


        return as_iajb_fly, as_ijab_fly, as_ibja_fly

    def gen_as_mv_fly(self, as_iajb_fly, as_ijab_fly, as_ibja_fly):

        def TDA_as_mv(V):
            ''' return AX
                for RSH, a_x  =1
                AV = delta_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
            '''
            V = V.reshape(n_occ, max_vir, -1)
            MV = delta_max_vir_fly(V) + 2*as_iajb_fly(V) - a_x*as_ijab_fly(V)
            MV = MV.reshape(n_occ*max_vir, -1)
            return MV

        def TDDFT_as_mv(X, Y):
            '''return AX+BY and AY+BX
                for RSH, a_x = 1
                AV = delta_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
                BV = 2*iajb_fly(V) - a_x*ibja_fly(V)
            '''
            X = X.reshape(n_occ, max_vir, -1)
            Y = Y.reshape(n_occ, max_vir, -1)

            X_max_vir = X[:,:max_vir,:]
            Y_max_vir = Y[:,:max_vir,:]

            iajb_X = as_iajb_fly(X_max_vir)
            iajb_Y = as_iajb_fly(Y_max_vir)

            ijab_X = as_ijab_fly(X_max_vir)
            ijab_Y = as_ijab_fly(Y_max_vir)

            ibja_X = as_ibja_fly(X_max_vir)
            ibja_Y = as_ibja_fly(Y_max_vir)

            delta_X = delta_max_vir_fly(X_max_vir)
            delta_Y = delta_max_vir_fly(Y_max_vir)

            AX = delta_X + 2*iajb_X - a_x*ijab_X
            AY = delta_Y + 2*iajb_Y - a_x*ijab_Y

            BX = 2*iajb_X - a_x*ibja_X
            BY = 2*iajb_Y - a_x*ibja_Y

            U1 = np.zeros_like(X)
            U2 = np.zeros_like(X)

            U1[:,:max_vir,:] = AX + BY
            U2[:,:max_vir,:] = AY + BX

            U1 = U1.reshape(n_occ*max_vir,-1)
            U2 = U2.reshape(n_occ*max_vir,-1)

            return U1, U2

        def TDDFT_as_spolar_mv(X):

            ''' for RSH, a_x=1
                (A+B)X = delta_fly(V) + 4*iajb_fly(V) - a_x*ijab_fly(V) - a_x*ibja_fly(V)
            '''

            X = X.reshape(n_occ, max_vir, -1)
            U = delta_max_vir_fly(X) + 4*as_iajb_fly(X) - a_x*as_ijab_fly(X) - a_x*as_ibja_fly(X)
            U = U.reshape(n_occ*max_vir, -1)
            return U

        return TDA_as_mv, TDDFT_as_mv, TDDFT_as_spolar_mv



    def build(self):

        auxmol = self.gen_auxmol()

        eri2c, eri3c = self.gen_electron_int(mol=mol, auxmol=auxmol, omega=0)

        eri2c_erf, eri3c_erf = self.gen_electron_int(mol=mol, auxmol=auxmol, omega=self.omega)

        ''' alpha + beta = 1
        '''
        alpha_RSH = 0.157706
        beta_RSH = 0.842294

        ''' the 2c2e and 3c2e integrals with RSH
        '''
        eri2c_RSH = alpha_RSH*eri2c + beta_RSH*eri2c_erf
        eri3c_RSH = alpha_RSH*eri3c + beta_RSH*eri3c_erf

        '''the GAMMA tensors without RSH
        '''
        (GAMMA_ij,
        GAMMA_ab,
        GAMMA_ia,
        GAMMA_J_ia,
        GAMMA_J_ij) = self.gen_GAMMA(eri2c=eri2c, eri3c=eri3c)

        '''the GAMMA tensors with RSH
        '''
        (GAMMA_ij_RSH,
        GAMMA_ab_RSH,
        GAMMA_ia_RSH,
        GAMMA_J_ia_RSH,
        GAMMA_J_ij_RSH) = self.gen_GAMMA(eri2c=eri2c_RSH, eri3c=eri3c_RSH)

        ''' the 4c2e integrals without RSH
        '''
        (as_iajb_fly,
        as_ijab_fly,
        as_ibja_fly) = self.gen_as_2e_fly(
                        GAMMA_ij=GAMMA_ij,
                        GAMMA_ab=GAMMA_ab,
                        GAMMA_ia=GAMMA_ia,
                        GAMMA_J_ia=GAMMA_J_ia,
                        GAMMA_J_ij=GAMMA_J_ij)

        ''' the 4c2e integrals with RSH
        '''
        (as_iajb_fly_RSH,
        as_ijab_fly_RSH,
        as_ibja_fly_RSH) = self.gen_as_2e_fly(
                        GAMMA_ij=GAMMA_ij_RSH,
                        GAMMA_ab=GAMMA_ab_RSH,
                        GAMMA_ia=GAMMA_ia_RSH,
                        GAMMA_J_ia=GAMMA_J_ia_RSH,
                        GAMMA_J_ij=GAMMA_J_ij_RSH)

        ''' use columb type integral without RSH,
            only the exchange type integral with RSH
        '''
        (TDA_as_mv,
        TDDFT_as_mv,
        TDDFT_as_spolar_mv) = self.gen_as_mv_fly(
                                as_iajb_fly=as_iajb_fly,
                                as_ijab_fly=as_ijab_fly_RSH,
                                as_ibja_fly=as_ibja_fly_RSH)

        return TDA_as_mv, TDDFT_as_mv, TDDFT_as_spolar_mv
