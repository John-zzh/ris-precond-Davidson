from pyscf import gto, scf, dft, tddft, data, lib
from arguments import args
from collections import Counter
from opt_einsum import contract as einsum
import numpy as np
from SCF_calc import (atom_coordinates, mol, n_occ, n_vir, N_bf, N_atm,
                                    un_ortho_C_matrix, delta_hdiag, a_x)
# U_list = [0.125]*N_atm

#PBE0
# U_list = [
# 0.08076662560543,
# 0.09590984529187832,
# 0.124,
# 0.124,
# 0.124,
# 0.11471069052940008]

#wb97x
# U_list = [
# 1.2161373541063307,
# 2.1903830225866874,
# 0.124,
# 0.124,
# 0.124,
# 0.17504205914879986]

if args.U == True:
    with open(glob.glob('*_U.txt')[0]) as f:
        U_file = f.readlines()
        U_list = [float(i) for i in U_file]
else:
    U_list = [0.123]*N_atm

class TDDFT_as(object):

    def __init__(self, U_list, omega=0):
        self.U_list = U_list
        self.omega = omega

    def gen_auxmol(self):
        print('asigning auxiliary basis set')
        '''
        U_list is a list of gaussian index for each atom
        '''
        U_list=self.U_list


        auxmol = gto.M(atom=atom_coordinates)

        atom_count = Counter(auxmol.elements)

        auxmol_basis_keys = []
        for key in atom_count:
            for i in range(atom_count[key]):
                if i > 0:
                    auxmol_basis_keys.append(key+'^'+str(i+1))
                else:
                    auxmol_basis_keys.append(key)

        basis = [[[0, [i, 1.0]]] for i in U_list]
        aux_basis = dict(zip(auxmol_basis_keys, basis))
        [print("{:<5s} {:<5.3f}".format(key, value[0][1][0])) for key, value in aux_basis.items()]
        '''
        aux_basis = {
        'O': [[0, [0.1235, 1.0]]],
        'C': [[0, [0.1235, 1.0]]],
        'H': [[0, [0.1510, 1.0]]],
        'H^2': [[0, [0.1510, 1.0]]],
        'H^3': [[0, [0.1510, 1.0]]],
        'H^4': [[0, [0.1510, 1.0]]]
        }
        '''
        auxmol.basis = aux_basis

        auxmol.build()

        return auxmol


    def gen_electron_int(self, mol, auxmol, omega=0):

        nao = mol.nao_nr()
        naux = auxmol.nao_nr()

        auxmol.set_range_coulomb(omega)
        mol.set_range_coulomb(omega)


        '''2 center 2 electron integral
            N_atm * N_atm
        '''
        eri2c = auxmol.intor('int2c2e_sph')

        '''3 center 2 electron integral
            N_bf * N_bf * N_atm
        '''
        pmol = mol + auxmol

        eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,
                                                mol.nbas,mol.nbas+auxmol.nbas))

        return eri2c, eri3c

    def gen_GAMMA(self, eri2c, eri3c):

        eri2c_inv = np.linalg.inv(eri2c)

        Delta = einsum("PQ, uvQ -> uvP", eri2c_inv, eri3c)
        GAMMA = einsum("up, vq, uvP -> pqP", un_ortho_C_matrix, un_ortho_C_matrix, Delta)

        GAMMA_ij = np.zeros((n_occ, n_occ, N_atm))
        GAMMA_ij[:,:,:] = GAMMA[:n_occ,:n_occ,:]

        GAMMA_ab = np.zeros((n_vir, n_vir, N_atm))
        GAMMA_ab[:,:,:] = GAMMA[n_occ:,n_occ:,:]

        GAMMA_ia = np.zeros((n_occ, n_vir, N_atm))
        GAMMA_ia[:,:,:] = GAMMA[:n_occ,n_occ:,:]

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

        return as_iajb_fly, as_ijab_fly

    def gen_as_mv_fly(self, as_iajb_fly, as_ijab_fly):

        def as_mv(V):
            '''return AX'''
            # print('using as')
            # a_x=1
            V = V.reshape(n_occ, n_vir, -1)
            '''AX =  delta_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)'''
            MV = einsum("ia,iam->iam", delta_hdiag, V) + 2*as_iajb_fly(V) - a_x*as_ijab_fly(V)
            MV = MV.reshape(n_occ*n_vir,-1)
            return MV

        return as_mv

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
        as_ijab_fly) = self.gen_as_2e_fly(
                        GAMMA_ij=GAMMA_ij,
                        GAMMA_ab=GAMMA_ab,
                        GAMMA_ia=GAMMA_ia,
                        GAMMA_J_ia=GAMMA_J_ia,
                        GAMMA_J_ij=GAMMA_J_ij)

        ''' the 4c2e integrals with RSH
        '''
        (as_iajb_fly_RSH,
        as_ijab_fly_RSH) = self.gen_as_2e_fly(
                        GAMMA_ij=GAMMA_ij_RSH,
                        GAMMA_ab=GAMMA_ab_RSH,
                        GAMMA_ia=GAMMA_ia_RSH,
                        GAMMA_J_ia=GAMMA_J_ia_RSH,
                        GAMMA_J_ij=GAMMA_J_ij_RSH)

        ''' use columb type integral without RSH,
            only the exchange type integral with RSH
        '''
        self.TDA_as_mv = self.gen_as_mv_fly(
                        as_iajb_fly=as_iajb_fly,
                        as_ijab_fly=as_ijab_fly_RSH)


TDDFT_as = TDDFT_as(U_list=U_list, omega=args.RSH_omega)
TDDFT_as.build()
as_mv = TDDFT_as.TDA_as_mv
