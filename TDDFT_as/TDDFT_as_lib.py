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
    '''
    alpha + beta = 1
    wb97x
    '''
    a_x = 1
    alpha_RSH = 0.157706
    beta_RSH = 0.842294


class TDDFT_as(object):
    def __init__(self, Uk = args.Uk):
        if args.Uread == True:
            '''
            read Uk from txt
            '''
            file = os.popen('ls *_Uk.txt').readlines()[0].replace('\n', '')
            self.Uk = float(np.loadtxt(file))

        else:
            self.Uk = Uk

    def gen_auxmol(self, add_p = True):
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
        # print('auxmol_basis_keys', auxmol_basis_keys)
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
                exp = Uk/(parameter.RADII[atom])**2

            if atom != 'H' and add_p == True:
                aux_basis[atom_index] = [[0, [exp, 1.0]],[1, [exp, 1.0]]]
            else:
                aux_basis[atom_index] = [[0, [exp, 1.0]]]

        auxmol.basis = aux_basis
        auxmol.build()
        [print(k, v) for k, v in auxmol.basis.items()]

        return auxmol


    def gen_electron_int(self, mol, auxmol, RS_omega=0):

        nao = mol.nao_nr()
        naux = auxmol.nao_nr()

        auxmol.set_range_coulomb(RS_omega)
        mol.set_range_coulomb(RS_omega)


        '''
        (pq|rs) = Σ_PQ (pq|P)(P|Q)^-1(Q|rs)
        2 center 2 electron integral (P|Q)
        N_auxbf * N_auxbf
        '''
        eri2c = auxmol.intor('int2c2e_sph')
        '''3 center 2 electron integral (pq|P)
            N_bf * N_bf * N_auxbf
        '''
        pmol = mol + auxmol

        eri3c = pmol.intor('int3c2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,
                            mol.nbas,mol.nbas+auxmol.nbas))

        return eri2c, eri3c

    def gen_GAMMA(self, eri2c, eri3c, max_vir=max_vir, calc='all'):

        N_auxbf = eri2c.shape[0]

        eri2c_inv = np.linalg.inv(eri2c)

        '''PQ is eri2c shape, N_auxbf'''
        Delta = einsum("PQ, uvQ -> uvP", eri2c_inv, eri3c)
        GAMMA = einsum("up, vq, uvP -> pqP", un_ortho_C_matrix, un_ortho_C_matrix, Delta)

        if calc in ['coulomb','all']:
            '''(ia|jb) columb'''
            GAMMA_ia = np.zeros((n_occ, max_vir, N_auxbf))
            GAMMA_ia[:,:,:] = GAMMA[:n_occ,n_occ:n_occ+max_vir,:]
            GAMMA_J_ia = einsum("iaA , AB -> iaB", GAMMA_ia, eri2c)


        if calc in ['exchange','all']:
            '''(ij|ab) exchange'''
            GAMMA_ij = np.zeros((n_occ, n_occ, N_auxbf))
            GAMMA_ab = np.zeros((max_vir, max_vir, N_auxbf))
            GAMMA_ij[:,:,:] = GAMMA[:n_occ,:n_occ,:]
            GAMMA_ab[:,:,:] = GAMMA[n_occ:n_occ+max_vir,n_occ:n_occ+max_vir,:]
            GAMMA_J_ij = einsum("ijA , AB -> ijB", GAMMA_ij, eri2c)


        if calc == 'all':
            return GAMMA_ia, GAMMA_J_ia, GAMMA_ab, GAMMA_J_ij
        elif calc == 'coulomb':
            return GAMMA_ia, GAMMA_J_ia
        elif calc == 'exchange':
            return GAMMA_ab, GAMMA_J_ij

        '''
        use coulomb type integral without RSH,
        only the exchange type integral with RSH
        '''

    # def gen_2e_fly(self, GAMMA_ia, GAMMA_J_ia, GAMMA_ij, GAMMA_ab, GAMMA_J_ij):
    #
    #     def iajb_fly(V):
    #         '''(ia|jb)'''
    #         GAMMA_jb_V = einsum("iaA, iam -> Am", GAMMA_ia, V)
    #         iajb_V  = einsum("iaA, Am -> iam", GAMMA_J_ia, GAMMA_jb_V)
    #         return iajb_V
    #
    #     def ijab_fly(V):
    #         '''(ij|ab)'''
    #         GAMMA_ab_V = einsum("abA, jbm -> jAam", GAMMA_ab, V)
    #         ijab_V  = einsum("ijA, jAam -> iam", GAMMA_J_ij, GAMMA_ab_V)
    #         return ijab_V
    #
    #     def ibja_fly(V):
    #         '''
    #         the Forck exchange energy in B matrix
    #         (ib|ja)
    #         '''
    #         GAMMA_ja_V = einsum("ibA,jbm->Ajim", GAMMA_ia, V)
    #         ibja_V = einsum("jaA,Ajim->iam", GAMMA_J_ia, GAMMA_ja_V)
    #         return ibja_V
    #
    #     return iajb_fly, ijab_fly, ibja_fly

    def gen_coulomb(self, GAMMA_ia, GAMMA_J_ia):
        def iajb_fly(V):
            '''(ia|jb)'''
            GAMMA_jb_V = einsum("iaA, iam -> Am", GAMMA_ia, V)
            iajb_V  = einsum("iaA, Am -> iam", GAMMA_J_ia, GAMMA_jb_V)
            return iajb_V
        return iajb_fly

    def gen_exchange(self, GAMMA_ia, GAMMA_J_ia, GAMMA_ab, GAMMA_J_ij):
        def ijab_fly(V):
            '''(ij|ab)'''
            GAMMA_ab_V = einsum("abA, jbm -> jAam", GAMMA_ab, V)
            ijab_V  = einsum("ijA, jAam -> iam", GAMMA_J_ij, GAMMA_ab_V)
            return ijab_V
        def ibja_fly(V):
            '''
            the Forck exchange energy in B matrix
            (ib|ja)
            '''
            GAMMA_ja_V = einsum("ibA,jbm->Ajim", GAMMA_ia, V)
            ibja_V = einsum("jaA,Ajim->iam", GAMMA_J_ia, GAMMA_ja_V)
            return ibja_V
        return ijab_fly, ibja_fly

    def gen_mv_fly(self, iajb_fly, ijab_fly, ibja_fly, a_x=a_x):

        def TDA_mv(V):
            ''' return AX
                for RSH, a_x  =1
                AV = delta_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
            '''
            V = V.reshape(n_occ, max_vir, -1)
            MV = delta_max_vir_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
            MV = MV.reshape(n_occ*max_vir, -1)
            return MV

        def TDDFT_mv(X, Y):
            '''return AX+BY and AY+BX
                for RSH, a_x = 1
                AV = delta_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
                BV = 2*iajb_fly(V) - a_x*ibja_fly(V)
            '''
            X = X.reshape(n_occ, max_vir, -1)
            Y = Y.reshape(n_occ, max_vir, -1)

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

        def TDDFT_spolar_mv(X):

            ''' for RSH, a_x=1
                (A+B)X = delta_fly(V) + 4*iajb_fly(V) - a_x*ijab_fly(V) - a_x*ibja_fly(V)
            '''

            X = X.reshape(n_occ, max_vir, -1)
            U = delta_max_vir_fly(X) + 4*iajb_fly(X) - a_x*ijab_fly(X) - a_x*ibja_fly(X)
            U = U.reshape(n_occ*max_vir, -1)
            return U

        return TDA_mv, TDDFT_mv, TDDFT_spolar_mv

    def build(self):

        if args.coulomb_ex != 'none':
            auxmol_with_p = self.gen_auxmol(add_p = True)
        if args.coulomb_ex != 'all':
            auxmol_without_p = self.gen_auxmol(add_p = False)

        if args.coulomb_ex == 'none':
            auxmol_cl = auxmol_without_p
            auxmol_ex = auxmol_without_p

        elif args.coulomb_ex == 'all':
            auxmol_cl = auxmol_with_p
            auxmol_ex = auxmol_with_p

        elif args.coulomb_ex == 'coulomb':
            auxmol_cl = auxmol_with_p
            auxmol_ex = auxmol_without_p

        elif args.coulomb_ex == 'exchange':
            auxmol_cl = auxmol_without_p
            auxmol_ex = auxmol_with_p

        '''
        the 2c2e and 3c2e integrals with/without RSH
        (ij|ab) -> alpha*(ij|1/r|ab)  + beta*(ij|erf(oemga)/r|ab)
        '''
        eri2c_cl, eri3c_cl = self.gen_electron_int(mol=mol, auxmol=auxmol_cl, RS_omega=0)
        eri2c_ex, eri3c_ex = self.gen_electron_int(mol=mol, auxmol=auxmol_ex, RS_omega=0)

        if args.functional in parameter.RSH_F:
            eri2c_erf, eri3c_erf = self.gen_electron_int(mol=mol, auxmol=auxmol_ex, RS_omega=parameter.RSH_omega)
            eri2c_ex = alpha_RSH*eri2c_ex + beta_RSH*eri2c_erf
            eri3c_ex = alpha_RSH*eri3c_ex + beta_RSH*eri3c_erf


        print('eri2c_cl.shape', eri2c_cl.shape)
        # print('eri3c_cl.shape', eri3c_cl.shape)
        print('eri2c_ex.shape', eri2c_ex.shape)
        # print('eri3c_ex.shape', eri3c_ex.shape)

        '''
        (ia|jb) tensors for coulomb always have no RSH, might have s/p orbit
        (ij|ab) tensors for exchange might have RSH, might have s/p orbit
        '''
        GAMMA_ia_cl, GAMMA_J_ia_cl = self.gen_GAMMA(
                                eri2c=eri2c_cl, eri3c=eri3c_cl, calc='coulomb')

        GAMMA_ia_ex, GAMMA_J_ia_ex, GAMMA_ab_ex, GAMMA_J_ij_ex = self.gen_GAMMA(
                                eri2c=eri2c_ex, eri3c=eri3c_ex, calc='all')


        '''(pq|rs)'''

        iajb_fly = self.gen_coulomb(GAMMA_ia=GAMMA_ia_cl,
                                  GAMMA_J_ia=GAMMA_J_ia_cl)

        ijab_fly, ibja_fly = self.gen_exchange(GAMMA_ia=GAMMA_ia_ex,
                                             GAMMA_J_ia=GAMMA_J_ia_ex,
                                               GAMMA_ab=GAMMA_ab_ex,
                                             GAMMA_J_ij=GAMMA_J_ij_ex)



        (TDA_mv,
        TDDFT_mv,
        TDDFT_spolar_mv) = self.gen_mv_fly(
                            iajb_fly=iajb_fly,
                            ijab_fly=ijab_fly,
                            ibja_fly=ibja_fly)


        return TDA_mv, TDDFT_mv, TDDFT_spolar_mv

'''
40eV
wb97x
coulomb_p  5.36e-06  31
[ 7.5765133   9.60608489  9.67092002 10.46669613 10.8205826 ]
exchange_p 3.70e-06    32
[ 7.35490183  9.46473101  9.48484662 10.43569971 10.79267799]
all 3.70e-06   32
[ 7.58112216  9.57737997  9.66995834 10.47484652 10.82029651]
all standard, March26 3.70e-06 32
[ 7.58112214  9.57737996  9.66995833 10.47484651 10.82029649]
none  3.82e-06 32
[ 7.35398054  9.46621718  9.51521824 10.42833804 10.79068673]
none standard, March26 3.82e-06 32
[ 7.35398052  9.46621717  9.51521823 10.42833802 10.79068672]




pbe0
coulomb_p   7.93e-06  27
[ 7.21230947  8.92134037  9.12940136  9.85071589 10.13094993]
exchange_p  6.41e-06 28
[ 7.02818028  8.75818829  9.03433726  9.8330746  10.1162566 ]
all  7.57e-06 27
[ 7.21583381  8.92161957  9.11060839  9.86052239 10.13364727]
all standard, March26 7.57e-06 27
[ 7.21584106  8.92162812  9.11061339  9.86053042 10.13365608]
none  6.42e-06 28
[ 7.02841663  8.75901625  9.05576802  9.82355389 10.11311337]
none standard, March26  6.42e-06 28
[ 7.02842094  8.75902107  9.05577121  9.82355838 10.11311842]




1000000eV
pbe0
coulomb  7.74e-06   26
[ 7.18168397  8.90493142  9.11059227  9.8432434  10.12418766]
exchange  6.04e-06 27
[ 7.02218677  8.75441354  9.02156881  9.827227   10.10848812]
all  9.06e-06 26
[ 7.18339121  8.90467917  9.09077771  9.85266457 10.12651742]
none 5.84e-06 27
[ 7.02332443  8.75547601  9.04225908  9.81788812 10.10547884]


'''
