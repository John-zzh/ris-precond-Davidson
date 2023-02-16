from pyscf import gto, scf, dft, tddft, data, lib

from collections import Counter
# from opt_einsum import contract as einsum
import numpy as np
import gc
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from arguments import args
from SCF_calc import (atom_coordinates, mol, n_occ, N_bf, N_atm, delta_hdiag2_fly,
                    un_ortho_C_matrix, delta_hdiag, delta_hdiag2, a_x, delta_fly,
                    n_occ, n_vir, A_size, cl_A_rest_size,
                    cl_rest_vir, cl_truc_occ, cl_rest_occ, cl_rest_vir,
                    ex_rest_vir, ex_truc_occ)
from mathlib import parameter, math

# print('type(delta_hdiag2_fly)',type(delta_hdiag2_fly))

if args.einsum == 'opt':
    from opt_einsum import contract as einsum
elif args.einsum == 'pyscf':
    einsum = lib.einsum
elif args.einsum == 'parallel':
    from einsum2 import einsum2 as einsum

if args.functional in parameter.RSH_F:
    '''
    alpha + beta = 1
    wb97x
    '''
    a_x = 1
    alpha_RSH = 0.157706
    beta_RSH = 0.842294


class TDDFT_as(object):
    # def __init__(self, Uk = args.Uk):
    #     # if args.Uread == True:
    #     #     '''
    #     #     read Uk from txt
    #     #     '''
    #     #     file = os.popen('ls *_Uk.txt').readlines()[0].replace('\n', '')
    #     #     self.Uk = float(np.loadtxt(file))
    #     #
    #     # else:
    #     self.Uk = Uk

    def gen_auxmol(self, U=1, add_p=True, full_fitting=False):
        print('asigning auxiliary basis set, add p function =', add_p)
        print('U =', U)


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

            exp = parameter.as_exp[atom] * U

            if atom != 'H' and add_p == True:
                aux_basis[atom_index] = [[0, [exp, 1.0]],[1, [exp, 1.0]]]
            else:
                aux_basis[atom_index] = [[0, [exp, 1.0]]]

        if full_fitting:
            print('full aux_basis')
            aux_basis = args.basis_set+"-jkfit"
        auxmol.basis = aux_basis
        auxmol.build()
        print(auxmol._basis)
        # [print(k, v) for k, v in auxmol.basis.items()]

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
        '''
        3 center 2 electron integral (pq|P)
        N_bf * N_bf * N_auxbf
        '''
        pmol = mol + auxmol

        eri3c = pmol.intor('int3c2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,
                            mol.nbas,mol.nbas+auxmol.nbas))

        return eri2c, eri3c
    #
    # def gen_diag_iajb(self, GAMMA_J_ia, GAMMA_ia, n_occ=n_occ,n_vir=n_vir):
    #     '''
    #     generate the diagonal elements of (ia|jb), utilizing the RIJK
    #     '''
    #
    #     diag = einsum("iaA,iaA->ia", GAMMA_J_ia, GAMMA_ia)
    #     #
    #     # full_ijab = einsum("iaA,jbA-> iajb", GAMMA_J_ia, GAMMA_ia)
    #     # standard_diag = np.zeros((n_occ,n_vir))
    #     # for i in range(n_occ):
    #     #     for a in range(n_vir):
    #     #         standard_diag[i][a] = full_ijab[i][a][i][a]
    #     #
    #     # check = np.linalg.norm(standard_diag - diag)
    #     # print('check diag iajb accuracy', check)
    #
    #     return diag
    #
    # def gen_diag_ijab(self, GAMMA_J_ij, GAMMA_ab, n_occ=n_occ,n_vir=n_vir):
    #     '''
    #     generate the diagonal elements of (ij|ab),utilizing the RIJK
    #     '''
    #
    #     diag = einsum("iiA,aaA->ia", GAMMA_J_ij, GAMMA_ab)
    #     #
    #     # full_ijab = einsum("ijA,abA-> ijab", GAMMA_J_ij, GAMMA_ab)
    #     # standard_diag = np.zeros((n_occ,n_vir))
    #     # for i in range(n_occ):
    #     #     for a in range(n_vir):
    #     #         standard_diag[i][a] = full_ijab[i][i][a][a]
    #     #
    #     # check = np.linalg.norm(standard_diag - diag)
    #     # print('check diag ijab accuracy', check)
    #
    #     return diag



    def gen_GAMMA(self, eri2c, eri3c, truc_occ, rest_vir, calc, n_occ=n_occ):

        N_auxbf = eri2c.shape[0]

        eri2c_inv = np.linalg.inv(eri2c)

        '''
        PQ is eri2c shape, N_auxbf
        GAMMA.shape = N_bf, N_bf, N_auxbf
        '''
        Delta = einsum("PQ,uvQ->uvP", eri2c_inv, eri3c)
        GAMMA = einsum("up,vq,uvP->pqP", un_ortho_C_matrix, un_ortho_C_matrix, Delta)


        print('eri3c.shape', eri3c.shape)

        '''
        N_bf:
            truc_occ           rest_occ           rest_vir      truc_vir
      ==============#======================|---------------#------------------------------
                     n_occ                         n_vir

                    truc_occ     rest_occ       rest_vir    truc_vir
                -------------|-------------||-------------|-------------
                |            |             ||             |            |
                |            |             ||             |            |
        truc_occ|            |             ||             |            |
                |            |             ||             |            |
                -------------|-------------||-------------|-------------
                |            |             ||             |            |
        rest_occ|            |   GAMMA_ij  ||  GAMMA_ia   |            |
                |            |             ||             |            |
                |            |             ||             |            |
        n_occ   =============|=============||=============|=============-
                |            |             ||             |            |
        rest_vir|            |             ||  GAMMA_ab   |            |
                |            |             ||             |            |
                |            |             ||             |            |
                -------------|-------------||-------------|-------------
                |            |             ||             |            |
        truc_vir|            |             ||             |            |
                |            |             ||             |            |
                |            |             ||             |            |
                -------------|-------------||-------------|-------------

        '''

        GAMMA_ia = math.copy_array(GAMMA[truc_occ:n_occ,n_occ:n_occ+rest_vir,:])
        GAMMA_J_ia = einsum("iaA,AB->iaB", GAMMA_ia, eri2c)

        if calc == 'coulomb':
            '''(ia|jb) coulomb term'''

            diag_cl = None
            if args.woodbury:
                # diag_cl = self.gen_diag_iajb(GAMMA_J_ia=GAMMA_J_ia, GAMMA_ia=GAMMA_ia)
                diag_cl = einsum("iaA,iaA->ia", GAMMA_J_ia, GAMMA_ia)

            return GAMMA_ia, GAMMA_J_ia, diag_cl

        if calc == 'exchange':
            '''(ij|ab) exchange term '''

            GAMMA_ij = math.copy_array(GAMMA[truc_occ:n_occ, truc_occ:n_occ,:])
            GAMMA_ab = math.copy_array(GAMMA[n_occ:n_occ+rest_vir,n_occ:n_occ+rest_vir,:])

            GAMMA_J_ij = einsum("ijA,AB->ijB", GAMMA_ij, eri2c)
            # print('eri2c.shape in buiding Gamma', eri2c.shape)

            diag_ex = None
            if args.woodbury:
                # diag_ex = self.gen_diag_ijab(GAMMA_J_ij=GAMMA_J_ij, GAMMA_ab=GAMMA_ab)
                diag_ex = einsum("iiA,aaA->ia", GAMMA_J_ij, GAMMA_ab)

            return GAMMA_ia, GAMMA_J_ia, GAMMA_ab, GAMMA_J_ij, diag_ex

    '''
    use coulomb type integral without RSH,
    only the exchange type integral with RSH
    '''
    def gen_coulomb(self, GAMMA_ia, GAMMA_J_ia):
        def iajb_fly(V):
            '''(ia|jb)'''
            GAMMA_jb_V = einsum("iaA,iam->Am", GAMMA_ia, V)
            iajb_V  = einsum("iaA,Am->iam", GAMMA_J_ia, GAMMA_jb_V)
            return iajb_V
        return iajb_fly

    def gen_exchange(self, GAMMA_ia, GAMMA_J_ia, GAMMA_ab, GAMMA_J_ij):
        def ijab_fly(V):
            '''(ij|ab)'''
            GAMMA_ab_V = einsum("abA,jbm->jAam", GAMMA_ab, V)
            ijab_V  = einsum("ijA,jAam->iam", GAMMA_J_ij, GAMMA_ab_V)
            return ijab_V

        def ibja_fly(V):
            '''
            the Forck exchange energy in B matrix
            (ib|ja)
            '''
            GAMMA_ja_V = einsum("ibA,jbm->Ajim", GAMMA_ia, V)
            ibja_V = einsum("jaA,Ajim->iam", GAMMA_J_ia, GAMMA_ja_V)
            # if args.woodbury:
            #     ibja_V = einsum("iaA,iaA,iam -> iam", GAMMA_J_ia, GAMMA_ia, V)
            return ibja_V
        return ijab_fly, ibja_fly


    def gen_mv_fly(self, iajb_fly, ijab_fly, ibja_fly, a_x=a_x, delta_hdiag2_fly=delta_hdiag2_fly):

        def TDA_mv(X):
            ''' return AX
                for RSH, a_x = 1
                AV = delta_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
            '''
            X_cl = X.reshape(cl_rest_occ, cl_rest_vir, -1)
            X_ex = X_cl[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:]

            AX = delta_hdiag2_fly(X_cl)
            AX += 2*iajb_fly(X_cl)
            AX[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:] -= a_x*ijab_fly(X_ex)
            AX = AX.reshape(cl_A_rest_size, -1)
            return AX

        def TDDFT_mv(X, Y):
            '''
            [A B][X] = [AX+BY] = [U1]
            [B A][Y]   [AY+BX]   [U2]
            we want AX+BY and AY+BX
            instead of computing AX+BY and AY+BX directly
            we compute (A+B)(X+Y) and (A-B)(X-Y)
            it can save one tensor contraction then the former method

            for RSH, a_x = 1
            (A+B)V = delta_fly(V) + 4*iajb_fly(V) - a_x*[ibja_fly(V) + ijab_fly(V)]
            (A-B)V = delta_fly(V) + a_x[ibja_fly(V) - ijab_fly(V)]

                            --------------#-----------------------
            truc_occ        |    hdiag1   |                      |
                            #-------------#                      |
                            |             |        hdiag3        |
            rest_occ        |    hdiag2   |                      |
                            |             |                      |
                            --------------#-----------------------
                              rest_vir      truc_vir
            '''
            X = X.reshape(cl_rest_occ, cl_rest_vir, -1)
            Y = Y.reshape(cl_rest_occ, cl_rest_vir, -1)


            X_p_Y_cl = X + Y
            X_m_Y_cl = X - Y

            X_p_Y_ex = X_p_Y_cl[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:]
            X_m_Y_ex = X_m_Y_cl[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:]


            A_p_B_X_p_Y = delta_hdiag2_fly(X_p_Y_cl)
            A_p_B_X_p_Y += 4*iajb_fly(X_p_Y_cl)
            A_p_B_X_p_Y[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:] -= a_x*(ibja_fly(X_p_Y_ex) + ijab_fly(X_p_Y_ex))

            A_m_B_X_m_Y = delta_hdiag2_fly(X_m_Y_cl)
            A_m_B_X_m_Y[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:] += a_x*(ibja_fly(X_m_Y_ex) - ijab_fly(X_m_Y_ex))

            U1 = (A_p_B_X_p_Y + A_m_B_X_m_Y)/2
            U2 = (A_p_B_X_p_Y - A_m_B_X_m_Y)/2

            U1 = U1.reshape(cl_A_rest_size,-1)
            U2 = U2.reshape(cl_A_rest_size,-1)

            return U1, U2

        def TDDFT_spolar_mv(X):

            ''' for RSH, a_x=1
                (A+B)X = delta_fly(V) + 4*iajb_fly(V) - a_x*ijab_fly(V) - a_x*ibja_fly(V)
            '''
            X_cl = X.reshape(cl_rest_occ, cl_rest_vir, -1)
            X_ex = X_cl[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:]


            ABX = delta_hdiag2_fly(X_cl)
            ABX +=  4*iajb_fly(X_cl)
            ABX[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:] -= a_x*ijab_fly(X_ex) - a_x*ibja_fly(X_ex)
            ABX = ABX.reshape(cl_A_rest_size, -1)
            return ABX

        return TDA_mv, TDDFT_mv, TDDFT_spolar_mv

    def build(self, delta_hdiag2_fly=delta_hdiag2_fly):

        auxmol_cl = self.gen_auxmol(U=args.coulomb_U, add_p=args.coulomb_aux_add_p, full_fitting=False)
        auxmol_ex = self.gen_auxmol(U=args.exchange_U, add_p=args.exchange_aux_add_p, full_fitting=False)

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
        print('eri2c_ex.shape', eri2c_ex.shape)

        '''
        (ia|jb) tensors for coulomb always have no RSH, might have s/p orbit
        (ij|ab) tensors for exchange might have RSH, might have s/p orbit
        '''
        GAMMA_ia_cl, GAMMA_J_ia_cl,diag_cl_risp = self.gen_GAMMA(
                                eri2c=eri2c_cl, eri3c=eri3c_cl,
                                truc_occ=cl_truc_occ,
                                rest_vir=cl_rest_vir,
                                calc='coulomb')

        GAMMA_ia_ex, GAMMA_J_ia_ex, GAMMA_ab_ex, GAMMA_J_ij_ex, diag_ex_risp = self.gen_GAMMA(
                                eri2c=eri2c_ex, eri3c=eri3c_ex,
                                truc_occ=ex_truc_occ,
                                rest_vir=ex_rest_vir,
                                calc='exchange')

        print('type(delta_hdiag2_fly)',type(delta_hdiag2_fly))
        delta_hdiag2_fly = delta_hdiag2_fly
        if args.woodbury and args.full_fitting:
            '''
            rebuild the auxmol and rebuild the 3c2e and 2c2e with full auxbasis
            and grep the diagonal elements
            '''
            auxmol_full = self.gen_auxmol(U=args.exchange_U, add_p=args.exchange_aux_add_p, full_fitting=True)
            eri2c_full, eri3c_full = self.gen_electron_int(mol=mol, auxmol=auxmol_full, RS_omega=0)

            *_, diag_cl_full = self.gen_GAMMA(eri2c=eri2c_full, eri3c=eri3c_full,
                                                truc_occ=0,
                                                rest_vir=n_vir,
                                                calc='coulomb')
            *_, diag_ex_full = self.gen_GAMMA(eri2c=eri2c_full, eri3c=eri3c_full,
                                                truc_occ=0,
                                                rest_vir=n_vir,
                                                calc='exchange')

            diag_cl_correction = diag_cl_full - diag_cl_risp
            diag_ex_correction = diag_ex_full - diag_ex_risp


            def delta_hdiag2_fly(V):
                '''
                make sure no trunction on the coulomb, so that delta_hdiag2 = delta_hdiag
                delta_hdiag2 is KS orbital energy diff
                '''
                hidag_merge = delta_hdiag + diag_cl_correction - diag_ex_correction
                hidag_merge_v = einsum("ia,iam->iam", hidag_merge, V)
                return hidag_merge_v



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
                            ibja_fly=ibja_fly,
                    delta_hdiag2_fly=delta_hdiag2_fly)

        return TDA_mv, TDDFT_mv, TDDFT_spolar_mv
