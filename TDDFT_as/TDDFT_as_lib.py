from pyscf import gto, scf, dft, tddft, data, lib

from collections import Counter
# from opt_einsum import contract as einsum
import numpy as np
import gc
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from arguments import args
from SCF_calc import (mf, atom_coordinates, mol, n_occ, N_bf, N_atm, delta_hdiag2_fly,
                    un_ortho_C_matrix, delta_hdiag, delta_hdiag2, a_x, delta_fly,
                    n_occ, n_vir, A_size, cl_A_rest_size,
                    cl_rest_vir, cl_truc_occ, cl_rest_occ, cl_rest_vir,
                    ex_rest_vir, ex_truc_occ)
from mathlib import parameter, math
from functools import partial

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
    print('RSH, a_x = 1')
    alpha_RSH = 0.157706
    beta_RSH = 0.842294


class TDDFT_as(object):

    def gen_auxmol(self, U=1, add_p=False, add_d=False, full_fitting=False):
        print('asigning auxiliary basis set, add p function =', add_p)
        print('U =', U)
        '''
        parse_arg = False turns off PySCF built-in output file
        '''
        auxmol = gto.M(atom=atom_coordinates, parse_arg = False)

        if not full_fitting:
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

                aux_basis[atom_index] = [[0, [exp, 1.0]]]


                if atom != 'H':
                    if add_p:
                        aux_basis[atom_index].append([1, [exp, 1.0]])
                    if add_d:
                        aux_basis[atom_index].append([2, [exp, 1.0]])



        else:
            print('full aux_basis')
            aux_basis = args.basis_set+"-jkfit"
        auxmol.basis = aux_basis
        auxmol.build()
        # print(auxmol._basis)
        [print(k, v) for k, v in auxmol._basis.items()]

        return auxmol


    def gen_2c_3c(self, mol, auxmol, RSH_omega=0):

        nao = mol.nao_nr()
        naux = auxmol.nao_nr()

        mol.set_range_coulomb(RSH_omega)
        auxmol.set_range_coulomb(RSH_omega)

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

    def gen_electron_int(self, mol, auxmol_cl, auxmol_ex):

        eri2c_cl, eri3c_cl = self.gen_2c_3c(mol=mol, auxmol=auxmol_cl, RSH_omega=0)
        eri2c_ex, eri3c_ex = self.gen_2c_3c(mol=mol, auxmol=auxmol_ex, RSH_omega=0)

        if args.functional in parameter.RSH_F:
            print('2c2e and 2c2e for RSH (ij|ab)')
            eri2c_erf, eri3c_erf = self.gen_2c_3c(mol=mol, auxmol=auxmol_ex, RSH_omega=parameter.RSH_omega)

            eri2c_ex = alpha_RSH*eri2c_ex + beta_RSH*eri2c_erf
            eri3c_ex = alpha_RSH*eri3c_ex + beta_RSH*eri3c_erf
        else:
            print('2c2e and 2c2e for (ia|jb)')

        return eri2c_cl, eri3c_cl, eri2c_ex, eri3c_ex


    def gen_GAMMA(self, eri2c, eri3c, truc_occ, rest_vir, calc, n_occ=n_occ):

        N_auxbf = eri2c.shape[0]

        '''
        PQ is eri2c shape, N_auxbf
        GAMMA.shape = N_bf, N_bf, N_auxbf
        '''
        Delta = einsum("PQ,uvQ->uvP", np.linalg.inv(eri2c), eri3c)
        GAMMA = einsum("up,vq,uvP->pqP", un_ortho_C_matrix, un_ortho_C_matrix, Delta)

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

            diag_cl = einsum("iaA,iaA->ia", GAMMA_J_ia, GAMMA_ia)

            return GAMMA_ia, GAMMA_J_ia, diag_cl

        if calc == 'exchange':
            '''(ij|ab) exchange term '''

            GAMMA_ij = math.copy_array(GAMMA[truc_occ:n_occ, truc_occ:n_occ,:])
            GAMMA_ab = math.copy_array(GAMMA[n_occ:n_occ+rest_vir,n_occ:n_occ+rest_vir,:])

            GAMMA_J_ij = einsum("ijA,AB->ijB", GAMMA_ij, eri2c)
            # print('eri2c.shape in buiding Gamma', eri2c.shape)

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


    def gen_mv_fly(self, iajb_fly, ijab_fly, ibja_fly, a_x=a_x,
                    diag_cl_correct=None,
                    diag_ex_correct=None,
                             fxc_on=False,
                         max_memory=4000):


        from pyscf.dft import numint
        ni = mf._numint
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, [mf.mo_coeff]*2, [mf.mo_occ*.5]*2, spin=1)
        fxc_kernel = partial(ni.nr_rks_fxc_st, mol=mol,
                                             grids=mf.grids,
                                            xc_code=mf.xc,
                                                dm0=None,
                                         relativity=0,
                                            singlet=True,
                                              rho0=rho0,
                                              vxc=vxc,
                                              fxc=fxc,
                                        max_memory=max_memory)
        orbv = mf.mo_coeff[:,n_occ:]
        orbo = mf.mo_coeff[:,:n_occ]
        def fxc_kernel_fly(X):
            X = X.reshape(cl_rest_occ, cl_rest_vir, -1)
            # x = lib.einsum('Nai->Nia',x)
            # dm = reduce(np.dot, (orbv, 2*x.reshape(:,n_vir,n_occ), orbo.T))
            dm = lib.einsum('pa,iaN,iq->Nqp',orbv,X,orbo.T)
            dmT = lib.einsum('Nqp->Npq',dm)
            v1ao = fxc_kernel(dms_alpha = dm+dmT)
            v1vo = lib.einsum('ap,Npq,qi->Nia',orbv.T, v1ao, orbo)
            v1vo = v1vo.reshape(-1,cl_A_rest_size)
            v1vo = 0.5*v1vo.T
            return v1vo

        def TDA_mv(X):
            ''' return AX
                for RSH, a_x = 1
                AV = delta_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
            '''
            # print('a_x=', a_x)
            X_cl = X.reshape(cl_rest_occ, cl_rest_vir, -1)
            X_ex = X_cl[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:]

            AX = delta_hdiag2_fly(X_cl)
            AX += 2*iajb_fly(X_cl)
            AX[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:] -= a_x*ijab_fly(X_ex)

            if diag_cl_correct and diag_ex_correct:
                # pass
                # print('diag correction')
                AX += 2*diag_cl_correct(X_cl) - a_x*diag_ex_correct(X_cl)
            AX = AX.reshape(cl_A_rest_size, -1)

            if fxc_on:
                AX += fxc_kernel_fly(X)
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
            A_p_B_X_p_Y[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:] -= ( a_x*(ibja_fly(X_p_Y_ex) + ijab_fly(X_p_Y_ex)) )

            A_m_B_X_m_Y = delta_hdiag2_fly(X_m_Y_cl)
            A_m_B_X_m_Y[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:] += ( a_x*(ibja_fly(X_m_Y_ex) - ijab_fly(X_m_Y_ex)) )

            if diag_cl_correct and diag_ex_correct:
                A_p_B_X_p_Y += ( (4-a_x)*diag_cl_correct(X_p_Y_cl) - a_x*diag_ex_correct(X_p_Y_cl) )
                A_m_B_X_m_Y += ( a_x*(diag_cl_correct(X_m_Y_cl) - diag_ex_correct(X_m_Y_cl)) )

            U1 = (A_p_B_X_p_Y + A_m_B_X_m_Y)/2
            U2 = (A_p_B_X_p_Y - A_m_B_X_m_Y)/2

            U1 = U1.reshape(cl_A_rest_size,-1)
            U2 = U2.reshape(cl_A_rest_size,-1)

            return U1, U2

        def TDDFT_spolar_mv(X):

            ''' for RSH, a_x=1
                (A+B)X = delta_fly(V) + 4*iajb_fly(V) - a_x*[ijab_fly(V) + ibja_fly(V)]
            '''
            X_cl = X.reshape(cl_rest_occ, cl_rest_vir, -1)
            X_ex = X_cl[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:]


            ABX = delta_hdiag2_fly(X_cl)
            ABX +=  4*iajb_fly(X_cl)
            ABX[ex_truc_occ-cl_truc_occ:,:ex_rest_vir,:] -= ( a_x* (ibja_fly(X_ex) + ijab_fly(X_ex)) )
            if diag_cl_correct and diag_ex_correct:
                ABX += ( (4 - a_x)*diag_cl_correct(X_cl) - a_x*diag_ex_correct(X_cl) )
            ABX = ABX.reshape(cl_A_rest_size, -1)
            return ABX

        return TDA_mv, TDDFT_mv, TDDFT_spolar_mv

    def build(self):

        auxmol_cl = self.gen_auxmol(U=args.coulomb_U, add_p=args.coulomb_aux_add_p, add_d=args.coulomb_aux_add_d, full_fitting=args.full_fitting)
        auxmol_ex = self.gen_auxmol(U=args.exchange_U, add_p=args.exchange_aux_add_p, full_fitting=args.full_fitting)

        '''
        the 2c2e and 3c2e integrals with/without RSH
        (ij|ab) -> alpha*(ij|1/r|ab)  + beta*(ij|erf(oemga)/r|ab)
        '''
        # eri2c_cl, eri3c_cl = self.gen_electron_int(mol=mol, auxmol=auxmol_cl)
        # eri2c_ex, eri3c_ex = self.gen_electron_int(mol=mol, auxmol=auxmol_ex)

        eri2c_cl, eri3c_cl, eri2c_ex, eri3c_ex = self.gen_electron_int(mol=mol,
                                                                 auxmol_cl=auxmol_cl,
                                                                 auxmol_ex=auxmol_ex)
        # if args.functional in parameter.RSH_F:
        #     eri2c_erf, eri3c_erf = self.gen_electron_int(mol=mol, auxmol=auxmol_ex, RS_omega=parameter.RSH_omega)
        #     eri2c_ex = alpha_RSH*eri2c_ex + beta_RSH*eri2c_erf
        #     eri3c_ex = alpha_RSH*eri3c_ex + beta_RSH*eri3c_erf

        print('eri2c_cl.shape', eri2c_cl.shape)
        print('eri2c_ex.shape', eri2c_ex.shape)

        '''
        (ia|jb) tensors for coulomb always have no RSH,  have sp orbit
        (ij|ab) tensors for exchange might have RSH, might have sp orbit
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

        diag_cl_correct = None
        diag_ex_correct = None
        if args.woodbury and args.full_fitting:
            '''
            rebuild the auxmol and rebuild the 3c2e and 2c2e with full auxbasis
            and grep the diagonal elements
            '''
            auxmol_full = self.gen_auxmol(full_fitting=True)

            eri2c_cl_full, eri3c_cl_full, eri2c_ex_full, eri3c_ex_full = self.gen_electron_int(mol=mol,
                                                                                         auxmol_cl=auxmol_full,
                                                                                         auxmol_ex=auxmol_full)

            *_, diag_cl_full = self.gen_GAMMA(eri2c=eri2c_cl_full, eri3c=eri3c_cl_full,
                                                truc_occ=0,
                                                rest_vir=n_vir,
                                                calc='coulomb')
            *_, diag_ex_full = self.gen_GAMMA(eri2c=eri2c_ex_full, eri3c=eri3c_ex_full,
                                                truc_occ=0,
                                                rest_vir=n_vir,
                                                calc='exchange')

            def diag_cl_correct(V):
                return einsum("ia,iam->iam", args.mix_c*(diag_cl_full - diag_cl_risp), V)

            def diag_ex_correct(V):
                return einsum("ia,iam->iam", args.mix_c*(diag_ex_full - diag_ex_risp), V)


        '''(pq|rs)V'''

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
                     diag_cl_correct=diag_cl_correct,
                     diag_ex_correct=diag_ex_correct,
                             fxc_on=args.fxc_on,
                         max_memory=0.5*args.memory)

        return TDA_mv, TDDFT_mv, TDDFT_spolar_mv
