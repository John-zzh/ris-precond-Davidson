# -*- coding: utf-8 -*-

from arguments import args
from pyscf import gto, scf, dft, tddft, data, lib
import numpy as np
import time
import mathlib.parameter as parameter
import mathlib.math as math
import os, sys
import psutil

from opt_einsum import contract as einsum

basename = args.xyzfile.split('.',1)[0]

def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024**3
    print('{} memory used: {:<.2f} GB'.format(hint, memory))

def SCF_kernel(xyzfile = args.xyzfile,
                   mol = gto.Mole(),
             basis_set = args.basis_set,
               verbose = args.verbose,
            max_memory = args.memory,
                method = args.method,
             checkfile = args.checkfile,
           density_fit = args.density_fit,
            functional = args.functional,
            grid_level = args.grid_level):
    kernel_0 = time.time()

    '''read the xyzfile but ommit its first two lines'''
    with open(xyzfile) as f:
        coordinate = f.read().splitlines()[2:]
        atom_coordinates = [i for i in coordinate if i != '']
    '''build geometry in PySCF'''

    mol.atom = atom_coordinates
    mol.basis = basis_set
    mol.verbose = verbose
    mol.max_memory = max_memory
    print('mol.max_memory =', mol.max_memory)

    mol.build(parse_arg = False)

    '''DFT or HF'''
    if method == 'RKS':
        mf = dft.RKS(mol)
    elif method == 'UKS':
        mf = dft.UKS(mol)
    elif method == 'RHF':
        mf = scf.RHF(mol)
    elif method == 'UHF':
        mf = scf.UHF(mol)

    if 'KS' in args.method:
        print('RKS')
        mf.xc = functional
        mf.grids.level = grid_level
    else:
        print('HF')
    if density_fit:
        mf = mf.density_fit()
        print('Density fitting turned on')
    if checkfile == True:
        '''use the *.chk file as scf input'''
        mf.chkfile = basename + '_' + functional + '.chk'
        mf.init_guess = 'chkfile'
    mf.conv_tol = 1e-10
    print ('Molecule built')
    print ('Calculating SCF Energy...')
    mf.kernel()
    # [print(k, v,'\n') for k, v in mol._basis.items()]

    kernel_1 = time.time()
    kernel_t = kernel_1 - kernel_0
    print ('SCF Done after %.2f'%kernel_t, 'seconds')

    return atom_coordinates, mol, mf, kernel_t

atom_coordinates, mol, mf, kernel_t = SCF_kernel()


show_memory_info('after SCF')

'''Collect everything needed from PySCF'''

def gen_global_var(tddft, mf, mol, functional=args.functional):
    '''
    TDA_vind & TDDFT_vind are ab-initio matrix vector multiplication function
    '''
    td = tddft.TDA(mf)
    TD = tddft.TDDFT(mf)
    '''
    hdiag is one dinension matrix, (A_size,)
    '''
    TDA_vind, hdiag = td.gen_vind(mf)
    TDDFT_vind, Hdiag = TD.gen_vind(mf)

    N_atm = mol.natm
    mo_occ = mf.mo_occ
    '''
    mf.mo_occ is an array of occupance [2,2,2,2,2,0,0,0,0.....]
    N_bf is the total amount of MOs
    if no truncation, then max_vir = n_vir and n_occ + max_vir = N_bf
    '''

    '''
    produce orthonormalized coefficient matrix C, N_bf * N_bf
    mf.mo_coeff is the unorthonormalized coefficient matrix
    S = mf.get_ovlp()  is basis overlap matrix
    S = np.dot(np.linalg.inv(c.T), np.linalg.inv(c))

    C_matrix is the orthonormalized coefficient matrix
    np.dot(C_matrix.T,C_matrix) is a an identity matrix
    '''
    S = mf.get_ovlp()
    X = math.matrix_power(S, 0.5)
    un_ortho_C_matrix = mf.mo_coeff
    C_matrix = np.dot(X,mf.mo_coeff)

    N_bf = len(mo_occ)
    n_occ = len(np.where(mo_occ > 0)[0])
    n_vir = len(np.where(mo_occ == 0)[0])
    delta_hdiag = hdiag.reshape(n_occ, n_vir)
    A_size = n_occ * n_vir

    tol_eV = args.truncate_virtual/parameter.Hartree_to_eV
    homo_vir = delta_hdiag[-1,:]
    max_vir = len(np.where(homo_vir <= tol_eV)[0])

    max_vir_hdiag = delta_hdiag[:,:max_vir]
    rst_vir_hdiag = delta_hdiag[:,max_vir:]

    A_reduced_size = n_occ * max_vir

    '''
    R_array is inter-particle distance array
    unit == ’Bohr’, 5.29177210903(80)×10^(−11) m
    '''
    R_array = gto.mole.inter_distance(mol, coords=None)

    a_x, beta, alpha = parameter.gen_alpha_beta_ax(functional)

    if args.beta != None:
        beta = args.beta
        alpha = args.alpha

    print('a_x =', a_x)
    print('beta =', beta)
    print('alpha =', alpha)
    print('N_bf =', N_bf)
    print('n_occ = ', n_occ)
    print('n_vir = ', n_vir)
    print('max_vir = ', max_vir)
    print('A_size = ', A_size)
    print('A_reduced_size =', A_reduced_size)

    return (TDA_vind, TDDFT_vind, hdiag, delta_hdiag, max_vir_hdiag,
        rst_vir_hdiag, N_atm, un_ortho_C_matrix, C_matrix, N_bf, n_occ, n_vir,
        max_vir, A_size, A_reduced_size, R_array, a_x, beta, alpha)

(TDA_vind, TDDFT_vind, hdiag, delta_hdiag, max_vir_hdiag, rst_vir_hdiag, N_atm,
un_ortho_C_matrix, C_matrix, N_bf, n_occ, n_vir, max_vir, A_size,
A_reduced_size, R_array, a_x, beta, alpha) = gen_global_var(tddft=tddft, mf=mf,
mol=mol, functional=args.functional)

def TDA_matrix_vector(V):
    '''
    return AX
    '''
    return TDA_vind(V.T).T

def TDDFT_matrix_vector(X, Y):
    '''return AX + BY and AY + BX'''
    XY = np.vstack((X,Y)).T
    U = TDDFT_vind(XY)
    U1 = U[:,:A_size].T
    U2 = -U[:,A_size:].T
    return U1, U2

def static_polarizability_matrix_vector(X):
    '''
    return (A+B)X
    this is not the optimum way, but the only way in PySCF
    '''
    U1, U2 = TDDFT_matrix_vector(X,X)
    return U1

def delta_fly(V):
    '''
    delta_hdiag.shape = (n_occ, n_vir)
    '''
    delta_v = einsum("ia,iam->iam", delta_hdiag, V)
    return delta_v

def delta_max_vir_fly(V):
    '''
    max_vir_hdiag.shape = (n_occ, max_vir)
    '''
    delta_max_vir_v = einsum("ia,iam->iam", max_vir_hdiag, V)
    return delta_max_vir_v

def delta_rst_vir_fly(V):
    '''
    max_vir_hdiag.shape = (n_occ, n_vir-max_vir)
    '''
    delta_rst_vir_v = einsum("ia,iam->iam", rst_vir_hdiag, V)
    return delta_rst_vir_v

def gen_P():
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]
    int_r= mol.intor_symmetric('int1e_r')
    P = lib.einsum("xpq,pi,qa->iax", int_r, orbo, orbv.conj())
    return P

def gen_ip_func(diag_i, iter_i, diag_p, iter_p):
    dict = {}
    dict[0] = (iter_i, iter_p)
    dict[1] = (diag_i, diag_p)
    dict[2] = (diag_i, iter_p)
    dict[3] = (iter_i, diag_p)
    return dict

ip_name = [            # option
['iter','iter'],       # 0
['diag','diag'],       # 1
['diag','iter'],       # 2
['iter','diag']]       # 3

def gen_calc():
    dict={}
    dict['TDA'] = args.TDA
    dict['TDDFT'] = args.TDDFT
    dict['dpolar'] = args.dpolar
    dict['spolar'] = args.spolar
    dict['sTDA'] = args.sTDA
    dict['sTDDFT'] = args.sTDDFT
    dict['PySCF_TDDFT'] = args.pytd
    for calc in dict.keys():
        if dict[calc] == True:
            print(calc)
            return calc
calc_name = gen_calc()

global_var = (basename, atom_coordinates, mol, mf, kernel_t, TDA_vind,
TDDFT_vind, hdiag, delta_hdiag, max_vir_hdiag, rst_vir_hdiag, N_atm,
un_ortho_C_matrix, C_matrix, N_bf, n_occ, n_vir, max_vir, A_size,
A_reduced_size, R_array, a_x, beta, alpha, TDA_matrix_vector,
TDDFT_matrix_vector, static_polarizability_matrix_vector)
