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
                charge = args.charge,
             basis_set = args.basis_set,
               verbose = args.verbose,
            max_memory = args.memory,
                method = args.method,
             checkfile = args.checkfile,
                  dscf = args.dscf,
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
    mol.charge = charge
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
        if args.dscf == True:
            mf.max_cycle = 0
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

'''
Collect everything needed from PySCF
'''


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
if no truncation, then rest_vir = n_vir and n_occ + rest_vir = N_bf
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


'''
N_bf:
    trunced_occ                    rest_occ          rest_vir      truncated_vir
n_occ*args.truncate_occupied                                          n_vir*args.truncate_virtual
============================#======================|---------------#------------------------------
                         n_occ                                   n_vir
'''



homo_vir = delta_hdiag[-1,:]
occ_lumo = delta_hdiag[:,0]

'''
the truncation thresholds for coulomb term must be larger than
that of the exchange term
'''
vir_tol_hartree = [i/parameter.Hartree_to_eV for i in args.truncate_virtual]
cl_vir_tol_hartree = vir_tol_hartree[0]
ex_vir_tol_hartree = vir_tol_hartree[1]

occ_tol_hartree = [i/parameter.Hartree_to_eV for i in args.truncate_occupied]
cl_occ_tol_hartree = occ_tol_hartree[0]
ex_occ_tol_hartree = occ_tol_hartree[1]

'''
cl_rest_vir, cl_truc_vir,
cl_rest_occ, cl_truc_occ,
ex_rest_vir, ex_truc_vir,
ex_rest_occ, ex_truc_occ,
cl_A_rest_size, ex_A_rest_size

'''
cl_rest_vir = len(np.where(homo_vir <= cl_vir_tol_hartree)[0])
ex_rest_vir = len(np.where(homo_vir <= ex_vir_tol_hartree)[0])

cl_rest_occ = len(np.where(occ_lumo <= cl_occ_tol_hartree)[0])
ex_rest_occ = len(np.where(occ_lumo <= ex_occ_tol_hartree)[0])

cl_truc_vir = n_vir - cl_rest_vir
cl_truc_occ = n_occ - cl_rest_occ

ex_truc_vir = n_vir - ex_rest_vir
ex_truc_occ = n_occ - ex_rest_occ

cl_A_rest_size = cl_rest_occ * cl_rest_vir
ex_A_rest_size = ex_rest_occ * ex_rest_vir

rest_occ, rest_vir = cl_rest_occ, cl_rest_vir
reduced_occ, reduced_vir = cl_rest_occ, cl_rest_vir
trunced_occ, trunced_vir = cl_truc_occ, cl_truc_vir
A_reduced_size = cl_A_rest_size
'''
delta_hdiag:

                --------------#-----------------------
cl_truc_occ     |    hdiag1   |                      |
                #-------------#                      |
                |             |        hdiag3        |
cl_rest_occ     |    hdiag2   |                      |
                |             |                      |
                --------------#-----------------------
                  cl_rest_vir      cl_truc_vir
'''

delta_hdiag1 = delta_hdiag[:cl_truc_occ,:cl_rest_vir].copy()
delta_hdiag2 = delta_hdiag[cl_truc_occ:,:cl_rest_vir].copy()
delta_hdiag3 = delta_hdiag[:,cl_rest_vir:].copy()

print('delta_hdiag2',delta_hdiag2.shape)


'''
R_array is inter-particle distance array
unit == ’Bohr’, 5.29177210903(80)×10^(−11) m
'''
R_array = gto.mole.inter_distance(mol, coords=None)

a_x, beta, alpha = parameter.gen_alpha_beta_ax(args.functional)

# if args.beta != None:
#     beta = args.beta
#     alpha = args.alpha

print('a_x =', a_x)
# print('beta =', beta)
# print('alpha =', alpha)
print('N_bf =', N_bf)
print('N_atm =', N_atm)

print('n_occ =', n_occ)
print('cl_rest_occ =', cl_rest_occ)
print('ex_rest_occ =', ex_rest_occ)

print('n_vir =', n_vir)
print('cl_rest_vir = ', cl_rest_vir)
print('ex_rest_vir = ', ex_rest_vir)

print('A_size = ', A_size)
print('cl_A_rest_size =', cl_A_rest_size)
print('ex_A_rest_size =', ex_A_rest_size)

print('reduced_vir =', reduced_vir)

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
    V = V.reshape(A_size,-1)
    delta_v = V*hdiag.reshape(-1,1)
    delta_v = delta_v.reshape(n_occ, n_vir, -1)
    # delta_v = einsum("ia,iam->iam", delta_hdiag, V)
    return delta_v

def delta_hdiag1_fly(V):
    '''
    hdiag3.shape = (rest_occ, rest_vir)
    '''
    delta_hdiag1_v = einsum("ia,iam->iam", delta_hdiag1, V)
    return delta_hdiag1_v

def delta_hdiag2_fly(V):
    '''
    rest_vir_hdiag.shape = (rest_occ, rest_vir)
    '''
    delta_hdiag2_v = einsum("ia,iam->iam", delta_hdiag2, V)
    return delta_hdiag2_v

def delta_hdiag3_fly(V):
    '''
    rest_vir_hdiag.shape = (n_occ, trunced_vir)
    '''
    delta_hdiag3_v = einsum("ia,iam->iam", delta_hdiag3, V)
    return delta_hdiag3_v

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
    dict['CPKS'] = args.CPKS
    dict['sTDA'] = args.sTDA
    dict['sTDDFT'] = args.sTDDFT
    dict['PySCF_TDDFT'] = args.pytd

    for calc in dict.keys():
        if dict[calc] == True:
            print(calc)
            return calc
calc_name = gen_calc()

# print(mol._basis)
