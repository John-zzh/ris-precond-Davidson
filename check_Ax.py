import numpy as np
import pyscf
from pyscf import gto, scf, dft, tddft, lib
from functools import reduce

mol = pyscf.gto.M(
    atom='''
C         -4.89126        3.29770        0.00029
H         -5.28213        3.05494       -1.01161
O         -3.49307        3.28429       -0.00328
H         -5.28213        2.58374        0.75736
H         -5.23998        4.31540        0.27138
H         -3.22959        2.35981       -0.24953
    ''',
    basis='def2-SVP',
    verbose=3,
)


mf = dft.RKS(mol)
mf.xc = 'pbe0'
mf.kernel()

mo_occ = mf.mo_occ
n_occ = len(np.where(mo_occ > 0)[0])
n_vir = len(np.where(mo_occ == 0)[0])
A_size = n_occ * n_vir

td = tddft.TDA(mf)
TD = tddft.TDDFT(mf)

TDA_vind, hdiag = td.gen_vind(mf)
TDDFT_vind, Hdiag = TD.gen_vind(mf)

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
    this the hack way in PySCF
    '''
    U1, U2 = TDDFT_matrix_vector(X,X)
    return U1





def fvind(x):
    vresp = mf.gen_response(singlet=None, hermi=1)
    orbv = mf.mo_coeff[:,n_occ:]
    orbo = mf.mo_coeff[:,:n_occ]

    e_a = mf.mo_energy[mo_occ==0]
    e_i = mf.mo_energy[mo_occ>0]

    evo = lib.direct_sum('a-i->ai', e_a, e_i)

    dm = reduce(np.dot, (orbv, 2*x.reshape(n_vir,n_occ), orbo.T))
    v1ao = vresp(dm+dm.T)
    v1vo = reduce(np.dot, (orbv.T, v1ao, orbo)).reshape(n_vir, n_occ)
    diag = np.einsum("ai,ai->ai", evo, x.reshape(n_vir,n_occ))
    ans = (v1vo + diag).T
    return ans

def CPHF(X):
    '''
    return (A+B)X ?
    '''

    print('X shape',X.shape)
    A_p_B_X = fvind(X.T)
    A_p_B_X = A_p_B_X.reshape(A_size,-1)
    print('A_p_B_X.shape',A_p_B_X.shape)
    return A_p_B_X

test_X = np.ones((A_size,1))

A_p_B_X = CPHF(test_X)
A_p_B_X_standard = static_polarizability_matrix_vector(test_X)
print('A_p_B_X_standard.shape',A_p_B_X_standard.shape)

diff = A_p_B_X - A_p_B_X_standard

print('difference from standard (A+B)X = ', np.linalg.norm(diff))
# print('difference from standard AX = ', np.linalg.norm(diffA))
# print('ratio',np.linalg.norm(ratio))
#
#
# test_V = np.random.rand(1,A_size)
#
# vind_V = TDDFT_vind(np.hstack((test_V,test_V)))[0, A_size:]
# CPHF_V = fvind(test_V)
# print('difference from standard vind_V = ', np.linalg.norm(vind_V-CPHF_V))
