import numpy as np
import pyscf
from pyscf import gto, scf, dft, tddft


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

tol      = 1e-8
max_iter = 100

mf = pyscf.scf.RHF(mol)
mf.max_cycle = max_iter
mf.conv_tol  = tol
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


vresp = mf.gen_response(singlet=None, hermi=1)
orbv = mf.mo_coeff[:,n_occ:]
orbo = mf.mo_coeff[:,:n_occ]
from functools import reduce

def static_polarizability_matrix_vector2(X):
    '''
    return (A+B)X ?
    '''
    origin_X = X.copy()
    # X = X.reshape(n_occ,n_vir,1)[:,:,0].T
    print('X.shape',X.shape)
    dm = reduce(np.dot, (orbv, 2*X.reshape(n_vir,n_occ), orbo.T))
    v1ao = vresp(dm+dm.T)
    A_p_B_X = reduce(np.dot, (orbv.T, v1ao, orbo))
    print('A_p_B_X.shape',A_p_B_X.shape)
    A_p_B_X = A_p_B_X.ravel().reshape(n_vir*n_occ,-1)
    print('A_p_B_X.shape',A_p_B_X.shape)
    A_p_B_X_standard = static_polarizability_matrix_vector(origin_X)
    print('difference from standard (A+B)X = ', np.linalg.norm(A_p_B_X - A_p_B_X_standard))

    AX_standard = TDA_matrix_vector(origin_X)
    print('difference from standard AX = ', np.linalg.norm(A_p_B_X - AX_standard))

    return A_p_B_X

test_X = np.random.rand(n_occ*n_vir,1)

static_polarizability_matrix_vector2(test_X)
