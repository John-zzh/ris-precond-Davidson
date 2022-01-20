import numpy as np
import mathlib
import parameterlib
import time
from arguments import args
from SCF_calc import n_occ, n_vir, max_vir, max_vir_hdiag, A_size
from sTDA_lib import sTDA_mv


''' [A 0] X = XΩ
    [0 h]
'''

def TDA_A_diag_initial_guess(k, hdiag=max_vir_hdiag):
    '''m is the amount of initial guesses'''
    hdiag = hdiag.reshape(-1,)
    V_size = hdiag.shape[0]

    Dsort = hdiag.argsort()
    energies = hdiag[Dsort][:k]*parameterlib.Hartree_to_eV
    V = np.zeros((V_size, k))
    for j in range(k):
        V[Dsort[j], j] = 1.0
    return V, energies

def TDA_A_diag_preconditioner(residual, sub_eigenvalue,
                                        hdiag=max_vir_hdiag):
    '''DX = XΩ'''
    k = np.shape(residual)[1]
    t = 1e-14
    D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_eigenvalue
    '''force all small values not in [-t,t]'''
    D = np.where( abs(D) < t, np.sign(D)*t, D)
    new_guess = residual/D

    return new_guess

def Davidson_eigensolver(matrix_vector_product=sTDA_mv, k=args.nstates,
                                tol=args.initial_TOL):
    '''AX = XΩ'''
    print('nstate =', k)
    Davidson_start = time.time()
    max = 35

    '''m is size of subspace'''
    m = 0
    new_m = min([k+8, 2*k, A_size])
    V = np.zeros((A_size, (max+1)*k + m))
    W = np.zeros_like(V)

    '''V is subsapce basis
       W is transformed guess vectors'''
    V[:, :new_m], initial_energies = TDA_A_diag_initial_guess(k=new_m,
                                                          hdiag=max_vir_hdiag)
    for i in range(max):
        '''create subspace'''
        W[:, m:new_m] = matrix_vector_product(V[:, m:new_m])
        sub_A = np.dot(V[:,:new_m].T, W[:,:new_m])
        sub_A = mathlib.symmetrize(sub_A)

        '''Diagonalize the subspace Hamiltonian, and sorted.
        sub_eigenvalue[:k] are smallest k eigenvalues'''
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        full_guess = np.dot(V[:,:new_m], sub_eigenket[:, :k])

        '''residual = AX - XΩ = AVx - XΩ = Wx - XΩ'''
        residual = np.dot(W[:,:new_m], sub_eigenket[:,:k])
        residual -= full_guess*sub_eigenvalue[:k]

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break

        '''index for unconverged residuals'''
        index = [r_norms.index(i) for i in r_norms if i > tol]
        '''precondition the unconverged residuals'''
        new_guess = TDA_A_diag_preconditioner(
                        residual = residual[:,index],
                  sub_eigenvalue = sub_eigenvalue[:k][index],
                           hdiag = max_vir_hdiag)

        '''orthonormalize the new guess against basis and put into V holder'''
        m = new_m
        V, new_m = mathlib.Gram_Schmidt_fill_holder(V, m, new_guess)

    Davidson_end = time.time()
    Davidson_time = Davidson_end - Davidson_start
    print('A diagonalized in', i, 'steps; ', '%.2f'%Davidson_time, 'seconds' )
    print('threshold =', tol)
    print('excitation energies:')
    print(sub_eigenvalue[:k]*parameterlib.Hartree_to_eV)

    U = np.zeros((n_occ,n_vir,k))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,k)
    U = U.reshape(A_size, k)
    omega = sub_eigenvalue[:k]*parameterlib.Hartree_to_eV
    return U, omega
