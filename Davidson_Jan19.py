#!/usr/bin/python

import time
import numpy as np
import mathlib

import TDDFT_as_lib
from sTDA_lib import sTDA_mv, full_sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv
import TDDFT_as_lib

from opt_einsum import contract as einsum
import Davidson_eigensolver_lib
import os, sys
from pyscf import gto, scf, dft, tddft, data, lib
import parameterlib
from arguments import args
from Davidson_eigensolver_lib import Davidson_eigensolver as sTDA_eigen_solver

# share args object across all py files


import yaml
import scipy
from scipy import optimize
from SCF_calc import global_var


'''wb97x  methanol, 1e-5
        TDA          [7.60466912 9.60759452 9.65620573 10.54964748 10.84658266]
  sTDDFT no truncate [6.46636611 8.18031534 8.38140651 9.45011415 9.5061059 ]
            40 eV    [6.46746642 8.18218267 8.38314651 9.45214869 9.5126739 ]
    sTDA no truncate [6.46739711 8.18182208 8.38358473 9.45195554 9.52133129]
            40 eV    [6.46827111 8.18334703 8.38483801 9.45361525 9.52562255]

   PBE0  methanol, 1e-5
    TDA no truncate  [ 7.2875264   8.93645089  9.18027002  9.92054961 10.16937337]

'''
print('curpath', os.getcwd())
print('lib.num_threads() = ', lib.num_threads())

(basename, atom_coordinates, mol, mf, kernel_t, TDA_vind, TDDFT_vind, hdiag,
delta_hdiag, max_vir_hdiag, rst_vir_hdiag, N_atm, un_ortho_C_matrix, C_matrix,
N_bf, n_occ, n_vir, max_vir, A_size, A_reduced_size, R_array, a_x, beta, alpha,
eta, show_memory_info, TDA_matrix_vector, TDDFT_matrix_vector,
static_polarizability_matrix_vector) = global_var


show_memory_info('at beginning')



if args.TDDFT_as == True:

    sTDA_mv = TDDFT_as_lib.as_mv

def TDA_A_diag_initial_guess(k, hdiag=max_vir_hdiag, matrix_vector_product=None):
    '''m is the amount of initial guesses'''
    hdiag = hdiag.reshape(-1,)
    V_size = hdiag.shape[0]
    Dsort = hdiag.argsort()
    energies = hdiag[Dsort][:k]*parameterlib.Hartree_to_eV
    V = np.zeros((V_size, k))
    for j in range(k):
        V[Dsort[j], j] = 1.0
    return V, energies

def TDA_A_diag_preconditioner(residual, sub_eigenvalue, hdiag=max_vir_hdiag,
                                                    matrix_vector_product=None):
    '''DX = XΩ'''
    k = np.shape(residual)[1]
    t = 1e-14
    D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_eigenvalue
    '''force all small values not in [-t,t]'''
    D = np.where( abs(D) < t, np.sign(D)*t, D)
    new_guess = residual/D

    return new_guess

def sTDA_preconditioner(residual, sub_eigenvalue, matrix_vector_product=sTDA_mv,
                                            tol=args.precond_TOL, max=30):
    '''sTDA preconditioner
       (A - Ω*I)^-1 P = X
       AX - XΩ = P
       P is residuals (in big Davidson's loop) to be preconditioned
    '''
    p_start = time.time()

    '''number of vectors to be preconditioned'''
    N_vectors = residual.shape[1]
    Residuals = residual.reshape(n_occ,n_vir,-1)
    omega = sub_eigenvalue
    P = Residuals[:,:max_vir,:]
    P = P.reshape(A_reduced_size,-1)

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P = P/pnorm

    start = time.time()
    tol = args.precond_TOL

    V = np.zeros((A_reduced_size, (max+1)*N_vectors))
    W = np.zeros((A_reduced_size, (max+1)*N_vectors))
    count = 0

    '''now V and W are empty holders, 0 vectors
       W = sTDA_mv(V)
       count is the amount of vectors that already sit in the holder
       in each iteration, V and W will be filled/updated with new guess basis
       which is the preconditioned residuals
    '''

    '''initial guess: DX - XΩ = P
       Dp is the preconditioner
       <t: returns np.sign(D)*t; else: D
    '''
    t = 1e-10
    Dp = np.repeat(hdiag.reshape(-1,1), N_vectors, axis=1) - omega
    Dp = np.where(abs(Dp)<t, np.sign(Dp)*t, Dp)
    Dp = Dp.reshape(n_occ, n_vir, -1)
    D = Dp[:,:max_vir,:].reshape(A_reduced_size,-1)
    inv_D = 1/D

    '''generate initial guess'''
    Xig = P*inv_D
    count = 0
    V, new_count = mathlib.Gram_Schmidt_fill_holder(V, count, Xig)


    mvcost = 0
    GScost = 0
    subcost = 0
    subgencost = 0

    for i in range(max):

        '''project sTDA_A matrix and vector P into subspace'''
        mvstart = time.time()
        W[:, count:new_count] = matrix_vector_product(V[:, count:new_count])
        mvend = time.time()
        mvcost += mvend - mvstart

        substart = time.time()
        sub_P= np.dot(V[:,:new_count].T, P)
        sub_A = np.dot(V[:,:new_count].T, W[:,:new_count])
        subend = time.time()
        subgencost += subend - substart

        sub_A = mathlib.symmetrize(sub_A)
        m = np.shape(sub_A)[0]

        substart = time.time()
        sub_guess = mathlib.solve_AX_Xla_B(sub_A, omega, sub_P)
        subend = time.time()
        subcost += subend - substart

        full_guess = np.dot(V[:,:new_count], sub_guess)
        residual = np.dot(W[:,:new_count], sub_guess) - full_guess*omega - P

        r_norms = np.linalg.norm(residual, axis=0).tolist()

        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break

        '''index of unconverged states'''
        index = [r_norms.index(i) for i in r_norms if i > tol]

        '''precondition the unconverged residuals'''
        new_guess = residual[:,index]*inv_D[:,index]


        GSstart = time.time()
        count = new_count
        V, new_count = mathlib.Gram_Schmidt_fill_holder(V, count, new_guess)
        GSend = time.time()
        GScost += GSend - GSstart

    p_end = time.time()
    p_cost = p_end - p_start

    if i == (max -1):
        print('_____sTDA Preconditioner Failed Due to Iteration Limit _______')
        print('failed after ', i, 'steps,', '%.4f'%p_cost,'s')
        print('current residual norms', r_norms)
    else:
        print('sTDA precond Done after', i, 'steps;', '%.4f'%p_cost,'seconds')

    print('max_norm = ', '%.2e'%max_norm)
    for enrty in ['subgencost', 'mvcost', 'GScost', 'subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/p_cost))
    full_guess = full_guess*pnorm

    U = np.zeros((n_occ,n_vir,N_vectors))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,-1)

    if max_vir < n_vir:
        ''' DX2 - X2*Omega = P2'''
        P2 = Residuals[:,max_vir:,:]
        P2 = P2.reshape(n_occ*(n_vir-max_vir),-1)

        D2 = Dp[:,max_vir:,:]
        D2 = D2.reshape(n_occ*(n_vir-max_vir),-1)
        X2 = (P2/D2).reshape(n_occ,n_vir-max_vir,-1)
        U[:,max_vir:,:] = X2

    U = U.reshape(A_size, -1)

    '''if we want to know more about the preconditioning process,
        return the current_dic, rather than origin_dic'''
    return U

def gen_TDA_lib():
    i_lib={}
    p_lib={}
    i_lib['sTDA']   = sTDA_eigen_solver
    i_lib['Adiag']  = TDA_A_diag_initial_guess
    p_lib['sTDA']   = sTDA_preconditioner
    p_lib['Adiag']  = TDA_A_diag_preconditioner
    # p_lib['Jacobi'] = Jacobi_preconditioner
    # p_lib['new_ES'] = new_ES
    return i_lib, p_lib

def fill_dictionary(dic,init,prec,k,icost,pcost,wall_time,N_itr,N_mv,
            initial_energies=None,energies=None,difference=None,overlap=None,
            tensor_alpha=None, initial_tensor_alpha=None):
    dic['initial guess'] = init
    dic['preconditioner'] = prec
    dic['nstate'] = k
    dic['molecule'] = basename
    dic['method'] = args.method
    dic['functional'] = args.functional
    dic['threshold'] = args.conv_tolerance
    dic['SCF time'] = kernel_t
    dic['Initial guess time'] = icost
    dic['initial guess threshold'] = args.initial_TOL
    dic['New guess generating time'] = pcost
    dic['preconditioner threshold'] = args.precond_TOL
    dic['total time'] = wall_time
    dic['excitation energy(eV)'] = energies
    dic['iterations'] = N_itr
    dic['A matrix size'] = A_size
    dic['final subspace size'] = N_mv
    dic['ax'] = a_x
    dic['alpha'] = alpha
    dic['beta'] = beta
    dic['virtual truncation tol'] = args.truncate_virtual
    dic['n_occ'] = n_occ
    dic['n_vir'] = n_vir
    dic['max_vir'] = max_vir
    dic['semiempirical_difference'] = difference
    dic['overlap'] = overlap
    dic['initial_energies'] = initial_energies
    dic['Dynamic polarizability wavelength'] = args.dynpol_omega
    dic['Dynamic polarizability tensor alpha'] = tensor_alpha
    dic['Dynamic polarizability initial tensor alpha'] = initial_tensor_alpha
    return dic

def on_the_fly_tuning(sub_A, V):
    print('checking commutator_se norm')
    # print('{:<8s}{:<8s}{:<20s}'.format('beta','alpha','commutator_se_norm'))
    smallest_norm=1000
    opt_alpha=0
    opt_beta=0
    for try_beta in args.beta_list:
        for try_alpha in args.alpha_list:
            tensors = on_the_fly_tensors(alpha=try_alpha, beta=try_beta)
            sub_A_se = np.dot(V.T, tensors.full_sTDA_mv(V))
            commutator = np.dot(sub_A,sub_A_se) - np.dot(sub_A_se,sub_A)
            commutator_se_norm = np.linalg.norm(commutator)
            if commutator_se_norm < smallest_norm :
                smallest_norm = commutator_se_norm
                opt_alpha=try_alpha
                opt_beta=try_beta
            # print("{:<8.2f}{:<8.2f}{:<20.15f}".format(try_beta,try_alpha,commutator_se_norm))
    print('the predicted best alpha beta pairs is:')
    print("{:<8.2f}{:<8.2f}{:<20.15f}".format(opt_beta,opt_alpha,smallest_norm))
    return opt_alpha, opt_beta

def Davidson(init, prec, k=args.nstates, tol=args.conv_tolerance):
    '''Davidson frame, we can use different initial guess and preconditioner'''
    D_start = time.time()
    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    TDA_i_lib, TDA_p_lib = gen_TDA_lib()
    initial_guess = TDA_i_lib[init]
    new_guess_generator = TDA_p_lib[prec]

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)

    init_start = time.time()
    max = args.max
    m = 0
    new_m = min([k + args.extrainitial, 2*k, A_size])
    V = np.zeros((A_size, max*k + new_m))
    W = np.zeros_like(V)
    V[:, :new_m], initial_energies = initial_guess(k=new_m, matrix_vector_product=sTDA_mv)
    init_end = time.time()

    init_time = init_end - init_start
    print('initial guess time %.4f seconds'%init_time)

    Pcost = 0
    MVcost = 0
    for ii in range(max):
        print('\n')
        print('Iteration ', ii)
        istart = time.time()

        MV_start = time.time()
        W[:, m:new_m] = TDA_matrix_vector(V[:,m:new_m])
        MV_end = time.time()
        iMVcost = MV_end - MV_start
        MVcost += iMVcost
        sub_A = np.dot(V[:,:new_m].T, W[:,:new_m])
        sub_A = mathlib.symmetrize(sub_A)
        print('subspace size: ', np.shape(sub_A)[0])

        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        full_guess = np.dot(V[:,:new_m], sub_eigenket[:,:k])
        AV = np.dot(W[:,:new_m], sub_eigenket[:,:k])
        residual = AV - full_guess * sub_eigenvalue[:k]

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms

        print('maximum residual norm %.2e'%max_norm)
        if max_norm < tol or ii == (max-1):
            iend = time.time()
            icost = iend - istart
            current_dic['iteration total cost'] = icost
            current_dic['iteration MV cost'] = iMVcost
            iteration_list[ii] = current_dic
            print('iMVcost %.4f'%iMVcost)
            print('icost %.4f'%icost)
            print('Davidson procedure Done \n')
            break

        index = [r_norms.index(i) for i in r_norms if i > tol]

        P_start = time.time()

        new_guess = new_guess_generator(
                                residual = residual[:,index],
                          sub_eigenvalue = sub_eigenvalue[:k][index],
                   matrix_vector_product = sTDA_mv)

        P_end = time.time()

        iteration_list[ii] = current_dic

        Pcost += P_end - P_start

        m = new_m
        V, new_m = mathlib.Gram_Schmidt_fill_holder(V, m, new_guess)
        print('new generated guesses:', new_m - m)

        iend = time.time()
        icost = iend - istart
        current_dic['iteration cost'] = icost
        current_dic['iteration MV cost'] = iMVcost
        iteration_list[ii] = current_dic
        print('iMVcost %.4f'%iMVcost)
        print('icost %.4f'%icost)

    energies = sub_eigenvalue[:k]*parameterlib.Hartree_to_eV
    V_basis = V[:,:new_m]
    W_basis = W[:,:new_m]

    D_end = time.time()
    Dcost = D_end - D_start

    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=k,
                                icost=init_time, pcost=Pcost, wall_time=Dcost,
            energies = energies.tolist(), N_itr=ii+1, N_mv=np.shape(sub_A)[0],
            initial_energies=initial_energies.tolist())
    if ii == max-1:
        print('========== Davidson Failed Due to Iteration Limit ============')
        print('current residual norms', r_norms)
    else:
        print('------- Davidson done -------')
    print('max_norm = ', max_norm)
    print('Total steps =', ii+1)
    print('Total time: %.4f seconds'%Dcost)
    print('MVcost %.4f'%MVcost)
    print('Final subspace shape = %s'%np.shape(sub_A)[0])
    print('Precond time: %.4f seconds'%Pcost, '{:.2%}'.format(Pcost/Dcost))
    I_AAV = gen_forecaster(eta, sub_eigenvalue[:k], full_guess)
    print('I_AAV =', I_AAV)
    Davidson_dic['preconditioner forecaster'] = I_AAV
    return (energies, full_guess, AV, residual, Davidson_dic, V_basis, W_basis,
                                                                          sub_A)

def TDDFT_A_diag_initial_guess(V_holder, W_holder, new_m, hdiag=hdiag):
    hdiag = hdiag.reshape(-1,)
    Dsort = hdiag.argsort()
    V_holder[:,:new_m], energies = TDA_A_diag_initial_guess(new_m, hdiag=hdiag)
    return (V_holder, W_holder, new_m, energies, V_holder[:,:new_m],
                                                            W_holder[:,:new_m])

def TDDFT_A_diag_preconditioner(R_x, R_y, omega, hdiag, tol=None):
    '''preconditioners for each corresponding residual (state)'''
    hdiag = hdiag.reshape(-1,1)
    k = R_x.shape[1]
    t = 1e-14
    d = np.repeat(hdiag.reshape(-1,1), k, axis=1)

    D_x = d - omega
    D_x = np.where( abs(D_x) < t, np.sign(D_x)*t, D_x)
    D_x_inv = D_x**-1

    D_y = d + omega
    D_y = np.where( abs(D_y) < t, np.sign(D_y)*t, D_y)
    D_y_inv = D_y**-1

    X_new = R_x*D_x_inv
    Y_new = R_y*D_y_inv

    return X_new, Y_new

def sTDDFT_eigen_solver(k, tol=args.initial_TOL):
    '''[ A' B' ] X - [1   0] Y Ω = 0
       [ B' A' ] Y   [0  -1] X   = 0
    '''
    max = 30
    sTDDFT_start = time.time()
    print('setting initial guess')
    print('sTDDFT Convergence tol = %.2e'%tol)
    m = 0
    new_m = min([k+8, 2*k, A_reduced_size])
    V_holder = np.zeros((A_reduced_size, (max+1)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    '''set up initial guess V W, transformed vectors U1 U2'''
    V_holder, W_holder, new_m, energies, Xig, Yig = TDDFT_A_diag_initial_guess(
        V_holder=V_holder, W_holder=W_holder, new_m=new_m, hdiag=max_vir_hdiag)

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]
        '''U1 = AV + BW
           U2 = AW + BV'''

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = sTDDFT_mv(
                                            X=V[:, m:new_m], Y=W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)


        a = mathlib.symmetrize(a)
        b = mathlib.symmetrize(b)
        sigma = mathlib.symmetrize(sigma)
        pi = mathlib.anti_symmetrize(pi)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''solve the eigenvalue omega in the subspace'''
        subcost_start = time.time()
        omega, x, y = mathlib.TDDFT_subspace_eigen_solver(a, b, sigma, pi, k)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R_x = U1x + U2y - X_full*omega
           R_y = U2x + U1y + Y_full*omega
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''

        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega

        residual = np.vstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or ii == (max -1):
            break

        index = [r_norms.index(i) for i in r_norms if i > tol]

        '''preconditioning step'''
        X_new, Y_new = TDDFT_A_diag_preconditioner(R_x=R_x[:,index],
                    R_y=R_y[:,index], omega=omega[index], hdiag = max_vir_hdiag)

        '''GS and symmetric orthonormalization'''
        m = new_m
        GScost_start = time.time()
        V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder\
                                        (V_holder, W_holder, m, X_new, Y_new)
        GScost_end = time.time()
        GScost += GScost_end - GScost_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    sTDDFT_end = time.time()

    sTDDFT_cost = sTDDFT_end - sTDDFT_start

    if ii == (max -1):
        print('========= sTDDFT Failed Due to Iteration Limit=================')
        print('sTDDFT diagonalization failed')
        print('current residual norms', r_norms)
    else:
        print('sTDDFT diagonalization Converged' )

    print('after ', ii+1, 'iterations; %.4f'%sTDDFT_cost, 'seconds')
    print('final subspace', sigma.shape[0])
    print('max_norm = ', '%.2e'%max_norm)
    for enrty in ['MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/sTDDFT_cost))
    X = np.zeros((n_occ,n_vir,k))
    Y = np.zeros((n_occ,n_vir,k))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,-1)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,-1)

    X = X.reshape(A_size, -1)
    Y = Y.reshape(A_size, -1)

    energies = omega*parameterlib.Hartree_to_eV
    print('sTDDFT excitation energy:')
    print(energies)
    return energies, X, Y

def sTDDFT_initial_guess(V_holder, W_holder, new_m):
    energies, X_new_backup, Y_new_backup = sTDDFT_eigen_solver(new_m)
    V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(
                            V_holder, W_holder, 0,  X_new_backup, Y_new_backup)
    return V_holder, W_holder, new_m, energies, X_new_backup, Y_new_backup

def sTDDFT_preconditioner(Rx, Ry, omega, tol=args.precond_TOL):
    ''' [ A' B' ] - [1  0]X  Ω = P'''
    ''' [ B' A' ]   [0 -1]Y    = Q'''
    ''' P = Rx '''
    ''' Q = Ry '''

    print('sTDDFT_preconditioner conv', tol)
    max = 30
    sTDDFT_start = time.time()
    k = len(omega)
    m = 0

    Rx = Rx.reshape(n_occ,n_vir,-1)
    Ry = Ry.reshape(n_occ,n_vir,-1)

    P = Rx[:,:max_vir,:].reshape(A_reduced_size,-1)
    Q = Ry[:,:max_vir,:].reshape(A_reduced_size,-1)

    initial_start = time.time()
    V_holder = np.zeros((A_reduced_size, (max+1)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    '''normalzie the RHS'''
    PQ = np.vstack((P,Q))
    pqnorm = np.linalg.norm(PQ, axis=0, keepdims = True)

    P /= pqnorm
    Q /= pqnorm

    X_new, Y_new  = TDDFT_A_diag_preconditioner(R_x=P, R_y=Q, omega=omega,
                                                            hdiag=max_vir_hdiag)
    V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(
                                    V_holder, W_holder, 0,  X_new, Y_new)
    initial_end = time.time()
    initial_cost = initial_end - initial_start

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]

        '''U1 = AV + BW
           U2 = AW + BV'''

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = sTDDFT_mv(
                                            X=V[:, m:new_m], Y=W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)

        '''p = VP + WQ
           q = WP + VQ'''
        p = np.dot(V.T, P)
        p += np.dot(W.T, Q)

        q = np.dot(W.T, P)
        q += np.dot(V.T, Q)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        a = mathlib.symmetrize(a)
        b = mathlib.symmetrize(b)
        sigma = mathlib.symmetrize(sigma)
        pi = mathlib.anti_symmetrize(pi)

        '''solve the x & y in the subspace'''
        subcost_start = time.time()
        x, y = mathlib.TDDFT_subspace_liear_solver(a, b, sigma, pi, p, q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R_x = U1x + U2y - X_full*omega - P
           R_y = U2x + U1y + Y_full*omega - Q
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''

        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega
        R_x -= P

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega
        R_y -= Q

        residual = np.vstack((R_x,R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or ii == (max -1):
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]

        '''preconditioning step'''
        Pstart = time.time()
        X_new, Y_new = TDDFT_A_diag_preconditioner(R_x[:,index], R_y[:,index],
                                            omega[index], hdiag = max_vir_hdiag)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(\
                                            V_holder, W_holder, m, X_new, Y_new)
        GS_end = time.time()
        GScost += GS_end - GS_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    sTDDFT_end = time.time()

    P_cost = sTDDFT_end - sTDDFT_start

    if ii == (max -1):
        print('========== sTDDFT_precond Failed Due to Iteration Limit========')
        print('sTDDFT preconditioning failed')
        print('current residual norms', r_norms)
    else:
        print('sTDDFT preconditioning Done')
    print('after',ii+1,'steps; %.4f'%P_cost,'s')
    print('final subspace', sigma.shape[0])
    print('max_norm = ', '%.2e'%max_norm)
    for enrty in ['initial_cost','MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/P_cost))

    X_full = X_full*pqnorm
    Y_full = Y_full*pqnorm

    X = np.zeros((n_occ,n_vir,k))
    Y = np.zeros((n_occ,n_vir,k))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,k)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,k)

    if max_vir < n_vir:
        P2 = Rx[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)
        Q2 = Ry[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)

        X2, Y2 = TDDFT_A_diag_preconditioner(R_x=P2, R_y=Q2, omega=omega,
                                                hdiag=delta_hdiag[:,max_vir:])
        X[:,max_vir:,:] = X2.reshape(n_occ,n_vir-max_vir,-1)
        Y[:,max_vir:,:] = Y2.reshape(n_occ,n_vir-max_vir,-1)

    X = X.reshape(A_size,-1)
    Y = Y.reshape(A_size,-1)

    return X, Y

def gen_TDDFT_lib():
    i_lib={}
    p_lib={}
    i_lib['sTDDFT'] = sTDDFT_initial_guess
    i_lib['Adiag']  = TDDFT_A_diag_initial_guess
    p_lib['sTDDFT'] = sTDDFT_preconditioner
    p_lib['Adiag']  = TDDFT_A_diag_preconditioner
    return i_lib, p_lib

def TDDFT_eigen_solver(init, prec, k=args.nstates, tol=args.conv_tolerance):
    '''[ A B ] X - [1   0] Y Ω = 0
       [ B A ] Y   [0  -1] X   = 0
    '''
    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)
    print('A matrix size = ', A_size)

    TDDFT_start = time.time()
    max = args.max
    m = 0

    new_m = min([k + args.extrainitial, 2*k, A_size])

    TDDFT_i_lib, TDDFT_p_lib = gen_TDDFT_lib()

    initial_guess = TDDFT_i_lib[init]
    new_guess_generator = TDDFT_p_lib[prec]

    V_holder = np.zeros((A_size, (max+3)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    init_start = time.time()
    V_holder, W_holder, new_m, initial_energies, X_ig, Y_ig =\
                                    initial_guess(V_holder, W_holder, new_m)
    init_end = time.time()
    init_time = init_end - init_start

    initial_energies = initial_energies.tolist()[:k]

    print('new_m =', new_m)
    print('initial guess done')

    Pcost = 0
    for ii in range(max):
        print('\niteration', ii)
        show_memory_info('beginning of step '+ str(ii))

        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]
        '''U1 = AV + BW
           U2 = AW + BV'''
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] =\
                            TDDFT_matrix_vector(V[:, m:new_m], W[:, m:new_m])

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)

        a = mathlib.symmetrize(a)
        b = mathlib.symmetrize(b)
        sigma = mathlib.symmetrize(sigma)
        pi = mathlib.anti_symmetrize(pi)

        print('subspace size: %s' %sigma.shape[0])

        omega, x, y = mathlib.TDDFT_subspace_eigen_solver(a, b, sigma, pi, k)

        '''compute the residual
           R_x = U1x + U2y - X_full*omega
           R_y = U2x + U1y + Y_full*omega
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''

        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega

        residual = np.vstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        max_norm = np.max(r_norms)
        print('Maximum residual norm: ', '%.2e'%max_norm)
        if max_norm < tol or ii == (max -1):
            print('TDDFT precedure Done\n')
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]
        index = [i for i,R in enumerate(r_norms) if R > tol]
        print('unconverged states', index)

        P_start = time.time()
        X_new, Y_new = new_guess_generator(\
                            R_x[:,index], R_y[:,index], omega[index])
        P_end = time.time()
        Pcost += P_end - P_start

        m = new_m
        V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(\
                                        V_holder, W_holder, m, X_new, Y_new)
        print('m & new_m', m, new_m)
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    omega = omega*parameterlib.Hartree_to_eV

    difference = np.mean((np.array(initial_energies) - np.array(omega))**2)
    difference = float(difference)

    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)) \
                    + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    TDDFT_end = time.time()
    TDDFT_cost = TDDFT_end - TDDFT_start

    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=k,
            icost=init_time, pcost=Pcost, wall_time=TDDFT_cost,
            energies=omega.tolist(), N_itr=ii+1, N_mv=np.shape(sigma)[0],
            initial_energies=initial_energies, difference=difference,
            overlap=overlap)
    if ii == (max -1):
        print('===== TDDFT Failed Due to Iteration Limit============')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('============= TDDFT Calculation Done ==============')

    print('after', ii+1,'iterations','%.2f'%TDDFT_cost,'s')
    print('Final subspace ', sigma.shape[0])
    print('preconditioning cost', '%.4f'%Pcost, '%.2f'%(Pcost/TDDFT_cost),"%")
    print('max_norm = ', '%.2e'%max_norm)

    show_memory_info('Total TDDFT')
    return omega, X_full, Y_full, Davidson_dic

def gen_dynpol_lib():
    i_lib={}
    p_lib={}
    i_lib['sTDDFT'] = sTDDFT_preconditioner
    i_lib['Adiag']  = TDDFT_A_diag_preconditioner
    p_lib['sTDDFT'] = sTDDFT_preconditioner
    p_lib['Adiag']  = TDDFT_A_diag_preconditioner
    return i_lib, p_lib

def gen_P():
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]
    int_r= mol.intor_symmetric('int1e_r')
    P = lib.einsum("xpq,pi,qa->iax", int_r, orbo, orbv.conj())
    return P

def dynamic_polarizability(init, prec):
    ''' [ A B ] - [1  0]X  w = -P'''
    ''' [ B A ]   [0 -1]Y    = -Q'''
    dp_start = time.time()

    dynpol_i_lib, dynpol_p_lib = gen_dynpol_lib()
    initial_guess = dynpol_i_lib[init]
    new_guess_generator = dynpol_p_lib[prec]

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)
    print('A matrix size = ', A_size)

    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    k = len(args.dynpol_omega)
    omega =  np.zeros([3*k])
    for jj in range(k):
        '''if have 3 ω, [ω1 ω1 ω1, ω2 ω2 ω2, ω3 ω3 ω3]
           convert nm to Hartree'''
        omega[3*jj:3*(jj+1)] = 45.56337117/args.dynpol_omega[jj]

    P = gen_P()
    P = P.reshape(-1,3)

    P_origin = np.zeros_like(P)
    Q = np.zeros_like(P)

    P_origin[:,:] = P[:,:]
    Q[:,:] = P[:,:]

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    pqnorm = pnorm * (2**0.5)
    print('pqnorm', pqnorm)
    P /= pqnorm

    P = np.tile(P,k)

    max = args.max
    tol = args.conv_tolerance
    m = 0
    V_holder = np.zeros((A_size, (max+1)*k*3))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    init_start = time.time()
    X_ig, Y_ig = initial_guess(-P, -Q, omega, tol=args.initial_TOL)

    alpha_omega_ig = []
    X_p_Y = X_ig + Y_ig
    X_p_Y = X_p_Y*np.tile(pqnorm,k)
    for jj in range(k):
        '''*-1 from the definition of dipole moment. *2 for double occupancy'''
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        alpha_omega_ig.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)
    print('initial guess of tensor alpha')
    for i in range(k):
        print(args.dynpol_omega[i],'nm')
        print(alpha_omega_ig[i])

    V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(\
                                            V_holder, W_holder, 0, X_ig, Y_ig)
    init_end = time.time()
    initial_cost = init_end - init_start
    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        print('Iteration', ii)

        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = TDDFT_matrix_vector(\
                                                V[:, m:new_m], W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)

        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)

        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)

        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)

        '''p = VP + WQ
           q = WP + VQ'''
        p = np.dot(V.T, P)
        p += np.dot(W.T, Q)

        q = np.dot(W.T, P)
        q += np.dot(V.T, Q)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        print('sigma.shape', sigma.shape)

        subcost_start = time.time()
        x, y = mathlib.TDDFT_subspace_liear_solver(a, b, sigma, pi, -p, -q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R_x = U1x + U2y - X_full*omega + P
           R_y = U2x + U1y + Y_full*omega + Q
           X_full = Vx + Wy
           Y_full = Wx + Vy
        '''
        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega
        R_x += P

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega
        R_y += Q

        residual = np.vstack((R_x,R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        print('maximum residual norm: ', '%.3e'%np.max(r_norms))

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        if np.max(r_norms) < tol or ii == (max -1):
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]

        Pstart = time.time()
        X_new, Y_new = new_guess_generator(R_x[:,index],
                            R_y[:,index], omega[index], tol=args.precond_TOL)
        Pend = time.time()
        Pcost += Pend - Pstart

        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = mathlib.VW_Gram_Schmidt_fill_holder(\
                                        V_holder, W_holder, m, X_new, Y_new)
        GS_end = time.time()
        GScost += GS_end - GS_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    dp_end = time.time()
    dp_cost = dp_end - dp_start

    if ii == (max -1):
        print('======= Dynamic polarizability Failed Due to Iteration Limit=====')
        print('dynamic polarizability failed after ', ii+1, 'iterations  ', round(dp_cost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('Dynamic polarizability Converged after ', ii+1, 'iterations  ', round(dp_cost, 4), 'seconds')
        print('initial_cost', round(initial_cost,4), round(initial_cost/dp_cost * 100,2),'%')
        print('Pcost', round(Pcost,4), round(Pcost/dp_cost * 100,2),'%')
        print('MVcost', round(MVcost,4), round(MVcost/dp_cost * 100,2),'%')
        print('GScost', round(GScost,4), round(GScost/dp_cost * 100,2),'%')
        print('subcost', round(subcost,4), round(subcost/dp_cost * 100,2),'%')
        print('subgencost', round(subgencost,4), round(subgencost/dp_cost * 100,2),'%')

    print('Wavelength we look at', args.dynpol_omega)
    alpha_omega = []

    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full))\
                    + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    X_p_Y = X_full + Y_full

    X_p_Y = X_p_Y*np.tile(pqnorm,k)

    for jj in range(k):
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        alpha_omega.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)

    difference = 0
    for i in range(k):
        difference += np.mean((alpha_omega_ig[i] - alpha_omega[i])**2)

    difference = float(difference)

    show_memory_info('Total Dynamic polarizability')
    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=3*k,
            icost=initial_cost, pcost=Pcost, wall_time=dp_cost,
            energies=omega.tolist(), N_itr=ii+1, N_mv=np.shape(sigma)[0],
            difference=difference, overlap=overlap,
            tensor_alpha=[i.tolist() for i in alpha_omega],
            initial_tensor_alpha=[i.tolist() for i in alpha_omega_ig])
    return alpha_omega, Davidson_dic

def stapol_A_diag_initprec(P, hdiag=hdiag, tol=None):
    d = hdiag.reshape(-1,1)
    P = -P/d
    # P /= -d
    return P

def stapol_sTDDFT_initprec(Pr, tol=args.initial_TOL, matrix_vector_product=sTDDFT_stapol_mv):
    '''(A* + B*)X = -P
       note the negative sign of P!
       residual = (A* + B*)X + P
       X_ig = -P/d
       X_new = residual/D
    '''
    ssp_start = time.time()
    max = 30
    m = 0
    npvec = Pr.shape[1]

    P = Pr.reshape(n_occ,n_vir,-1)[:,:max_vir,:]
    P = P.reshape(A_reduced_size,-1)
    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P /= pnorm

    V_holder = np.zeros((A_reduced_size, (max+1)*npvec))
    U_holder = np.zeros_like(V_holder)

    '''setting up initial guess'''
    init_start = time.time()
    X_ig = stapol_A_diag_initprec(P, hdiag=max_vir_hdiag)
    V_holder, new_m = mathlib.Gram_Schmidt_fill_holder(V_holder, m, X_ig)
    init_end = time.time()
    initial_cost = init_end - init_start

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        '''creating the subspace'''
        MV_start = time.time()
        '''U = AX + BX = (A+B)X'''
        U_holder[:, m:new_m] = matrix_vector_product(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = mathlib.symmetrize(a_p_b)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''solve the x in the subspace'''
        subcost_start = time.time()
        x = np.linalg.solve(a_p_b, -p)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''compute the residual
           R = Ux + P'''
        Ux = np.dot(U,x)
        residual = Ux + P

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        index = [r_norms.index(i) for i in r_norms if i > tol]
        if np.max(r_norms) < tol or ii == (max -1):
            print('Static polarizability procedure aborted')
            break

        Pstart = time.time()
        X_new = stapol_A_diag_initprec(-residual[:,index], hdiag=max_vir_hdiag)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, new_m = mathlib.Gram_Schmidt_fill_holder(V_holder, m, X_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break
    X_full = np.dot(V,x)
    '''alpha = np.dot(X_full.T, P)*-4'''

    ssp_end = time.time()
    ssp_cost = ssp_end - ssp_start

    if ii == (max -1):
        print('== sTDDFT Stapol precond Failed Due to Iteration Limit======')
        print('current residual norms', r_norms)
    else:
        print('sTDDFT Stapol precond Converged' )
    print('after', ii+1, 'steps', '%.4f'%ssp_cost,'s')
    print('conv threhsold = %.2e'%tol)
    print('final subspace:', a_p_b.shape[0])
    print('max_norm = ', '%.2e'%np.max(r_norms))
    for enrty in ['initial_cost','MVcost','GScost','subgencost','subcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/ssp_cost))

    X_full = X_full*pnorm

    U = np.zeros((n_occ,n_vir,npvec))
    U[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,-1)[:,:,:]

    if max_vir < n_vir:
        ''' DX2 = -P2'''
        P2 = Pr.reshape(n_occ,n_vir,-1)[:,max_vir:,:]
        P2 = P2.reshape(n_occ*(n_vir-max_vir),-1)
        D2 = delta_hdiag[:,max_vir:]
        D2 = D2.reshape(n_occ*(n_vir-max_vir),-1)
        X2 = (-P2/D2).reshape(n_occ,n_vir-max_vir,-1)
        U[:,max_vir:,:] = X2
    U = U.reshape(A_size, npvec)
    return U

def gen_stapol_lib():
    i_lib={}
    p_lib={}
    i_lib['sTDDFT'] = stapol_sTDDFT_initprec
    i_lib['Adiag']  = stapol_A_diag_initprec
    p_lib['sTDDFT'] = stapol_sTDDFT_initprec
    p_lib['Adiag']  = stapol_A_diag_initprec
    return i_lib, p_lib

def static_polarizability(init, prec):
    '''(A+B)X = -P
       residual = (A+B)X + P
    '''
    print('initial guess', init)
    print('preconditioner', prec)
    sp_start = time.time()

    P = gen_P()
    P = P.reshape(-1,3)

    P_origin = np.zeros_like(P)
    P_origin[:,:] = P[:,:]

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    P /= pnorm

    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    stapol_i_lib, stapol_p_lib = gen_stapol_lib()
    initial_guess = stapol_i_lib[init]
    new_guess_generator = stapol_p_lib[prec]

    max = args.max
    tol = args.conv_tolerance
    m = 0

    V_holder = np.zeros((A_size, (max+1)*3))
    U_holder = np.zeros_like(V_holder)

    init_start = time.time()
    X_ig = initial_guess(P, tol=args.initial_TOL)

    alpha_init = np.dot((X_ig*pnorm).T, P_origin)*-4
    print('alpha tensor of initial guess:')
    print(alpha_init)

    V_holder, new_m = mathlib.Gram_Schmidt_fill_holder(V_holder, 0, X_ig)
    print('new_m =', new_m)
    init_end = time.time()
    initial_cost = init_end - init_start

    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        print('\nIteration', ii)
        MV_start = time.time()
        U_holder[:, m:new_m] = \
                    static_polarizability_matrix_vector(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = mathlib.symmetrize(a_p_b)
        subgenend = time.time()

        '''solve the x in the subspace'''
        x = np.linalg.solve(a_p_b, -p)

        '''compute the residual
           R = Ux + P'''
        Ux = np.dot(U,x)
        residual = Ux + P

        r_norms = np.linalg.norm(residual, axis=0).tolist()

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        '''index for unconverged residuals'''
        index = [r_norms.index(i) for i in r_norms if i > tol]
        max_norm = np.max(r_norms)
        print('max_norm = %.2e'%max_norm)
        if max_norm < tol or ii == (max -1):
            # print('static polarizability precodure aborted\n')
            break

        '''preconditioning step'''
        Pstart = time.time()

        X_new = new_guess_generator(-residual[:,index], tol=args.precond_TOL)
        Pend = time.time()
        Pcost += Pend - Pstart

        '''GS and symmetric orthonormalization'''
        m = new_m
        GS_start = time.time()
        V_holder, new_m = mathlib.Gram_Schmidt_fill_holder(V_holder, m, X_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break

    X_full = np.dot(V,x)
    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)))

    X_full = X_full*pnorm

    tensor_alpha = np.dot(X_full.T, P_origin)*-4
    sp_end = time.time()
    sp_cost = sp_end - sp_start

    if ii == (max -1):
        print('==== Static polarizability Failed Due to Iteration Limit ======')
        print('current residual norms', r_norms)
        print('max_norm = ', max_norm)
    else:
        print('Static polarizability Converged')

    print('after', ii+1, 'steps; %.4f'%sp_cost,'s')
    print('final subspace', a_p_b.shape)
    print('max_norm = ', '%.2e'%np.max(r_norms))
    for enrty in ['initial_cost','MVcost','Pcost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s  {:<5.2%}".format(enrty, cost, cost/sp_cost))

    difference = np.mean((alpha_init - tensor_alpha)**2)
    difference = float(difference)

    sp_end = time.time()
    spcost = sp_end - sp_start
    Davidson_dic = fill_dictionary(Davidson_dic, init=init, prec=prec, k=3,
            icost=initial_cost, pcost=Pcost, wall_time=sp_cost,
            N_itr=ii+1, N_mv=np.shape(a_p_b)[0], difference=difference,
            overlap=overlap, tensor_alpha=[i.tolist() for i in tensor_alpha],
            initial_tensor_alpha=[i.tolist() for i in alpha_init])
    return tensor_alpha, Davidson_dic

def gen_calc():
    name_dic={}
    name_dic['TDA'] = args.TDA
    name_dic['TDDFT'] = args.TDDFT
    name_dic['dynpol'] = args.dynpol
    name_dic['stapol'] = args.stapol
    name_dic['sTDA'] = args.sTDA
    name_dic['sTDDFT'] = args.sTDDFT
    name_dic['Truncate_test'] = args.Truncate_test
    name_dic['PySCF_TDDFT'] = args.pytd
    for calc in ['TDA','TDDFT','dynpol','stapol',
                        'sTDA','sTDDFT','Truncate_test','PySCF_TDDFT']:
        if name_dic[calc] == True:
            print(calc)
            return calc

def dump_yaml(Davidson_dic, calc, init, prec):
    curpath = os.getcwd()
    yamlpath = os.path.join(\
                   curpath,basename+'_'+calc+'_i_'+init+'_p_'+prec+'.yaml')
    with open(yamlpath, "w", encoding="utf-8") as f:
        yaml.dump(Davidson_dic, f)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def gen_metrics(beta, alpha, ab_initio_V, ab_initio_E,
                    full_sTDA_mv, sTDA_mv, AV, residual,
                                V_basis, W_basis, sub_A):
    V = ab_initio_V #ab-initio eigenkets
    lambda_matrix = np.diag(ab_initio_E)

    with HiddenPrints():
        '''sTDA V and E
           energies_se is in eV
        '''
        V_se, energies_se = sTDA_eigen_solver(k=args.nstates,
                                            tol=args.conv_tolerance,
                          matrix_vector_product=sTDA_mv)
    '''energy difference
    '''
    energy_diff = np.linalg.norm(energies_se - ab_initio_E*parameterlib.Hartree_to_eV)

    V_overlap = np.linalg.norm(np.dot(V_se.T, V))

    if args.AV == True:
        ab_initio_AV = AV
    else:
        ab_initio_AV = ab_initio_E * V

    ''' commutator norm
        |V.T[A, A^se]V|
       =[λ, V.T*A^se*V]
    '''
    VAseV = np.dot(V.T, full_sTDA_mv(V))
    commutator = mathlib.commutator(lambda_matrix, VAseV)
    VcAAV = np.linalg.norm(commutator)

    ''' 2nd promising!
        condition number
        |V.T * A^{se,-1}A * V|
        AV = λ * V
        A^{se,-1} AV = X => A^se X = AV
    '''
    with HiddenPrints():
        '''A^se X=W
        '''
        X = stapol_sTDDFT_initprec(Pr=-ab_initio_AV,
                                  tol=1e-6,
                matrix_vector_product=sTDA_mv)
    sub_condition = np.dot(V.T, X)
    k_VA_1AV = mathlib.cond_number(sub_condition)

    '''commutator basis V norm
        here, V_b is the Kryove space basis, W = AV
        |V_b.T [A, A^se] V_b|
       = V_b.T (A A^se - A^se A) V_b
       = V_b.T A A^se V_b - V_b.T A^se A V_b
       = W_b.T W^se - W^se.T W_b
    '''
    AseVb = full_sTDA_mv(V_basis)
    WbAseVb = np.dot(W_basis.T, AseVb)
    VbcAAVb = np.linalg.norm(WbAseVb - WbAseVb.T)

    ''' most promising!
        quality norm
        |(I - A^{se,-1}A) V |
        = V - X
    '''
    I_AAV = np.linalg.norm(V - X)

    metric_list = [
        beta,
        alpha,
        I_AAV,
        k_VA_1AV,
        VcAAV,
        VbcAAVb,
        energy_diff,
        V_overlap]

    return metric_list

def gen_I_AAV(ab_initio_E:list, ab_initio_V:np.array,
                            matrix_vector_product=sTDA_mv) -> float:
    with HiddenPrints():
        '''A^se X= AV
        '''
        AV = ab_initio_E * ab_initio_V
        X = stapol_sTDDFT_initprec(Pr=-AV, tol=1e-6,
                            matrix_vector_product=matrix_vector_product)

    I_AAV = float(np.linalg.norm(ab_initio_V - X))
    return I_AAV

def gen_forecaster(eta, *args):
    '''  |(I - A^{se,-1}A) V |  = V - X
        a is chemical hardness for each atom, in shape (N_atm,)
    '''
    ab_initio_E, ab_initio_V = args


    I_AAV = gen_I_AAV(ab_initio_E=ab_initio_E, ab_initio_V=ab_initio_V,
                                                matrix_vector_product=sTDA_mv)

    return I_AAV

def gen_as_forecaster(U:np.array, *args):
    U = U.tolist()
    '''  |(I - A^{se,-1}A) V |  = V - X
        U is index for each atom, 1d array (N_atm,)
    '''
    ab_initio_E, ab_initio_V = args
    TDDFT_as = TDDFT_as_lib.TDDFT_as(U_list=U_list)

    TDDFT_as.build()
    as_mv = TDDFT_as.TDA_as_mv

    I_AAV = gen_I_AAV(ab_initio_E=ab_initio_E, ab_initio_V=ab_initio_V,
                                                matrix_vector_product=as_mv)

    return I_AAV

if __name__ == "__main__":
    calc = gen_calc()
    TDA_combo = [            # option
    ['sTDA','sTDA'],         # 0
    ['Adiag','Adiag'],       # 1
    ['Adiag','sTDA'],        # 2
    ['sTDA','Adiag'],        # 3
    ['sTDA','Jacobi'],       # 4
    ['Adiag','Jacobi'],      # 5
    ['Adiag','new_ES'],      # 6
    ['sTDA','new_ES']]       # 7
    TDDFT_combo = [          # option
    ['sTDDFT','sTDDFT'],     # 0
    ['Adiag','Adiag'],       # 1
    ['Adiag','sTDDFT'],      # 2
    ['sTDDFT','Adiag']]      # 3
    print('|-------- In-house Developed {0} Starts ---------|'.format(calc))
    print('Residual conv =', args.conv_tolerance)
    if args.TDA == True:
        for option in args.ip_options:
            init, prec = TDA_combo[option]
            print('\n','Number of excited states = ', args.nstates)
            (Excitation_energies, eigenkets, AV, residual, Davidson_dic,
                                V_basis, W_basis, sub_A) = Davidson(init,prec)
            print('Excited State energies (eV) = ','\n', Excitation_energies)

            dump_yaml(Davidson_dic, calc, init, prec)
    if args.traceAA == True:
        metric_name_list = [
        'beta',
        'alpha',
        '|(I - A^se,-1 A)V|',
        'k(V.T(A^se,-1 A)V)',
        '|V.T[A, A^se]V|',
        '|Vb.T[A, A^se]Vb|',
        '|E^se - E^ab|',
        '|V^se,T V|']

        metric_name_format = "{0[0]:<8s}{0[1]:<8s}{0[2]:<20s}{0[3]:<20s}{0[4]:<20s}{0[5]:<20s}{0[6]:<20s}{0[7]:<20s}"
        metric_value_format = "{0[0]:<8.2f}{0[1]:<8.2f}{0[2]:<20.8f}{0[3]:<20.8f}{0[4]:<20.8f}{0[5]:<20.8f}{0[6]:<20.8f}{0[7]:<20.8f}"

        with open("data.txt", "w") as data_file:
            print(metric_name_format.format(metric_name_list),file=data_file)
        print(metric_name_format.format(metric_name_list))

        smallest_I_AAV = 10000
        opt_beta, opt_alpha = 0, 0
        for beta in args.beta_list:
            for alpha in args.alpha_list:
                GammaJ, GammaK = gen_gammaJK(alpha=alpha, beta=beta)
                sTDA_mv, full_sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv = gen_mv_fly(
                                                                GammaJ=GammaJ,
                                                                GammaK=GammaK)

                metric_list = gen_metrics(beta=beta,
                                         alpha=alpha,
                                   ab_initio_V=eigenkets,
             ab_initio_E=Excitation_energies/parameterlib.Hartree_to_eV,
                                       V_basis=V_basis,
                                       W_basis=W_basis,
                                         sub_A=sub_A,
                                  full_sTDA_mv=full_sTDA_mv,
                                       sTDA_mv=sTDA_mv,
                                            AV=AV,
                                      residual=residual)
                if metric_list[2] < smallest_I_AAV:
                    smallest_I_AAV = metric_list[2]
                    opt_beta, opt_alpha = metric_list[0], metric_list[1]

                with open("data.txt", "a") as data_file:
                    print(metric_value_format.format(metric_list), file=data_file)
                print(metric_value_format.format(metric_list))

        print('smallest_I_AAV =', smallest_I_AAV)
        print('opt_beta = ', opt_beta)
        print('opt_alpha = ', opt_alpha)
    if args.etatune == True:

        eta=gen_eta()

        print(mol.elements)

        print('eta =')
        [print(i) for i in eta[0,:].tolist()]
        I_AAV_initial = gen_forecaster(eta, Excitation_energies/parameterlib.Hartree_to_eV, eigenkets)
        print('I_AAV_initial =', I_AAV_initial)
        bnds = []
        for i in range(eta.shape[1]):
            bnds.append((eta[0,i]*(1-args.bounds), eta[0,i]*(1+args.bounds)))
        print('bnds =')
        [print(i) for i in bnds]

        for m in ['SLSQP']:
            print('method =', m)
            result = scipy.optimize.minimize(fun=gen_forecaster,
                                              x0=eta,
                    args=(Excitation_energies/parameterlib.Hartree_to_eV, eigenkets),
                                          method=m,
                                          bounds=bnds,
                                          options={
                                          'maxiter': 100,
                                          'ftol': args.ftol,
                                          'iprint': 99,
                                          'disp': True,
                                          'eps': args.step})
            print('result.x =')
            [print(i) for i in result.x]

            print('result.success =', result.success)
            print('result.message =', result.message)

            with open(m+'_eta.txt', 'wb') as f:
                np.savetxt(f, result.x)

            I_AAV_min = gen_forecaster(result.x,
                    Excitation_energies/parameterlib.Hartree_to_eV, eigenkets)
            Davidson_dic['I_AAV_min'] = I_AAV_min
            dump_yaml(Davidson_dic, calc, init, prec)
            print('I_AAV_min =', I_AAV_min)
            print()
    if args.Utune == True:

        # U_list = [0.124]*N_atm
        U_list=np.array(U_list)
        print('U_list shape =', U_list.shape)
        bnds = []
        for i in range(N_atm):
            bnds.append((U_list[i,]*0.1, U_list[i,]*30))

        print('boundary =', bnds)
        I_AAV_initial = gen_as_forecaster(U_list, Excitation_energies/parameterlib.Hartree_to_eV, eigenkets)
        print('I_AAV_initial =', I_AAV_initial)

        print('starting scipy optimization')
        with HiddenPrints():
            result = scipy.optimize.minimize(fun=gen_as_forecaster,
                                              x0=U_list,
                    args=(Excitation_energies/parameterlib.Hartree_to_eV, eigenkets),
                                          method='SLSQP',
                                          bounds=bnds,
                                          options={
                                          'maxiter': 100,
                                          'ftol': args.ftol,
                                          'iprint': 99,
                                          'disp': False,
                                          'eps': args.step})
        print('scipy optimization finished')
        print('result.x =')
        [print(i) for i in result.x]

        print('result.success =', result.success)
        print('result.status =', result.status)
        print('result.message =', result.message)
        print('result.fun =', result.fun)
        print('result.nit =', result.nit)


        with open('SLSQP_U.txt', 'wb') as f:
            np.savetxt(f, result.x)

        Davidson_dic['I_AAV_min'] = result.fun
        dump_yaml(Davidson_dic, calc, init, prec)

    if args.TDDFT == True:
        for option in args.ip_options:
            init, prec = TDDFT_combo[option]
            print('\nNumber of excited states =', args.nstates)
            Excitation_energies,X,Y,Davidson_dic = TDDFT_eigen_solver(init,prec)
            print('Excited State energies (eV) =\n',Excitation_energies)
            dump_yaml(Davidson_dic, calc, init, prec)
    if args.dynpol == True:
        for option in args.ip_options:
            init,prec = TDDFT_combo[option]
            print('\nPerturbation wavelength omega (nm) =', args.dynpol_omega)
            alpha_omega, Davidson_dic = dynamic_polarizability(init,prec)
            print('Dynamic polarizability tensor alpha')
            dump_yaml(Davidson_dic, calc, init, prec)
            for i in range(len(args.dynpol_omega)):
                print(args.dynpol_omega[i],'nm')
                print(alpha_omega[i])
    if args.stapol == True:
        for option in args.ip_options:
            init,prec = TDDFT_combo[option]
            print('\n')
            tensor_alpha, Davidson_dic = static_polarizability(init,prec)
            print('Static polarizability tensor alpha')
            print(tensor_alpha)
            dump_yaml(Davidson_dic, calc, init, prec)
    if args.sTDA == True:
        X, energies = sTDA_eigen_solver(k=args.nstates, tol=args.conv_tolerance, matrix_vector_product=sTDA_mv)
    if args.TDDFT_as == True:
        X, energies = sTDA_eigen_solver(k=args.nstates, tol=args.conv_tolerance, matrix_vector_product=sTDA_mv)
    if args.sTDDFT == True:
        energies,X,Y = sTDDFT_eigen_solver(k=args.nstates,tol=args.conv_tolerance)

    if args.pytd == True:
        TD.nstates = args.nstates
        TD.conv_tol = args.conv_tolerance
        TD.kernel()
        end = time.time()
    if args.verbose > 3:
        for key in vars(args):
            print(key,'=', vars(args)[key])
    print('|-------- In-house Developed {0} Ends ----------|'.format(calc))
