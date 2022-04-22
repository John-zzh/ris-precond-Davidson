# -*- coding: utf-8 -*-


import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import time
import numpy as np
from mathlib import math, parameter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Start reading database')
# read database here
records = {'john': 55, 'tom': 66}
logger.debug('Records: %s', records)
logger.info('Updating records ...')
# update records here
logger.info('Finish updating records')

def Davidson_symmetric(N_states,
                  initial_guess,
                 preconditioner,
          matrix_vector_product,
                         A_size,
                          hdiag,
                  # max_vir_hdiag,
                  #         n_occ,
                  #         n_vir,
                  #       max_vir,
                            max = 35,
                       conv_tol = 1e-5,
                   extrainitial = 8,
                  Davidson_dict = {}):
    '''
    AX = XΩ
    Davidson frame, can use different initial guess and preconditioner
    initial_guess is a function, takes the number of initial guess as input
    preconditioner is a function, takes the residual as the input
    '''
    logger.info('====== Davidson Diagonalization Starts ======')

    D_start = time.time()
    Davidson_dict['iteration'] = []
    iteration_list = Davidson_dict['iteration']


    m = 0
    new_m = min([N_states + extrainitial, 2*N_states, A_size])
    V = np.zeros((A_size, max*N_states + new_m))
    W = np.copy(V)

    '''
    generate the initial guesss and put into the basis holder V
    '''

    init_start = time.time()
    initial_vectors, initial_energies = initial_guess(N_states = new_m,
                                                         hdiag = hdiag,
                                         matrix_vector_product = sTDA_mv)
    V[:,:new_m] = initial_vectors[:,:]
    init_end = time.time()
    init_cost = init_end - init_start

    initial_energies = initial_energies[:N_states]
    logger.info('initial excitation energies:', initial_energies)
    logger.info('initial guess time {:.2f} seconds'.format(init_cost))

    Pcost = 0
    MVcost = 0
    for ii in range(max):
        logger.debug('Iteration: {:d}'.format(ii+1))
        istart = time.time()

        MV_start = time.time()
        W[:, m:new_m] = matrix_vector_product(V[:,m:new_m])
        MV_end = time.time()
        iMVcost = MV_end - MV_start
        MVcost += iMVcost

        sub_A = np.dot(V[:,:new_m].T, W[:,:new_m])
        sub_A = math.symmetrize(sub_A)
        logger.debug('subspace size: {:d}'.format(sub_A.shape[0]))

        '''
        Diagonalize the subspace A matrix, return sorted eigenvalues & vectors.
        sub_eigenvalue[:N_states] are smallest N_states eigenvalues
        '''
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        sub_eigenvalue = sub_eigenvalue[:N_states]
        sub_eigenket = sub_eigenket[:,:N_states]

        full_guess = np.dot(V[:,:new_m], sub_eigenket)
        AV = np.dot(W[:,:new_m], sub_eigenket)
        residual = AV - full_guess * sub_eigenvalue

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)

        iteration_list.append({})
        current_dict = iteration_list[ii]
        current_dict['residual norms'] = r_norms

        logger.debug('maximum residual norm {:.2e}'.format(max_norm))
        if max_norm < conv_tol or ii == (max-1):
            iend = time.time()
            icost = iend - istart
            current_dict['iteration total cost'] = icost
            current_dict['iteration MV cost'] = iMVcost
            iteration_list[ii] = current_dict
            logger.debug('Davidson Diagonalization Done')
            break

        index = [r_norms.index(i) for i in r_norms if i>conv_tol]

        P_start = time.time()
        new_guess = preconditioner(residual = residual[:,index],
                             sub_eigenvalue = sub_eigenvalue[index],
                                      hdiag = hdiag)
        P_end = time.time()
        Pcost += P_end - P_start

        m = new_m
        V, new_m = math.Gram_Schmidt_fill_holder(V, m, new_guess)
        logger.debug('amount of newly generated guesses:', new_m - m)

        iend = time.time()
        icost = iend - istart
        current_dict['iteration cost'] = icost
        current_dict['iteration MV cost'] = iMVcost
        iteration_list[ii] = current_dict
        logger.debug('iteration MV time {:.2f} seconds'.format(iMVcost))
        logger.debug('iteration total time {:.2f} seconds'.format(icost))

    energies = sub_eigenvalue*parameter.Hartree_to_eV

    D_end = time.time()
    Dcost = D_end - D_start

    energy_diff = float(np.linalg.norm(initial_energies - energies))
    X_overlap = float(np.linalg.norm(np.dot(initial_vectors.T, full_guess)))

    Davidson_dict = fill_dictionary(Davidson_dict,
                           N_itr = ii+1,
                           pcost = Pcost,
                           icost = init_cost,
                            N_mv = sub_A.shape[0],
                       wall_time = Dcost,
                initial_solution = initial_energies.tolist(),
                  final_solution = energies.tolist(),
                      difference = energy_diff,
                         overlap = X_overlap)

    if ii == max-1:
        logger.info('=== TDA Failed Due to Iteration Limit ===')
        logger.info('current residual norms', r_norms)
    else:
        logger.info('========== TDA Calculation Done==========')
    logger.info('energies:')
    logger.info(energies)
    logger.info('Maximum residual norm = {:.2e}'.format(max_norm))
    logger.info('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, Dcost))
    logger.info('Final subspace size = {:d}'.format(sub_A.shape[0]))
    logger.info('Total precondition time: {:.2f} seconds, {:.2%}'.format(Pcost, Pcost/Dcost))
    logger.info('Total Matrix-vector product cost {:.2f} seconds'.format(MVcost))

    return energies, full_guess, Davidson_dict, AV
