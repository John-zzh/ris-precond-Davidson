from pyscf import gto, scf, dft, tddft, data, lib

from opt_einsum import contract as einsum
import numpy as np
import functools
import os
import yaml
import scipy
from scipy import optimize

from arguments import args

from TDA.TDA_iter_initial_guess import TDA_iter_initial_guess
from TDA.TDA_iter_preconditioner import TDA_iter_preconditioner

from TDDFT_as import TDDFT_as_lib
from TDA.TDA_solver import TDA_solver

from SCF_calc import basename


def gen_RMSE(Uk):
    '''
    RMSE of TDA-as energy and TDA energy
    '''


    TDDFT_as = TDDFT_as_lib.TDDFT_as(Uk=Uk)
    TDA_as_mv, TDDFT_as_mv, TDDFT_as_spolar_mv = TDDFT_as.build()

    path = os.popen('ls *yaml').readlines()[0].replace('\n', '')
    with open(path, 'r') as y:
        temp = yaml.safe_load(y)

    abinitio_energy = np.array(temp['final solution'])
    U, as_energy = TDA_iter_initial_guess(N_states = args.nstates,
                                    conv_tol = args.conv_tolerance,
                       matrix_vector_product = TDA_as_mv)

    print('ab initio:')
    print(abinitio_energy)

    print('as:')
    print(as_energy)
    RMSE = float(np.linalg.norm(as_energy - abinitio_energy))

    print('RMSE:', RMSE)
    return RMSE

def main():
    # result = scipy.optimize.minimize(fun=gen_RMSE,
    #                                   x0=[args.Uk],
    #                               method='BFGS',
    #                               tol = args.ftol,
    #                               options={
    #                               'maxiter': 100,
    #                               'disp': True})

    result = scipy.optimize.minimize_scalar(fun = gen_RMSE,
                                        bounds = (0.1, 10),
                                        method = 'bounded',
                                       options = {
                                       'maxiter': 100,
                                       'disp': True,
                                       'xatol':args.ftol})
    '''
    fun: Function value.
    status: Termination status of the optimizer.
    success: Whether or not the optimizer exited successfully.
    message: Description of the cause of the termination.
    x: The solution of the optimization.
    nfev: Number of evaluations of the objective functions.
    '''
    for k,v in result.items():
        print(k,v)
    # print(vars(result))

    with open(basename+'_Uk.txt', 'w') as f:
        np.savetxt(f, np.array([result.x]))

if __name__ == '__main__':
    main()
