# -*- coding: utf-8 -*-

import numpy as np

import os, sys
from pyscf import lib
from arguments import args

import scipy
from scipy import optimize
from SCF_calc import show_memory_info, calc_name

from TDA import TDA
from TDDFT import TDDFT
from dpolar import dpolar
from spolar import spolar


print('curpath', os.getcwd())
print('lib.num_threads() = ', lib.num_threads())

show_memory_info('at beginning')

if __name__ == "__main__":

    print('|-------- In-house Developed {0} Starts ---------|'.format(calc_name))
    print('Residual conv =', args.conv_tolerance)
    if args.TDA == True:
        TDA.main()
    if args.TDDFT == True:
        TDDFT.main()
    if args.dpolar == True:
        dpolar.main()
    if args.spolar == True:
        spolar.main()

    if args.sTDA == True:
        pass
        # X, energies = sTDA_eigen_solver(k=args.nstates, tol=args.conv_tolerance, matrix_vector_product=sTDA_mv)
    if args.TDDFT_as == True:
        pass
        # X, energies = sTDA_eigen_solver(k=args.nstates, tol=args.conv_tolerance, matrix_vector_product=sTDA_mv)
    if args.sTDDFT == True:
        pass
        # energies,X,Y = sTDDFT_eigen_solver(k=args.nstates,tol=args.conv_tolerance)

    if args.pytd == True:
        TD.nstates = args.nstates
        TD.conv_tol = args.conv_tolerance
        TD.kernel()

    if args.verbose > 3:
        for key in vars(args):
            print(key,'=', vars(args)[key])
    print('|-------- In-house Developed {0} Ends ----------|'.format(calc_name))
