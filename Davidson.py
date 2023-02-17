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
from CPKS import CPKS
import TDA_as_Uk, spectra
np.set_printoptions(linewidth=220, precision=5)
print('curpath', os.getcwd())

print('lib.num_threads() = ', lib.num_threads())

show_memory_info('at beginning')

if __name__ == "__main__":
    if args.verbose >= 5:
        for key in vars(args):
            print(key,'=', vars(args)[key])
    print('|-------- In-house Developed {0} Starts ---------|'.format(calc_name))

    a = [args.TDA, args.TDDFT, args.dpolar, args.spolar, args.spectra, args.CPKS]
    b = [TDA, TDDFT, dpolar, spolar, spectra, CPKS]
    # print(a,b)
    for i in range(len(a)):
        if a[i] == True:
            b[i].main()

    if args.test == True:
        import profile
        profile.main()

    if args.pytd == True:
        TD.nstates = args.nstates
        TD.conv_tol = args.conv_tolerance
        TD.kernel()


    print('|-------- In-house Developed {0} Ends ----------|'.format(calc_name))
