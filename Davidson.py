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
import TDA_as_Uk, spectra

print('curpath', os.getcwd())
print('lib.num_threads() = ', lib.num_threads())

show_memory_info('at beginning')

if __name__ == "__main__":

    print('|-------- In-house Developed {0} Starts ---------|'.format(calc_name))

    a = [args.TDA, args.TDDFT, args.dpolar, args.spolar, args.spectra, args.Uk_tune]
    b = [TDA, TDDFT, dpolar, spolar, spectra, TDA_as_Uk]
    # print(a,b)
    for i in range(len(a)):
        if a[i] == True:
            b[i].main()


    if args.pytd == True:
        TD.nstates = args.nstates
        TD.conv_tol = args.conv_tolerance
        TD.kernel()

    if args.verbose > 3:
        for key in vars(args):
            print(key,'=', vars(args)[key])
    print('|-------- In-house Developed {0} Ends ----------|'.format(calc_name))
