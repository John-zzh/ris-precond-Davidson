# -*- coding: utf-8 -*-

import os
import numpy as np
import time
from SCF_calc import gen_P, basename
from mathlib import parameter

from arguments import args
from sTDA import sTDA_mv_lib
from TDDFT_as import TDDFT_as_lib
from TDA.TDA_iter_initial_guess import TDA_iter_initial_guess
from TDDFT.TDDFT_iter_initial_guess import TDDFT_iter_initial_guess_solver
from spolar.spolar_iter_initprec import spolar_iter_initprec
# from approx_mv import approx_TDDFT_mv



def profile(current, standard):
    print('current ', current)
    print('standard', standard)

    difference = float(np.linalg.norm(standard - current))
    print('difference {:.2e}'.format(difference))
    print('difference < 1e-5 ?', difference < 1e-5)


def main():
    '''
    PBE0/def2-SVP/notruncation/columb with p, exchange no p
    '''
    if args.TDA_as_profile == True:
        print('args.TDA_as_profile == True')
        U, current = TDA_iter_initial_guess(args.nstates, conv_tol = args.conv_tolerance)

        standard = np.array([7.21125852,8.90473556,9.11320684,9.85079603,
                            10.13011112,10.62269141,10.64366598,11.53150119,
                            11.72485107,11.95051588,12.17699229,12.70232019,
                            13.19763166,13.65743867,13.99173964,14.10157737,
                            14.29360538,14.3237626, 15.29586278,15.44461689])
        profile(current, standard[:current.shape[0]])

    if args.TDDFT_as_profile == True:
        print('args.TDDFT_as_profile == True')
        current, X, Y = TDDFT_iter_initial_guess_solver(
                                  N_states = args.nstates,
                                  conv_tol = args.conv_tolerance)

        standard = np.array([7.17783877,8.87950997,9.06679987,9.83950791,
                            10.08954883,10.57475193,10.62195665,11.5106705,
                            11.69914005,11.87475773,12.16976521,12.65722489,
                            13.16464999,13.63114457,13.98035901,14.03851621,
                            14.24505458,14.31506715,15.22499659,15.31655733])
        profile(current, standard[:current.shape[0]])
        np.savetxt('TDDFT-s', current, fmt='%.8f')




    if args.spolar_as_profile == True or args.dpolar_as_profile == True:
        P_origin = gen_P().reshape(-1,3)
        pnorm = np.linalg.norm(P_origin, axis=0, keepdims = True)
        print('pnorm', pnorm)

    if args.spolar_as_profile == True:
        X_full = spolar_iter_initprec(RHS = -P_origin,
                                conv_tol = args.conv_tolerance)

        current = np.dot((X_full).T, P_origin)*-4

        standard = np.array([[17.84620839, -1.12906387, -0.30081435],
                             [-1.12906387, 17.5735604,   0.45302561],
                             [-0.30081435,  0.45302561, 15.99329249]])

        profile(current, standard)


    if args.dpolar_as_profile == True:
        pass






if __name__ == '__main__':
    main()
