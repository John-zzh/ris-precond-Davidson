# -*- coding: utf-8 -*-

'''
absolute import
'''
import os,sys
import numpy as np
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from TDDFT.TDDFT_iter_initial_guess import TDDFT_iter_initial_guess
from TDDFT.TDDFT_iter_preconditioner import TDDFT_iter_preconditioner
from TDDFT.TDDFT_solver import TDDFT_solver

from arguments import args
from mathlib.diag_ip import TDDFT_diag_initial_guess, TDDFT_diag_preconditioner
from SCF_calc import gen_ip_func, ip_name, calc_name
from dump_yaml import dump_yaml


def main():
    TDA_ip_dict = gen_ip_func(diag_i = TDDFT_diag_initial_guess,
                              iter_i = TDDFT_iter_initial_guess,
                              diag_p = TDDFT_diag_preconditioner,
                              iter_p = TDDFT_iter_preconditioner)

    for option in args.ip_options:
        initial_guess, preconditioner = TDA_ip_dict[option]
        (Excitation_energies, X, Y, Davidson_dict) = TDDFT_solver(
                         N_states = args.nstates,
                    initial_guess = initial_guess,
                   preconditioner = preconditioner,
                         conv_tol = args.conv_tolerance,
                     extrainitial = args.extrainitial,
                              max = args.max)

        dump_yaml(Davidson_dict, option)
        np.savetxt('TDDFT', Excitation_energies, fmt='%.8f')

if __name__ == '__main__':
    main()
