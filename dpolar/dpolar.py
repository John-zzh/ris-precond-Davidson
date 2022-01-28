# -*- coding: utf-8 -*-

'''
absolute import
'''
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from arguments import args
from mathlib.diag_ip import TDDFT_diag_preconditioner

from SCF_calc import gen_ip_func
from dump_yaml import dump_yaml

from TDDFT.TDDFT_iter_preconditioner import TDDFT_iter_preconditioner
from dpolar.dpolar_solver import dpolar_solver

def main():
    ip_dict = gen_ip_func(diag_i = TDDFT_diag_preconditioner,
                          iter_i = TDDFT_iter_preconditioner,
                          diag_p = TDDFT_diag_preconditioner,
                          iter_p = TDDFT_iter_preconditioner)

    for option in args.ip_options:
        initial_guess, preconditioner = ip_dict[option]
        (alpha_omega, Davidson_dict) = dpolar_solver(
                                  initial_guess = initial_guess,
                                 preconditioner = preconditioner,
                                       conv_tol = args.conv_tolerance,
                                            max = args.max,
                                    initial_TOL = args.initial_TOL,
                                    precond_TOL = args.precond_TOL,
                                   dpolar_omega = args.dpolar_omega)
        dump_yaml(Davidson_dict, option)
if __name__ == '__main__':
    main()
