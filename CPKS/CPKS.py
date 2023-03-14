# -*- coding: utf-8 -*-

'''
absolute import
'''
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import numpy as np
from arguments import args
from mathlib.diag_ip import spolar_diag_initprec
from SCF_calc import gen_ip_func, ip_name, calc_name, n_occ, n_vir, static_polarizability_matrix_vector, static_polarizability_matrix_vector2
from dump_yaml import dump_yaml

from spolar.spolar_iter_initprec import spolar_iter_initprec
from spolar.spolar_solver import spolar_solver

def main():
    ip_dict = gen_ip_func(diag_i = spolar_diag_initprec,
                          iter_i = spolar_iter_initprec,
                          diag_p = spolar_diag_initprec,
                          iter_p = spolar_iter_initprec)

    '''wvo.txt is the actually the negative RHS, but P is also negative, just trust the sign'''

    for option in args.ip_options:
        wvo = np.loadtxt('wvo.txt')
        print('wvo.shape', wvo.shape)
        wvo = -wvo.T.reshape(-1,1)
        initial_guess, preconditioner = ip_dict[option]
        (tensor_alpha, X_full, Davidson_dict) = spolar_solver(
                                initial_guess = initial_guess,
                               preconditioner = preconditioner,
                        matrix_vector_product = static_polarizability_matrix_vector,
                                     conv_tol = args.conv_tolerance,
                                          max = args.max,
                                          RHS = wvo,
                                  calc_name = 'CPKS')
        toy_z = X_full.reshape(n_occ, n_vir).T
        np.savetxt('X_full.txt', toy_z)
        dump_yaml(Davidson_dict, option)

        if args.cpks_test:
            standard_z = np.loadtxt('z1.txt')
            print('compare to pyscf difference = ', np.linalg.norm(toy_z-standard_z))

if __name__ == '__main__':
    main()
