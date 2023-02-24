# -*- coding: utf-8 -*-

'''
absolute import
'''
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)


import spectra
from TDA.TDA_solver import TDA_solver
from TDA.TDA_iter_initial_guess import TDA_iter_initial_guess
from TDA.TDA_iter_preconditioner import TDA_iter_preconditioner
from TDA.Jacobi_preconditioner import Jacobi_preconditioner
from TDA.projector_preconditioner import projector_preconditioner

from arguments import args
from mathlib.diag_ip import TDA_diag_initial_guess, TDA_diag_preconditioner
from SCF_calc import gen_ip_func, ip_name, calc_name
from approx_mv import approx_TDA_mv
from dump_yaml import dump_yaml


'''
wb97x  methanol, 1e-5
    TDA          [7.60466912 9.60759452 9.65620573 10.54964748 10.84658266]
sTDDFT no truncate [6.46636611 8.18031534 8.38140651 9.45011415 9.5061059 ]
        40 eV    [6.46746642 8.18218267 8.38314651 9.45214869 9.5126739 ]
sTDA no truncate [6.46739711 8.18182208 8.38358473 9.45195554 9.52133129]
        40 eV    [6.46827111 8.18334703 8.38483801 9.45361525 9.52562255]

PBE0  methanol, 1e-5
sTDA no trunc [7.03457351 8.57113829 8.85193968 9.77450474 9.89962781]
TDA           [7.2875264  8.93645089 9.18027002 9.92054961 10.16937337]
'''
def main():
    TDA_ip_dict = gen_ip_func(diag_i = TDA_diag_initial_guess,
                              iter_i = TDA_iter_initial_guess,
                              diag_p = TDA_diag_preconditioner,
                              iter_p = TDA_iter_preconditioner)
    if args.jacobi:
        TDA_ip_dict[0] = (TDA_iter_initial_guess, Jacobi_preconditioner)
    if args.projector:
        TDA_ip_dict[0] = (TDA_iter_initial_guess, projector_preconditioner)

    for option in args.ip_options:
        initial_guess, preconditioner = TDA_ip_dict[option]
        (Excitation_energies,
        transition_vector_X,
        Davidson_dict,
        AV) = TDA_solver(N_states = args.nstates,
                    initial_guess = initial_guess,
                   preconditioner = preconditioner,
                         conv_tol = args.conv_tolerance,
                     extrainitial = args.extrainitial,
                              max = args.max)

        dump_yaml(Davidson_dict, option)

    '''
    spectra comparation
    '''
    spectra.gen_spectra(energies = Excitation_energies,
               transition_vector = transition_vector_X,
                            name = 'TDA')

if __name__ == '__main__':
    main()
        # I_AAV = gen_forecaster(eta, sub_eigenvalue, full_guess)
        # print('I_AAV =', I_AAV)
        # Davidson_dict['preconditioner forecaster'] = I_AAV
