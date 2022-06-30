# -*- coding: utf-8 -*-

import os
import numpy as np

from SCF_calc import gen_P, basename
from mathlib import parameter

from arguments import args

from TDA.TDA_iter_initial_guess import TDA_iter_initial_guess
from TDDFT.TDDFT_iter_initial_guess import TDDFT_iter_initial_guess_solver

P = gen_P()
P = P.reshape(-1,3)

def gen_spectra(energies, transition_vector, name):
    '''
    E = hν
    c = λ·ν
    E = hc/λ = hck   k in cm-1

    energy in unit eV
    1240.7011/ xyz eV = xyz nm

    for TDA,   f = 2/3 E |<P|X>|**2     ???
    for TDDFT, f = 2/3 E |<P|X+Y>|

    '''
    energies = energies.reshape(-1,)

    eV = energies.copy()
    # print(energies, energies.shape)
    cm_1 = eV*8065.544
    nm = 1240.7011/eV

    '''
    P is right-hand-side of polarizability
    transition_vector is eigenvector of A matrix
    '''

    hartree = energies/parameter.Hartree_to_eV
    trans_dipole = np.dot(P.T, transition_vector)

    trans_dipole = 2*trans_dipole**2
    '''
    2* because alpha and beta spin
    '''
    oscillator_strength = 2/3 * hartree * np.sum(trans_dipole, axis=0)

    '''
    eV, oscillator_strength, cm_1, nm
    '''
    entry = [eV, oscillator_strength, cm_1, nm]
    data = np.zeros((eV.shape[0],len(entry)))
    for i in range(4):
        data[:,i] = entry[i]

    filename = name + '_spectra.txt'
    with open(filename, 'w') as f:
        np.savetxt(f, data)
    print('spectra written to', filename, '\n')

def main():


    from sTDA import sTDA_mv_lib
    from TDDFT_as import TDDFT_as_lib
    '''
    generate spectra peaks for sTDA and TDA-as
    '''

    a = ['sTDA', 'as']
    b = [sTDA_mv_lib.sTDA(), TDDFT_as_lib.TDDFT_as()]

    for i in range(len(a)):
        print('computing', a[i], 'spectra')
        approx_TDA_mv, approx_TDDFT_mv, approx_spolar_mv = b[i].build()

        U, omega = TDA_iter_initial_guess(N_states = args.nstates,
                                          conv_tol = args.conv_tolerance,
                             matrix_vector_product = approx_TDA_mv)
        gen_spectra(energies = omega,
           transition_vector = U,
                        name = a[i]+'_TDA')

        energies, X, Y = TDDFT_iter_initial_guess_solver(
                                      N_states = args.nstates,
                                      conv_tol = args.conv_tolerance,
                         matrix_vector_product = approx_TDDFT_mv)

        gen_spectra(energies = energies,
            transition_vector = X + Y,
                         name = a[i]+'_TDDFT')




#
# if __name__ == '__main__':
#     main()
