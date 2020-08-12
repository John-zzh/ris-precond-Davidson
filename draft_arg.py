import time
import numpy as np
import pyscf
from pyscf import gto, scf, dft, tddft, data
import argparse

parser = argparse.ArgumentParser(description='methods')

parser.add_argument('-c', '--filename',       type=str, default='methanol.xyz', help='input filename')    # coordinates from input file
parser.add_argument('-m', '--method',         type=str, default='RHF', help='method')                  # RKS, RHF, UKS, UHF
parser.add_argument('-f', '--functional',     type=str, default='b3lyp', help='exchange-correlation functional')
parser.add_argument('-b', '--basis_set',      type=str, default='def2-SVP', help='basis sets')
parser.add_argument('-i', '--initial_guess',  type=str, default='diag_A', help='initial_guess: diag_A or sTDA_A')
parser.add_argument('-p', '--preconditioner', type=str, default='diag_A', help='preconditioner: diag_A or sTDA_A')
args = parser.parse_args()


#print (args.filename)

# f = open(args.filename)
# coordinates = f.readlines()

print (args.filename)

# pyscf.gto.mole.fromfile('methanol.xyz',  format='xyz')



# del coordinates[:2]
#
# ###########################################################################
# mol = gto.Mole()
# mol.build(atom = coordinates, basis = args.basis_set, symmetry = True)
# ###########################################################################
