import time
import numpy as np
from opt_einsum import contract as einsum
import pyscf
from pyscf import gto, scf, dft, tddft, data, lib
import argparse
import os
import yaml
from pyscf.tools import molden

print ('lib.num_threads() = ', lib.num_threads())

parser = argparse.ArgumentParser(description='Davidson')
parser.add_argument('-x', '--xyzfile',        type=str, default='NA', help='xyz filename (molecule.xyz)')
parser.add_argument('-m', '--method',         type=str, default='NA', help='RHF RKS UHF UKS')
parser.add_argument('-f', '--functional',     type=str, default='NA', help='xc functional')
parser.add_argument('-b', '--basis_set',      type=str, default='NA', help='basis sets')
parser.add_argument('-df', '--density_fit',   type=bool, default=False, help='density fitting turn on')
parser.add_argument('-g', '--grid_level',     type=int, default='3', help='0-9, 9 is best')
parser.add_argument('-t', '--tolerance',      type=float, default= 1e-5, help='residual norm convergence threshold')
parser.add_argument('-n', '--nstates',        type=int, default= 4, help='number of excited states')
parser.add_argument('-C', '--compare',        type=bool, default = False , help='whether to compare with PySCF TDA-TDDFT')
args = parser.parse_args()
################################################
# read xyz file and delete its first two lines
basename = args.xyzfile.split('.',1)[0]
f = open(args.xyzfile)
atom_coordinates = f.readlines()
del atom_coordinates[:2]
###########################################################################
# build geometry in PySCF
mol = gto.Mole()
mol.atom = atom_coordinates
mol.basis = args.basis_set
mol.verbose = 5
mol.max_memory = 16000
mol.build(parse_arg = False)
###########################################################################
###################################################
#DFT or HF?
if args.method == 'RKS':
    mf = dft.RKS(mol)
elif args.method == 'UKS':
    mf = dft.UKS(mol)
elif args.method == 'RHF':
    mf = scf.RHF(mol)
elif args.method == 'UHF':
    mf = scf.UHF(mol)

if 'KS' in args.method:
    mf.xc = args.functional
    mf.grids.level = args.grid_level
    # 0-9, big number for large mesh grids, default is 3

if args.density_fit:
    mf = mf.density_fit()




mf.conv_tol = 1e-10
mf.chkfile = basename + '.chk'
print('Molecule built')
print('Calculating SCF Energy...')

kernel_0 = time.time()
mf.kernel()
kernel_1 = time.time()
kernel = round (kernel_1 - kernel_0, 4)

print('SCF Done after ', kernel, 'seconds')



# if args.compare == True:
#     print ('-----------------------------------------------------------------')
#     print ('|--------------------    PySCF TDA-TDDFT    ---------------------|')
#     td.nstates = args.nstates
#     td.conv_tol = 1e-10
#     td.verbose = 5
#     start = time.time()
#     td.kernel()
#     end = time.time()
#     pyscf_time = end-start
#     print ('Built-in Davidson time:', round(pyscf_time, 4), 'seconds')
#     print ('|---------------------------------------------------------------|')
