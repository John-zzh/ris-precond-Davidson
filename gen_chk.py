import time
import pyscf
import tempfile
from pyscf import gto, scf, dft, tddft, data, lib
import argparse
import os
import psutil
import yaml
from pyscf.tools import molden
from pyscf.dft import xcfun

# python PySCF_scf.py -x methanol.xyz -f pbe0 -b def2-TZVP

print('curpath', os.getcwd())
print('lib.num_threads() = ', lib.num_threads())

parser = argparse.ArgumentParser(description='Davidson')
parser.add_argument('-x', '--xyzfile',                      type=str,   default='NA', help='xyz filename (molecule.xyz)')
# parser.add_argument('-chk', '--checkfile',                  type=str,   default='NA', help='checkpoint filename (.chk)')
parser.add_argument('-m', '--method',                       type=str,   default='RKS', help='RHF RKS UHF UKS')
parser.add_argument('-f', '--functional',                   type=str,   default='NA',  help='xc functional')
parser.add_argument('-b', '--basis_set',                    type=str,   default='NA',  help='basis set')
parser.add_argument('-df', '--density_fit',                 type=bool,  default=True, help='density fitting turn on')
parser.add_argument('-g', '--grid_level',                   type=int,   default=3,   help='0-9, 9 is best')
parser.add_argument('-c', '--charge',                       type=int,   default=0,   help='molecular net charge')

parser.add_argument('-M',  '--memory',                      type=int,   default= 4000, help='max_memory')
parser.add_argument('-v',  '--verbose',                     type=int,   default= 5,    help='mol.verbose = 3,4,5')

args = parser.parse_args()
################################################

print(args)

########################################################
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024 / 1024
    print('{} memory used: {} MB'.format(hint, memory))
########################################################

info = psutil.virtual_memory()
print(info)

show_memory_info('at beginning')

# read xyz file and delete its first two lines
basename = args.xyzfile.split('.',1)[0]

f = open(args.xyzfile)
atom_coordinates = f.readlines()
del atom_coordinates[:2]
###########################################################################
# build geometry in PySCF
mol = gto.Mole()
mol.charge = args.charge
mol.atom = atom_coordinates
mol.basis = args.basis_set
mol.verbose = args.verbose
mol.max_memory = args.memory
print('mol.max_memory', mol.max_memory)
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
    print('RKS')
    mf.xc = args.functional
    mf.grids.level = args.grid_level
    # 0-9, big number for large mesh grids, default is 3
else:
    print('HF')

if args.density_fit:
    mf = mf.density_fit()
    print('Density fitting turned on')

# if args.checkfile != 'NA':
#     mf.chkfile = args.checkfile
#     mf.init_guess = 'chkfile'

mf.conv_tol = 1e-10

#
# Save the density matrix as the initial guess for the next calculation
#

mf.chkfile = basename + '_' + args.functional + '.chk'

# mf.init_guess = 'chkfile' # to use the chk file as input


print ('Molecule built')
print ('Calculating SCF Energy...')
kernel_0 = time.time()
mf.kernel()
kernel_1 = time.time()
kernel_t = kernel_1 - kernel_0
print ('SCF Done after ', round(kernel_t, 4), 'seconds')

show_memory_info('after SCF')
