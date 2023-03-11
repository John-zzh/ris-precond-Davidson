import time
import pyscf
import tempfile
from pyscf import lib, gto, scf, dft, tddft, grad
import argparse
import os
import psutil

import numpy

def str2bool(str):
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

print('curpath', os.getcwd())
print('lib.num_threads() = ', lib.num_threads())

parser = argparse.ArgumentParser(description='Davidson')
parser.add_argument('-x', '--xyzfile',                      type=str,   default='NA', help='xyz filename (molecule.xyz)')
parser.add_argument('-chk', '--checkfile',                  type=str2bool,  default=True, help='checkpoint filename (.chk)')
parser.add_argument('-m', '--method',                       type=str,   default='RKS', help='RHF RKS UHF UKS')
parser.add_argument('-f', '--functional',                   type=str,   default=None,  help='xc functional')
parser.add_argument('-b', '--basis_set',                    type=str,   default='def2-TZVP',  help='basis set')
parser.add_argument('-df', '--density_fit',                 type=str2bool,  default='True', help='density fitting turn on')
parser.add_argument('-g', '--grid_level',                   type=int,   default=3,   help='0-9, 9 is best')

parser.add_argument('-M',  '--memory',                      type=int,   default= 4000, help='max_memory')
parser.add_argument('-v',  '--verbose',                     type=int,   default= 5,    help='mol.verbose = 3,4,5')
parser.add_argument('-grad',  '--grad',                     type=str2bool,   default= 'False', help='perform S1 grad calculation')

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

with open(args.xyzfile) as f:
    coordinate = f.read().splitlines()[2:]
    atom_coordinates = [i for i in coordinate if i != '']
###########################################################################
# build geometry in PySCF
mol = gto.Mole()
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

if args.checkfile:
    mf.chkfile = basename + '_' + args.functional + '.chk'
    mf.init_guess = 'chkfile'

mf.conv_tol = 1e-10

#
# Save the density matrix as the initial guess for the next calculation
#



# mf.init_guess = 'chkfile' # to use the chk file as input


print ('Molecule built')
print ('Calculating SCF Energy...')
kernel_0 = time.time()
mf.kernel()
kernel_1 = time.time()
scf_time = kernel_1 - kernel_0
print ('SCF Done after ', round(scf_time, 4), 'seconds')

show_memory_info('after SCF')

if args.grad:
    TD_start = time.time()
    td = tddft.TDA(mf)
    td.nstates = 3
    e, z = td.kernel()
    # z = numpy.array(z)
    X = z[0][0]
    X = numpy.array(X)
    print('X.shape',X.shape)
    print('e.shape',' z.shape')
    print(len(e), type(z[0][0]))
    # print(z)

    TD_end = time.time()
    TD_time = TD_end - TD_start

    # z = grad_elec(mf, (X, numpy.zeros_like(X)))
    # attrs = vars(td)
    # for key in attrs:
    #     print(key, attrs[key])
    # tdg = td.nuc_grad_method()
    TDG_start = time.time()
    tdg = td.Gradients()
    tdg.cphf_max_cycle=50
    tdg.cphf_conv_tol=1e-8
    g1 = tdg.kernel(state=1)
    TDG_end = time.time()
    TDG_time = TDG_end - TDG_start
    print(g1)

    print('scf_time = {:.2f}'.format(scf_time))
    print('TD_time = {:.2f}'.format(TD_time))
    print('TDG_time = {:.2f}'.format(TDG_time))
