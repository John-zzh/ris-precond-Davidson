import time
import numpy as np
import scipy
from opt_einsum import contract as einsum
#from einsum2 import einsum2 as einsum
import pyscf
from pyscf import gto, scf, dft, tddft, data, lib
import argparse
import os
import psutil
import yaml

# from pyscf.tools import molden
# from pyscf.dft import xcfun

# wb97x methanol, sTDA[ 6.46739329  8.18181719  8.38358186  9.45195086  9.52132657  9.98634407
#  10.57628203 11.22108803 11.23872459 11.57456576]

# wb97x methanol, sTDDFT [6.46636223 8.18031038 8.38140358 9.45010938 9.50610109]
# truncate 40 eV         [6.46748128 8.18219018 8.38315179 9.45216734 9.51268853]
#
print('curpath', os.getcwd())
print('lib.num_threads() = ', lib.num_threads())

parser = argparse.ArgumentParser(description='Davidson')
parser.add_argument('-x', '--xyzfile',                      type=str,   default='NA',  help='xyz filename (molecule.xyz)')
parser.add_argument('-chk', '--checkfile',                  type=bool,  default=False, help='checkpoint filename (.chk)')
parser.add_argument('-m', '--method',                       type=str,   default='RKS', help='RHF RKS UHF UKS')
parser.add_argument('-f', '--functional',                   type=str,   default='NA',  help='xc functional')
parser.add_argument('-b', '--basis_set',                    type=str,   default='NA',  help='basis set')
parser.add_argument('-df', '--density_fit',                 type=bool,  default=True,  help='density fitting turn on')
parser.add_argument('-g', '--grid_level',                   type=int,   default='3',   help='0-9, 9 is best')

parser.add_argument('-n',  '--nstates',                     type=int,   default = 4,      help='number of excited states')
parser.add_argument('-pytd',  '--pytd',                     type=bool,  default = False , help='whether to compare with PySCF TDDFT')

parser.add_argument('-TDA','--TDA',                         type=bool,  default = False, help='perform TDA')
parser.add_argument('-TDDFT','--TDDFT',                     type=bool,  default = False, help='perform TDDFT')
parser.add_argument('-dynpol','--dynpol',                   type=bool,  default = False, help='perform dynamic polarizability')
parser.add_argument('-omega','--dynpol_omega',              type=float, default = [], nargs='+', help='dynamic polarizability with perurbation omega, a list')
parser.add_argument('-stapol','--stapol',                   type=bool,  default = False, help='perform static polarizability')

parser.add_argument('-sTDA','--sTDA',                       type=bool,  default = False, help='perform sTDA calculation')
parser.add_argument('-sTDDFT','--sTDDFT',                   type=bool,  default = False, help='perform sTDDFT calculation')
parser.add_argument('-TT','--Truncate_test',                type=bool,  default = False, help='test the wall time for different virtual truncation')

parser.add_argument('-TV','--truncate_virtual',             type=float, default = 0, help='the threshold to truncate virtual orbitals, in eV')

parser.add_argument('-o1', '--TDA_options',                 type=int,   default = [0], nargs='+', help='0-7')
parser.add_argument('-o2', '--TDDFT_options',               type=int,   default = [0], nargs='+', help='0-3')
parser.add_argument('-o3', '--dynpol_options',              type=int,   default = [0], nargs='+', help='0-3')
parser.add_argument('-o4', '--stapol_options',              type=int,   default = [0], nargs='+', help='0-3')

parser.add_argument('-t1',  '--TDA_tolerance',              type=float, default= 1e-5, help='residual norm conv for TDA')
parser.add_argument('-t2',  '--TDDFT_tolerance',            type=float, default= 1e-5, help='residual norm conv for TDDFT')
parser.add_argument('-t3',  '--dynpol_tolerance',           type=float, default= 1e-5, help='residual norm conv for dynamic polarizability')
parser.add_argument('-t4',  '--stapol_tolerance',           type=float, default= 1e-5, help='residual norm conv for static polarizability')

parser.add_argument('-max',  '--max',                       type=int,   default= 30,   help='max iterations')

parser.add_argument('-it1', '--TDA_initialTOL',             type=float, default= 1e-3, help='conv for sTDA inital guess')
parser.add_argument('-it2', '--TDDFT_initialTOL',           type=float, default= 1e-3, help='conv for sTDDFT inital guess')
parser.add_argument('-it3', '--dynpol_initprecTOL',         type=float, default= 1e-2, help='conv for s-dynamic polarizability inital guess and precond')
parser.add_argument('-it4', '--stapol_initprecTOL',         type=float, default= 1e-2, help='conv for s-static polarizability inital guess and precond')

parser.add_argument('-pt1', '--TDA_precondTOL',             type=float, default= 1e-2, help='conv for TDA preconditioner')
parser.add_argument('-pt2', '--TDDFT_precondTOL',           type=float, default= 1e-2, help='conv for TDDFT preconditioner')

parser.add_argument('-ei1', '--TDA_extrainitial',           type=int,   default= 8,    help='number of extral TDA initial guess vectors, 0-8')
parser.add_argument('-ei2', '--TDDFT_extrainitial',         type=int,   default= 8,    help='number of extral TDDFT initial guess vectors, 0-8')
parser.add_argument('-ei3n','--TDDFT_extrainitial_3n',      type=bool,  default= False,help='number of extral TDDFT initial guess vectors, 3state')

parser.add_argument('-et', '--eigensolver_tol',             type=float, default= 1e-5, help='conv for new guess generator in new_ES')
parser.add_argument('-M',  '--memory',                      type=int,   default= 4000, help='max_memory')
parser.add_argument('-v',  '--verbose',                     type=int,   default= 5,    help='mol.verbose = 3,4,5')

parser.add_argument('-be', '--beta',                        type=float, default= [],    nargs='+', help='beta = 0.83')
parser.add_argument('-al', '--alpha',                       type=float, default= [],    nargs='+', help='alpha = 0.83')

args = parser.parse_args()
################################################


########################################################
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024 / 1024
    print('{} memory used: {} MB'.format(hint, memory))
########################################################

# info = psutil.virtual_memory()
# print(info)

show_memory_info('at beginning')

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

if args.checkfile == True:
    mf.chkfile = basename + '_' + args.functional + '.chk'
    mf.init_guess = 'chkfile'
    # to use the chk file as scf input


mf.conv_tol = 1e-10

print ('Molecule built')
print ('Calculating SCF Energy...')
kernel_0 = time.time()
mf.kernel()
kernel_1 = time.time()
kernel_t = kernel_1 - kernel_0
print ('SCF Done after ', round(kernel_t, 4), 'seconds')

show_memory_info('after SCF')

#################################################################
# Collect everything needed from PySCF
Qstart = time.time()
Natm = mol.natm
mo_occ = mf.mo_occ
n_occ = len(np.where(mo_occ > 0)[0])
#mf.mo_occ is an array of occupance [2,2,2,2,2,0,0,0,0.....]
n_vir = len(np.where(mo_occ == 0)[0])
#################################################
# generate matrix vector multiplication function
td = tddft.TDA(mf)
TDA_vind, hdiag = td.gen_vind(mf)

TD = tddft.TDDFT(mf)
TDDFT_vind, Hdiag = TD.gen_vind(mf)

A_size = n_occ * n_vir

# return AV
def TDA_matrix_vector(V):
    return TDA_vind(V.T).T

def TDDFT_matrix_vector(X, Y):
    '''return AX + BY and AY + BX'''
    XY = np.vstack((X,Y)).T
    U = TDDFT_vind(XY)
    U1 = U[:,:A_size].T
    U2 = -U[:,A_size:].T
    return U1, U2

def static_polarizability_matrix_vector(X):
    '''return (A+B)X '''
    U1, U2 = TDDFT_matrix_vector(X,X)
    return U1
#################################################

##################################################################################################
# create a function for dictionary of chemical hardness, by mappig two iteratable subject, list
# list of elements
elements = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca','Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr','Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn','Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd','Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb','Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu']
#list of chemical hardness, they are floats, containing elements 1-94, in Hartree
hardness = [0.47259288,0.92203391,0.17452888,0.25700733,0.33949086,0.42195412,0.50438193,0.58691863,0.66931351,0.75191607,0.17964105,0.22157276,0.26348578,0.30539645,0.34734014,0.38924725,0.43115670,0.47308269,0.17105469,0.20276244,0.21007322,0.21739647,0.22471039,0.23201501,0.23933969,0.24665638,0.25398255,0.26128863,0.26859476,0.27592565,0.30762999,0.33931580,0.37235985,0.40273549,0.43445776,0.46611708,0.15585079,0.18649324,0.19356210,0.20063311,0.20770522,0.21477254,0.22184614,0.22891872,0.23598621,0.24305612,0.25013018,0.25719937,0.28784780,0.31848673,0.34912431,0.37976593,0.41040808,0.44105777,0.05019332,0.06762570,0.08504445,0.10247736,0.11991105,0.13732772,0.15476297,0.17218265,0.18961288,0.20704760,0.22446752,0.24189645,0.25932503,0.27676094,0.29418231,0.31159587,0.32902274,0.34592298,0.36388048,0.38130586,0.39877476,0.41614298,0.43364510,0.45104014,0.46848986,0.48584550,0.12526730,0.14268677,0.16011615,0.17755889,0.19497557,0.21240778,0.07263525,0.09422158,0.09920295,0.10418621,0.14235633,0.16394294,0.18551941,0.22370139]
HARDNESS = dict(zip(elements,hardness))
#function to return chemical hardness from dictionary HARDNESS
def Hardness(atom_id):
    atom = mol.atom_pure_symbol(atom_id)
    return HARDNESS[atom]

################################################################################
# This block is the function to produce orthonormalized coefficient matrix C
def matrix_power(S,a):
    s,ket = np.linalg.eigh(S)
    s = s**a
    X = np.linalg.multi_dot([ket,np.diag(s),ket.T])
    #X == S^a
    return X

def matrix_power2(S):
    s,ket = np.linalg.eigh(S)
    s_sqrt = s**0.5
    s_inv = 1/s_sqrt

    S1 = np.dot(ket*s_sqrt,ket.T)
    S2 = np.dot(ket*s_inv,ket.T)

    #S1 == S^1/2
    #S2 == S^-1/2
    return S1, S2

def orthonormalize(C):
    X = matrix_power(mf.get_ovlp(), 0.5)
    # S = mf.get_ovlp() #.get_ovlp() is basis overlap matrix
    # S = np.dot(np.linalg.inv(c.T), np.linalg.inv(c))
    C = np.dot(X,C)
    return C

################################################################################
def gen_alpha_beta_ax():
    RSH_F = [
    'lc-b3lyp',
    'wb97',
    'wb97x',
    'wb97x-d3',
    'cam-b3lyp']
    RSH_paramt = [
    [0.53, 8.00, 4.50],
    [0.61, 8.00, 4.41],
    [0.56, 8.00, 4.58],
    [0.51, 8.00, 4.51],
    [0.38, 1.86, 0.90]]
    RSH_F_paramt = dict(zip(RSH_F, RSH_paramt))

    hybride_F = ['b3lyp', 'tpssh', 'm05-2x', 'pbe0', 'm06', 'm06-2x', 'NA']# NA is for Hartree-Fork
    hybride_paramt = [0.2, 0.1, 0.56, 0.25, 0.27, 0.54, 1]
    DF_ax = dict(zip(hybride_F, hybride_paramt))
    #Zhao, Y. and Truhlar, D.G., 2006. Density functional for spectroscopy: no long-range self-interaction error, good performance for Rydberg and charge-transfer states, and better performance on average than B3LYP for ground states. The Journal of Physical Chemistry A, 110(49), pp.13126-13130.

    if args.functional in RSH_F:
        a_x, beta, alpha = RSH_F_paramt[args.functional]

    elif args.functional in hybride_F:
        beta1 = 0.2
        beta2 = 1.83
        alpha1 = 1.42
        alpha2 = 0.48

        a_x = DF_ax[args.functional]
        beta = beta1 + beta2 * a_x
        alpha = alpha1 + alpha2 * a_x

    if args.beta != []:
        beta = args.beta[0]

    if args.alpha != []:
        alpha = args.alpha[0]

    print('a_x =', a_x)
    print('beta =', beta)
    print('alpha =', alpha)
    return a_x, beta, alpha

a_x, beta, alpha = gen_alpha_beta_ax()

# creat \eta matrix

def gen_gammaJK():
    Natm = mol.natm
    a = [Hardness(atom_id) for atom_id in range(Natm)]
    a = np.asarray(a).reshape(1,-1)
    eta = (a+a.T)/2

    #Inter-particle distance array
    # unit == ’Bohr’, Its value is 5.29177210903(80)×10^(−11) m
    R = pyscf.gto.mole.inter_distance(mol, coords=None)
    # creat GammaK and GammaK matrix
    GammaJ = (R**beta + (a_x * eta)**(-beta))**(-1/beta)
    GammaK = (R**alpha + eta**(-alpha)) **(-1/alpha)
    return GammaJ, GammaK

GammaJ, GammaK = gen_gammaJK()

def generateQ():
    N_bf = len(mo_occ)
    C = mf.mo_coeff
    # mf.mo_coeff is the coefficient matrix
    C = orthonormalize(C)
    # C is orthonormalized coefficient matrix, N_bf * N_bf
    # np.dot(C.T,C) is a an identity matrix
    aoslice = mol.aoslice_by_atom()
    q = np.zeros([Natm, N_bf, N_bf])
    #N_bf is number Atomic orbitals = n_occ+n_vir
    for atom_id in range(Natm):
        shst, shend, atstart, atend = aoslice[atom_id]
        q[atom_id,:, :] = np.dot(C[atstart:atend, :].T, C[atstart:atend, :])
    return q

q_tensors = generateQ()

''' if no truncation, max_vir = n_vir and n_occ + max_vir = N_bf '''
print('hdiag', hdiag.shape)
delta_diag_A = hdiag.reshape(n_occ, n_vir)

def gen_maxvir(tol_eV = args.truncate_virtual):
    if tol_eV != 0:
        homo_vir = delta_diag_A[-1,:].tolist()
        # print(delta_diag_A[-1,:]*27.211386245988)
        tol = tol_eV/27.211386245988
        for i in range(len(homo_vir)):
            if homo_vir[i] >= tol:
                max_vir = i
                break
            else:
                max_vir = n_vir
    else:
        max_vir = n_vir
    return max_vir

max_vir = gen_maxvir()
print('n_occ = ', n_occ)
print('n_vir = ', n_vir)
print('max_vir = ', max_vir)

A_reduced_size = n_occ * max_vir

def gen_QJK(max_vir=max_vir):

    q_ij = np.zeros((Natm, n_occ, n_occ))
    q_ij[:,:,:] = q_tensors[:,:n_occ,:n_occ]

    q_ab = np.zeros((Natm, max_vir, max_vir))
    q_ab[:,:,:] = q_tensors[:,n_occ:n_occ+max_vir,n_occ:n_occ+max_vir]

    q_ia = np.zeros((Natm, n_occ, max_vir))
    q_ia[:,:,:] = q_tensors[:,:n_occ,n_occ:n_occ+max_vir]

    GK_q_jb = einsum("Bjb,AB->Ajb", q_ia, GammaK)
    GJ_q_ab = einsum("Bab,AB->Aab", q_ab, GammaJ)


    return q_ij, q_ab, q_ia , GK_q_jb, GJ_q_ab

q_ij, q_ab, q_ia , GK_q_jb, GJ_q_ab = gen_QJK()
# pre-calculate and store the Q-Gamma rank 3 tensor
Qend = time.time()

Q_time = Qend - Qstart
print('Q-Gamma tensors building time =', round(Q_time, 4))
################################################################################

show_memory_info('after Q matrix')


################################################################################
# This block is to define on-the-fly two electron intergeral (pq|rs)
# A_iajb * v = delta_ia_ia*v + 2(ia|jb)*v - (ij|ab)*v

# iajb_v = einsum('Aia,Bjb,AB,jbm -> iam', q_ia, q_ia, GammaK, V)
# ijab_v = einsum('Aij,Bab,AB,jbm -> iam', q_ij, q_ab, GammaJ, V)

max_vir_hdiag = delta_diag_A[:,:max_vir]

def gen_iajb_ijab_ibja_delta_fly(max_vir=max_vir, \
                                q_ij=q_ij, \
                                q_ab=q_ab, \
                                q_ia=q_ia , \
                                GK_q_jb=GK_q_jb, \
                                GJ_q_ab=GJ_q_ab):

    def iajb_fly(V):
        start = time.time()
        GK_q_jb_V = einsum("Ajb,jbm->Am", GK_q_jb, V)
        iajb_V = einsum("Aia,Am->iam", q_ia, GK_q_jb_V)
        end = time.time()
        # print('iajb =', round(end - start,4))
        return iajb_V


    def ijab_fly(V):
        start = time.time()
        GJ_q_ab_V = einsum("Aab,jbm->Ajam", GJ_q_ab, V)
        ijab_V = einsum("Aij,Ajam->iam", q_ij, GJ_q_ab_V)
        end = time.time()
        # print('ijab =', round(end - start,4))
        return ijab_V

    def ibja_fly(V):
        start = time.time()
        # the Forck exchange energy in B matrix
        ''' (ib|ja) '''
        # GK_q_jb_V = einsum("Aja,jbm->Abam", GK_q_jb, V)
        # ibja_V = einsum("Aib,Abam->iam", q_ia, GK_q_jb_V)
        # [ 6.46636595  8.18031516  8.38140638  9.45011397  9.50610571  9.979084 10.57322403 11.2055107  11.23419534 11.57308169]

        q_ib_V = einsum("Aib,jbm->Ajim", q_ia, V)
        ibja_V = einsum("Aja,Ajim->iam", GK_q_jb, q_ib_V)
        #[ 6.46636595  8.18031515  8.38140638  9.45011397  9.50610571  9.979084 10.57322403 11.2055107  11.23419534 11.57308169]
        end = time.time()
        # print('ibja =', round(end - start,4))
        return ibja_V

    def delta_fly(V):
        '''delta_diag_A.shape = (n_occ, n_vir)'''
        delta_v = einsum("ia,iam->iam", delta_diag_A, V)
        return delta_v

    def delta_max_vir_fly(V):
        '''delta_diag_A.shape = (n_occ, n_vir)'''
        delta_max_vir_v = einsum("ia,iam->iam", max_vir_hdiag, V)
        return delta_max_vir_v

    return iajb_fly, ijab_fly, ibja_fly, delta_fly, delta_max_vir_fly

iajb_fly, ijab_fly, ibja_fly, delta_fly, delta_max_vir_fly = \
                                                gen_iajb_ijab_ibja_delta_fly()

def gen_sTDA_sTDDFT_stapol_fly(max_vir=max_vir, \
                            iajb_fly = iajb_fly, \
                            ijab_fly = ijab_fly, \
                            ibja_fly = ibja_fly, \
                            delta_max_vir_fly = delta_max_vir_fly):

    def sTDA_mv(V):
        # sTDA_A * V
        V = V.reshape(n_occ, max_vir, -1)
        '''MV =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)'''
        MV = delta_max_vir_fly(V) + 2*iajb_fly(V) - ijab_fly(V)
        MV = MV.reshape(n_occ*max_vir,-1)
        return MV

    def sTDDFT_mv(X, Y):
        '''''return AX+BY and AY+BX'''''
        X = X.reshape(n_occ, max_vir,-1)
        Y = Y.reshape(n_occ, max_vir,-1)

        X_max_vir = X[:,:max_vir,:]
        Y_max_vir = Y[:,:max_vir,:]

        iajb_X = iajb_fly(X_max_vir)
        iajb_Y = iajb_fly(Y_max_vir)

        ijab_X = ijab_fly(X_max_vir)
        ijab_Y = ijab_fly(Y_max_vir)

        ibja_X = ibja_fly(X_max_vir)
        ibja_Y = ibja_fly(Y_max_vir)

        delta_X = delta_max_vir_fly(X_max_vir)
        delta_Y = delta_max_vir_fly(Y_max_vir)

        '''sTDA_A =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)'''
        '''sTDDFT_B = 2*iajb_fly(V) - a_x*ibja_fly(V)'''

        AX = delta_X + 2*iajb_X - ijab_X
        AY = delta_Y + 2*iajb_Y - ijab_Y

        BX = 2*iajb_X - a_x*ibja_X
        BY = 2*iajb_Y - a_x*ibja_Y

        U1 = np.zeros_like(X)
        U2 = np.zeros_like(X)

        U1[:,:max_vir,:] = AX + BY
        U2[:,:max_vir,:] = AY + BX

        U1 = U1.reshape(n_occ*max_vir,-1)
        U2 = U2.reshape(n_occ*max_vir,-1)

        return U1, U2

    def sTDDFT_stapol_mv(X):
        ''' return (A+B)X '''
        ''' sTDA_A = delta_fly(V) + 4*iajb_fly(V) \
                    - ijab_fly(V) - a_x*ibja_fly(V)'''
        X = X.reshape(n_occ, max_vir, -1)

        U = delta_max_vir_fly(X) + 4*iajb_fly(X) - ijab_fly(X) - a_x*ibja_fly(X)

        U = U.reshape(n_occ*max_vir, -1)
        return U

    return sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv

sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv = gen_sTDA_sTDDFT_stapol_fly()

################################################################################

################################################################################
# orthonormalization of guess_vectors
def Gram_Schmidt_bvec(A, bvec):
    ''' orthonormalize vector b against all A vectors'''
    ''' b = b - A*(A.T*b)'''
    if A.shape[1] != 0:
        # suppose A is orthonormalized
        projections_coeff = np.dot(A.T, bvec)
        bvec = bvec - np.dot(A, projections_coeff)
    return bvec

def VW_Gram_Schmidt(x, y, V, W):
    m = np.dot(V.T,x) + np.dot(W.T,y)
    n = np.dot(W.T,x) + np.dot(V.T,y)

    x_new = x - np.dot(V,m) - np.dot(W,n)
    y_new = y - np.dot(W,m) - np.dot(V,n)

    return x_new, y_new

def Gram_Schmidt_fill_holder(V, count, vecs):
    # V is a vectors holder
    # count is the amount of vectors that already sit in the holder
    nvec = np.shape(vecs)[1]
    # amount of new vectors intended to fill in the V
    # count will be final amount of vectors in V
    for j in range(nvec):
        vec = vecs[:, j]
        vec = Gram_Schmidt_bvec(V[:, :count], vec)   #single orthonormalize
        vec = Gram_Schmidt_bvec(V[:, :count], vec)   #double orthonormalize

        norm = np.linalg.norm(vec)
        if  norm > 1e-14:
            vec = vec/norm
            V[:, count] = vec
            count += 1
    new_count = count

    return V, new_count

def S_symmetry_orthogonal(x,y):
    S = np.zeros((2,2))
    S[0,0] = np.dot(x.T,x) + np.dot(y.T,y)
    S[0,1] = np.dot(x.T,y) + np.dot(y.T,x)
    S[1,0] = S[0,1]
    S[1,1] = S[0,0]

    ss = matrix_power(S, -0.5)
    a = ss[0,0]
    b = ss[0,1]
    # x = S^-1/2
    x_new = x*a + y*b
    y_new = y*a + x*b
    return x_new, y_new

def symmetrize(A):
    A = (A + A.T)/2
    return A

def anti_symmetrize(A):
    A = (A - A.T)/2
    return A


################################################################################
# define the orthonormality of a matrix A as the norm of (A.T*A - I)
def check_orthonormal(A):
    n = np.shape(A)[1]
    B = np.dot(A.T, A)
    c = np.linalg.norm(B - np.eye(n))
    return c
################################################################################

################################################################################
def VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new):
    # put X_new into V, and Y_new into W
    # m is the amount of vectors that already on V or W
    nvec = np.shape(X_new)[1]
    # amount of new vectors intended to fill in the V_holder and W_holder
    for j in range(0, nvec):

        V = V_holder[:,:m]
        W = W_holder[:,:m]

        x_tmp = X_new[:,j].reshape(-1,1)
        y_tmp = Y_new[:,j].reshape(-1,1)

        x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)
        x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)

        x_tmp,y_tmp = S_symmetry_orthogonal(x_tmp,y_tmp)
        # x_tmp,y_tmp = S_symmetry_orthogonal(x_tmp,y_tmp)

        # x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)

        xy_norm = (np.dot(x_tmp.T, x_tmp) +  np.dot(y_tmp.T, y_tmp))**0.5

        if  xy_norm > 1e-14:
            x_tmp = x_tmp/xy_norm
            y_tmp = y_tmp/xy_norm

            V_holder[:,m] = x_tmp[:,0]
            W_holder[:,m] = y_tmp[:,0]

            m += 1
        else:
            print('vector kicked out during GS orthonormalization')

    # print('check VW orthonormalization')
    # VW = np.vstack((V_holder[:,:m], W_holder[:,:m]))
    # WV = np.vstack((W_holder[:,:m], V_holder[:,:m]))
    # VWWV = np.hstack((VW,WV))
    # print('check_orthonormal VWWV:',check_orthonormal(VWWV))

    return V_holder, W_holder, m
################################################################################



################################################################################
def solve_AX_Xla_B(A, omega, Q):
    ''' AX - XΩ  = Q '''

    Qnorm = np.linalg.norm(Q, axis=0, keepdims = True)
    Q /= Qnorm
    N_vectors = len(omega)
    a, u = np.linalg.eigh(A)
    ub = np.dot(u.T, Q)
    ux = np.zeros_like(Q)
    for k in range(N_vectors):
        ux[:, k] = ub[:, k]/(a - omega[k])
    X = np.dot(u, ux)

    X *= Qnorm
    return X
################################################################################
def TDA_A_diag_initial_guess(m, V, hdiag = hdiag):
    # m is the amount of initial guesses
    hdiag = hdiag.reshape(-1,)
    Dsort = hdiag.argsort()
    for j in range(m):
        V[Dsort[j], j] = 1.0
    return V

def sTDA_initial_guess(m, V):
    sTDA_A_eigenkets = Davidson0(m)
    print('sTDA_A_eigenkets', sTDA_A_eigenkets.shape)
    #diagonalize sTDA_A amtrix
    V[:, :m] = sTDA_A_eigenkets[:,:m]
    return V
################################################################################

################################################################################
def TDA_A_diag_preconditioner(arg1, arg2, arg3=None, arg4=None, arg5=None, arg6=None, arg7=None, arg8=None, hdiag = hdiag):
    """ residual[:,index], sub_eigenvalue[:k][index], current_dic, full_guess[:,index], index, W[:,:m], V[:,:m], sub_A """
    # preconditioners for each corresponding residual
    residual = arg1
    sub_eigenvalue = arg2
    current_dic = arg3

    k = np.shape(residual)[1]
    t = 1e-14

    # print('hdiag in preconditioner', hdiag.reshape(-1,1).shape)
    D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_eigenvalue
    D = np.where( abs(D) < t, np.sign(D)*t, D) # force all values not in domain (-t, t)

    new_guess = residual/D

    return new_guess, current_dic
################################################################################


################################################################################
# sTDA preconditioner
def sTDA_preconditioner(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8):
    """ residual[:,index], sub_eigenvalue[:k][index], current_dic, full_guess[:,index], index, W[:,:m], V[:,:m], sub_A """
    """ (sTDA_A - λ*I)^-1 B = X """
    """ AX - Xλ = B """
    """ columns in B are residuals (in Davidson's loop) to be preconditioned, """
    residuals = arg1
    eigen_lambda = arg2
    current_dic = arg3

    precondition_start = time.time()

    # N_rows = np.shape(B)[0]
    A_reduced_size = n_occ*max_vir
    residuals = residuals.reshape(n_occ,n_vir,-1)
    B = residuals[:,:max_vir,:]


    B = B.reshape(A_reduced_size,-1)
    # B = B.reshape(A_reduced_size, -1)
    N_vectors = B.shape[1]

    #number of vectors to be preconditioned
    bnorm = np.linalg.norm(B, axis=0, keepdims = True)
    #norm of each vectors in B, shape (1,-1)
    B = B/bnorm

    start = time.time()
    tol = args.TDA_precondTOL # Convergence tolerance
    max = 30   # Maximum number of iterations

    V = np.zeros((A_reduced_size, (max+1)*N_vectors))
    W = np.zeros((A_reduced_size, (max+1)*N_vectors))
    count = 0

    # now V and W are empty holders, 0 vectors
    # W = sTDA_mv(V)
    # count is the amount of vectors that already sit in the holder
    # in each iteration, V and W will be filled/updated with new guess vectors

    ###########################################
    #initial guess: (diag(A) - λ)^-1 B.
    # D is preconditioner for each state
    t = 1e-10
    hdiag_m_lambda = np.repeat(hdiag.reshape(-1,1), N_vectors, axis=1) - eigen_lambda
    hdiag_m_lambda = np.where( abs(hdiag_m_lambda) < t, np.sign(hdiag_m_lambda)*t, hdiag_m_lambda) # <t: returns np.sign(D)*t; else: D

    hdiag_m_lambda = hdiag_m_lambda.reshape(n_occ, n_vir, -1)
    D = hdiag_m_lambda[:,:max_vir,:].reshape(A_reduced_size,-1)
    inv_D = 1/D

    # generate initial guess
    init = B*inv_D
    V, new_count = Gram_Schmidt_fill_holder(V, count, init)
    W[:, count:new_count] = sTDA_mv(V[:, count:new_count])
    count = new_count

    current_dic['preconditioning'] = []
    ####################################################################################
    for i in range(max):
        sub_B = np.dot(V[:,:count].T, B)
        sub_A = np.dot(V[:,:count].T, W[:,:count])
        #project sTDA_A matrix and vector B into subspace
        sub_A = symmetrize(sub_A)
        # size of subspace
        m = np.shape(sub_A)[0]

        sub_guess = solve_AX_Xla_B(sub_A, eigen_lambda, sub_B)

        full_guess = np.dot(V[:,:count], sub_guess)
        residual = np.dot(W[:,:count], sub_guess) - full_guess*eigen_lambda - B

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        current_dic['preconditioning'].append({'precondition residual norms': r_norms})

        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break

        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
        # preconditioning step
        # only generate new guess from unconverged residuals
        new_guess = residual[:,index]*inv_D[:,index]

        V, new_count = Gram_Schmidt_fill_holder(V, count, new_guess)
        W[:, count:new_count] = sTDA_mv(V[:, count:new_count])
        count = new_count

    precondition_end = time.time()
    precondition_time = precondition_end - precondition_start
    if i == (max -1):
        print('_________________ sTDA Preconditioner Failed Due to Iteration Limit _________________')
        print('sTDA preconditioning failed after ', i, 'steps; ', round(precondition_time, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', max_norm)
        print('orthonormality of V', check_orthonormal(V[:,:count]))
    else:
        print('sTDA Preconditioning Done after ', i, 'steps; ', \
                round(precondition_time, 4), 'seconds')

    full_guess *= bnorm

    U = np.zeros((n_occ,n_vir,N_vectors))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,-1)

    if max_vir < n_vir:
        ''' DX2 - X2*Omega = B2'''
        B2 = residuals[:,max_vir:,:]
        B2 = B2.reshape(n_occ*(n_vir-max_vir),-1)

        D2 = hdiag_m_lambda[:,max_vir:,:]
        D2 = D2.reshape(n_occ*(n_vir-max_vir),-1)
        X2 = (B2/D2).reshape(n_occ,n_vir-max_vir,-1)
        U[:,max_vir:,:] = X2

    U = U.reshape(A_size, -1)
    return U, current_dic
################################################################################

################################################################################
# K_inv # exacty the same function with sTDA_preconditioner, just no dic
# used in Jacobi_precodnitioner
def K_inv(B, eigen_lambda):
    # to solve K^(-1)y and K^(-1)u
    # K = A-λ*I
    # (sTDA_A - eigen_lambda*I)^-1 B = X
    # AX - Xλ = B
    # columns in B are residuals or current guess
    precondition_start = time.time()

    N_rows = np.shape(B)[0]
    B = B.reshape(N_rows, -1)
    N_vectors = np.shape(B)[1]

    #number of vectors to be preconditioned
    bnorm = np.linalg.norm(B, axis=0, keepdims = True)
    #norm of each vectors in B, shape (1,-1)
    B = B/bnorm

    start = time.time()
    tol = 1e-2    # Convergence tolerance
    max = 30   # Maximum number of iterations

    V = np.zeros((N_rows, (max+1)*N_vectors))
    W = np.zeros_like(V)
    count = 0

    # now V and W are empty holders, 0 vectors
    # W = sTDA_mv(V)
    # count is the amount of vectors that already sit in the holder
    # in each iteration, V and W will be filled/updated with new guess vectors
    ###########################################
    #initial guess: (diag(A) - λ)^-1 B.
    # D is preconditioner for each state
    t = 1e-10
    D = np.repeat(hdiag.reshape(-1,1), N_vectors, axis=1) - eigen_lambda
    D= np.where( abs(D) < t, np.sign(D)*t, D) # <t: returns np.sign(D)*t; else: D
    inv_D = 1/D

    # generate initial guess
    init = B*inv_D
    V, new_count = Gram_Schmidt_fill_holder(V, count, init)
    W[:, count:new_count] = sTDA_mv(V[:, count:new_count])
    count = new_count
    ############################################################################
    for i in range(0, max):
        sub_B = np.dot(V[:,:count].T, B)
        sub_A = np.dot(V[:,:count].T, W[:,:count])
        #project sTDA_A matrix and vector B into subspace
        # size of subspace
        m = np.shape(sub_A)[0]
        sub_guess = solve_AX_Xla_B(sub_A, eigen_lambda, sub_B)
        full_guess = np.dot(V[:,:count], sub_guess)
        residual = np.dot(W[:,:count], sub_guess) - full_guess*eigen_lambda - B
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)

        if max_norm < tol or i == (max-1):
            break

        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]

        # preconditioning step
        # only generate new guess from unconverged residuals
        new_guess = residual[:,index]*inv_D[:,index]

        V, new_count = Gram_Schmidt_fill_holder(V, count, new_guess)
        W[:, count:new_count] = sTDA_mv(V[:, count:new_count])
        count = new_count

    precondition_end = time.time()
    precondition_time = precondition_end - precondition_start
    if i == (max -1):
        print('_________________ K inverse Failed Due to Iteration Limit _________________')
        print('K inverse  failed after ', i, 'steps; ', round(precondition_time, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', max_norm)
        print('orthonormality of V', check_orthonormal(V[:,:count]))
    else:
        print('K inverse Done after ', i, 'steps; ', round(precondition_time, 4), 'seconds')
    return full_guess*bnorm
################################################################################

################################################################################
def Jacobi_preconditioner(arg1, arg2, arg3, arg4, arg5=None, arg6=None, arg7=None, arg8=None):
    """ residual[:,index], sub_eigenvalue[:k][index], current_dic, full_guess[:,index], index, W[:,:m], V[:,:m], sub_A """
    """    (1-uu*)(A-λ*I)(1-uu*)t = -B"""
    """    B is residual, we want to solve t """
    """    z approximates t """
    """    z = (A-λ*I)^(-1)*(-B) + α(A-λ*I)^(-1) * u"""
    """    where α = [u*(A-λ*I)^(-1)y]/[u*(A-λ*I)^(-1)u] """
    """    first is to solve (A-λ*I)^(-1)y and (A-λ*I)^(-1)u """

    B = arg1
    eigen_lambda = arg2
    current_dic = arg3
    current_guess = arg4

    u = current_guess
    K_inv_y = K_inv(-B, eigen_lambda)
    K_inv_u = K_inv(current_guess, eigen_lambda)
    n = np.multiply(u, K_inv_y).sum(axis=0)
    d = np.multiply(u, K_inv_u).sum(axis=0)
    Alpha = n/d

    z = K_inv_y -  Alpha*K_inv_u
    return z, current_dic
################################################################################

################################################################################
# original simple Davidson, to solve eigenvalues and eigenkets of sTDA_A matrix
def Davidson0(k):
    print('sTDA nstate =', k)
    sTDA_D_start = time.time()
    tol = args.TDA_initialTOL # Convergence tolerance
    max = 30
    #################################################
    # m is size of subspace
    m = min([k+8, 2*k, A_size])
    V = np.zeros((A_reduced_size, max*k + m))
    W = np.zeros_like(V)
    # positions of hdiag with lowest values set as 1

    V = TDA_A_diag_initial_guess(m, V, hdiag = max_vir_hdiag)
    W[:, :m] = sTDA_mv(V[:, :m])
    # create transformed guess vectors

    #generate initial guess and put in holders V and W
    ############################################################################
    for i in range(max):
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_A = symmetrize(sub_A)
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        # Diagonalize the subspace Hamiltonian, and sorted.
        #sub_eigenvalue[:k] are smallest k eigenvalues
        full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])
        residual = np.dot(W[:,:m], sub_eigenket[:,:k]) - full_guess * sub_eigenvalue[:k]

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break
        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
        ########################################
        # preconditioning step
        # only generate new guess from unconverged residuals
        new_guess, Y = TDA_A_diag_preconditioner(residual[:,index], sub_eigenvalue[:k][index], hdiag = max_vir_hdiag)
        # orthonormalize the new guesses against old guesses and put into V holder
        V, new_m = Gram_Schmidt_fill_holder(V, m, new_guess)
        W[:, m:new_m] = sTDA_mv(V[:, m:new_m])
        m = new_m
    ############################################################################


    sTDA_D_end = time.time()
    sTDA_D = sTDA_D_end - sTDA_D_start
    print('sTDA A diagonalization:','threshold =', tol, '; in', i, 'steps ', round(sTDA_D, 4), 'seconds' )
    print('sTDA excitation energies:')
    print(sub_eigenvalue[:k]*27.211386245988)

    # U = np.zeros((n_occ * n_vir, k))
    # U[:n_occ * max_vir,:] = full_guess

    U = np.zeros((n_occ,n_vir,k))
    U[:,:max_vir,:] = full_guess.reshape(n_occ,max_vir,k)
    U = U.reshape(A_size, k)
    return U
################################################################################


################################################################################
def sTDDFT_preconditioner_subspace_eigen_solver(a, b, sigma, pi, p, q, omega):
    ''' [ a b ] x - [ σ   π] x  Ω = p '''
    ''' [ b a ] y   [-π  -σ] y    = q '''

    ##############################
    ''' normalize the RHS '''
    pq = np.vstack((p,q))
    pqnorm = np.linalg.norm(pq, axis=0, keepdims = True)
    # print('pqnorm', pqnorm)
    p /= pqnorm
    q /= pqnorm
    ##############################
    d = abs(np.diag(sigma))
    # d is an one-dimension matrix
    d_mh = d**(-0.5)
    # d_h = d**0.5

    ''' LU = d^−1/2 (σ − π) d^−1/2 '''
    ''' A = PLU '''
    ''' P is identity matrix only when A is diagonally dominant '''
    s_m_p = d_mh.reshape(-1,1) * (sigma - pi) * d_mh.reshape(1,-1)
    P_permutation, L, U = scipy.linalg.lu(s_m_p)
    # print(np.diag(P_permutation))
    L = np.dot(P_permutation, L)

    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    p_p_q_tilde = np.dot(L_inv, d_mh.reshape(-1,1)*(p+q))
    p_m_q_tilde = np.dot(U_inv.T, d_mh.reshape(-1,1)*(p-q))

    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T '''
    GGT = np.linalg.multi_dot([U_inv.T, d_mh.reshape(-1,1)*(a-b)*d_mh.reshape(1,-1), U_inv])

    G = scipy.linalg.cholesky(GGT, lower=True) # lower triangle matrix
    G_inv = np.linalg.inv(G)

    ''' a ̃+ b ̃= L^−1 d^−1/2 (a+b) d^−1/2 L^−T '''
    ''' M = G^T (a ̃+ b ̃) G '''
    a_p_b_tilde = np.linalg.multi_dot([L_inv, d_mh.reshape(-1,1)*(a+b)*d_mh.reshape(1,-1), L_inv.T])
    M = np.linalg.multi_dot([G.T, a_p_b_tilde, G])
    T = np.dot(G.T, p_p_q_tilde) + np.dot(G_inv, p_m_q_tilde * omega.reshape(1,-1))

    Z = solve_AX_Xla_B(M, omega**2, T)

    ''' (x ̃+ y ̃) = GZ '''
    ''' x + y = d^-1/2 L^-T (x ̃+ y ̃) '''
    ''' x - y = d^-1/2 U^-1 (x ̃- y ̃) '''
    x_p_y_tilde = np.dot(G,Z)
    x_p_y = d_mh.reshape(-1,1) * np.dot(L_inv.T, x_p_y_tilde)
    # x_m_y = d_mh.reshape(-1,1) * np.linalg.multi_dot([U_inv, G_inv.T, Z])
    x_m_y_tilde = (np.dot(a_p_b_tilde, x_p_y_tilde) - p_p_q_tilde)/omega
    x_m_y = d_mh.reshape(-1,1) * np.dot(U_inv, x_m_y_tilde)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x

    x *= pqnorm
    y *= pqnorm

    return x, y
################################################################################

################################################################################
def Qx(V, x):
    """ Qx = (1 - V*V.T)*x = x - V*V.T*x  """
    return x - einsum('ij, jk, kl -> il', V, V.T, x)
################################################################################

################################################################################
def on_the_fly_Hx(W, V, sub_A, x):
    """ on-the-fly compute H'x """
    """ H′ ≡ W*V.T + V*W.T − V*a*V.T + Q*K*Q"""
    """ K approximates H, here K = sTDA_A"""
    """ H′ ≡ W*V.T + V*W.T − V*a*V.T + (1-V*V.T)sTDA_A(1-V*V.T)"""
    """ H′x ≡ a + b − c + d """
    a = einsum('ij, jk, kl -> il', W, V.T, x)
    b = einsum('ij, jk, kl -> il', V, W.T, x)
    c = einsum('ij, jk, kl, lm -> im', V, sub_A, V.T, x)
    d = Qx(V, sTDA_mv(Qx(V, x)))
    Hx = a + b - c + d
    return Hx
################################################################################

################################################################################
# to diagonalize the H'
def new_ES(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8):
    """ residual[:,index], sub_eigenvalue[:k][index], current_dic, full_guess[:,index], index, W[:,:m], V[:,:m], sub_A """
    current_dic = arg3
    return_index = arg5
    W_H = arg6
    V_H = arg7
    sub_A_H = arg8

    """ new eigenvalue solver """
    """ the traditional Davidson to diagonalize the H' matrix """
    """ W_H, V_H, sub_A_H are from the exact H """

    sTDA_D_start = time.time()
    tol = args.eigensolver_tol # Convergence tolerance
    max = 30
    #################################################
    # m is size of subspace
    k = args.nstates
    m = min([k+8, 2*k, A_size])
    # m is the amount of initial guesses
    V = np.zeros((A_size, max*k + m))
    W = np.zeros_like(V)
    # positions of hdiag with lowest values set as 1

    # sTDA as initial guess
    V = sTDA_initial_guess(m, V)

    W[:,:m] = on_the_fly_Hx(W_H, V_H, sub_A_H, V[:, :m])
    # create transformed guess vectors

    #generate initial guess and put in holders V and W
    ###########################################################################################
    for i in range(max):
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
#             print(sub_eigenvalue[:k]*27.211386245988)
        # Diagonalize the subspace Hamiltonian, and sorted.
        #sub_eigenvalue[:k] are smallest k eigenvalues

        residual = np.dot(W[:,:m], sub_eigenket[:,:k]) - np.dot(V[:,:m], sub_eigenket[:,:k] * sub_eigenvalue[:k])

        r_norms = np.linalg.norm(residual, axis=0).tolist()
#             print(r_norms)
        # largest residual norm
        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break
        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
        ########################################
        # preconditioning step
        # only generate new guess from unconverged residuals

        new_guess, Y = TDA_A_diag_preconditioner(residual[:,index], sub_eigenvalue[:k][index])
        # Y doesn't matter

        # orthonormalize the new guesses against old guesses and put into V holder
        V, new_m = Gram_Schmidt_fill_holder(V, m, new_guess)
#             print(check_orthonormal(V[:,:new_m]))
        W[:, m:new_m] = on_the_fly_Hx(W_H, V_H, sub_A_H, V[:, m:new_m])
        m = new_m
    ###########################################################################################
    full_guess = np.dot(V[:,:m], sub_eigenket[:,:k])

    sTDA_D_end = time.time()
    sTDA_D = sTDA_D_end - sTDA_D_start
    print('H_app diagonalization:','threshold =', tol, '; in', i, 'steps ', round(sTDA_D, 2), 'seconds' )
#         print('H_app', sub_eigenvalue[:k]*27.211386245988)

    return full_guess[:,return_index], current_dic
################################################################################

################################################################################
# a dictionary for TDA initial guess and precodnitioner
TDA_i_key = ['sTDA', 'Adiag']
TDA_i_func = [sTDA_initial_guess, TDA_A_diag_initial_guess]
TDA_i_lib = dict(zip(TDA_i_key, TDA_i_func))

TDA_p_key = ['sTDA', 'Adiag', 'Jacobi', 'new_ES']
TDA_p_func = [sTDA_preconditioner, TDA_A_diag_preconditioner, \
Jacobi_preconditioner, new_ES]
TDA_p_lib = dict(zip(TDA_p_key, TDA_p_func))
################################################################################

################################################################################
#Davidson frame, where we can choose different initial guess and preconditioner
def Davidson(k, tol, init,prec):
    D_start = time.time()
    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    initial_guess = TDA_i_lib[init]
    new_guess_generator = TDA_p_lib[prec]

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)
    print('A matrix size = ', A_size,'*', A_size)
    max = args.max
    # Maximum number of iterations

    m = min([k + args.TDA_extrainitial, 2*k, A_size])
    # amount of initila guess
    #################################################
    # generate initial guess

    V = np.zeros((A_size, max*k + m))
    W = np.zeros_like(V)

    init_start = time.time()
    V = initial_guess(m, V)
    init_end = time.time()
    init_time = init_end - init_start

    print('Intial guess time:', round(init_time, 4), 'seconds')
    #generate initial guess and put in holders V and W
    # m is size of subspace

    # W = Av, create transformed guess vectors
    W[:, :m] = TDA_matrix_vector(V[:, :m])

    # time cost for preconditioning
    Pcost = 0
    ###########################################################################################
    for ii in range(max):
        print('Iteration ', ii)

        # sub_A is subspace A matrix
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_A = symmetrize(sub_A)
        print('subspace size: ', np.shape(sub_A)[0])

        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        # print(sub_eigenvalue[:k])
        # Diagonalize the subspace Hamiltonian, and sorted.
        #sub_eigenvalue[:k] are smallest k eigenvalues
        full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])

        residual = np.dot(W[:,:m], sub_eigenket[:,:k]) - full_guess * sub_eigenvalue[:k]

        r_norms = np.linalg.norm(residual, axis=0).tolist()

        # largest residual norm
        max_norm = np.max(r_norms)

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms

        print('maximum residual norm', max_norm)
        if max_norm < tol or ii == (max-1):
            print('Davidson procedure aborted')
            break

        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]

        ########################################
        # generate new guess
        P_start = time.time()
        new_guess, current_dic = new_guess_generator(
                                    arg1 = residual[:,index],
                                    arg2 = sub_eigenvalue[:k][index],
                                    arg3 = current_dic,
                                    arg4 = full_guess[:,index],
                                    arg5 = index,
                                    arg6 = W[:,:m],
                                    arg7 = V[:,:m],
                                    arg8 = sub_A)
        P_end = time.time()

        iteration_list[ii] = current_dic

        Pcost += P_end - P_start

        # orthonormalize the new guesses against old guesses and put into V holder
        V, new_m = Gram_Schmidt_fill_holder(V, m, new_guess)
        print('m,new_m',m,new_m)
        W[:, m:new_m] = TDA_matrix_vector(V[:, m:new_m])
        print('new generated guesses:', new_m - m)
        m = new_m

    energies = sub_eigenvalue[:k]*27.211386245988


    D_end = time.time()
    Dcost = D_end - D_start
    Davidson_dic['initial guess'] = init
    Davidson_dic['preconditioner'] = prec
    Davidson_dic['nstate'] = k
    Davidson_dic['molecule'] = basename
    Davidson_dic['method'] = args.method
    Davidson_dic['functional'] = args.functional
    Davidson_dic['threshold'] = tol
    Davidson_dic['SCF time'] = kernel_t
    Davidson_dic['Initial guess time'] = init_time
    Davidson_dic['initial guess threshold'] = args.TDA_initialTOL
    Davidson_dic['New guess generating time'] = Pcost
    Davidson_dic['preconditioner threshold'] = args.TDA_precondTOL
    Davidson_dic['total time'] = Dcost
    Davidson_dic['iterations'] = ii+1
    Davidson_dic['A matrix size'] = A_size
    Davidson_dic['final subspace size'] = np.shape(sub_A)[0]
    Davidson_dic['excitation energy(eV)'] = energies.tolist()
    # Davidson_dic['semiempirical_difference'] = difference
    # Davidson_dic['overlap'] = overlap
    Davidson_dic['ax'] = a_x
    Davidson_dic['alpha'] = alpha
    Davidson_dic['beta'] = beta
    Davidson_dic['virtual truncation tol'] = args.truncate_virtual
    Davidson_dic['n_occ'] = n_occ
    Davidson_dic['n_vir'] = n_vir
    Davidson_dic['max_vir'] = max_vir
    ###########################################################################################
    if ii == max-1:
        print('============ Davidson Failed Due to Iteration Limit ==============')
        print('Davidson failed after ', round(Dcost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', max_norm)
    else:
        print('Davidson done after ', round(Dcost, 4), 'seconds')
        print('Total steps =', ii+1)
        print('Final subspace shape = ', np.shape(sub_A))
        print('Preconditioning time:', round(Pcost, 4), round(Pcost/Dcost*100,2), '%')
    return energies, full_guess, Davidson_dic
################################################################################



def TDDFT_A_diag_initial_guess(V_holder, W_holder, new_m, hdiag = hdiag):
    hdiag = hdiag.reshape(-1,)
    Dsort = hdiag.argsort()
    V_holder = TDA_A_diag_initial_guess(new_m, V_holder, hdiag = hdiag)
    energies = hdiag[Dsort][:new_m]*27.211386245988
    return V_holder, W_holder, new_m, energies, V_holder[:,:new_m], W_holder[:,:new_m]

################################################################################
def TDDFT_A_diag_preconditioner(R_x, R_y, omega, hdiag = hdiag):
    # preconditioners for each corresponding residual
    hdiag = hdiag.reshape(-1,1)
    # print('hdiag', hdiag.shape)
    k = R_x.shape[1]
#     print('omega.shape',omega.shape)
    t = 1e-14

    d = np.repeat(hdiag.reshape(-1,1), k, axis=1)

    D_x = d - omega
    D_x = np.where( abs(D_x) < t, np.sign(D_x)*t, D_x)
    D_x_inv = D_x**-1

    D_y = d + omega
    D_y = np.where( abs(D_y) < t, np.sign(D_y)*t, D_y)
    # force all values not in domain (-t, t)
    D_y_inv = D_y**-1

#     print('R_x.shape, D_x.shape',R_x.shape, D_x.shape)
    X_new = R_x*D_x_inv
    Y_new = R_y*D_y_inv

    return X_new, Y_new
################################################################################

################################################################################
def TDDFT_subspace_eigen_solver(a, b, sigma, pi, k):
    ''' [ a b ] x - [ σ   π] x  w = 0 '''
    ''' [ b a ] y   [-π  -σ] y    = 0 '''

    d = abs(np.diag(sigma))
    # print(d)
    # d is an one-dimension matrix
    d_mh = d**(-0.5)


    s_m_p = d_mh.reshape(-1,1) * (sigma - pi) * d_mh.reshape(1,-1)

    '''LU = d^−1/2 (σ − π) d^−1/2'''
    ''' A = PLU '''
    ''' if A is diagonally dominant, P is identity matrix (in fact not always) '''
    P_permutation, L, U = scipy.linalg.lu(s_m_p)
    # print(np.diag(P_permutation))

    L = np.dot(P_permutation, L)

    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T '''
    GGT = np.linalg.multi_dot([U_inv.T, d_mh.reshape(-1,1)*(a-b)*d_mh.reshape(1,-1), U_inv])

    G = scipy.linalg.cholesky(GGT, lower=True)
    # lower triangle matrix
    G_inv = np.linalg.inv(G)

    ''' M = G^T L^−1 d^−1/2 (a+b) d^−1/2 L^−T G '''
    M = np.linalg.multi_dot([G.T, L_inv, d_mh.reshape(-1,1)*(a+b)*d_mh.reshape(1,-1), L_inv.T, G])

    omega2, Z = np.linalg.eigh(M)
    omega = (omega2**0.5)[:k]
    # print(omega*27.211386245988)
    Z = Z[:,:k]

    ''' It requires Z^T Z = 1/Ω '''
    ''' x+y = d^−1/2 L^−T GZ Ω^-0.5 '''
    ''' x−y = d^−1/2 U^−1 G^−T Z Ω^0.5 '''

    x_p_y = d_mh.reshape(-1,1) * np.linalg.multi_dot([L_inv.T, G, Z])     * (np.array(omega)**-0.5).reshape(1,-1)
    x_m_y = d_mh.reshape(-1,1) * np.linalg.multi_dot([U_inv, G_inv.T, Z]) * (np.array(omega)**0.5).reshape(1,-1)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x

    # norm = np.linalg.multi_dot([x.T, sigma, x])
    # norm += np.linalg.multi_dot([x.T, pi, y])
    # norm -= np.linalg.multi_dot([y.T, pi, x])
    # norm -= np.linalg.multi_dot([y.T, sigma, y])
    # print(norm)

    return omega, x, y
################################################################################

################################################################################
def sTDDFT_eigen_solver(k):
    ''' [ A' B' ] x - [ σ   π] x  w = 0'''
    ''' [ B' A' ] y   [-π  -σ] y    = 0'''
    tol=args.TDDFT_initialTOL
    max = 30
    sTDDFT_start = time.time()
    print('setting initial guess')
    m = 0
    new_m = min([k+8, 2*k, A_reduced_size])
    # new_m = min([3*k, A_size])
    # print('initial new_m = ', new_m)
    V_holder = np.zeros((A_reduced_size, (max+1)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)
    # set up initial guess VW, transformed vectors U1&U2

    V_holder, W_holder, new_m, energies, Xig, Yig = \
    TDDFT_A_diag_initial_guess(V_holder, W_holder, new_m, hdiag = max_vir_hdiag)
#     print('initial guess done')
    ##############################

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        ###############################################################
        # print('ii = ', ii)
        # creating the subspace
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]

        # U1 = AV + BW
        # U2 = AW + BV

        # show_memory_info('before sTDDFT_mv')
        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = \
                                        sTDDFT_mv(V[:, m:new_m], W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start
        # show_memory_info('after sTDDFT_mv')

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1) + np.dot(W.T, U2)
        b = np.dot(V.T, U2) + np.dot(W.T, U1)
        sigma = np.dot(V.T, V) - np.dot(W.T, W)
        pi = np.dot(V.T, W) - np.dot(W.T, V)

        a = symmetrize(a)
        b = symmetrize(b)
        sigma = symmetrize(sigma)
        pi = anti_symmetrize(pi)

        subgenend = time.time()
        subgencost += subgenend - subgenstart
        ###############################################################

        ###############################################################
#         solve the eigenvalue omega in the subspace
        subcost_start = time.time()
        omega, x, y = TDDFT_subspace_eigen_solver(a, b, sigma, pi, k)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start
        ###############################################################
        # compute the residual
        # R_x = U1x + U2y - (Vx + Wy)omega
        # R_y = U2x + U1y + (Wx + Vy)omega
        # R_x = np.dot(U1,x) + np.dot(U2,y) - (np.dot(V,x) + np.dot(W,y))*omega
        # R_y = np.dot(U2,x) + np.dot(U1,y) + (np.dot(W,x) + np.dot(V,y))*omega

        U1x = np.dot(U1,x)
        U2x = np.dot(U2,x)
        Vx = np.dot(V,x)
        Wx = np.dot(W,x)

        U1y = np.dot(U1,y)
        U2y = np.dot(U2,y)
        Vy = np.dot(V,y)
        Wy = np.dot(W,y)

        X_full = Vx + Wy
        Y_full = Wx + Vy

        R_x = U1x + U2y - X_full*omega
        R_y = U2x + U1y + Y_full*omega

        residual = np.vstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
#         print('r_norms', r_norms)

        if np.max(r_norms) < tol or ii == (max -1):
            break
        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
#         print('index', index)
        ###############################################################

        ########################################################################
        # preconditioning step
        X_new, Y_new = TDDFT_A_diag_preconditioner(\
                R_x[:,index], R_y[:,index], omega[index], hdiag = max_vir_hdiag)
        ########################################################################

        ###############################################################
        # GS and symmetric orthonormalization
        m = new_m
        GScost_start = time.time()
        V_holder, W_holder, new_m = \
                VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new)
        GScost_end = time.time()
        GScost += GScost_end - GScost_start

        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break
        ###############################################################

    sTDDFT_end = time.time()

    sTDDFT_cost = sTDDFT_end - sTDDFT_start

    if ii == (max -1):
        print('========= sTD-DFT Failed Due to Iteration Limit==================')
        print('sTD-DFT failed after ', ii+1, 'iterations  ', round(sTDDFT_cost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('sTDDFT Converged after ', ii+1, 'iterations  ', round(sTDDFT_cost, 4), 'seconds')
        print('final subspace', sigma.shape)
        print('Pcost', round(Pcost,4), round(Pcost/sTDDFT_cost * 100,2),'%')
        print('MVcost', round(MVcost,4), round(MVcost/sTDDFT_cost * 100,2),'%')
        print('GScost', round(GScost,4), round(GScost/sTDDFT_cost * 100,2),'%')
        print('subcost', round(subcost,4), round(subcost/sTDDFT_cost * 100,2),'%')
        print('subgencost', round(subgencost,4), round(subgencost/sTDDFT_cost * 100,2),'%')

    X = np.zeros((n_occ,n_vir,k))
    Y = np.zeros((n_occ,n_vir,k))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,-1)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,-1)

    X = X.reshape(A_size, -1)
    Y = Y.reshape(A_size, -1)

    energies = omega*27.211386245988
    print('sTDDFT excitation energy:')
    print(energies)
    return energies, X, Y
################################################################################

################################################################################
def sTDDFT_initial_guess(V_holder, W_holder, new_m):
    energies, X_new_backup, Y_new_backup = sTDDFT_eigen_solver(new_m)
    # energies, X_new, Y_new = sTDDFT_eigen_solver(new_m)
    V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder\
                            (V_holder, W_holder, 0,  X_new_backup, Y_new_backup)
    return V_holder, W_holder, new_m, energies, X_new_backup, Y_new_backup
################################################################################

################################################################################
def sTDDFT_preconditioner(Rx, Ry, omega):
    ''' [ A B ] - [1  0]X  w = P'''
    ''' [ B A ]   [0 -1]Y    = Q'''
    ''' P = Rx '''
    ''' Q = Ry '''
    tol = args.TDDFT_precondTOL
    print('sTDDFT_preconditioner conv', tol)
    max = 30
    sTDDFT_start = time.time()
    k = len(omega)
    m = 0

    Rx = Rx.reshape(n_occ,n_vir,-1)
    Ry = Ry.reshape(n_occ,n_vir,-1)

    P = Rx[:,:max_vir,:].reshape(A_reduced_size,-1)
    Q = Ry[:,:max_vir,:].reshape(A_reduced_size,-1)

    initial_start = time.time()
    V_holder = np.zeros((A_reduced_size, (max+1)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    ##############################
    '''normalzie the RHS'''
    PQ = np.vstack((P,Q))
    pqnorm = np.linalg.norm(PQ, axis=0, keepdims = True)

    # print('pqnorm', pqnorm)
    P /= pqnorm
    Q /= pqnorm

    ##############################

    ##############################
    # setting up initial guess
    X_new, Y_new  = TDDFT_A_diag_preconditioner(P, Q, omega, hdiag = max_vir_hdiag)
    V_holder, W_holder, new_m = \
            VW_Gram_Schmidt_fill_holder(V_holder, W_holder, 0,  X_new, Y_new)
    initial_end = time.time()
    initial_cost = initial_end - initial_start
    print('new_m =', new_m)
    print('diagonal initial guess done')
    ##############################
    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
#         print('||||____________________Iteration', ii, '_________')
        ###############################################################
        # creating the subspace
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]

        # U1 = AV + BW
        # U2 = AW + BV

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = \
                                        sTDDFT_mv(V[:, m:new_m], W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1) + np.dot(W.T, U2)
        b = np.dot(V.T, U2) + np.dot(W.T, U1)
        sigma = np.dot(V.T, V) - np.dot(W.T, W)
        pi = np.dot(V.T, W) - np.dot(W.T, V)

        a = symmetrize(a)
        b = symmetrize(b)
        sigma = symmetrize(sigma)
        pi = anti_symmetrize(pi)

        # p = VP + WQ
        # q = WP + VQ
        p = np.dot(V.T, P) + np.dot(W.T, Q)
        q = np.dot(W.T, P) + np.dot(V.T, Q)

        subgenend = time.time()
        subgencost += subgenend - subgenstart
#         print('sigma.shape', sigma.shape)
        ###############################################################

        ###############################################################
#         solve the x & y in the subspace
        subcost_start = time.time()
        x, y = sTDDFT_preconditioner_subspace_eigen_solver(\
                                                  a, b, sigma, pi, p, q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start
        ###############################################################

        ###############################################################
        # compute the residual
        # R_x = U1x + U2y - (Vx + Wy)omega
        # R_y = U2x + U1y + (Wx + Vy)omega
        # R_x = np.dot(U1,x) + np.dot(U2,y) - (np.dot(V,x) + np.dot(W,y))*omega - P
        # R_y = np.dot(U2,x) + np.dot(U1,y) + (np.dot(W,x) + np.dot(V,y))*omega - Q

        U1x = np.dot(U1,x)
        U2x = np.dot(U2,x)
        Vx = np.dot(V,x)
        Wx = np.dot(W,x)

        U1y = np.dot(U1,y)
        U2y = np.dot(U2,y)
        Vy = np.dot(V,y)
        Wy = np.dot(W,y)

        X_full = Vx + Wy
        Y_full = Wx + Vy

        R_x = U1x + U2y - X_full*omega - P
        R_y = U2x + U1y + Y_full*omega - Q

        residual = np.vstack((R_x,R_y))
#         print('residual.shape', residual.shape)
        r_norms = np.linalg.norm(residual, axis=0).tolist()
#         print('r_norms', r_norms)
        if np.max(r_norms) < tol or ii == (max -1):
            break
        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
#         print('index', index)
        ########################################################################

        ########################################################################
        # preconditioning step
        Pstart = time.time()
        X_new, Y_new = TDDFT_A_diag_preconditioner(R_x[:,index], R_y[:,index], \
                                            omega[index], hdiag = max_vir_hdiag)
        Pend = time.time()
        Pcost += Pend - Pstart
        ########################################################################

        ########################################################################
        # GS and symmetric orthonormalization
        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(\
                                            V_holder, W_holder, m, X_new, Y_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
#         print('m & new_m', m, new_m)
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break
        ###############################################################

    sTDDFT_end = time.time()

    sTDDFT_precond_cost = sTDDFT_end - sTDDFT_start

    if ii == (max -1):
        print('========== sTDDFT_precond Failed Due to Iteration Limit==========')
        print('sTDDFT_precond failed after ', ii+1, 'iterations  ', round(sTDDFT_precond_cost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('sTDDFT_precond Converged after ', ii+1, 'iterations  ', round(sTDDFT_precond_cost, 4), 'seconds')
        print('initial_cost', round(initial_cost,4), round(initial_cost/sTDDFT_precond_cost * 100,2),'%')
        print('Pcost', round(Pcost,4), round(Pcost/sTDDFT_precond_cost * 100,2),'%')
        print('MVcost', round(MVcost,4), round(MVcost/sTDDFT_precond_cost * 100,2),'%')
        print('GScost', round(GScost,4), round(GScost/sTDDFT_precond_cost * 100,2),'%')
        print('subcost', round(subcost,4), round(subcost/sTDDFT_precond_cost * 100,2),'%')
        print('subgencost', round(subgencost,4), round(subgencost/sTDDFT_precond_cost * 100,2),'%')
        print('final subspace', sigma.shape)

    show_memory_info('preconditioner at end')
    X_full *=  pqnorm
    Y_full *=  pqnorm

    X = np.zeros((n_occ,n_vir,k))
    Y = np.zeros((n_occ,n_vir,k))

    X[:,:max_vir,:] = X_full.reshape(n_occ,max_vir,k)
    Y[:,:max_vir,:] = Y_full.reshape(n_occ,max_vir,k)

    if max_vir < n_vir:
        P2 = Rx[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)
        Q2 = Ry[:,max_vir:,:].reshape(n_occ*(n_vir-max_vir),-1)

        X2, Y2 = TDDFT_A_diag_preconditioner(\
                        P2, Q2, omega, hdiag = delta_diag_A[:,max_vir:])
        X[:,max_vir:,:] = X2.reshape(n_occ,n_vir-max_vir,-1)
        Y[:,max_vir:,:] = Y2.reshape(n_occ,n_vir-max_vir,-1)

    X = X.reshape(A_size,-1)
    Y = Y.reshape(A_size,-1)
    print(X.shape, Y.shape)
    return X, Y
################################################################################

################################################################################
# a dictionary for TDDFT initial guess and precodnitioner
TDDFT_i_key = ['sTDDFT', 'Adiag']
TDDFT_i_func = [sTDDFT_initial_guess, TDDFT_A_diag_initial_guess]
TDDFT_i_lib = dict(zip(TDDFT_i_key, TDDFT_i_func))

TDDFT_p_key = ['sTDDFT', 'Adiag']
TDDFT_p_func = [sTDDFT_preconditioner, TDDFT_A_diag_preconditioner]
TDDFT_p_lib = dict(zip(TDDFT_p_key, TDDFT_p_func))
################################################################################

################################################################################
def TDDFT_eigen_solver(init, prec, k=args.nstates, tol=args.TDDFT_tolerance):

    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)
    print('A matrix size = ', A_size,'*', A_size)

    TDDFT_start = time.time()
    max = args.max
    m = 0

    if args.TDDFT_extrainitial_3n == True:
        new_m = min([k + args.TDDFT_extrainitial, 3*k, A_size])
    else:
        new_m = min([k + args.TDDFT_extrainitial, 2*k, A_size])

    # new_m = k

    initial_guess = TDDFT_i_lib[init]
    new_guess_generator = TDDFT_p_lib[prec]

    V_holder = np.zeros((A_size, (max+3)*k))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)
    # set up initial guess VW, transformed vectors U1&U2

    ##############################
    # setting up initial guess
    init_start = time.time()
    V_holder, W_holder, new_m, initial_energies, X_ig, Y_ig = initial_guess(V_holder, W_holder, new_m)
    init_end = time.time()
    init_time = init_end - init_start

    initial_energies = initial_energies.tolist()[:k]

    print('new_m =', new_m)
    print('initial guess done')
    ##############################
    Pcost = 0
    VWGScost = 0
    for ii in range(max):
        print('||||____________________Iteration', ii, '____________________||||')
        show_memory_info('beginning of step '+ str(ii))
        ###############################################################
        # creating the subspace
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]
        # U1 = AV + BW
        # U2 = AW + BV
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = TDDFT_matrix_vector(V[:, m:new_m], W[:, m:new_m])

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        a = np.dot(V.T, U1) + np.dot(W.T, U2)
        b = np.dot(V.T, U2) + np.dot(W.T, U1)

        sigma = np.dot(V.T, V) - np.dot(W.T, W)
        pi = np.dot(V.T, W) - np.dot(W.T, V)

        a = symmetrize(a)
        b = symmetrize(b)
        sigma = symmetrize(sigma)
        pi = anti_symmetrize(pi)

        print('sigma.shape', sigma.shape)
        ###############################################################

        ###############################################################
#         solve the eigenvalue omega in the subspace
        omega, x, y = TDDFT_subspace_eigen_solver(a, b, sigma, pi, k)
        ###############################################################
        # compute the residual
        # R_x = U1x + U2y - (Vx + Wy)omega
        # R_y = U2x + U1y + (Wx + Vy)omega
        # R_x = np.dot(U1,x) + np.dot(U2,y) - (np.dot(V,x) + np.dot(W,y))*omega
        # R_y = np.dot(U2,x) + np.dot(U1,y) + (np.dot(W,x) + np.dot(V,y))*omega

        U1x = np.dot(U1,x)
        U2x = np.dot(U2,x)
        Vx = np.dot(V,x)
        Wx = np.dot(W,x)

        U1y = np.dot(U1,y)
        U2y = np.dot(U2,y)
        Vy = np.dot(V,y)
        Wy = np.dot(W,y)

        X_full = Vx + Wy
        Y_full = Wx + Vy

        R_x = U1x + U2y - X_full*omega
        R_y = U2x + U1y + Y_full*omega

        residual = np.vstack((R_x, R_y))

        r_norms = np.linalg.norm(residual, axis=0).tolist()

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        print('Maximum residual norm: ', np.max(r_norms))
        # print('r_norms', r_norms)
        if np.max(r_norms) < tol or ii == (max -1):
            break
        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
        index = [i for i,R in enumerate(r_norms) if R > tol]
        print('unconverged states', index)
        ###############################################################

        #####################################################################################
        # preconditioning step
        P_start = time.time()
        X_new, Y_new = new_guess_generator(R_x[:,index], R_y[:,index], omega[index])
        P_end = time.time()
        Pcost += P_end - P_start
        #####################################################################################

        ###############################################################
        # GS and symmetric orthonormalization
        m = new_m
        VWGScost_start = time.time()
        V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new)
        VWGScost_end = time.time()
        VWGScost += VWGScost_end - VWGScost_start
        print('m & new_m', m, new_m)
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break
        ###############################################################



    omega *= 27.211386245988

    difference = np.mean((np.array(initial_energies) - np.array(omega))**2)
    difference = float(difference)

    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)) + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    TDDFT_end = time.time()
    TDDFT_cost = TDDFT_end - TDDFT_start
    Davidson_dic['initial guess'] = init
    Davidson_dic['initial_energies'] = initial_energies
    Davidson_dic['semiempirical_difference'] = difference
    Davidson_dic['overlap'] = overlap
    Davidson_dic['preconditioner'] = prec
    Davidson_dic['nstate'] = k
    Davidson_dic['molecule'] = basename
    Davidson_dic['method'] = args.method
    Davidson_dic['functional'] = args.functional
    Davidson_dic['threshold'] = tol
    Davidson_dic['SCF time'] = kernel_t
    Davidson_dic['Initial guess time'] = init_time
    Davidson_dic['initial guess threshold'] = args.TDDFT_initialTOL
    Davidson_dic['New guess generating time'] = Pcost
    Davidson_dic['preconditioner threshold'] = args.TDDFT_precondTOL
    Davidson_dic['total time'] = TDDFT_cost
    Davidson_dic['iterations'] = ii + 1
    Davidson_dic['A matrix size'] = A_size
    Davidson_dic['final subspace size'] = np.shape(sigma)[0]
    Davidson_dic['excitation energy(eV)'] = omega.tolist()
    Davidson_dic['ax'] = a_x
    Davidson_dic['alpha'] = alpha
    Davidson_dic['beta'] = beta
    Davidson_dic['virtual truncation tol'] = args.truncate_virtual
    Davidson_dic['n_occ'] = n_occ
    Davidson_dic['n_vir'] = n_vir
    Davidson_dic['max_vir'] = max_vir
    if ii == (max -1):
        print('============================ TDDFT Failed Due to Iteration Limit============================')
        print('TDDFT failed after ', ii+1, 'iterations  ', round(TDDFT_cost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('================================== TDDFT Calculation Done ==================================')
        print('TDDFT Converged after ', ii+1, 'iterations  ', round(TDDFT_cost, 4), 'seconds')
        print('Initial guess',init)
        print('preconditioner', prec)
        print('Final subspace ', sigma.shape)
        print('preconditioning cost', round(Pcost,4), round(Pcost/TDDFT_cost * 100, 2), "%")
        print('VWGScost',round(VWGScost,4))
        # print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))

    # need normalize
    show_memory_info('Total TDDFT')
    return omega, X_full, Y_full, Davidson_dic
# end of TDDFT module
################################################################################

################################################################################
# a dictionary for TDDFT initial guess and precodnitioner
dynpol_i_key = ['sTDDFT', 'Adiag']
dynpol_i_func = [sTDDFT_preconditioner, TDDFT_A_diag_preconditioner]
dynpol_i_lib = dict(zip(dynpol_i_key, dynpol_i_func))

dynpol_p_key = ['sTDDFT', 'Adiag']
dynpol_p_func = [sTDDFT_preconditioner, TDDFT_A_diag_preconditioner]
dynpol_p_lib = dict(zip(dynpol_p_key, dynpol_p_func))
################################################################################

################################################################################
def gen_P():
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]
    int_r= mol.intor_symmetric('int1e_r')
    P = lib.einsum("xpq,pi,qa->iax", int_r, orbo, orbv.conj())
    return P
################################################################################

################################################################################
def dynamic_polarizability(init, prec):
    ''' [ A B ] - [1  0]X  w = -P'''
    ''' [ B A ]   [0 -1]Y    = -Q'''
    dp_start = time.time()
    initial_guess = dynpol_i_lib[init]
    new_guess_generator = dynpol_p_lib[prec]

    print('Initial guess:  ', init)
    print('Preconditioner: ', prec)
    print('A matrix size = ', A_size,'*', A_size)

    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    k = len(args.dynpol_omega)
    omega =  np.zeros([3*k])
    for jj in range(k):
        # 0,1,2   3,4,5   6,7,8
        omega[3*jj:3*(jj+1)] = 45.56337117/args.dynpol_omega[jj]
        # convert nm to Hartree

    P = gen_P()
    P = P.reshape(-1,3)

    P_origin = np.zeros_like(P)
    P_origin[:,:] = P[:,:]

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    pqnorm = pnorm * (2**0.5)
    print('pqnorm', pqnorm)
    P /= pqnorm

    P = np.tile(P,k)
    Q = P
    #P, Q, omega

    max = args.max
    tol = args.dynpol_tolerance

    m = 0
    V_holder = np.zeros((A_size, (max+1)*k*3))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    ##############################
    # setting up initial guess
    init_start = time.time()
    X_ig, Y_ig = initial_guess(-P, -Q, omega)

    alpha_omega_ig = []
    X_p_Y = X_ig + Y_ig
    X_p_Y *= np.tile(pqnorm,k)
    for jj in range(k):
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        # *-1 from the definition of dipole moment. *2 for double occupancy
        alpha_omega_ig.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)
    print('initial guess of tensor alpha')
    for i in range(k):
        print(args.dynpol_omega[i],'nm')
        print(alpha_omega_ig[i])

    V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(V_holder, W_holder, 0, X_ig, Y_ig)
    init_end = time.time()
#     print('initial guess done')
    initial_cost = init_end - init_start
    ##############################
    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        print('||||____________________Iteration', ii, '_________')
        ###############################################################
        # creating the subspace
        V = V_holder[:,:new_m]
        W = W_holder[:,:new_m]
        # U1 = AV + BW
        # U2 = AW + BV

        MV_start = time.time()
        U1_holder[:, m:new_m], U2_holder[:, m:new_m] = TDDFT_matrix_vector(V[:, m:new_m], W[:, m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:new_m]
        U2 = U2_holder[:,:new_m]

        subgenstart = time.time()
        a = np.dot(V.T, U1) + np.dot(W.T, U2)
        b = np.dot(V.T, U2) + np.dot(W.T, U1)

        sigma = np.dot(V.T, V) - np.dot(W.T, W)
        pi = np.dot(V.T, W) - np.dot(W.T, V)

        a = symmetrize(a)
        b = symmetrize(b)
        sigma = symmetrize(sigma)
        pi = anti_symmetrize(pi)
        # p = VP + WQ
        # q = WP + VQ
        p = np.dot(V.T, P) + np.dot(W.T, Q)
        q = np.dot(W.T, P) + np.dot(V.T, Q)
        subgenend = time.time()
        subgencost += subgenend - subgenstart
        print('sigma.shape', sigma.shape)
        ###############################################################

        ###############################################################
#         solve the x & y in the subspace
        subcost_start = time.time()
        x, y = sTDDFT_preconditioner_subspace_eigen_solver(a, b, sigma, pi, -p, -q, omega)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start
        ###############################################################

        ###############################################################
        # compute the residual
#         R_x = U1x + U2y - (Vx + Wy)omega - P
#         R_y = U2x + U1y + (Wx + Vy)omega - Q
#         R_x = np.dot(U1,x) + np.dot(U2,y) - (np.dot(V,x) + np.dot(W,y))*omega - P
#         R_y = np.dot(U2,x) + np.dot(U1,y) + (np.dot(W,x) + np.dot(V,y))*omega - Q

        U1x = np.dot(U1,x)
        U2x = np.dot(U2,x)
        Vx = np.dot(V,x)
        Wx = np.dot(W,x)

        U1y = np.dot(U1,y)
        U2y = np.dot(U2,y)
        Vy = np.dot(V,y)
        Wy = np.dot(W,y)

        X_full = Vx + Wy
        Y_full = Wx + Vy

        R_x = U1x + U2y - X_full*omega + P
        R_y = U2x + U1y + Y_full*omega + Q

        residual = np.vstack((R_x,R_y))
#         print('residual.shape', residual.shape)
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        print('r_norms', r_norms)
        print('maximum residual norm: ', np.max(r_norms))

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        if np.max(r_norms) < tol or ii == (max -1):
            break
        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
#         print('index', index)
        ###############################################################

        #####################################################################################
        # preconditioning step
        Pstart = time.time()
        X_new, Y_new = new_guess_generator(R_x[:,index], R_y[:,index], omega[index])
        Pend = time.time()
        Pcost += Pend - Pstart
        #####################################################################################

        ###############################################################
        # GS and symmetric orthonormalization
        m = new_m
        GS_start = time.time()
        V_holder, W_holder, new_m = VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
#         print('m & new_m', m, new_m)
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break
        ###############################################################

    dp_end = time.time()
    dp_cost = dp_end - dp_start

    if ii == (max -1):
        print('============================ Dynamic polarizability Failed Due to Iteration Limit============================')
        print('dynamic polarizability failed after ', ii+1, 'iterations  ', round(dp_cost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('Dynamic polarizability Converged after ', ii+1, 'iterations  ', round(dp_cost, 4), 'seconds')
        print('initial_cost', round(initial_cost,4), round(initial_cost/dp_cost * 100,2),'%')
        print('Pcost', round(Pcost,4), round(Pcost/dp_cost * 100,2),'%')
        print('MVcost', round(MVcost,4), round(MVcost/dp_cost * 100,2),'%')
        print('GScost', round(GScost,4), round(GScost/dp_cost * 100,2),'%')
        print('subcost', round(subcost,4), round(subcost/dp_cost * 100,2),'%')
        print('subgencost', round(subgencost,4), round(subgencost/dp_cost * 100,2),'%')

    print('Wavelength we look at', args.dynpol_omega)
    alpha_omega = []

    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)) + np.linalg.norm(np.dot(Y_ig.T, Y_full)))

    X_p_Y = X_full + Y_full

    X_p_Y *= np.tile(pqnorm,k)

    for jj in range(k):
        X_p_Y_tmp = X_p_Y[:,3*jj:3*(jj+1)]
        # *-1 from the definition of dipole moment. *2 for double occupancy
        alpha_omega.append(np.dot(P_origin.T, X_p_Y_tmp)*-2)

     # np.dot(-P.T, X_full) + np.dot(-P.T, Y_full)

    difference = 0
    for i in range(k):
        difference += np.mean((alpha_omega_ig[i] - alpha_omega[i])**2)

    difference = float(difference)


    Davidson_dic['initial guess'] = init
    Davidson_dic['preconditioner'] = prec
    Davidson_dic['molecule'] = basename
    Davidson_dic['method'] = args.method
    Davidson_dic['functional'] = args.functional
    Davidson_dic['threshold'] = args.dynpol_tolerance
    Davidson_dic['SCF time'] = kernel_t
    Davidson_dic['Initial guess time'] = initial_cost
    Davidson_dic['initial guess threshold'] = args.dynpol_initprecTOL
    Davidson_dic['New guess generating time'] = Pcost
    Davidson_dic['preconditioner threshold'] = args.dynpol_initprecTOL
    Davidson_dic['total time'] = dp_cost
    Davidson_dic['iterations'] = ii + 1
    Davidson_dic['A matrix size'] = A_size
    Davidson_dic['final subspace size'] = np.shape(sigma)[0]
    Davidson_dic['Dynamic polarizability wavelength'] = args.dynpol_omega
    Davidson_dic['Dynamic polarizability tensor alpha'] = [i.tolist() for i in alpha_omega]
    Davidson_dic['Dynamic polarizability tensor alpha initial guess '] = [i.tolist() for i in alpha_omega_ig]
    Davidson_dic['overlap'] = overlap
    Davidson_dic['semiempirical_difference'] = difference
    Davidson_dic['ax'] = a_x
    Davidson_dic['alpha'] = alpha
    Davidson_dic['beta'] = beta
    Davidson_dic['virtual truncation tol'] = args.truncate_virtual
    Davidson_dic['n_occ'] = n_occ
    Davidson_dic['n_vir'] = n_vir
    Davidson_dic['max_vir'] = max_vir
    show_memory_info('Total Dynamic polarizability')

    return alpha_omega, Davidson_dic
################################################################################

###############################################################################
def stapol_A_diag_initprec(P):
    d = hdiag.reshape(-1,1)
    return -P/d
###############################################################################

###############################################################################
def stapol_sTDDFT_initprec(P):
    '''(A* + B*)X = -P'''
    '''residual = (A* + B*)X + P'''
    '''X_ig = -P/d'''
    '''X_new = (residual - P)/D = Ux/D'''
    ssp_start = time.time()
    max = 30
    tol = args.stapol_initprecTOL
    m = 0
    npvec = P.shape[1]

    P_origin = np.zeros_like(P)
    P_origin[:,:] = P[:,:]
    pnorm = np.linalg.norm(P_origin, axis=0, keepdims = True)
    # print('pnorm', pnorm)
    P /= pnorm

    # print('vectors to be preconditioned', npvec)
    V_holder = np.zeros((A_size, (max+1)*npvec))
    U_holder = np.zeros_like(V_holder)

    ##############################
    # setting up initial guess
    init_start = time.time()

    X_ig = stapol_A_diag_initprec(P)

    V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, m, X_ig)
    # V_holder[:,0] = X_new[:,0]/np.linalg.norm(X_new[:,0])
    # if npvec >= 2:
    #     V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, 1, X_new[:, 1:])
    # else:
    #     new_m = 1
    # print('new_m =', new_m)
    init_end = time.time()
#     print('initial guess done')
    initial_cost = init_end - init_start
    ##############################
    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        # creating the subspace
        MV_start = time.time()
        U_holder[:, m:new_m] = sTDDFT_stapol_mv(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        # U = AX + BX = (A+B)X
#         print(check_orthonormal(V))
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = symmetrize(a_p_b)

        subgenend = time.time()
        subgencost += subgenend - subgenstart
        ###############################################################

        ###############################################################
#         solve the x in the subspace
        subcost_start = time.time()
        x = np.linalg.solve(a_p_b, -p)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start
        ###############################################################

        ###############################################################
        # compute the residual
#         R = Ux + P
        Ux = np.dot(U,x)
        residual = Ux + P

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
        ###############################################################
        if np.max(r_norms) < tol or ii == (max -1):
            break

        ########################################################################
        # preconditioning step
        Pstart = time.time()

        X_new = stapol_A_diag_initprec(-residual[:,index])

        Pend = time.time()
        Pcost += Pend - Pstart
        ########################################################################

        ###############################################################
        # GS and symmetric orthonormalization
        m = new_m
        GS_start = time.time()
        V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, m, X_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
#         print('m & new_m', m, new_m)
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break
        ###############################################################
    X_full = np.dot(V,x)
    # alpha = np.dot(X_full.T, P)*-4

    ssp_end = time.time()
    ssp_cost = ssp_end - ssp_start

    if ii == (max -1):
        print('============================ sTDDFT Static polarizability Failed Due to Iteration Limit============================')
        print('sTDDFT Static polarizability failed after ', ii+1, 'iterations  ', round(ssp_cost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('sTDDFT Static polarizability Converged after ', ii+1, 'iterations  ', round(ssp_cost, 4), 'seconds')
        # print('initial_cost', round(initial_cost,4), round(initial_cost/ssp_cost * 100,2),'%')
        # print('Pcost', round(Pcost,4), round(Pcost/ssp_cost * 100,2),'%')
        print('MVcost', round(MVcost,4), round(MVcost/ssp_cost * 100,2),'%')
        # print('GScost', round(GScost,4), round(GScost/ssp_cost * 100,2),'%')
        # print('subcost', round(subcost,4), round(subcost/ssp_cost * 100,2),'%')
        # print('subgencost', round(subgencost,4), round(subgencost/ssp_cost * 100,2),'%')

    X_full *= pnorm
    return X_full
################################################################################

################################################################################
# a dictionary for static_polarizability initial guess and precodnitioner
stapol_i_key = ['sTDDFT', 'Adiag']
stapol_i_func = [stapol_sTDDFT_initprec, stapol_A_diag_initprec]
stapol_i_lib = dict(zip(stapol_i_key, stapol_i_func))

stapol_p_key = ['sTDDFT', 'Adiag']
stapol_p_func = [stapol_sTDDFT_initprec, stapol_A_diag_initprec]
stapol_p_lib = dict(zip(stapol_p_key, stapol_p_func))
################################################################################

################################################################################
# Static polarizability
def static_polarizability(init, prec):
    '''(A+B)X = -P'''
    '''residual = (A+B)X + P'''
    '''X_new = (residual - P)/D'''
    print('initial guess', init)
    print('preconditioner', prec)
    sp_start = time.time()

    P = gen_P()
    P = P.reshape(-1,3)

    P_origin = np.zeros_like(P)
    P_origin[:,:] = P[:,:]

    pnorm = np.linalg.norm(P, axis=0, keepdims = True)
    # pnorm is constant
    print('pnorm', pnorm)

    P /= pnorm

    Davidson_dic = {}
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    initial_guess = stapol_i_lib[init]
    new_guess_generator = stapol_p_lib[prec]

    max = args.max
    tol = args.stapol_tolerance
    m = 0

    V_holder = np.zeros((A_size, (max+1)*3))
    U_holder = np.zeros_like(V_holder)

    ##############################
    # setting up initial guess
    init_start = time.time()

#     X_new = -P/d3
    X_ig = initial_guess(P)

    alpha_init = np.dot((X_ig*pnorm).T, P_origin)*-4
    print('alpha tensor of initial guess:')
    print(alpha_init)

    V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, 0, X_ig)
    print('new_m =', new_m)
    init_end = time.time()
#     print('initial guess done')
    initial_cost = init_end - init_start
    ##############################
    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    for ii in range(max):
        print('||||____________________Iteration', ii, '_________')
        ########################################################################
        # creating the subspace

        MV_start = time.time()
        U_holder[:, m:new_m] = static_polarizability_matrix_vector(V_holder[:,m:new_m])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        V = V_holder[:,:new_m]
        # print('check_orthonormal(V)',check_orthonormal(V))
        # U = AX + BX = (A+B)X
#         print(check_orthonormal(V))
        U = U_holder[:,:new_m]

        subgenstart = time.time()
        p = np.dot(V.T, P)
        a_p_b = np.dot(V.T,U)
        a_p_b = symmetrize(a_p_b)
        subgenend = time.time()
        subgencost += subgenend - subgenstart

        # print('a_p_b.shape', a_p_b.shape)
        ########################################################################

        ########################################################################
#         solve the x in the subspace
        subcost_start = time.time()
        x = np.linalg.solve(a_p_b, -p)
        # print('x.shape', x.shape)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start
        ########################################################################

        ########################################################################
        # compute the residual
#         R = Ux + P
        Ux = np.dot(U,x)
        residual = Ux + P

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        print('r_norms', r_norms)

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['residual norms'] = r_norms
        iteration_list[ii] = current_dic

        # index for unconverged residuals
        index = [r_norms.index(i) for i in r_norms if i > tol]
#         print('index', index)
        ########################################################################
        if np.max(r_norms) < tol or ii == (max -1):
            break

        ########################################################################
        # preconditioning step
        Pstart = time.time()

        X_new = new_guess_generator(-residual[:,index])
        # X_new = X_new/np.linalg.norm(X_new, axis=0)
        Pend = time.time()
        Pcost += Pend - Pstart
        ########################################################################

        ########################################################################
        # GS and symmetric orthonormalization
        m = new_m
        GS_start = time.time()
        V_holder, new_m = Gram_Schmidt_fill_holder(V_holder, m, X_new)
        GS_end = time.time()
        GScost += GS_end - GS_start
#         print('m & new_m', m, new_m)
        if new_m == m:
            print('All new guesses kicked out during GS orthonormalization')
            break
        ###############################################################
    print('V.shape', V.shape)
    X_full = np.dot(V,x)

    overlap = float(np.linalg.norm(np.dot(X_ig.T, X_full)))

    X_full *= pnorm

    tensor_alpha = np.dot(X_full.T, P_origin)*-4
    sp_end = time.time()
    sp_cost = sp_end - sp_start



    if ii == (max -1):
        print('============================ Static polarizability Failed Due to Iteration Limit============================')
        print('Static polarizability failed after ', ii+1, 'iterations  ', round(sp_cost, 4), 'seconds')
        print('current residual norms', r_norms)
        print('max_norm = ', np.max(r_norms))
    else:
        print('Static polarizability Converged after ', ii+1, 'iterations  ', round(sp_cost, 4), 'seconds')
        print('initial_cost', round(initial_cost,4), round(initial_cost/sp_cost * 100,2),'%')
        print('Pcost', round(Pcost,4), round(Pcost/sp_cost * 100,2),'%')
        print('MVcost', round(MVcost,4), round(MVcost/sp_cost * 100,2),'%')
        print('GScost', round(GScost,4), round(GScost/sp_cost * 100,2),'%')
        print('subcost', round(subcost,4), round(subcost/sp_cost * 100,2),'%')
        print('subgencost', round(subgencost,4), round(subgencost/sp_cost * 100,2),'%')

    difference = np.mean((alpha_init - tensor_alpha)**2)
    difference = float(difference)

    sp_end = time.time()
    spcost = sp_end - sp_start
    Davidson_dic['initial guess'] = init
    Davidson_dic['preconditioner'] = prec
    Davidson_dic['molecule'] = basename
    Davidson_dic['method'] = args.method
    Davidson_dic['functional'] = args.functional
    Davidson_dic['threshold'] = args.stapol_tolerance
    Davidson_dic['SCF time'] = kernel_t
    Davidson_dic['Initial guess time'] = initial_cost
    Davidson_dic['initial guess threshold'] = args.stapol_initprecTOL
    Davidson_dic['New guess generating time'] = Pcost
    Davidson_dic['preconditioner threshold'] = args.stapol_initprecTOL
    Davidson_dic['total time'] = sp_cost
    Davidson_dic['iterations'] = ii+1
    Davidson_dic['A matrix size'] = A_size
    Davidson_dic['final subspace size'] = np.shape(a_p_b)[0]
    Davidson_dic['semiempirical_difference'] = difference
    Davidson_dic['overlap'] = overlap
    Davidson_dic['ax'] = a_x
    Davidson_dic['alpha'] = alpha
    Davidson_dic['beta'] = beta
    Davidson_dic['virtual truncation tol'] = args.truncate_virtual
    Davidson_dic['n_occ'] = n_occ
    Davidson_dic['n_vir'] = n_vir
    Davidson_dic['max_vir'] = max_vir
    return tensor_alpha, Davidson_dic
################################################################################

TDA_combo = [            # option
['sTDA','sTDA'],     # 0
['Adiag','Adiag'],   # 1
['Adiag','sTDA'],    # 2
['sTDA','Adiag'],    # 3
['sTDA','Jacobi'],   # 4
['Adiag','Jacobi'],  # 5
['Adiag','new_ES'],  # 6
['sTDA','new_ES']]   # 7

TDDFT_combo = [         # option
['sTDDFT','sTDDFT'],     # 0
['Adiag','Adiag'],       # 1
['Adiag','sTDDFT'],      # 2
['sTDDFT','Adiag']]      # 3

dynpol_combo = [         # option
['sTDDFT','sTDDFT'],                     # 0
['Adiag','Adiag'],                       # 1
['Adiag','sTDDFT'],                      # 2
['sTDDFT','Adiag']]                      # 3

stapol_combo = [         # option
['sTDDFT','sTDDFT'],                     # 0
['Adiag','Adiag'],                       # 1
['Adiag','sTDDFT'],                      # 2
['sTDDFT','Adiag']]                      # 3

if args.TDA == True:
    for option in args.TDA_options:
        init, prec = TDA_combo[option]
        print('-------------------------------------------------------------------')
        print('|---------------   In-house Developed Davidson Starts   -----------|')
        print('Residual conv =', args.TDA_tolerance)
        print('Number of excited states =', args.nstates)

        total_start = time.time()
        Excitation_energies, eigenkets, Davidson_dic = Davidson(args.nstates, args.TDA_tolerance, init,prec)
        total_end = time.time()
        total_time = total_end - total_start

        print('In-house Davidson time:', round(total_time, 4), 'seconds')

        print('Excited State energies (eV) =')
        print(Excitation_energies)

        curpath = os.getcwd()
        yamlpath = os.path.join(curpath, basename + '_TDA_i_' + init + '_p_'+ prec + '.yaml')

        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(Davidson_dic, f)

        print('|---------------   In-house Developed Davidson Done   -----------|')

if args.TDDFT == True:
    for option in args.TDDFT_options:
        init, prec = TDDFT_combo[option]
        print('----------------------------------------------------------------------------')
        print('|---------------   In-house Developed TDDFT Eigensolver Starts   -----------|')
        print('Residual conv =', args.TDDFT_tolerance)
        print('Number of excited states =', args.nstates)

        total_start = time.time()
        Excitation_energies, X, Y, Davidson_dic = TDDFT_eigen_solver(init,prec)
        total_end = time.time()
        total_time = total_end - total_start

        print('In-house TDDFT Eigensolver time:', round(total_time, 4), 'seconds')

        print('Excited State energies (eV) =')
        print(Excitation_energies)

        curpath = os.getcwd()
        yamlpath = os.path.join(curpath, basename + '_TDDFT_i_' + init + '_p_'+ prec + '.yaml')

        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(Davidson_dic, f)

        print('|---------------   In-house Developed TDDFT Eigensolver Done   -----------|')

if args.dynpol == True:
    args.TDDFT_precondTOL = args.dynpol_initprecTOL
    print('args.TDDFT_precondTOL', args.TDDFT_precondTOL)
    for option in args.dynpol_options:
        init,prec = dynpol_combo[option]
        print('---------------------------------------------------------------------------------')
        print('|---------------   In-house Developed Dynamic Polarizability Starts   -----------|')
        print('Residual conv =', args.dynpol_tolerance)
        print('Perturbation wavelength omega (nm) =', args.dynpol_omega)

        total_start = time.time()
        alpha_omega, Davidson_dic = dynamic_polarizability(init,prec)
        total_end = time.time()
        total_time = total_end - total_start

        print('In-house Dynamic Polarizability time:', round(total_time, 4), 'seconds')
        print('Dynamic polarizability tensor alpha')
        for i in range(len(args.dynpol_omega)):
            print(args.dynpol_omega[i],'nm')
            print(alpha_omega[i])

        curpath = os.getcwd()
        yamlpath = os.path.join(curpath, basename + '_Dynamic_Polarizability_i_' + init + '_p_'+ prec + '.yaml')

        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(Davidson_dic, f)

        print('|---------------   In-house Developed Dynamic Polarizability Done   -----------|')

if args.stapol == True:
    for option in args.stapol_options:
        init,prec = stapol_combo[option]
        print('--------------------------------------------------------------------------------')
        print('|---------------   In-house Developed Static Polarizability Starts   -----------|')
        print('Residual conv =', args.stapol_tolerance)

        total_start = time.time()
        tensor_alpha, Davidson_dic = static_polarizability(init,prec)
        total_end = time.time()
        total_time = total_end - total_start

        print('In-house Static Polarizability time:', round(total_time, 4), 'seconds')
        print('Static polarizability tensor alpha')
        print(tensor_alpha)

        curpath = os.getcwd()
        yamlpath = os.path.join(curpath, basename + '_Static_Polarizability_i_' + init + '_p_'+ prec + '.yaml')

        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(Davidson_dic, f)

        print('|---------------   In-house Developed Static Polarizability Done   -----------|')


if args.sTDA == True:
    print('--------------------------------------------------------------------------------')
    print('|---------------------------   In-house Developed sTDA   --------------------|')
    print('Convergence =', args.TDA_initialTOL)

    total_start = time.time()
    X = Davidson0(k=args.nstates)
    total_end = time.time()
    total_time = total_end - total_start

    print('In-house sTDA time:', round(total_time, 4), 'seconds')
    print('|---------------------------   In-house Developed sTDA done  ----------------|')

if args.sTDDFT == True:
    print('--------------------------------------------------------------------------------')
    print('|---------------------------   In-house Developed sTDDFT   --------------------|')
    print('Convergence =', args.TDDFT_initialTOL)

    total_start = time.time()
    energies, X, Y = sTDDFT_eigen_solver(k=args.nstates)
    total_end = time.time()
    total_time = total_end - total_start

    print('In-house sTDDFT time:', round(total_time, 4), 'seconds')

    print('|---------------------------   In-house Developed sTDDFT done  ----------------|')


if args.Truncate_test == True:
    print('--------------------------------------------------------------------------------')
    print('|---------------------------   Test the Truncation efficiency  ----------------|')

    n_state = 40
    X = np.random.rand(A_size,n_state)
    Y = np.random.rand(A_size,n_state)
    print('n_vir = ', n_vir)
    print('A_size =', A_size)
    print('n_state =', n_state)
    print("virtual tuning: -------------------------------------------- ")
    print("{:<8} {:<8} {:<8} {:<8}".format(\
            'eV', 'max_vir', 'sTDA_t', 'sTDDFT_t'))
    for vir_trunc in [40, 50, 60, 70, 10000000]:
        del max_vir, sTDA_mv, sTDDFT_mv
        max_vir = gen_maxvir(tol_eV = vir_trunc)
        q_ij, q_ab, q_ia , GK_q_jb, GJ_q_ab = gen_QJK(max_vir=max_vir)
        # print('q_ab', q_ab.shape, 'GK_q_jb', GK_q_jb.shape)
        iajb_fly, ijab_fly, ibja_fly, delta_fly = gen_iajb_ijab_ibja_delta_fly(\
                                        max_vir=max_vir, \
                                        q_ij = q_ij, \
                                        q_ab = q_ab, \
                                        q_ia = q_ia , \
                                        GK_q_jb = GK_q_jb, \
                                        GJ_q_ab = GJ_q_ab)


        sTDA_mv, sTDDFT_mv, sTDDFT_stapol_mv = gen_sTDA_sTDDFT_stapol_fly(\
                                        max_vir=max_vir, \
                                        iajb_fly = iajb_fly, \
                                        ijab_fly = ijab_fly, \
                                        ibja_fly = ibja_fly, \
                                        delta_fly = delta_fly)


        sTDA_start = time.time()
        sTDA_X = sTDA_mv(X)
        sTDA_end = time.time()
        sTDA_mv_time = sTDA_end - sTDA_start

        sTDDFT_start = time.time()
        sTDDFT_X, sTDDFT_Y = sTDDFT_mv(X, Y)
        sTDDFT_end = time.time()
        sTDDFT_mv_time = sTDDFT_end - sTDDFT_start

        print("{:<8} {:<8} {:<8.4f} {:<8.4f}".format(\
                vir_trunc, max_vir, sTDA_mv_time, sTDDFT_mv_time))

    print('|-------------------------------------   Test done  --------------------------|')

if args.pytd == True:
    print('-----------------------------------------------------------------')
    print('|----------------------    PySCF TTDDFT    ----------------------|')
    TD.nstates = args.nstates
    TD.conv_tol = 1e-10
    start = time.time()
    TD.kernel()
    end = time.time()
    pyscf_time = end-start
    print('Built-in TDDFT time:', round(pyscf_time, 4), 'seconds')
    print('|---------------------------------------------------------------|')

if args.verbose > 3:
    for key in vars(args):
        print(key,'=', vars(args)[key])
