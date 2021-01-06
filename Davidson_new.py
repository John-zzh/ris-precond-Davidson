import time
import numpy as np
from opt_einsum import contract as einsum
import pyscf
from pyscf import gto, scf, dft, tddft, data, lib
import argparse
import os
import yaml
from pyscf.tools import molden
from pyscf.dft import xcfun

print('curpath', os.getcwd())
print('lib.num_threads() = ', lib.num_threads())

parser = argparse.ArgumentParser(description='Davidson')
parser.add_argument('-x', '--xyzfile',        type=str, default='NA', help='xyz filename (molecule.xyz)')
# parser.add_argument('-chk', '--checkfile',    type=str, default='NA', help='checkpoint filename (.chk)')
parser.add_argument('-m', '--method',         type=str, default='NA', help='RHF RKS UHF UKS')
parser.add_argument('-f', '--functional',     type=str, default='NA', help='xc functional')
parser.add_argument('-b', '--basis_set',      type=str, default='NA', help='basis set')
parser.add_argument('-df', '--density_fit',   type=bool, default=True, help='density fitting turn on')
parser.add_argument('-g', '--grid_level',     type=int, default='3', help='0-9, 9 is best')
# parser.add_argument('-i', '--initial_guess',  type=str, default='sTDA', help='initial guess: Adiag or sTDA')
# parser.add_argument('-p', '--preconditioner', type=str, default='sTDA', help='preconditioner: Adiag or sTDA')
parser.add_argument('-t', '--tolerance',      type=float, default= 1e-5, help='residual norm convergence threshold')
parser.add_argument('-n', '--nstates',        type=int, default= 4, help='number of excited states')
parser.add_argument('-C', '--compare',        type=bool, default = False , help='whether to compare with PySCF TDA-TDDFT')
parser.add_argument('-o', '--options',        type=int, default ='NA', nargs='+', help='isis=0, iAiA=1, iAis=2, isiA=3')
parser.add_argument('-it', '--initialTOL',    type=float, default= 1e-4, help='convergence threshold for sTDA inital guess')
parser.add_argument('-pt', '--precondTOL',    type=float, default= 1e-2, help='convergence threshold for sTDA preconditioner')
parser.add_argument('-M',  '--memory',        type=int, default= 4000, help='max_memory')
parser.add_argument('-ei', '--extrainitial',  type=int, default= 8, help='number of extral initial guess vectors, [0, 8]')
parser.add_argument('-et', '--eigensolver_tol',type=float, default= 1e-5, help='convergence threshold for new guess generator in new_ES')
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


print ('Molecule built')
print ('Calculating SCF Energy...')
kernel_0 = time.time()
mf.kernel()
kernel_1 = time.time()
kernel_t = kernel_1 - kernel_0
print ('SCF Done after ', round(kernel_t, 4), 'seconds')

mo_occ = mf.mo_occ



########################################################################
# Collect everything needed from PySCF
Qstart = time.time()
# extract vind() function
td = tddft.TDA(mf)

vind, hdiag = td.gen_vind(mf)

# vind (V) = A*V
def matrix_vector(V):
    return vind(V.T).T

Natm = mol.natm


occupied = len(np.where(mo_occ > 0)[0])
#mf.mo_occ is an array of occupance [2,2,2,2,2,0,0,0,0.....]
virtual = len(np.where(mo_occ == 0)[0])

# AO = [int(i.split(' ',1)[0]) for i in mol.ao_labels()]
# # .split(' ',1) is to split each element by space, split once.
# # mol.ao_labels() it is Labels of AO basis functions
# # AO is a list of corresponding atom_id

N_bf = len(mo_occ)
R = pyscf.gto.mole.inter_distance(mol, coords=None)
#Inter-particle distance array
# unit == ’Bohr’, Its value is 5.29177210903(80)×10^(−11) m
########################################################################


##################################################################################################
# create a function for dictionary of chemical hardness, by mappig two iteratable subject, list
# list of elements



elements = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca',
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U' , 'Np', 'Pu']
#list of chemical hardness, they are floats, containing elements 1-94, in Hartree
hardness = [
0.47259288,
0.92203391,
0.17452888,
0.25700733,
0.33949086,
0.42195412,
0.50438193,
0.58691863,
0.66931351,
0.75191607,
0.17964105,
0.22157276,
0.26348578,
0.30539645,
0.34734014,
0.38924725,
0.43115670,
0.47308269,
0.17105469,
0.20276244,
0.21007322,
0.21739647,
0.22471039,
0.23201501,
0.23933969,
0.24665638,
0.25398255,
0.26128863,
0.26859476,
0.27592565,
0.30762999,
0.33931580,
0.37235985,
0.40273549,
0.43445776,
0.46611708,
0.15585079,
0.18649324,
0.19356210,
0.20063311,
0.20770522,
0.21477254,
0.22184614,
0.22891872,
0.23598621,
0.24305612,
0.25013018,
0.25719937,
0.28784780,
0.31848673,
0.34912431,
0.37976593,
0.41040808,
0.44105777,
0.05019332,
0.06762570,
0.08504445,
0.10247736,
0.11991105,
0.13732772,
0.15476297,
0.17218265,
0.18961288,
0.20704760,
0.22446752,
0.24189645,
0.25932503,
0.27676094,
0.29418231,
0.31159587,
0.32902274,
0.34592298,
0.36388048,
0.38130586,
0.39877476,
0.41614298,
0.43364510,
0.45104014,
0.46848986,
0.48584550,
0.12526730,
0.14268677,
0.16011615,
0.17755889,
0.19497557,
0.21240778,
0.07263525,
0.09422158,
0.09920295,
0.10418621,
0.14235633,
0.16394294,
0.18551941,
0.22370139]
HARDNESS = dict(zip(elements,hardness))
#function to return chemical hardness from dictionary HARDNESS
def Hardness(atom_id):
    atom = mol.atom_pure_symbol(atom_id)
    return HARDNESS[atom]
# mol.atom_pure_symbol(atom_id) returns pure element symbol, no special characters


########################################################################
# This block is the function to produce orthonormalized coefficient matrix C
def matrix_power(S,a):
    s,ket = np.linalg.eigh(S)
    s = s**a
    X = np.linalg.multi_dot([ket,np.diag(s),ket.T])
    #X == S^1/2
    return X

def orthonormalize(C):
    X = matrix_power(mf.get_ovlp(), 0.5)
    # S = mf.get_ovlp() #.get_ovlp() is basis overlap matrix
    # S = np.dot(np.linalg.inv(c.T), np.linalg.inv(c))
    C = np.dot(X,C)
    return C

C = mf.mo_coeff
# mf.mo_coeff is the coefficient matrix

C = orthonormalize(C)
# C is orthonormalized coefficient matrix
# np.dot(C.T,C) is a an identity matrix
########################################################################
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

hybride_F = [
'b3lyp',
'tpssh',
'm05-2x',
'pbe0',
'm06',
'm06-2x',
'NA']# NA is for Hartree-Fork
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


# creat \eta matrix
a = [Hardness(atom_id) for atom_id in range(Natm)]
a = np.asarray(a).reshape(1,-1)
eta = (a+a.T)/2

# creat GammaK and GammaK matrix
GammaJ = (R**beta + (a_x * eta)**(-beta))**(-1/beta)
GammaK = (R**alpha + eta**(-alpha)) **(-1/alpha)


Natm = mol.natm
def generateQ():
    aoslice = mol.aoslice_by_atom()
    q = np.zeros([Natm, N_bf, N_bf])
    #N_bf is number Atomic orbitals, occupied+virtual, q is same size with C
    for atom_id in range(Natm):
        shst, shend, atstart, atend = aoslice[atom_id]
        q[atom_id,:, :] = np.dot(C[atstart:atend, :].T, C[atstart:atend, :])
    return q

q_tensors = generateQ()


q_tensor_ij = np.zeros((Natm, occupied, occupied))
q_tensor_ij[:,:,:] = q_tensors[:, :occupied,:occupied]

q_tensor_ab = np.zeros((Natm, virtual, virtual))
q_tensor_ab[:,:,:] = q_tensors[:, occupied:,occupied:]

q_tensor_ia = np.zeros((Natm, occupied, virtual))
q_tensor_ia[:,:,:] = q_tensors[:, :occupied,occupied:]


Q_K = einsum('Bjb, AB -> Ajb', q_tensor_ia, GammaK)
Q_J = einsum('Bab, AB -> Aab', q_tensor_ab, GammaJ)
# pre-calculate and store the Q-Gamma rank 3 tensor
Qend = time.time()

Q_time = Qend - Qstart
print('Q-Gamma tensors building time =', round(Q_time, 4))
##################################################################################################


###################################################################################################
# This block is to define on-the-fly two electron intergeral (pq|rs)
# A_iajb * v = delta_ia_ia*v + 2(ia|jb)*v - (ij|ab)*v

# iajb_v = einsum('Aia, Bjb, AB, jbm -> iam', q_tensor_ia, q_tensor_ia, GammaK, V)
# ijab_v = einsum('Aij, Bab, AB, jbm -> iam', q_tensor_ij, q_tensor_ab, GammaJ, V)

def iajb_fly(V):
    V = V.reshape(occupied, virtual, -1)
    Q_K_V = einsum('Ajb, jbm -> Am', Q_K, V)
    iajb_V = einsum('Aia, Am -> iam', q_tensor_ia, Q_K_V).reshape(occupied*virtual, -1)

    return iajb_V

def ijab_fly(V):
    V = V.reshape(occupied, virtual, -1)
    # (-1, occupied, virtual)
#     ijab_v = einsum('Aij, Aab, jbm -> iam', q_tensor_ij, Q_J,  V)

    # contract smaller index first
    # Aij_V = einsum('Aij, jbm -> Aibm', q_tensor_ij, V)
    # ijab_V = einsum('Aab, Aibm -> iam', Q_J, Aij_V).reshape(occupied*virtual, -1)

    # contract larger index first
    Aab_V = einsum('Aab, jbm -> jAam', Q_J, V)
    #('Aab, mjb -> mjaA')
    ijab_V = einsum('Aij, jAam -> iam', q_tensor_ij, Aab_V).reshape(occupied*virtual, -1)
    #('Aij, mjaA -> mia)
    return ijab_V

delta_diag_A = hdiag.reshape(occupied, virtual)



def delta_fly(V):
    V = V.reshape(occupied, virtual, -1)
    delta_v = einsum('ia,iam -> iam', delta_diag_A, V).reshape(occupied*virtual, -1)
    return delta_v

def sTDA_fly(V):
    # sTDA_A * V
    V = V.reshape(occupied*virtual,-1)
    # this feature can deal with multiple vectors
    sTDA_V =  delta_fly(V) + 2*iajb_fly(V) - ijab_fly(V)
    return sTDA_V
###################################################################################################





##############################################################################################
# orthonormalization of guess_vectors
def Gram_Schdmit_bvec(A, bvec):
    # suppose A is orthonormalized
    projections_coeff = np.dot(A.T, bvec)
    bvec = bvec - np.dot(A, projections_coeff)
    return bvec

def Gram_Schdmit(A):
    # A matrix has J columns, orthonormalize each columns
    # unualified vectors will be removed
    N_rows = np.shape(A)[0]
    N_vectors = np.shape(A)[1]
    A = A/np.linalg.norm(A, axis=0, keepdims = True)

    B = np.zeros((N_rows,N_vectors))
    count = 0
    ############b
    for j in range(N_vectors):
        bvec = Gram_Schdmit_bvec(B[:, :count], A[:, j])
        norm = np.linalg.norm(bvec)
        if norm > 1e-14:
            B[:, count] = bvec/np.linalg.norm(bvec)
            count +=1
    return B[:, :count]

def Gram_Schdmit_fill_holder(V, count, vecs):
    # V is a vectors holder
    # count is the amount of vectors that already sit in the holder

    nvec = np.shape(vecs)[1]
    # amount of new vectors intended to fill in the V

    # count will be final amount of vectors in V
    for j in range(nvec):
        vec = vecs[:, j]
        vec = Gram_Schdmit_bvec(V[:, :count], vec)   #single orthonormalize
        vec = Gram_Schdmit_bvec(V[:, :count], vec) #double orthonormalize

        norm = np.linalg.norm(vec)
        if  norm > 1e-14:
            vec = vec/norm
            V[:, count] = vec
            count += 1
    new_count = count

    return V, new_count
########################################################################


####################################################################
# define the orthonormality of a matrix A as the norm of (A.T*A - I)
def check_orthonormal(A):
    n = np.shape(A)[1]
    B = np.dot(A.T, A)
    c = np.linalg.norm(B - np.eye(n))
    return c
####################################################################


########################################################################
def solve_AX_Xla_B(sub_A, eigen_lambda, sub_B):
    # AX - XB  = Q
    N_vectors = len(eigen_lambda)
    a, u = np.linalg.eigh(sub_A)
    ub = np.dot(u.T, sub_B)
    ux = np.zeros_like(sub_B)
    for k in range(N_vectors):
        ux[:, k] = ub[:, k]/(a - eigen_lambda[k])
    sub_guess = np.dot(u, ux)
    return sub_guess
#########################################################################

########################################################################
# sTDA preconditioner
def sTDA_preconditioner(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8):
    """ residual[:,index], sub_eigenvalue[:k][index], current_dic, full_guess[:,index], index, W[:,:m], V[:,:m], sub_A """
    """ (sTDA_A - λ*I)^-1 B = X """
    """ AX - Xλ = B """
    """ columns in B are residuals (in Davidson's loop) to be preconditioned, """
    B = arg1
    eigen_lambda = arg2
    current_dic = arg3


    precondition_start = time.time()

    N_rows = np.shape(B)[0]
    B = B.reshape(N_rows, -1)
    N_vectors = np.shape(B)[1]


    #number of vectors to be preconditioned
    bnorm = np.linalg.norm(B, axis=0, keepdims = True)
    #norm of each vectors in B, shape (1,-1)
    B = B/bnorm

    start = time.time()
    tol = args.precondTOL    # Convergence tolerance
    max = 30   # Maximum number of iterations

    V = np.zeros((N_rows, (max+1)*N_vectors))
    W = np.zeros((N_rows, (max+1)*N_vectors))
    count = 0

    # now V and W are empty holders, 0 vectors
    # W = sTDA_fly(V)
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
    V, new_count = Gram_Schdmit_fill_holder(V, count, init)
    W[:, count:new_count] = sTDA_fly(V[:, count:new_count])
    count = new_count

    current_dic['preconditioning'] = []
    ####################################################################################
    for i in range(max):
        sub_B = np.dot(V[:,:count].T, B)
        sub_A = np.dot(V[:,:count].T, W[:,:count])
        #project sTDA_A matrix and vector B into subspace

        # size of subspace
        m = np.shape(sub_A)[0]

        sub_guess = solve_AX_Xla_B(sub_A, eigen_lambda, sub_B)

        full_guess = np.dot(V[:,:count], sub_guess)
        residual = np.dot(W[:,:count], sub_guess) - full_guess*eigen_lambda - B

        Norms_of_r = np.linalg.norm(residual, axis=0, keepdims = False)

        current_dic['preconditioning'].append({'precondition residual norms': Norms_of_r.tolist()})

        max_norm = np.max(Norms_of_r)

        if max_norm < tol:
            break

        # index for unconverged residuals
        index = [i for i in range(len(Norms_of_r)) if Norms_of_r[i] > tol]

        # preconditioning step
        # only generate new guess from unconverged residuals
        new_guess = residual[:,index]*inv_D[:,index]

        V, new_count = Gram_Schdmit_fill_holder(V, count, new_guess)
        W[:, count:new_count] = sTDA_fly(V[:, count:new_count])
        count = new_count

        # V_orthonormality = check_orthonormal(V[:,:count])
        # current_dic['step' + str(i)]['V_orthonormality'] = float(V_orthonormality)

    precondition_end = time.time()
    precondition_time = precondition_end - precondition_start
    if i == (max -1):
        print('_________________ sTDA Preconditioner Failed Due to Iteration Limit _________________')
        print('sTDA preconditioning failed after ', i, 'steps; ', round(precondition_time, 4), 'seconds')
        print('current residual norms', Norms_of_r)
        print('max_norm = ', max_norm)
        print('orthonormality of V', check_orthonormal(V[:,:count]))
    else:
        print('sTDA Preconditioning Done after ', i, 'steps; ', round(precondition_time, 4), 'seconds')

    return full_guess*bnorm, current_dic
###########################################################################################

########################################################################
# K_inv # exacty the same function with sTDA_preconditioner, just no dic
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
    # W = sTDA_fly(V)
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
    V, new_count = Gram_Schdmit_fill_holder(V, count, init)
    W[:, count:new_count] = sTDA_fly(V[:, count:new_count])
    count = new_count
    ####################################################################################
    for i in range(0, max):
        sub_B = np.dot(V[:,:count].T, B)
        sub_A = np.dot(V[:,:count].T, W[:,:count])
        #project sTDA_A matrix and vector B into subspace
        # size of subspace
        m = np.shape(sub_A)[0]
        sub_guess = solve_AX_Xla_B(sub_A, eigen_lambda, sub_B)
        full_guess = np.dot(V[:,:count], sub_guess)
        residual = np.dot(W[:,:count], sub_guess) - full_guess*eigen_lambda - B
        Norms_of_r = np.linalg.norm(residual, axis=0, keepdims = False)
        max_norm = np.max(Norms_of_r)

        if max_norm < tol:
            break

        # index for unconverged residuals
        index = [i for i in range(len(Norms_of_r)) if Norms_of_r[i] > tol]

        # preconditioning step
        # only generate new guess from unconverged residuals
        new_guess = residual[:,index]*inv_D[:,index]

        V, new_count = Gram_Schdmit_fill_holder(V, count, new_guess)
        W[:, count:new_count] = sTDA_fly(V[:, count:new_count])
        count = new_count

    precondition_end = time.time()
    precondition_time = precondition_end - precondition_start
    if i == (max -1):
        print('_________________ K inverse Failed Due to Iteration Limit _________________')
        print('K inverse  failed after ', i, 'steps; ', round(precondition_time, 4), 'seconds')
        print('current residual norms', Norms_of_r)
        print('max_norm = ', max_norm)
        print('orthonormality of V', check_orthonormal(V[:,:count]))
    else:
        print('K inverse Done after ', i, 'steps; ', round(precondition_time, 4), 'seconds')
    return full_guess*bnorm
###########################################################################################



###########################
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
############################



###############################################################################
n = occupied*virtual

################################################################################
# original simple Davidson, just to solve eigenvalues and eigenkets of sTDA_A matrix
def Davidson0(k):
    sTDA_D_start = time.time()
    tol = args.initialTOL # Convergence tolerance
    max = 30
    #################################################
    # m is size of subspace
    m = min([k+8, 2*k, n])
    V = np.zeros((n, max*k + m))
    W = np.zeros_like(V)
    # positions of hdiag with lowest values set as 1

    V = A_diag_initial_guess(m, V)
    W[:, :m] = sTDA_fly(V[:, :m])
    # create transformed guess vectors

    #generate initial guess and put in holders V and W
    ###########################################################################################
    for i in range(0, max):
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        # Diagonalize the subspace Hamiltonian, and sorted.
        #sub_eigenvalue[:k] are smallest k eigenvalues
        residual = np.dot(W[:,:m], sub_eigenket[:,:k]) - np.dot(V[:,:m], sub_eigenket[:,:k] * sub_eigenvalue[:k])

        Norms_of_r = np.linalg.norm(residual, axis=0, keepdims = True)
        # largest residual norm
        max_norm = np.max(Norms_of_r)
        if max_norm < tol:
            break
        # index for unconverged residuals
        index = [i for i in range(np.shape(Norms_of_r)[1]) if Norms_of_r[0,i] > tol]
        ########################################
        # preconditioning step
        # only generate new guess from unconverged residuals

        new_guess, Y = A_diag_preconditioner(residual[:,index], sub_eigenvalue[:k][index])
        # orthonormalize the new guesses against old guesses and put into V holder
        V, new_m = Gram_Schdmit_fill_holder(V, m, new_guess)
        W[:, m:new_m] = sTDA_fly(V[:, m:new_m])
        m = new_m
    ###########################################################################################
    full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])

    sTDA_D_end = time.time()
    sTDA_D = sTDA_D_end - sTDA_D_start
    print('sTDA A diagonalization:','threshold =', tol, '; in', i, 'steps ', round(sTDA_D, 4), 'seconds' )
    return full_guess
###########################################################################################





#############################################
# initial guesses
##########################################################################
def A_diag_initial_guess(m, V):
    # m is size of subspace A matrix, also is the amount of initial guesses
    sort = hdiag.argsort()
    for j in range(m):
        V[sort[j], j] = 1.0
    return V


def sTDA_initial_guess(m, V):
    #diagonalize sTDA_A amtrix
    V[:, :m] = Davidson0(m)
    return V
######################################################################################









#####################################################
def A_diag_preconditioner(arg1, arg2, arg3=None, arg4=None, arg5=None, arg6=None, arg7=None, arg8=None):
    """ residual[:,index], sub_eigenvalue[:k][index], current_dic, full_guess[:,index], index, W[:,:m], V[:,:m], sub_A """
    # preconditioners for each corresponding residual
    residual = arg1
    sub_eigenvalue = arg2
    current_dic = arg3

    k = np.shape(residual)[1]
    t = 1e-14

    D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_eigenvalue
    D = np.where( abs(D) < t, np.sign(D)*t, D) # force all values not in domain (-t, t)

    new_guess = residual/D

    return new_guess, current_dic
#######################################################


################################################################################
# original simple Davidson, just to solve eigenvalues and eigenkets of sTDA_A matrix
def Davidson0(k):
    sTDA_D_start = time.time()
    tol = args.initialTOL # Convergence tolerance
    max = 30
    #################################################
    # m is size of subspace
    m = min([k+8, 2*k, n])
    V = np.zeros((n, max*k + m))
    W = np.zeros_like(V)
    # positions of hdiag with lowest values set as 1

    V = A_diag_initial_guess(m, V)
    W[:, :m] = sTDA_fly(V[:, :m])
    # create transformed guess vectors

    #generate initial guess and put in holders V and W
    ###########################################################################################
    for i in range(0, max):
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        # Diagonalize the subspace Hamiltonian, and sorted.
        #sub_eigenvalue[:k] are smallest k eigenvalues
        residual = np.dot(W[:,:m], sub_eigenket[:,:k]) - np.dot(V[:,:m], sub_eigenket[:,:k] * sub_eigenvalue[:k])

        Norms_of_r = np.linalg.norm(residual, axis=0, keepdims = True)
        # largest residual norm
        max_norm = np.max(Norms_of_r)
        if max_norm < tol:
            break
        # index for unconverged residuals
        index = [i for i in range(np.shape(Norms_of_r)[1]) if Norms_of_r[0,i] > tol]
        ########################################
        # preconditioning step
        # only generate new guess from unconverged residuals

        new_guess, Y = A_diag_preconditioner(residual[:,index], sub_eigenvalue[:k][index])
        # orthonormalize the new guesses against old guesses and put into V holder
        V, new_m = Gram_Schdmit_fill_holder(V, m, new_guess)
        W[:, m:new_m] = sTDA_fly(V[:, m:new_m])
        m = new_m
    ###########################################################################################
    full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])

    sTDA_D_end = time.time()
    sTDA_D = sTDA_D_end - sTDA_D_start
    print('sTDA A diagonalization:','threshold =', tol, '; in', i, 'steps ', round(sTDA_D, 4), 'seconds' )
    return full_guess
###########################################################################################

##########################################################################
# we gonna use it anyway, just calcualte first and use it later
sTDA_A_eigenkets = Davidson0(min([args.nstates+8, 2*args.nstates, n]))
##########################################################################

###########################################################################################
def Qx(V, x):
    """ Qx = (1 - V*V.T)*x = x - V*V.T*x  """
    return x - einsum('ij, jk, kl -> il', V, V.T, x)

def on_the_fly_Hx(W, V, sub_A, x):
    """ on-the-fly compute H'x """
    """ H′ ≡ W*V.T + V*W.T − V*a*V.T + Q*K*Q"""
    """ K approximates H, here K = sTDA_A"""
    """ H′ ≡ W*V.T + V*W.T − V*a*V.T + (1-V*V.T)sTDA_A(1-V*V.T)"""
    """ H′x ≡ a + b − c + d """
    a = einsum('ij, jk, kl -> il', W, V.T, x)
    b = einsum('ij, jk, kl -> il', V, W.T, x)
    c = einsum('ij, jk, kl, lm -> im', V, sub_A, V.T, x)
    d = Qx(V, sTDA_fly(Qx(V, x)))
    Hx = a + b - c + d
    return Hx
###########################################################################################

###########################################################################################
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
    m = min([k+8, 2*k, n])
    # m is the amount of initial guesses
    V = np.zeros((n, max*k + m))
    W = np.zeros_like(V)
    # positions of hdiag with lowest values set as 1

    # sTDA as initial guess
    V[:,:m] = sTDA_A_eigenkets[:,:m]
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

        Norms_of_r = np.linalg.norm(residual, axis=0, keepdims = True)
#             print(Norms_of_r)
        # largest residual norm
        max_norm = np.max(Norms_of_r)
        if max_norm < tol:
            break
        # index for unconverged residuals
        index = [i for i in range(np.shape(Norms_of_r)[1]) if Norms_of_r[0,i] > tol]
        ########################################
        # preconditioning step
        # only generate new guess from unconverged residuals

        new_guess, Y = A_diag_preconditioner(residual[:,index], sub_eigenvalue[:k][index])
        # Y doesn't matter

        # orthonormalize the new guesses against old guesses and put into V holder
        V, new_m = Gram_Schdmit_fill_holder(V, m, new_guess)
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
# a dictionary for initial guess and precodnitioner
i_key = ['sTDA', 'Adiag']
i_func = [sTDA_initial_guess, A_diag_initial_guess]
i_lib = dict(zip(i_key, i_func))

p_key = ['sTDA', 'Adiag', 'Jacobi', 'new_ES']
p_func = [sTDA_preconditioner, A_diag_preconditioner, Jacobi_preconditioner, new_ES]
p_lib = dict(zip(p_key, p_func))
################################################################################


################################################################################
# Real Davidson frame, where we can choose different initial guess and preconditioner
def Davidson(k, tol, i, p):
    D_start = time.time()
    Davidson_dic = {}
    Davidson_dic['initial guess'] = i
    Davidson_dic['preconditioner'] = p
    Davidson_dic['nstate'] = k
    Davidson_dic['molecule'] = basename
    Davidson_dic['method'] = args.method
    Davidson_dic['functional'] = args.functional
    Davidson_dic['threshold'] = tol
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    initial_guess = i_lib[i]
    new_guess_generator = p_lib[p]

    print('Initial guess:  ', i)
    print('Preconditioner: ', p)
    print('A matrix size = ', n,'*', n)
    max = 30
    # Maximum number of iterations

    m = min([k + args.extrainitial, 2*k, n])

    #################################################
    # generate initial guess

    V = np.zeros((n, max*k + m))
    W = np.zeros_like(V)

    init_start = time.time()
    V = initial_guess(m, V)
    init_end = time.time()
    init_time = init_end - init_start

    print('Intial guess time:', round(init_time, 4), 'seconds')
    #generate initial guess and put in holders V and W
    # m is size of subspace

    # W = Av, create transformed guess vectors
    W[:, :m] = matrix_vector(V[:, :m])

    # time cost for preconditioning
    Pcost = 0
    ###########################################################################################
    for ii in range(0, max):
        print('Davidson', ii)

        # sub_A is subspace A matrix
        sub_A = np.dot(V[:,:m].T, W[:,:m])

        print('subspace size: ', np.shape(sub_A)[0])

        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        # Diagonalize the subspace Hamiltonian, and sorted.
        #sub_eigenvalue[:k] are smallest k eigenvalues
        full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])

        residual = np.dot(W[:,:m], sub_eigenket[:,:k]) - full_guess * sub_eigenvalue[:k]

        Norms_of_r = np.linalg.norm(residual, axis=0, keepdims = True)

        # largest residual norm
        max_norm = np.max(Norms_of_r)

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['Davidosn residual norms'] = Norms_of_r[0,:].tolist()

        print('checking the residual norm')
        if max_norm < tol:
            print('All guesses converged!')
            break

        # index for unconverged residuals
        index = [i for i in range(np.shape(Norms_of_r)[1]) if Norms_of_r[0,i] > tol]

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
        V, new_m = Gram_Schdmit_fill_holder(V, m, new_guess)
        W[:, m:new_m] = matrix_vector(V[:, m:new_m])
        print('new generated guesses:', new_m - m)
        m = new_m

    D_end = time.time()
    Dcost = D_end - D_start
    Davidson_dic['SCF time'] = kernel_t
    Davidson_dic['Initial guess time'] = init_time
    Davidson_dic['sTDA initial guess threshold'] = args.initialTOL
    Davidson_dic['New guess generating time'] = Pcost
    Davidson_dic['sTDA preconditioner threshold'] = args.precondTOL
    Davidson_dic['Davidson time'] = Dcost
    Davidson_dic['iterations'] = ii+1
    Davidson_dic['A matrix size'] = n
    Davidson_dic['final subspace size'] = np.shape(sub_A)[0]
    Davidson_dic['excitation energy(eV)'] = (sub_eigenvalue[:k]*27.211386245988).tolist()
    ###########################################################################################
    if ii == max-1:
        print('============ Davidson Failed Due to Iteration Limit ==============')
        print('Davidson failed after ', round(Dcost, 4), 'seconds')
        print('current residual norms', Norms_of_r)
        print('max_norm = ', max_norm)
    else:
        print('Davidson done after ', round(Dcost, 4), 'seconds')
        print('Total steps =', ii+1)
        print('Final subspace shape = ', np.shape(sub_A))

    print('Preconditioning time:', round(Pcost, 4), 'seconds')
    return sub_eigenvalue[:k]*27.211386245988, full_guess, Davidson_dic
################################################################################


if args.compare == True:
    print('-----------------------------------------------------------------')
    print('|--------------------    PySCF TDA-TDDFT    ---------------------|')
    td.nstates = args.nstates
    td.conv_tol = 1e-10
    td.verbose = 5
    start = time.time()
    td.kernel()
    end = time.time()
    pyscf_time = end-start
    print('Built-in Davidson time:', round(pyscf_time, 4), 'seconds')
    print('|---------------------------------------------------------------|')


combo = [            # option
['sTDA','sTDA'],     # 0
['Adiag','Adiag'],   # 1
['Adiag','sTDA'],    # 2
['sTDA','Adiag'],    # 3
['sTDA','Jacobi'],   # 4
['Adiag','Jacobi'],  # 5
['Adiag','new_ES'],  # 6
['sTDA','new_ES']]   # 7

for option in args.options:
    i,p = combo[option]
    print('-------------------------------------------------------------------')
    print('|---------------   In-house Developed Davidson Starts   -----------|')
    print('Residual convergence threshold =', args.tolerance)
    print('Number of excited states =', args.nstates)

    total_start = time.time()
    Excitation_energies, eigenkets, Davidson_dic = Davidson (args.nstates, args.tolerance, i, p)
    total_end = time.time()
    total_time = total_end - total_start

    print('In-house Davidson time:', round(total_time, 4), 'seconds')

    print('Excited State energies (eV) =')
    print(Excitation_energies)

    curpath = os.getcwd()
    yamlpath = os.path.join(curpath, basename + '_i_' + i + '_p_'+ p + '.yaml')

    with open(yamlpath, "w", encoding="utf-8") as f:
        yaml.dump(Davidson_dic, f)

    print('|---------------   In-house Developed Davidson Done   -----------|')
