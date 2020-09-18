import time
import numpy as np
from opt_einsum import contract as einsum
import pyscf
from pyscf import gto, scf, dft, tddft, data, lib
import argparse
import os
import yaml
from pyscf.tools import molden

lib.num_threads(1)
print ('lib.num_threads() = ', lib.num_threads())

parser = argparse.ArgumentParser(description='Davidson')
parser.add_argument('-c', '--filename',       type=str, default='*.xyz*', help='input filename')
parser.add_argument('-M', '--molden',         type=bool,default= False, help='load molden file, rather than calcule MOs')
parser.add_argument('-m', '--method',         type=str, default='RHF', help='RHF RKS UHF UKS')
parser.add_argument('-f', '--functional',     type=str, default='b3lyp', help='xc functional')
parser.add_argument('-b', '--basis_set',      type=str, default='def2-SVP', help='basis sets')
parser.add_argument('-g', '--grid_level',     type=int, default='3', help='0-9, 9 is best')
parser.add_argument('-i', '--initial_guess',  type=str, default='sTDA', help='initial_guess: Adiag or sTDA')
parser.add_argument('-p', '--preconditioner', type=str, default='sTDA', help='preconditioner: diag_A or sTDA_A')
parser.add_argument('-t', '--tolerance',      type=float, default= 1e-5, help='residual norm convergence threshold')
parser.add_argument('-n', '--nstates',        type=int, default= 4, help='number of excited states')
parser.add_argument('-C', '--compare',        type=bool, default = False , help='whether to compare with PySCF TDA-TDDFT')
args = parser.parse_args()
################################################
# read xyz file and delete its first two lines
if args.molden == False:
    f = open(args.filename)
    atom_coordinates = f.readlines()
    del atom_coordinates[:2]
    ###########################################################################
    # build geometry in PySCF
    mol = gto.Mole()
    mol.atom = atom_coordinates
    mol.basis = args.basis_set
    mol.max_memory = 1000
    mol.build(parse_arg = False)
    ###########################################################################
    ###################################################
    #DFT or UF?
    if args.method == 'RKS':
        mf = dft.RKS(mol)
        mf.xc = args.functional
        mf.grids.level = args.grid_level
        # 0-9, big number for large mesh grids, default is 3
    elif args.method == 'UKS':
        mf = dft.UKS(mol)
        mf.xc = args.functional
        mf.grids.level = args.grid_level

    elif args.method == 'RHF':
        mf = scf.RHF(mol)
    elif args.method == 'UHF':
        mf = scf.UHF(mol)

    mf.conv_tol = 1e-12
    print ('Molecule built')
    print ('Calculating SCF Energy...')
    kernel_0 = time.time()
    mf.kernel()
    kernel_1 = time.time()
    kernel = round (kernel_1 - kernel_0, 4)
    print ('SCF Done after ', kernel, 'seconds')
    # pyscf.tools.molden.from_mo(mol, molden, mo_coeff, spin=’Alpha’, symm=None, ene=None, occ=None, ignore_h=True)
    #single point energy
    ####################################################
else:
    pyscf.tools.molden.read('adpbo.molden', verbose=0)
    print ('Molden file loaded')
################################################







########################################################################
# Collect everything needed from PySCF
Qstart = time.time()
# extract vind() function
td = tddft.TDA(mf)
vind, hdiag = td.gen_vind(mf)


Natm = mol.natm
MOe = mf.mo_energy
#an array of MO energies, in Hartree

occupied = len(np.where(mf.mo_occ > 0)[0])
#mf.mo_occ is an array of occupance [2,2,2,2,2,0,0,0,0.....]
virtual = len(np.where(mf.mo_occ == 0)[0])

AO = [int(i.split(' ',1)[0]) for i in mol.ao_labels()]
# .split(' ',1) is to split each element by space, split once.
# mol.ao_labels() it is Labels of AO basis functions
# AO is a list of corresponding atom_id


N_bf = len(AO)
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
def Hardness (atom_id):
    atom = mol.atom_pure_symbol(atom_id)
    return HARDNESS[atom]
# mol.atom_pure_symbol(atom_id) returns pure element symbol, no special characters


########################################################################
# This block is the function to produce orthonormalized coefficient matrix C
def matrix_power (S,a):
    s,ket = np.linalg.eigh(S)
    s = s**a
    X = np.linalg.multi_dot([ket,np.diag(s),ket.T])
    #X == S^1/2
    return X

def orthonormalize (C):
    X = matrix_power(mf.get_ovlp(), 0.5)
    # S = mf.get_ovlp() #.get_ovlp() is basis overlap matrix
    # S = np.dot(np.linalg.inv(c.T), np.linalg.inv(c))
    C = np.dot(X,C)
    return C

C = mf.mo_coeff
# mf.mo_coeff is the coefficient matrix
C = orthonormalize (C)
# C is orthonormalized coefficient matrix
# np.dot(C.T,C) is a an identity matrix
########################################################################
Functionals = [
'lc-b3lyp',
'wb97',
'wb97x',
'wb97x-d3',
'cam-b3lyp',
'b3lyp']

parameters = [
[0.53, 8.00, 4.50],
[0.61, 8.00, 4.41],
[0.56, 8.00, 4.58],
[0.51, 8.00, 4.51],
[0.38, 1.86, 0.90],
[0.56, 8.00, 4.58]]
Functionals_parameters = dict(zip(Functionals, parameters))

a_x, beta, alpha = Functionals_parameters[args.functional]

# creat \eta matrix
a = [Hardness (atom_id) for atom_id in range (Natm)]
a = np.asarray(a).reshape(1,-1)
eta = (a+a.T)/2

# creat GammaK and GammaK matrix
GammaJ = (R**beta + (a_x * eta)**(-beta))**(-1/beta)
GammaK = (R**alpha + eta**(-alpha)) **(-1/alpha)


Natm = mol.natm
def generateQ ():
    aoslice = mol.aoslice_by_atom()
    q = np.zeros([Natm, N_bf, N_bf])
    #N_bf is number Atomic orbitals, occupied+virtual, q is same size with C
    for atom_id in range (0, Natm):
        shst, shend, atstart, atend = aoslice[atom_id]
        q[atom_id,:, :] = np.dot(C[atstart:atend, :].T, C[atstart:atend, :])
    return q

q_tensors   = generateQ    ()
q_tensor_ij = q_tensors [:, :occupied,:occupied]
q_tensor_ab = q_tensors [:, occupied:,occupied:]
q_tensor_ia = q_tensors [:, :occupied,occupied:]

Q_K = einsum('Bjb, AB -> Ajb', q_tensor_ia, GammaK)
Q_J = einsum('Bab, AB -> Aab', q_tensor_ab, GammaJ)
# pre-calculate and store the Q-Gamma rank 3 tensor
Qend = time.time()

Q_time = Qend - Qstart
print ('Q-Gamma tensors building time =', round(Q_time, 4))
##################################################################################################


###################################################################################################
# This block is to define on-the-fly two electron intergeral (pq|rs)
# A_iajb * v = delta_ia_ia*v + 2(ia|jb)*v - (ij|ab)*v

# iajb_v = einsum('Aia, Bjb, AB, jbm -> iam', q_tensor_ia, q_tensor_ia, GammaK, V)
# ijab_v = einsum('Aij, Bab, AB, jbm -> iam', q_tensor_ij, q_tensor_ab, GammaJ, V)

def iajb_fly (V):
    V = V.reshape(occupied, virtual, -1)
    Q_K_V = einsum('Ajb, jbm -> Am', Q_K, V)
    iajb_V = einsum('Aia, Am -> iam', q_tensor_ia, Q_K_V).reshape(occupied*virtual, -1)

    return iajb_V

def ijab_fly (V):
    V = V.reshape(occupied, virtual, -1)
#     ijab_v = einsum('Aij, Aab, jbm -> iam', q_tensor_ij, Q_J,  V)

    # contract smaller index first
    # Aij_V = einsum('Aij, jbm -> Aibm', q_tensor_ij, V)
    # ijab_V = einsum('Aab, Aibm -> iam', Q_J, Aij_V).reshape(occupied*virtual, -1)

    # contract larger index first
    Aab_V = einsum('Aab, jbm -> jAam', Q_J, V)
    ijab_V = einsum('Aij, jAam -> iam', q_tensor_ij, Aab_V).reshape(occupied*virtual, -1)

    return ijab_V

delta_diag_A = hdiag.reshape(occupied, virtual)



def delta_fly (V):
    V = V.reshape(occupied, virtual, -1)
    delta_v = einsum('ia,iam -> iam', delta_diag_A, V).reshape(occupied*virtual, -1)
    return delta_v

def sTDA_fly (V):
    V = V.reshape(occupied*virtual,-1)
    # this feature can deal with multiple vectors
    sTDA_V =  delta_fly (V) + 2*iajb_fly (V) - ijab_fly (V)
    return sTDA_V
###################################################################################################





##############################################################################################
# orthonormalization of guess_vectors
def Gram_Schdmit_bvec (A, bvec):
    # suppose A is orthonormalized
    projections_coeff = np.dot(A.T, bvec)
    bvec = bvec - np.dot(A, projections_coeff)
    return bvec

def Gram_Schdmit (A):
    # A matrix has J columns, orthonormalize each columns
    # unualified vectors will be removed
    N_rows = np.shape(A)[0]
    N_vectors = np.shape(A)[1]
    A = A/np.linalg.norm(A, axis=0, keepdims = True)

    B = np.zeros((N_rows,N_vectors))
    count = 0
    ############b
    for j in range (0, N_vectors):
        bvec = Gram_Schdmit_bvec (B[:, :count], A[:, j])
        norm = np.linalg.norm(bvec)
        if norm > 1e-14:
            B[:, count] = bvec/np.linalg.norm(bvec)
            count +=1
    return B[:, :count]

def Gram_Schdmit_fill_holder (V, count, vecs):
    # V is a vectors holder
    # count is the amount of vectors that already sit in the holder

    nvec = np.shape(vecs)[1]
    # amount of new vectors intended to fill in the V

    # count will be final amount of vectors in V
    for j in range (0, nvec):
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
def check_orthonormal (A):
    n = np.shape(A)[1]
    B = np.dot (A.T, A)
    c = np.linalg.norm(B - np.eye(n))
    return c
####################################################################


########################################################################
def solve_AX_Xla_B (sub_A, eigen_lambda, sub_B):
    N_vectors = len(eigen_lambda)
    a, u = np.linalg.eigh(sub_A)
    ub = np.dot(u.T, sub_B)
    ux = np.zeros_like(sub_B)
    for k in range (N_vectors):
        ux[:, k] = ub[:, k]/(a - eigen_lambda[k])
    sub_guess = np.dot(u, ux)
    return sub_guess
#########################################################################

########################################################################
# sTDA preconditioner
def on_the_fly_sTDA_preconditioner (B, eigen_lambda, current_dic):
    # (sTDA_A - eigen_lambda*I)^-1 B = X
    # AX - X\lambda = B
    # columns in B are residuals (in Davidson's loop) to be preconditioned,
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
    W = np.zeros((N_rows, (max+1)*N_vectors))
    count = 0

    # now V and W are empty holders, 0 vectors
    # W = sTDA_fly(V)
    # count is the amount of vectors that already sit in the holder
    # in each iteration, V and W will be filled/updated with new guess vectors

    ###########################################
    #initial guess: (diag(A) - \lambda)^-1 B.
    # D is preconditioner for each state
    t = 1e-10
    D = np.repeat(hdiag.reshape(-1,1), N_vectors, axis=1) - eigen_lambda
    D= np.where( abs(D) < t, np.sign(D)*t, D) # <t: returns np.sign(D)*t; else: D
    inv_D = 1/D

    # generate initial guess
    init = B*inv_D
    V, new_count = Gram_Schdmit_fill_holder (V, count, init)
    W[:, count:new_count] = sTDA_fly(V[:, count:new_count])
    count = new_count

    current_dic['step'] = []
    ####################################################################################
    for i in range (0, max):
        sub_B = np.dot(V[:,:count].T, B)
        sub_A = np.dot(V[:,:count].T, W[:,:count])
        #project sTDA_A matrix and vector B into subspace

        # size of subspace
        m = np.shape(sub_A)[0]

        sub_guess = solve_AX_Xla_B(sub_A, eigen_lambda, sub_B)

        full_guess = np.dot(V[:,:count], sub_guess)
        residual = np.dot(W[:,:count], sub_guess) - full_guess*eigen_lambda - B

        Norms_of_r = np.linalg.norm (residual, axis=0, keepdims = False)

        max_norm = np.max(Norms_of_r)

        if max_norm < tol:
            break

        # index for unconverged residuals
        index = [i for i in range(len(Norms_of_r)) if Norms_of_r[i] > tol]


        current_dic['step'].append({'r_norms': Norms_of_r.tolist()})
        current_dic['step'].append({'unconverged_r': len(index)})

        # preconditioning step
        # only generate new guess from unconverged residuals
        new_guess = residual[:,index]*inv_D[:,index]

        V, new_count = Gram_Schdmit_fill_holder (V, count, new_guess)
        W[:, count:new_count] = sTDA_fly(V[:, count:new_count])
        count = new_count

        # V_orthonormality = check_orthonormal(V[:,:count])
        # current_dic['step' + str(i)]['V_orthonormality'] = float(V_orthonormality)

    precondition_end = time.time()
    precondition_time = precondition_end - precondition_start
    if i == (max -1):
        print ('_________________ sTDA Preconditioner Failed Due to Iteration Limmit _________________')
        print ('sTDA preconditioning failed after ', i, 'steps; ', round(precondition_time, 4), 'seconds')
        print ('current residual norms', Norms_of_r)
        print ('max_norm = ', max_norm)
        print ('orthonormality of V', check_orthonormal(V[:,:count]))
    else:
        print ('sTDA Preconditioning Done after ', i, 'steps; ', round(precondition_time, 4), 'seconds')

    return (full_guess*bnorm, current_dic)
###########################################################################################



#############################################
# framework of Davidson's Algorithms
###############################################################################
n = occupied*virtual

def A_diag_initial_guess (k, V):
    # m is size of subspace A matrix, also is the amount of initial guesses
    # m = min([2*k, k+8, occupied*virtual])
    m = k
    sort = hdiag.argsort()
    for j in range(m):
        V[sort[j], j] = 1.0

    return (m, V)

def sTDA_initial_guess (k, V):
    m = k
    #diagonalize sTDA_A amtrix
    V[:, :m] = Davidson0(m)

    return (m, V)
######################################################################################

#####################################################
def A_diag_preconditioner (residual, sub_eigenvalue, current_dic):
    # preconditioners for each corresponding residual
    k = np.shape(residual)[1]

    t = 1e-14

    D = np.repeat(hdiag.reshape(-1,1), k, axis=1) - sub_eigenvalue
    D = np.where( abs(D) < t, np.sign(D)*t, D) # force all values not in domain (-t, t)

    new_guess = residual/D

    return new_guess, current_dic
#######################################################

################################################################################
# original simple Davidson, just to solve eigenvalues and eigenkets of sTDA_A matrix
def Davidson0 (k):

    sTDA_D_start = time.time()
    tol = 1e-2 # Convergence tolerance

    max = 30
    #################################################
    # m is size of subspace
    m = min([2*k, k+8, occupied*virtual])
    V = np.zeros((n, 30*k))
    W = np.zeros((n, 30*k))
    # positions of hdiag with lowest values set as 1

    m, V = A_diag_initial_guess(k, V)

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

        Norms_of_r = np.linalg.norm (residual, axis=0, keepdims = True)
        # largest residual norm
        max_norm = np.max(Norms_of_r)
        if max_norm < tol:
            break
        # index for unconverged residuals
        index = [i for i in range(np.shape(Norms_of_r)[1]) if Norms_of_r[0,i] > tol]
        ########################################
        # preconditioning step
        # only generate new guess from unconverged residuals
        Y = None
        new_guess, Y = A_diag_preconditioner (residual[:,index], sub_eigenvalue[:k][index], Y)
        # orthonormalize the new guesses against old guesses and put into V holder
        V, new_m = Gram_Schdmit_fill_holder (V, m, new_guess)
        W[:, m:new_m] = sTDA_fly (V[:, m:new_m])
        m = new_m
    ###########################################################################################
    full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])

    sTDA_D_end = time.time()
    sTDA_D = sTDA_D_end - sTDA_D_start
    print ('sTDA A diagonalization:','threshold =', tol, '; in', i, 'steps ', round(sTDA_D, 4), 'seconds' )
    return (full_guess)
###########################################################################################


################################################################################
# Real Davidson frame, where we can choose different initial guess and preconditioner
def Davidson (k, tol, i, p, Davidson_dic):
    Davidson_dic['nstate'] = k
    Davidson_dic['threshold'] = tol
    Davidson_dic['iteration'] = []
    iteration_list = Davidson_dic['iteration']

    if i == 'sTDA':
        initial_guess = sTDA_initial_guess
    elif i == 'Adiag':
        initial_guess = A_diag_initial_guess


    if p == 'sTDA':
        precondition = on_the_fly_sTDA_preconditioner
    elif p == 'Adiag':
        precondition = A_diag_preconditioner

    print ('Initial guess:  ', i)
    print ('Preconditioner: ', p)


    n = occupied*virtual
    print ('A matrix size = ', n)
    max = 50
    # Maximum number of iterations

    #################################################
    # generate initial guess

    V = np.zeros((n, (max+1)*k))
    W = np.zeros((n, (max+1)*k))
    # positions of hdiag with lowest values set as 1
    # hdiag is non-interacting A matrix

    init_start = time.time ()
    m, V = initial_guess(k, V)
    init_end = time.time ()
    init_time = init_end - init_start

    print ('Intial guess time:', round(init_time, 4), 'seconds')
    #generate initial guess and put in holders V and W
    # m is size of subspace

    # W = Av, create transformed guess vectors
    W[:, :m] = vind(V[:, :m].T).T

    # time cost for preconditioning
    Pcost = 0
    ###########################################################################################
    for ii in range(0, max):
        print ('Davidson', ii)

        # sub_A is subspace A matrix
        sub_A = np.dot(V[:,:m].T, W[:,:m])

        print ('subspace size: ', np.shape(sub_A)[0])

        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        # Diagonalize the subspace Hamiltonian, and sorted.
        #sub_eigenvalue[:k] are smallest k eigenvalues
        full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])

        residual = np.dot(W[:,:m], sub_eigenket[:,:k]) - full_guess * sub_eigenvalue[:k]

        Norms_of_r = np.linalg.norm (residual, axis=0, keepdims = True)

        # largest residual norm
        max_norm = np.max(Norms_of_r)

        if max_norm < tol:
            print ('All guesses converged!')
            break

        # index for unconverged residuals
        index = [i for i in range(np.shape(Norms_of_r)[1]) if Norms_of_r[0,i] > tol]

        iteration_list.append({})
        current_dic = iteration_list[ii]
        current_dic['total_residuals'] = len(index)
        ########################################
        # preconditioning step
        # only generate new guess from unconverged residuals
        P_start = time.time()
        new_guess, current_dic = precondition (residual[:,index], sub_eigenvalue[:k][index], current_dic)
        P_end = time.time()

        iteration_list[ii] = current_dic

        Pcost += P_end - P_start

        # orthonormalize the new guesses against old guesses and put into V holder
        V, new_m = Gram_Schdmit_fill_holder (V, m, new_guess)
        W[:, m:new_m] = vind (V[:, m:new_m].T).T
        print ('preconditioned guesses:', new_m - m)
        m = new_m

    ###########################################################################################
    if ii == (max -1):
        print ('============ Davidson Failed Due to Iteration Limmit ==============')
        print ('Davidson failed after ', ii, 'steps; ', round(Pcost, 4), 'seconds')
        print ('current residual norms', Norms_of_r)
        print ('max_norm = ', max_norm)
    else:
        print ('Davidson done after ', ii, 'steps; ', round(Pcost, 4), 'seconds')
        print ('Total steps =', ii+1)
        print ('Final subspace size = ', np.shape(sub_A))
        print ('Preconditioning time:', round(Pcost, 4), 'seconds')

    return (sub_eigenvalue[:k]*27.21138624598853, full_guess)
################################################################################







# print ('-------------------------------------------------------------------')
# print ('|---------------   In-house Developed Davidson Starts   -----------|')
# print ('Residual convergence threshold =', args.tolerance)
# print ('Number of excited states =', args.nstates)

# total_start = time.time()
# Davidson_dic = {}
# Excitation_energies, kets = Davidson (args.nstates, args.tolerance, args.initial_guess, args.preconditioner, Davidson_dic)
# total_end = time.time()
# total_time = total_end - total_start
# if args.initial_guess == 'Adiag' and args.preconditioner == 'Adiag':
#     print ('In-house Davidson time:', round(total_time - Q_time, 4), 'seconds')
# else:
#     print ('In-house Davidson time:', round(total_time, 4), 'seconds')
# print ('Excited State energies (eV) =')
# print (Excitation_energies)
#
# curpath = os.path.dirname(os.path.realpath(__file__))
# yamlpath = os.path.join(curpath, 'i_'+str(args.initial_guess) + '_p_'+str(args.preconditioner)+ '.yaml')
#
# with open(yamlpath, "w", encoding="utf-8") as f:
#     yaml.dump(Davidson_dic, f)
#
# print ('|---------------   In-house Developed Davidson Done   -----------|')


if args.compare == True:
    print ('-----------------------------------------------------------------')
    print ('|--------------------    PySCF TDA-TDDFT    ---------------------|')
    td.nstates = args.nstates
    td.conv_tol = 1e-10
    td.verbose = 5
    start = time.time()
    td.kernel()
    end = time.time()
    pyscf_time = end-start
    print ('Built-in Davidson time:', round(pyscf_time, 4), 'seconds')
    print ('|---------------------------------------------------------------|')


option = ['sTDA', 'Adiag']
for i in option:
    for p in option:
        print ('-------------------------------------------------------------------')
        print ('|---------------   In-house Developed Davidson Starts   -----------|')
        print ('Residual convergence threshold =', args.tolerance)
        print ('Number of excited states =', args.nstates)

        total_start = time.time()
        Davidson_dic = {}
        Excitation_energies, kets = Davidson (args.nstates, args.tolerance, i, p, Davidson_dic)
        total_end = time.time()
        total_time = total_end - total_start

        print ('In-house Davidson time:', round(total_time, 4), 'seconds')

        print ('Excited State energies (eV) =')
        print (Excitation_energies)

        curpath = os.path.dirname(os.path.realpath(__file__))
        yamlpath = os.path.join(curpath, 'i_'+ i + '_p_'+ p + '.yaml')

        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(Davidson_dic, f)

        print ('|---------------   In-house Developed Davidson Done   -----------|')
