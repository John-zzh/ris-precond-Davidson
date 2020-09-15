import time
import numpy as np
import pyscf
from pyscf import gto, scf, dft, tddft, data
import argparse
import scipy
import os
import ruamel
from ruamel import yaml

parser = argparse.ArgumentParser(description='Davidson')

parser.add_argument('-c', '--filename',       type=str, default='methanol.xyz', help='input filename')
parser.add_argument('-m', '--method',         type=str, default='RHF', help='RHF RKS UHF UKS')
parser.add_argument('-f', '--functional',     type=str, default='b3lyp', help='xc functional')
parser.add_argument('-b', '--basis_set',      type=str, default='def2-SVP', help='basis sets')
parser.add_argument('-g', '--grid_level',     type=int, default='3', help='0-9, 9 is best')
parser.add_argument('-i', '--initial_guess',  type=str, default='sTDA', help='initial_guess: diag_A or sTDA_A')
parser.add_argument('-p', '--preconditioner', type=str, default='sTDA', help='preconditioner: diag_A or sTDA_A')
parser.add_argument('-t', '--tolerance',      type=float, default= 1e-5, help='residual norm convergence threshold')
parser.add_argument('-n', '--nstates',        type=int, default= 4 , help='number of excited states')
args = parser.parse_args()
################################################
# read xyz file and delete its first two lines
f = open(args.filename)
atom_coordinates = f.readlines()
del atom_coordinates[:2]
################################################


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
mf.kernel()
#single point energy
####################################################


########################################################################
# Collect everything needed from PySCF
Natm = mol.natm
MOe = mf.mo_energy
#an array of MO energies, in Hartree

occupied = len(np.where(mf.mo_occ > 0)[0])
#mf.mo_occ is an array of occupance [2,2,2,2,2,0,0,0,0.....]
virtual = len(np.where(mf.mo_occ == 0)[0])

AO = [int(i.split(' ',1)[0]) for i in mol.ao_labels()]
# .split(' ',1) is to split each element by space, split once.
# mol.ao_labels() it is Labels of AO basis functions, AO is a list of corresponding atom_id
# AO == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

N_bf = len(AO)
R = pyscf.gto.mole.inter_distance(mol, coords=None)
#Inter-particle distance array
# unit == ’Bohr’, Its value is 5.29177210903(80)×10^(−11) m
########################################################################

total_start = time.time()
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
def eta (atom_A_id, atom_B_id):
    eta = (Hardness(atom_A_id) + Hardness(atom_B_id))/2
    return eta
R = pyscf.gto.mole.inter_distance(mol, coords=None)
#Inter-particle distance array
# unit == ’Bohr’, Its value is 5.29177210903(80)×10^(−11) m
start = time.time()

###############################
# extract vind() function
td = tddft.TDA(mf)
vind, hdiag = td.gen_vind(mf)
###############################


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
    # X = S^1/2
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
GammaJ = np.zeros([Natm, Natm])
for i in range (0, Natm):
    for j in range (0, Natm):
        GammaJ[i,j] = (R[i, j]**beta + (a_x * eta(i, j))**(-beta))**(-1/beta)

GammaK = np.zeros([Natm, Natm])
for i in range (0, Natm):
    for j in range (0, Natm):
        GammaK[i,j] = (R[i, j]**alpha + eta(i, j)**(-alpha)) **(-1/alpha)

Natm = mol.natm
def generateQ ():
    q = np.zeros([Natm, N_bf, N_bf])
    #N_bf is number Atomic orbitals, occupied+virtual, q is same size with C
    for atom_id in range (0, Natm):
        for i in range (0, N_bf):
            for p in range (0, N_bf):
                for mu in range (0, N_bf):
                    if AO[mu] == atom_id:
                        #collect all basis functions centered on atom_id
                        # the last loop is to sum up all C_mui*C_mup, calculate element q[i,p]
                        q[atom_id,i,p] += C[mu,i]*C[mu,p]
                        #q[i,p] += 2*C[i,mu]*C[p,mu]
    return q

q_tensors   = generateQ    ()
q_tensor_ij = q_tensors [:, :occupied,:occupied]
q_tensor_ab = q_tensors [:, occupied:,occupied:]
q_tensor_ia = q_tensors [:, :occupied,occupied:]

Q_K = np.einsum('Bjb, AB -> Ajb', q_tensor_ia, GammaK)
Q_J = np.einsum('Bab, AB -> Aab', q_tensor_ab, GammaJ)
# pre-calculate and store the Q-Gamma rank 3 tensor
end = time.time()

Q_gamma_tensors_building_time = end - start
print ('Q-Gamma tensors building time =', Q_gamma_tensors_building_time)
##################################################################################################

######################################
# set parapeters
# if  args.functional == 'lc-b3lyp':
#     a_x = 0.53
#     beta= 8.00
#     alpha1= 4.50
# elif args.functional == 'wb97':
#     a_x = 0.61
#     beta= 8.00
#     alpha= 4.41
# elif args.functional == 'wb97x':
#     a_x = 0.56
#     beta= 8.00
#     alpha= 4.58
# elif args.functional == 'wb97x-d3':
#     a_x = 0.51
#     beta= 8.00
#     alpha= 4.51
# elif args.functional == 'cam-b3lyp':
#     a_x = 0.38
#     beta= 1.86
#     alpha= 0.90
# else:
#     a_x = 0.56
#     beta= 8.00
#     alpha= 4.58


######################################





###################################################################################################
# This block is to define on-the-fly two electron intergeral (pq|rs)
# A_iajb * v = delta_ia_ia*v + 2(ia|jb)*v - (ij|ab)*v

# iajb_v = np.einsum('Aia, Bjb, AB, jbm -> iam', q_tensor_ia, q_tensor_ia, GammaK, V)
# ijab_v = np.einsum('Aij, Bab, AB, jbm -> iam', q_tensor_ij, q_tensor_ab, GammaJ, V)

def iajb_fly (V):
    V = V.reshape(occupied, virtual, -1)
    Q_K_V = np.einsum('Ajb, jbm -> Am', Q_K, V)
    iajb_V = np.einsum('Aia, Am -> iam', q_tensor_ia, Q_K_V).reshape(occupied*virtual, -1)

    return iajb_V

def ijab_fly (V):
    V = V.reshape(occupied, virtual, -1)
#     ijab_v = np.einsum('Aij, Aab, jbm -> iam', q_tensor_ij, Q_J,  V)

    # contract smaller index first
    # Aij_V = np.einsum('Aij, jbm -> Aibm', q_tensor_ij, V)
    # ijab_V = np.einsum('Aab, Aibm -> iam', Q_J, Aij_V).reshape(occupied*virtual, -1)

    # contract larger index first
    Aab_V = np.einsum('Aab, jbm -> jAam', Q_J, V)
    ijab_V = np.einsum('Aij, jAam -> iam', q_tensor_ij, Aab_V).reshape(occupied*virtual, -1)
    return ijab_V

# delta_diag_A = np.zeros((occupied, virtual))
# for i in range (0, occupied):
#     for a in range (0, virtual):
#         delta_diag_A[i,a] = (MOe[occupied+a] - MOe[i])
delta_diag_A = hdiag.reshape(occupied, virtual)



def delta_fly (V):
    V = V.reshape(occupied, virtual, -1)
    #delta_v = np.einsum('ij,ab,ia,jb -> ia',delta_ij,delta_ab,delta_diag_A, v)
    delta_v = np.einsum('ia,iam -> iam', delta_diag_A, V).reshape(occupied*virtual, -1)
    return delta_v

def sTDA_fly (V):
    V = V.reshape(occupied*virtual,-1)
    # -1 means shape of first dimension is not asigned, but can be inferred with the rest dimension
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
    # B[:,0] = A[:,0]
    count = 0

    ############bug!!!!!!
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
    # vecs = Gram_Schdmit(vecs)
    nvec = np.shape(vecs)[1]
    # amount of new vectors intended to fill in the V

    # count will be final amount of vectors in V
    for j in range (0, nvec):
        vec = vecs[:, j]

        #bug!!!!!!!!
        vec = Gram_Schdmit_bvec(V[:, :count], vec)   #single orthonormalize
        vec = Gram_Schdmit_bvec(V[:, :count], vec) #double orthonormalize
#         print ('shape of V[:, i:]', np.shape(V[:, :i]))
        norm = np.linalg.norm(vec)
#         print ('norm =', norm)
        if  norm > 1e-14:
            vec = vec/norm
            V[:, count] = vec
            count += 1
#
#             print ('count =', count)
    new_count = count
#     print ('norms of V =', np.linalg.norm(V, axis=0, keepdims = True))
#     print ('norms of W =', np.linalg.norm(W, axis=0, keepdims = True))
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

################################################################################
# original Davidson, to solve eigenvalues and eigenkets of sTDA_A matrix
def Davidson0 (k):
    tol = 1e-5 # Convergence tolerance
    n = occupied*virtual # size of sTDA_A matrix
    max = 90

    #################################################
    # generate initial guess
    # m is size of subspace
    m = min([2*k, k+8, occupied*virtual])
    # a container to hold guess vectors
    V = np.zeros((n, 30*k))
    W = np.zeros((n, 30*k))

    # positions of hdiag with lowest values set as 1
    # hdiag is non-interactiong A matrix
    sort = hdiag.argsort()
    for j in range(0,m):
        V[int(np.argwhere(sort == j)), j] = 1

    W[:, :m] = vind(V[:, :m].T).T
    #generate initial guess and put in holders V and W


    ###########################################################################################
    for i in range(0, max):
        # sub_A is subspace A matrix
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
        new_guess = A_diag_preconditioner (residual[:,index], sub_eigenvalue[:k][index], Y=None)

        # orthonormalize the new guesses against old guesses
        # and put into V holder
        V, new_m = Gram_Schdmit_fill_holder (V, m, new_guess)
        W[:, m:new_m] = sTDA_fly (V[:, m:new_m])
        m = new_m
        #########################################################################################

    full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])

    print ('Iteration steps =', i+1)
    print ('Final subspace size = ', np.shape(sub_A))
    # print ('Davidson time:', round(end-start,4))

    return (full_guess)
###########################################################################################

########################################################################
def solve_AX_Xla_B (sub_A, eigen_lambda, sub_B):
    # m = np.shape(sub_A)[0]
    # I = np.eye(m)
    # N_vectors = len(eigen_lambda)
    # X = np.zeros((m, N_vectors))
    # for i in range (0, N_vectors):
    #     X[:, i] = np.linalg.solve (sub_A - eigen_lambda[i]*I, sub_B[:,i])
    # return X
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
    Lambda = np.diag(eigen_lambda)
    #     print ('shape of Lambda',np.shape(Lambda))
    N_rows = np.shape(B)[0]
    B = B.reshape(N_rows, -1)
    N_vectors = np.shape(B)[1]
    current_dic['amount of residual to be preconditioned'] = N_vectors
#     print ('n_residuals: ', N_vectors)
    #number of vectors to be preconditioned
    bnorm = np.linalg.norm(B, axis=0, keepdims = True)
    #norm of each vectors in B, shape (1,-1)
    B = B/bnorm
#     print ('shape of B=', np.shape(B))
    start = time.time()
    tol = 1e-2    # Convergence tolerance
    max = 50   # Maximum number of iterations

    V = np.zeros((N_rows, (max+1)*N_vectors))
    W = np.zeros((N_rows, (max+1)*N_vectors))
    count = 0

    # now V and W are empty holders, 0 vectors
    # W = sTDA_fly(V)
    # count is the amount of vectors that already sit in the holder
    # at the end of each iteration, V and W will be filled/updated with new guess vectors

    ###########################################
    #initial guess: (diag(A) - \lambda)^-1 B.

    diag = delta_diag_A.flatten()
#     # delta_diag_A.flatten() is (\spsilon_a-\spsilon_i)

    # D is preconditioner for each state
    t = 1e-10
    D = np.zeros((N_rows, N_vectors))
    for i in range (0, N_vectors):
        D[:,i] = diag - eigen_lambda[i]
        D[:,i][(D[:,i] < t)&(D[:,i] >= 0)] = t
        D[:,i][(D[:,i] > -t)&(D[:,i] < 0)] = -t

    # generate initial guess
    init = B/D
    V, new_count = Gram_Schdmit_fill_holder (V, count, init)
    W[:, count:new_count] = sTDA_fly(V[:, count:new_count])
    count = new_count


    # init = Gram_Schdmit(init)
    # count = np.shape(init)[1]
    # V[:, :count] = init
    # W[:, :count] = sTDA_fly(V[:, :count])



    #########################################
    # init = Gram_Schdmit(init)
    # residual = sTDA_fly(init) - init*eigen_lambda - B
    # Norms_of_r = np.linalg.norm (residual, axis=0, keepdims = True)
    # index = [i for i in range(np.shape(Norms_of_r)[1]) if Norms_of_r[0,i] > tol]
    # new_guess = residual[:,index]/D[:,index]
    #
    # V, count = Gram_Schdmit_fill_holder (V, 0, new_guess)
    # W[:, :count] = sTDA_fly(V[:, :count])
    ####################################################################################

    for i in range (0, max):
#         print ('Iteration =', i)
        sub_B = np.dot(V[:,:count].T, B)
        sub_A = np.dot(V[:,:count].T, W[:,:count])
        #project sTDA_A matrix and vector B into subspace

        # size of subspace
        m = np.shape(sub_A)[0]

        # solve sub_A * X + X * (-Lambad) = sub_B
        # sub_guess = scipy.linalg.solve_sylvester(sub_A, - Lambda, sub_B)

        sub_guess = solve_AX_Xla_B(sub_A, eigen_lambda, sub_B)

        full_guess = np.dot(V[:,:count], sub_guess)
        residual = np.dot(W[:,:count], sub_guess) - full_guess*eigen_lambda - B
        # print ('shape of residual =', np.shape(residual))
        Norms_of_r = np.linalg.norm (residual, axis=0, keepdims = True)
        # print ('shape of Norms_of_r =', np.shape(Norms_of_r))
        if i == 0:
            initial_residual = Norms_of_r

        max_norm = np.max(Norms_of_r)

        if max_norm < tol:
            break

        # index for unconverged residuals
        index = [i for i in range(np.shape(Norms_of_r)[1]) if Norms_of_r[0,i] > tol]


        current_dic['step' + str(i)] = {}


        current_dic['step' + str(i)]['r_norms'] = str(Norms_of_r)

        current_dic['step' + str(i)]['index of unconverged residual'] = str(index)
        current_dic['step' + str(i)]['amount of unconverged residual'] = len(index)



        # preconditioning step
        # only generate new guess from unconverged residuals

        ## bug!!!!!!!
        # new_guess = (residual[:,index] + B[:,index])/D[:,index]
        new_guess = residual[:,index]/D[:,index]


        V, new_count = Gram_Schdmit_fill_holder (V, count, new_guess)
        W[:, count:new_count] = sTDA_fly(V[:, count:new_count])
        count = new_count

        V_orthonormality = check_orthonormal(V[:,:count])



        current_dic['step' + str(i)]['V_orthonormality'] = float(V_orthonormality)
        # data_dic['V_orthonormality in iteration ' + str(i)] = str(V_orthonormality)
        # print (m, count)
        # if V_orthonormality > 1e-5:
        #     print ('Warning! Orthonormalily of V breakes down after ',i, ' steps')
        #     print ('initial residual norms', initial_residual)
        #     print ('current residual norms', Norms_of_r)
        #     break

#         # an awful backup plan in case of V_orthonormality > 1e-5
#           # orthonormalize all existing guess_vectors
#         if i%10 == 0 and i!= 0:
#             V[:,:count] = Gram_Schdmit(V[:,:count])
    # print (data_dic)
    precondition_end = time.time()
    precondition_time = precondition_end - precondition_start
    if i == (max -1):
        print ('============sTDA preconditioner Failed due to iteration limmit==============')
        print ('sTDA preconditioning failed after ', i, 'steps; ', precondition_time, 'seconds')
        print ('initial residual norms', initial_residual)
        print ('current residual norms', Norms_of_r)
        print ('max_norm = ', max_norm)
        print ('orthonormality of V', check_orthonormal(V[:,:count]))
        print ('shape of residuals = ',np.shape(residual))
    elif max_norm < tol:
        print ('sTDA preconditioning done after ', i, 'steps; ', round(precondition_time, 3), 'seconds')
        pass
#         print ('======================Converged!=================')



    return (full_guess*bnorm, current_dic)
###########################################################################################







#############################################
# initiate framework of Davidson's Algorithms
###############################################################################
n = occupied*virtual
def A_diag_initial_guess (k):
    # m is size of subspace A matrix, also is the amount of initial guesses
    m = min([2*k, k+8, occupied*virtual])
    #array of zeros, a container to hold current guess vectors
    V = np.zeros((n, 30*k))
    W = np.zeros((n, 30*k))
    # positions of hdiag with lowest values set as 1
    # hdiag is non-interactiong A matrix
    sort = hdiag.argsort()
    for j in range(0,m):
        V[int(np.argwhere(sort == j)), j] = 1
    W[:, :m] = vind(V[:, :m].T).T
    # W = Av, create transformed guess vectors
    return (m, V, W)

def sTDA_initial_guess (k):

    m = min([2*k, k+8, n])

    V = np.zeros((n, 30*k))
    # array of zeros, a container to hold current guess vectors, v
    W = np.zeros((n, 30*k))
    # a container to hold transformed guess vectors, Av

    #!!!!!!!! diagonalize sTDA_A amtrix
    V[:, :m] = Davidson0(m)
#     V[:, :m]= Gram_Schdmit (V[:, :m])
    W[:, :m] = vind(V[:,:m].T).T
    return (m, V, W)
######################################################################################

#####################################################
def A_diag_preconditioner (residual, sub_eigenvalue, current_dic):
    # preconditioners for each corresponding residual
    k = np.shape(residual)[1]
    # force all values not in domain (-t, t)
    t = 1e-14
    D = np.zeros((n, k))
    for i in range (0, k):
        D[:,i] = hdiag - sub_eigenvalue[i]
        D[:,i][(D[:,i]<t)&(D[:,i]>=0)] = t
        D[:,i][(D[:,i]>-t)&(D[:,i]<0)] = -t
    new_guess = residual/D
    return new_guess, current_dic
#######################################################

################################################################################
# original simple Davidson, just to solve eigenvalues and eigenkets of sTDA_A matrix
def Davidson0 (k):
    tol = 1e-2 # Convergence tolerance

    max = 30
    #################################################
    # generate initial guess
    # m is size of subspace
    m = min([2*k, k+8, occupied*virtual])
    # a container to hold guess vectors
    V = np.zeros((n, 30*k))
    W = np.zeros((n, 30*k))
    # positions of hdiag with lowest values set as 1
    # hdiag is non-interactiong A matrix
    sort = hdiag.argsort()
    for j in range(0,m):
        V[int(np.argwhere(sort == j)), j] = 1

    W[:, :m] = sTDA_fly(V[:, :m])
    #generate initial guess and put in holders V and W
    ###########################################################################################
    for i in range(0, max):
        # sub_A is subspace A matrix
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        # Diagonalize the subspace Hamiltonian, and sorted.
        #sub_eigenvalue[:k] are smallest k eigenvalues
        residual = np.dot(W[:,:m], sub_eigenket[:,:k]) - np.dot(V[:,:m], sub_eigenket[:,:k] * sub_eigenvalue[:k])
#         print ('shape of residual', np.shape(residual))
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
        # orthonormalize the new guesses against old guesses
        # and put into V holder
        V, new_m = Gram_Schdmit_fill_holder (V, m, new_guess)
        W[:, m:new_m] = sTDA_fly (V[:, m:new_m])
        m = new_m
    ###########################################################################################
    full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])
    # print ('Iteration steps =', i+1)
    # print ('Final subspace size = ', np.shape(sub_A))
    # print ('Davidson time:', round(end-start,4))
    print ('sTDA as initial guess done!')
    return (full_guess)
###########################################################################################

Davidson_dic = {}
################################################################################
# Real Davidson frame, where we can choose different initial guess and preconditioner
def Davidson (k, tol, i, p):

    if i == 'sTDA':
        initial_guess = sTDA_initial_guess
        print ('Initial guess: sTDA')
    elif i == 'Adiag':
        initial_guess = A_diag_initial_guess
        print ('Initial guess: Diagonal of Pseudo A matrix')

    if p == 'sTDA':
        precondition = on_the_fly_sTDA_preconditioner
        print ('Preconditioner: on-the-fly sTDA A matrix')

    elif p == 'Adiag':
        precondition = A_diag_preconditioner
        print ('Preconditioner: Diagonal of Pseudo A matrix')
    start = time.time()

    #tol = 1e-5
    # Convergence tolerance
    n = occupied*virtual
    max = 31
    # Maximum number of iterations

    #################################################
    # generate initial guess
    m, V, W = initial_guess(k)
    #generate initial guess and put in holders V and W
    # m is size of subspace

    # time cost for preconditioning
    Pcost = 0
    ###########################################################################################
    for ii in range(0, max):
        print ('Davidson Step', ii)

        ##############################
        ## dictionary to collect data
        # Davidson_dic['Davidson iteration ' + str(ii)] = {}

        Davidson_dic['Davidson iteration ' + str(ii)] = {}
        current_dic = Davidson_dic['Davidson iteration ' + str(ii)]
        ##############################



        # sub_A is subspace A matrix
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
        sTDAP_start = time.time()
        new_guess, current_dic = precondition (residual[:,index], sub_eigenvalue[:k][index], current_dic)
        sTDAP_end = time.time()
        Pcost += sTDAP_end - sTDAP_start
        # orthonormalize the new guesses against old guesses
        # and put into V holder
        V, new_m = Gram_Schdmit_fill_holder (V, m, new_guess)
        W[:, m:new_m] = vind (V[:, m:new_m].T).T
        print ('preconditioned guesses:', new_m-m)
        m = new_m
        Davidson_dic['Davidson iteration ' + str(ii)] = current_dic
    ###########################################################################################

    full_guess = np.dot(V[:,:m], sub_eigenket[:, :k])

    print ('Iteration steps =', ii+1)
    print ('Final subspace size = ', np.shape(sub_A))
    print ('Preconditioning time:', round(Pcost,4))

    return (sub_eigenvalue[:k]*27.21138624598853, full_guess)
################################################################################







print ('-------------------------------------------------------------------')
print ('|---------------   In-house Developed Davdison codes   -----------|')
print ('Residual convergence threshold =', args.tolerance)
print ('Number of excited states =', args.nstates)

Excitation_energies, kets = Davidson (args.nstates, args.tolerance, args.initial_guess, args.preconditioner)
total_end = time.time()
print ('In-house Davidson time:', round(total_end - total_start,4))
print ('Excited State energies (eV) =')
print (Excitation_energies)




print ('-----------------------------------------------------------------')
print ('|-----------------    PySCF TDA-TDDFT codes   ------------------|')
td.nstates = args.nstates
td.conv_tol = 1e-10
td.verbose = 5
start = time.time()
td.kernel()
end = time.time()
print ('Built-in Davidson time:', round(end-start,4))
print ('|---------------------------------------------------------------|')



curpath = os.path.dirname(os.path.realpath(__file__))
yamlpath = os.path.join(curpath, "data.yaml")

with open(yamlpath, "w", encoding="utf-8") as f:
    yaml.dump(Davidson_dic, f, Dumper=yaml.RoundTripDumper)
