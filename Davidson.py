import time
import numpy as np
import pyscf
from pyscf import gto, scf, dft, tddft, data
import argparse

parser = argparse.ArgumentParser(description='Davidson')

parser.add_argument('-c', '--filename',       type=str, default='methanol.xyz', help='input filename')
parser.add_argument('-m', '--method',         type=str, default='RHF', help='RHF RKS UHF UKS')
parser.add_argument('-f', '--functional',     type=str, default='b3lyp', help='xc functional')
parser.add_argument('-b', '--basis_set',      type=str, default='def2-SVP', help='basis sets')
parser.add_argument('-g', '--grid_level',     type=int, default='3', help='0-9, 9 is best')
parser.add_argument('-i', '--initial_guess',  type=str, default='diag_A', help='initial_guess: diag_A or sTDA_A')
parser.add_argument('-p', '--preconditioner', type=str, default='diag_A', help='preconditioner: diag_A or sTDA_A')
parser.add_argument('-t', '--tolerance',      type=float, default='1e-5', help='residual norm convergence threshold')
parser.add_argument('-n', '--nstates',        type=int, default='4', help='number of excited states')

args = parser.parse_args()
################################################
# read xyz file and delete its first two lines
f = open(args.filename)
coordinates = f.readlines()
del coordinates[:2]
################################################


###########################################################################
# build geometry in PySCF
mol = gto.Mole()
mol.atom = coordinates
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
##################################################################################################


######################################
# set parapeters
if  args.functional == 'lc-b3lyp':
    a_x = 0.53
    beta= 8.00
    alpha1= 4.50
elif args.functional == 'wb97':
    a_x = 0.61
    beta= 8.00
    alpha= 4.41
elif args.functional == 'wb97x':
    a_x = 0.56
    beta= 8.00
    alpha= 4.58
elif args.functional == 'wb97x-d3':
    a_x = 0.51
    beta= 8.00
    alpha= 4.51
elif args.functional == 'cam-b3lyp':
    a_x = 0.38
    beta= 1.86
    alpha= 0.90
else:
    a_x = 0.38
    beta= 1.86
    alpha= 0.90
######################################


##################################################################################################
# This block is to generate GammaJ and GammaK
# R is inter-particle distance array
def eta (atom_A_id, atom_B_id):
    eta = (Hardness(atom_A_id) + Hardness(atom_B_id))/2
    return eta

def gammaJ (atom_A_id, atom_B_id):
    gamma_A_B_J = (R[atom_A_id, atom_B_id]**beta + (a_x * eta(atom_A_id, atom_B_id))**(-beta))**(-1/beta)
    return gamma_A_B_J

def gammaK (atom_A_id, atom_B_id):
    gamma_A_B_K = (R[atom_A_id, atom_B_id]**alpha + eta(atom_A_id, atom_B_id)**(-alpha)) **(-1/alpha)
    return gamma_A_B_K
##################################################################################################


###################################################################################################
# This block is to define two electron intergeral (pq|rs)
def ele_intJ (i,j,a,b):
    Natm = mol.natm
    #number of atoms
    ijab = 0
    for atom_A_id in range (0, Natm):
        for atom_B_id in range (0, Natm):
            ijab += Qmatrix[atom_A_id][i,j] * Qmatrix[atom_B_id][a,b] * GammaJ[atom_A_id, atom_B_id]
    return ijab

def ele_intK (i,a,j,b):
    Natm = mol.natm
    iajb = 0
    for atom_A_id in range (0, Natm):
        for atom_B_id in range (0, Natm):
            iajb += Qmatrix[atom_A_id][i,a] * Qmatrix[atom_B_id][j,b] * GammaK[atom_A_id, atom_B_id]
    return iajb
###################################################################################################


########################################################################
# This block is the function to produce orthonormalized coefficient matrix C
def matrix_power (S,a):
    s,ket = np.linalg.eigh(S)
    # S = mf.get_ovlp() #.get_ovlp() is basis overlap matrix
    # S = np.dot(np.linalg.inv(c.T), np.linalg.inv(c))
    # # s is eigenvalues, must be all positive
    s = s**a
    X = np.linalg.multi_dot([ket,np.diag(s),ket.T])
    #X == S^1/2
    return X

def orthonormalize (C):
    X = matrix_power(mf.get_ovlp(), 0.5)
    C = np.dot(X,C)
    return C
# now C is orthonormalized coefficient matrix
# np.dot(C.T,C) is a identity matrix

def coefficient_matrix ():
    C = mf.mo_coeff
    # mf.mo_coeff is the coefficient matrix
    C = orthonormalize (C)
    return C
# orthogonalized MO coefficients
########################################################################


########################################################################
# To generate q tensor for a certain atom
def generateQ (atom_id):
    q = np.zeros([N_bf, N_bf])
    #N_bf is number Atomic orbitals, q is same size with C

    C = coefficient_matrix ()
    for i in range (0, N_bf):
        for p in range (0, N_bf):
            for mu in range (0, N_bf):
                if AO[mu] == atom_id:
                    #collect all basis functions centered on atom_id
                    # the last loop is to sum up all C_mui*C_mup, calculate element q[i,p]
                    q[i,p] += C[mu,i]*C[mu,p]
                    #q[i,p] += 2*C[i,mu]*C[p,mu]
    return q
########################################################################


########################################################################
# This blcok is to build sTDA_A matrix
Qmatrix = [(generateQ(atom_id)) for atom_id in range (0, Natm)]
#a list of q matrix for all stoms

#pre-compute and restore gammaJ and gammaK matrix, readt to use
GammaJ = np.zeros([Natm, Natm])
for i in range (0, Natm):
    for j in range (0, Natm):
        GammaJ[i,j] = gammaJ (i,j)
GammaK = np.zeros([Natm, Natm])
for i in range (0, Natm):
    for j in range (0, Natm):
        GammaK[i,j] = gammaK (i,j)

def build_sTDA_A ():
    start = time.time()
    sTDA_A = np.zeros ([occupied*virtual, occupied*virtual])
    # container for matrix A

    m = -1
    for i in range (0, occupied):
        for a in range (occupied, N_bf):
            m += 1 #for each ia pair, it corresponds to a certain row
            n = -1
            for j in range (0, occupied):
                for b in range (occupied, N_bf):
                    n += 1 #for each jb pair, it corresponds to a certain column
                    if i==j and a==b:
                        sTDA_A[m,n] = (MOe[a]-MOe[i]) + 2*ele_intK(i,a,j,b) - ele_intJ(i,j,a,b)
                    else:
                        sTDA_A[m,n] = 2*ele_intK(i,a,j,b) - ele_intJ(i,j,a,b)
    end = time.time()
    print ('sTDA_A matrix built within =', round (end - start, 2), 'seconds')
    print ('Size =', np.shape(sTDA_A)[0])
    return sTDA_A
########################################################################


########################################################################
print ('-----------------------------------------------------------------')
sTDA_A = build_sTDA_A()
## prepare sTDA_A matrix
td = tddft.TDA(mf)
vind, hdiag = td.gen_vind(mf)
n = len(hdiag)
I = np.eye(n)
########################################################################

######################################################################################
def diag_A_initial_guess (k):
    m = 2*k
    # m is size of subspace Hamiltonian, amount of initial guesses
    # m=k works for H2, m=4k works for H2O
    V = np.zeros((n, m))
    #array of zeros, a container to hold current guess vectors
    W = np.zeros((n, m))
    sort = hdiag.argsort()
    for j in range(0,m):
        V[int(np.argwhere(sort == j)), j] = 1
        # positions with lowest values set as 1
        W[:, j] = vind(V[:, j])
    # W = Av, create transformed guess vectors
    return (m, V, W)

def sTDA_A_initial_guess (k):
    m = 2*k
    V = np.zeros((n, m))
    W = np.zeros((n, m))
    eigvalues, eigkets = np.linalg.eigh(sTDA_A)
    for j in range(0,m):
        V[:, j] = eigkets [:, j]
        W[:, j] = vind(V[:, j])
    return (m, V, W)
######################################################################################

###################################################################
def diag_A_preconditioner (residual, sub_eigenvalue, x):
    d = hdiag - sub_eigenvalue[x]
    d[(d<1e-16)&(d>=0)] = 1e-16
    d[(d>-1e-16)&(d<0)] = -1e-16
    #kick out all small values

    new_vec = residual/d
    return new_vec

def sTDA_A_preconditioner (residual, sub_eigenvalue, x):
    new_vec = np.dot(np.linalg.inv(sTDA_A - sub_eigenvalue[x]*I),residual)
    return new_vec
####################################################################

################################################################################
def Davidson (k, tol, initial_guess, preconditioner):
    if initial_guess == 'sTDA_A':
        initial_guess = sTDA_A_initial_guess
        print ('Initial guess: sTDA A matrix')
    elif initial_guess == 'diag_A':
        initial_guess = diag_A_initial_guess
        print ('Initial guess: Diagonal of full A matrix')

    if preconditioner == 'sTDA_A':
        preconditioner = sTDA_A_preconditioner
        print ('Preconditioner: sTDA A matrix')
    elif preconditioner == 'diag_A':
        preconditioner = diag_A_preconditioner
        print ('Preconditioner: Diagonal of full A matrix')
    start = time.time()

    max = 90
    # Maximum number of iterations
    ###########################################################################################
    for i in range(0, max):
        sum_convec = 0
        # total converged eigenvectors
        # breaf if sum_convec == k
        #################################################
        # generate initial guess
        if i == 0:
            m, V, W = initial_guess(k)
        #################################################
        sub_A = np.dot(V.T, W)
        # sub_A is subspace A matrix
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        # Diagonalize the subspace Hamiltonian, and sorted.
        lasit_newvec = 0
        # amount of new vectors added in last iteration, ranging from 1 to k
        # because not all new guess_vectors can survive the Gram-Schmidt
        ####################################################################################
        for x in range(0,k):
            #looking at first k vecrors one by one, check whether they are roots
            residual = np.dot((W[:,:m]- sub_eigenvalue[x]*V[:,:m]), sub_eigenket[:,x])
            # np.dotV([:,:m])s[:,x]) can project the subspace-eigenket back to full space
            norm = np.linalg.norm(residual)
            if norm <= tol:
                sum_convec += 1
            else:
                # current guess is not good enough,
                # so we use current guess (residual) to create new guess vectors
                #########################################################
                new_vec = preconditioner (residual, sub_eigenvalue, x)
                #########################################################
                # preconditioner
                new_vec = new_vec/np.linalg.norm (new_vec)
                # normalize before Gram-Schmidt
                for y in range (0, m + lasit_newvec):
                    new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]
                    # orthornormalize the new vector against all old vectors
                norm = np.linalg.norm (new_vec)
                if norm > 1e-16:
                    new_vec = new_vec/norm
                    # normalzie the new vector, now Gram-Schmidt is done

                    V = np.append (V, new_vec[:, None], axis=1)
                    # put the new guess into container
                    trans_new_vec = vind(new_vec)
                    # print ('Shape of trans_new_vec =', np.shape(trans_new_vec))
                    W = np.append (W, trans_new_vec.T, axis = 1)
                    # put transformed guess Av into container

                    lasit_newvec += 1
        ####################################################################################
        if sum_convec == k:
            break
        m += lasit_newvec
    ###########################################################################################

    Eigenkets = np.dot(V[:,:m], sub_eigenket[:, :k])

    end = time.time()
    print ('Iteration steps =', i+1)
    print ('Davidson time:', round(end-start,4))

    return (sub_eigenvalue[:k]*27.21138624598853, Eigenkets[:,:k])



print ('-----------------------------------------------------------------')
print ('------------------   In-house Davdison codes   ------------------')

k = args.nstates
tol = args.tolerance
initial_guess = args.initial_guess
preconditioner = args.preconditioner

print ('Number of excited states =', k)
print ('Residual convergence threshold =', tol)
Excitation_energies, kets = Davidson (k, tol, initial_guess, preconditioner)
print ('Excited State energies (eV) =')
print (Excitation_energies)
print ('-----------------------------------------------------------------')
print ('------------------    PySCF TDA-TDDFT codes   -------------------')
td.nstates = k
start = time.time()
td.kernel()
end = time.time()
print ('Built-in Davidson time:', round(end-start,4))
