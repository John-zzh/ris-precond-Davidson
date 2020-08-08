import time
import numpy as np
import pyscf
import matplotlib.pylab as plt
from pyscf import gto, scf, dft, tddft, data




###########################################################################
# Acetaldehyde, CH3CHO
mol = gto.Mole()
mol.build(atom = '\
C         -3.15617        2.59898        0.79547;\
C         -1.79169        2.11570        0.42917;\
O         -0.80893        2.56621        0.99508;\
H         -1.66947        1.36193       -0.34183;\
H         -3.35300        2.38970        1.86780;\
H         -3.91803        2.07820        0.17854;\
H         -3.22824        3.69190        0.61449',\
basis = 'def2-SVP', symmetry = True)
###########################################################################

###########################################################################
# Water, H2O
# mol = gto.Mole()
# mol.build(atom = 'O         -4.89126        3.29770        0.00029;\
# H         -3.49307        3.28429       -0.00328;\
# H         -5.28213        2.58374        0.75736', basis = 'def2-SVP', symmetry = True)
# # mol.atom is the atoms and coordinates!
# type(mol.atom) is a class <str>
###########################################################################

# ###########################################################################
# # Methanol, CH3OH
# mol = gto.Mole()
# mol.build(atom = 'C         -4.89126        3.29770        0.00029;\
# O         -3.49307        3.28429       -0.00328;\
# H         -5.28213        2.58374        0.75736;\
# H         -5.28213        3.05494       -1.01161;\
# H         -5.23998        4.31540        0.27138;\
# H         -3.22959        2.35981       -0.24953', basis = 'def2-SVP', symmetry = True)
# ###########################################################################


###########################################################################
# #DFT calculations

# mf = dft.RKS(mol)
# mf.conv_tol = 1e-14
# mf.grids.level = 9
# mf.xc = 'b3lyp'
# mf.kernel()  #single point energy

# mf = dft.RKS(mol)
# mf.conv_tol = 1e-12
# mf.grids.level = 9     # 0-9, big number for large mesh grids, default is 3
# mf.xc = 'cam-b3lyp'
# mf.kernel()  #single point energy
###########################################################################



###########################################################################
# RHF calculation, no grid dependence
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
###########################################################################

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
# create a functin for dictionary of chemical hardness, by mappig two iteratable subject, list
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

##################################################################################################
# This block is to generate GammaJ and GammaK
# R is inter-particle distance array
a_x = 0.38
beta1= 1.86
beta2=0
alpha1= 0.9
alpha2=0
beta = beta1 + beta2 * a_x
alpha = alpha1 + alpha2 * a_x

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
    # # each column of ket is a eigenket
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
# rthogonalized MO coefficients
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

##population analysis
#mf.mulliken_pop_meta_lowdin_ao()

##home_made population analysis to check whether q tensor is correct
# for atom_id in range (0, Natm):
#     print (check_symmetric(Qmatrix[atom_id], tol=1e-12))
#     m = 0
#     for i in range (0, occupied):
#         m += Qmatrix[atom_id][i,i]
#         #sum over occupied orbitals
#     print (m)
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


def build_A ():

    A = np.zeros ([occupied*virtual, occupied*virtual])
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
                        A[m,n] = (MOe[a]-MOe[i]) + 2*ele_intK(i,a,j,b) - ele_intJ(i,j,a,b)
                    else:
                        A[m,n] = 2*ele_intK(i,a,j,b) - ele_intJ(i,j,a,b)
    print (np.shape(A))
    print ('sTDA_A matrix done')
    return A

start = time.time()
A = build_A ()
end = time.time()
print ('A_sTDA building time =', round (end - start, 2))
########################################################################




########################################################################
# a home-made function to check whether a matrix is symmetric
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)
#print ('symmetry of A_sTDA matrix :', check_symmetric(A, tol=1e-8))
########################################################################



td = tddft.TDA(mf)
td.nstates = 5
td.conv_tol = 1e-13
td.kernel()

#davidso blcok
vind, hdiag = td.gen_vind(mf)
n = len(hdiag)
# print ('size of A is', n)
# eigenvalues, eigenkets = original_davidson.davidson_A_matrix (A,6)

########################################################################
# reproduce the A matrix with vind() function
# and use np.linalg.eigh() to solve standard eigenvalues,
# but only 1e-6 agreement with td.kernel() results !!!!!!!!!
n = occupied * virtual
I = np.eye(n)
H = np.zeros((n,n))

vind, hdiag = td.gen_vind(mf)
for i in range (0, n):
    H[:,i] = vind (I[:, i])
energies, vectors = np.linalg.eigh(H)
print (energies[:5]*27.21138624598853)
########################################################################

########################################################################
# This traditional Davidson is to solve eiegnvalues of pre-calculated sTDA-A matrix
# It uses sTDA_A.diag to build preconditioner
# Actually not necessary in test codes, but essential in production codes
def davidson_A_matrix (A, k): # matrix sTDA_A, and how many eignvalues to solve
    start = time.time()
    n = np.shape(A)[0]
    tol = 1e-5      # Convergence tolerance
    max = 90       # Maximum number of iterations
    V = np.zeros((n,30*k)) #array of zeros, a container to hold guess vectors
    W = np.zeros((n,30*k)) #array of zeros, a container to hold transformed guess vectors, Av

    # Begin iterations
    m = 2*k
    # m is size of subspace Hamiltonian, amount of initial guess vectors
    # m=k works for H2, m=4k works for H2O
    for i in range(0, max):
        sum_convec = 0
        #total converged eigenvectors
        if sum_convec == k:
            break
        # print ('Iteration = ', Iteration)
        if i == 0:
            #initial guess
            sort = np.diag(A).argsort()
            for j in range(0,m):
                V[int(np.argwhere(sort == j)), j] = 1   #positions with lowest values set as 1
        for j in range(0,m):
            W[:, j] = np.dot(A,V[:,j])   # W = Av, create transformed guess vectors
        T = np.dot(V[:,:m].T, W[:,:m])  # T is subspace Hamiltonian
        THETA,S = np.linalg.eigh(T)  #Diagonalize the subspace Hamiltonian.
        idx = THETA.argsort()
        theta = THETA[idx]    #eigenvalues
        s = S[:,idx]          #eigenkets, m*m

        lasit_newvec = 0
        #it records amount of new vector added in last iteration, ranging from 1 to k
        for x in range(0,k):
            #looking at first k vecrors one by one, check if they are roots
            residual = np.dot((W[:,:m]- theta[x]*V[:,:m]), s[:,x])
            # np.dotV([:,:m])s[:,x]) can transform the subspace-eigenket back into full space eigenket
            norm = np.linalg.norm(residual)

            if norm <= tol:
                sum_convec += 1

            else:
            # we didn't find correct eigenkets, so we create new guess vectors
                d = np.diag(A)-theta[x]
                d[(d<1e-16)&(d>=0)] = 1e-16
                d[(d>-1e-16)&(d<0)] = -1e-16   #kick out all small values
                new_vec = residual/d
                #new guess vectors, core step of Davidson method

                new_vec = new_vec/np.linalg.norm (new_vec)
                #normalize before GS

                for y in range (0, m + lasit_newvec):  #orthornormalize the new vector against all vectors
                    new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]   #/ np.linalg.norm(V[:,i])) = 1 should be after np.dot

                norm = np.linalg.norm (new_vec)
                if norm > 1e-16:
                    new_vec = new_vec/norm
                    V[:, m + lasit_newvec] = new_vec
                    lasit_newvec += 1

        m += lasit_newvec
        #now m is size of subspace Hamiltonian in next iteration


    print ('Iteration =', i+1)

    end = time.time()
    Eigenkets = np.dot(V[:,:m], s[:, :k])
    print ('Davidson time (no vind function):', round(end-start,4))
    print (theta[:k]*27.21138624598853)
    return (theta[:k], Eigenkets[:,:k])
############################################################################


############################################################################
# This Davidson use first few eigenkets of sDTA_A matrix as initila guesses for full TDA_A matrix
# and use sTDA_A matrix as preconditioner.                                                                                                                  zXCC zZZZ                      ZAZZ                            ZA`         ZA                ZA   use sTDA_A's first few eigenkets as initial guess
def sTDA_init_p_davidson (k, tol):
    # k = how many eignvalues to solve, tol = residual norm as tolerance
    start = time.time()

    #tol = 1e-12      # Convergence tolerance
    max = 90      # Maximum number of iterations
    V = np.zeros((n,30*k)) #array of zeros, a container to hold guess vectors
    W = np.zeros((n,30*k)) #array of zeros, a container to hold transformed guess vectors
    I = np.eye(n)
    # Begin iterations
    m = 2*k  # m is size of subspace Hamiltonian, amount of initial guesses   #m=k works for H2, m=4k works for H2O
    for i in range(0, max):
        if i == 0:
           #eigv, eigk = davidson_A_matrix (A, m)
            eigv, eigk = np.linalg.eigh(A)
            for j in range(0,m):
                V[:, j] = eigk [:, j]
            # first few eigenkets of sTDA_A as initial guess (first iteration)

        for j in range(0, m):
            W[:, j] = vind (V[:,j])   #Av, create transformed guess vectors

        T = np.dot(V[:,:m].T, W[:,:m])  # T is subspace Hamiltonian
        THETA,S = np.linalg.eigh(T)  #Diagonalize the subspace Hamiltonian.
        idx = THETA.argsort()
        theta = THETA[idx]    #eigenvalues
        s = S[:,idx]          #eigenkets, m*m

        sum_convec = 0
        lasit_newvec = 0  #it records amount of new vector added in last iteration, ranging from 1 to k
        for x in range(0,k):      #looking at first k vecrors one by one, check if they are roots
            residual = np.dot((W[:,:m]- theta[x]*V[:,:m]), s[:,x])
            norm = np.linalg.norm(residual)

            if norm > tol:
                new_vec = np.dot(np.linalg.inv(A - theta[x]*I),residual)
                #!!!!!preconditioner

                #print (np.shape (new_vec))
                new_vec = new_vec/np.linalg.norm (new_vec) #normalize before GS

                for y in range (0, m + lasit_newvec):
                    #orthornormalize the new_guess_vector against all previous guess_vectors
                    new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]
                    # np.linalg.norm(V[:,i])) = 1, should be after np.dot

                norm = np.linalg.norm (new_vec)
                if norm > 1e-16:
                    new_vec = new_vec/norm
                    V[:, m + lasit_newvec] = new_vec
                    lasit_newvec += 1
            else:
                sum_convec += 1

        m += lasit_newvec   #now m is size of subspace Hamiltonian in next iteration
        if sum_convec == k:
            break

    print ('Iteration =', i + 1)

    end = time.time()
    Eigenkets = np.dot(V[:,:m], s[:, :k])
    print ('sTDA as initial guess and preconditioner Davidson time:', round(end-start,4))
    return (theta[:k])
################################################################################################################


################################################################################################################
# This Davidson use sTDA_A magtrix as preconditioner
def sTDA_p_davidson (k, tol):
    # k = how many eignvalues to solve, tol = residual norm as tolerance

    start = time.time()

    max = 90      # Maximum number of iterations
    V = np.zeros((n,30*k)) #array of zeros, a container to hold guess vectors
    W = np.zeros((n,30*k)) #array of zeros, a container to hold transformed guess vectors
    I = np.eye(n)

    # Begin iterations
    m = 2*k  # m is size of subspace Hamiltonian, amount of initial guesses   #m=k works for H2, m=4k works for H2O
    for i in range(0, max):
        if i == 0:
            #initial guesses
            sort = hdiag.argsort()
            for j in range(0,m):
                V[int(np.argwhere(sort == j)),j] = 1
                #positions with lowest values set as 1

        for j in range(0, m):
            W[:, j] = vind (V[:,j])   #Av, create transformed guess vectors
        T = np.dot(V[:,:m].T, W[:,:m])  # T is subspace Hmailtonian
        THETA,S = np.linalg.eigh(T)  #Diagonalize the subspace Hamiltonian.
        idx = THETA.argsort()
        theta = THETA[idx]    #eigenvalues
        s = S[:,idx]          #eigenkets, m*m


        lasit_newvec = 0
        #it records amount of new vector added in last iteration, ranging from 1 to k
        sum_convec = 0
        for x in range(0,k):
            #looking at first k subspace_eigenkets, check if they are roots
            residual = np.dot((W[:,:m]- theta[x]*V[:,:m]), s[:,x])
            norm = np.linalg.norm(residual)

            if norm > tol:
                new_vec = np.dot(np.linalg.inv(A - theta[x]*I),residual)
                #new guess vectors, core step of Davidson method
                # A is sTDA_A matrix

                new_vec = new_vec/np.linalg.norm (new_vec)
                #normalize before GS

                for y in range (0, m + lasit_newvec):
                    # orthornormalize the new_guess_vector against all previous guess_vectors
                    new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]
                    # np.linalg.norm(V[:,i])) = 1, should be after np.dot

                norm = np.linalg.norm (new_vec)
                if norm > 1e-16:
                    new_vec = new_vec/norm
                    V[:, m + lasit_newvec] = new_vec
                    lasit_newvec += 1
            else:
                sum_convec += 1


        m += lasit_newvec
        #now m is size of subspace Hamiltonian in next iteration
        if sum_convec == k:
            break

    print ('Iteration =', i + 1)

    end = time.time()
    Eigenkets = np.dot(V[:,:m], s[:, :k])
    print ('sTDA as preconditioner Davidson time:', round(end-start,4))
    return (theta[:k])
################################################################################################################


################################################################################################################
# This is traditional Davidson, use A.daig as preconditioner
def diagA_davidson (k, tol):
    # k = how many eignvalues to solve, tol = residual norm as tolerance

    start = time.time()

    max = 90      # Maximum number of iterations
    V = np.zeros((n,30*k)) #array of zeros, a container to hold guess vectors
    W = np.zeros((n,30*k)) #array of zeros, a container to hold transformed guess vectors
    I = np.eye(n)

    # Begin iterations
    m = 2*k  # m is size of subspace Hamiltonian, amount of initial guesses   #m=k works for H2, m=4k works for H2O
    for i in range(0, max):
        if i == 0:
            #first step
            sort = hdiag.argsort()
            for j in range(0,m):
                V[int(np.argwhere(sort == j)),j] = 1
                #positions with lowest values set as 1

        for j in range(0, m):
            W[:, j] = vind (V[:,j])   #Av, create transformed guess vectors
        T = np.dot(V[:,:m].T, W[:,:m])  # T is subspace Hmailtonian
        THETA,S = np.linalg.eigh(T)  #Diagonalize the subspace Hamiltonian.
        idx = THETA.argsort()
        theta = THETA[idx]    #eigenvalues
        s = S[:,idx]          #eigenkets, m*m

        sum_convec = 0
        lasit_newvec = 0  #it records amount of new vector added in last iteration, ranging from 1 to k
        for x in range(0,k):      #looking at first k vecrors one by one, check if they are roots
            residual = np.dot((W[:,:m]- theta[x]*V[:,:m]), s[:,x])
            norm = np.linalg.norm(residual)

            if norm > tol:
                #norm > tol means we didn't find correct eigenkets, so we create new guess vectors
                d = hdiag-theta[i]
                #preconditioner
                d[(d<1e-8)&(d>=0)] = 1e-8
                d[(d>-1e-8)&(d<0)] = -1e-8
                #kick out all small values
                new_vec = residual/d

                #new_vec = np.dot(np.linalg.inv(A - theta[x]*I),residual)
                #new guess vectors, core step of Davidson method
                #print (np.shape (new_vec))
                new_vec = new_vec/np.linalg.norm (new_vec) #normalize before GS

                for y in range (0, m + lasit_newvec):
                    #orthornormalize the new_guess_vector against all previous guess_vectors
                    new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]
                    # np.linalg.norm(V[:,i])) = 1, should be after np.dot

                norm = np.linalg.norm (new_vec)
                if norm > 1e-16:
                    new_vec = new_vec/norm
                    V[:, m + lasit_newvec] = new_vec
                    lasit_newvec += 1
            else:
                sum_convec += 1


        m += lasit_newvec   #now m is size of subspace Hamiltonian in next iteration
        if sum_convec == k:
            break

    print ('Iteration =', i + 1)

    end = time.time()
    Eigenkets = np.dot(V[:,:m], s[:, :k])
    print ('A.diag as preconditioner Davidson time:', round(end-start,4))
    return (theta[:k])
########################################################################################################


########################################################################################################
# Testig block, varting residual norm tolerance and number of eigenvalues to solve
TOL = [1e-5]
for k in range (2,6):
    for tol in TOL:
        print ('####################')
        print ('Number of eigenvalues = ', k)
        print (27.21138624598853 * sTDA_init_p_davidson (k, tol))
        print (27.21138624598853 * sTDA_p_davidson (k, tol))
        print (27.21138624598853 * diagA_davidson (k, tol))
        print ('####################')
########################################################################################################
