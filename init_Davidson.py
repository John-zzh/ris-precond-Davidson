import time
import numpy as np
import pyscf
import matplotlib.pylab as plt
from pyscf import gto, scf, dft, tddft, data
import original_davidson

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
# list of elements

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
#list of chemical hardness, they are floats, containing elements 1-94
#in Hartree
HARDNESS = dict(zip(elements,hardness))
#print (HARDNESS)
#create a dictionary by mappig two iteratable subject, list

# mol = gto.Mole()
# mol.build(atom = 'O         -4.89126        3.29770        0.00029;\
# H         -3.49307        3.28429       -0.00328;\
# H         -5.28213        2.58374        0.75736', basis = 'def2-SVP', symmetry = True)
# # mol.atom is the atoms and coordinates!
# # type(mol.atom) is a class <str>

mol = gto.Mole()
mol.build(atom = 'C         -4.89126        3.29770        0.00029;\
O         -3.49307        3.28429       -0.00328;\
H         -5.28213        2.58374        0.75736;\
H         -5.28213        3.05494       -1.01161;\
H         -5.23998        4.31540        0.27138;\
H         -3.22959        2.35981       -0.24953', basis = 'def2-SVP', symmetry = True)

# mf = dft.RKS(mol)
# mf.conv_tol = 1e-14
# mf.grids.level = 9
# mf.xc = 'b3lyp'
# mf.kernel()  #single point energy

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# mf = dft.RKS(mol)
# mf.conv_tol = 1e-12
# mf.grids.level = 9     # 0-9, big number for large mesh grids, default is 3
# mf.xc = 'cam-b3lyp'
# mf.kernel()  #single point energy


td = tddft.TDA(mf)
td.nstates = 5
td.conv_tol = 1e-13

start = time.time()
td.kernel()    #compute first few excited states.
end = time.time()
print ('PySCF time =', round(end-start,4))

#mf.mulliken_pop_meta_lowdin_ao()
##population analysis


#mol.ao_labels()
AO = [int(i.split(' ',1)[0]) for i in mol.ao_labels()]
# .split(' ',1) is to split each element by space, split once.
# mol.ao_labels() it is a list of all AOs and corresponding atom_id, AO is a list of atom_id
# AO == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

N_bf = len(AO)
#print ('Atomic orbitals: ', AO)
# print ('Total number of AOs =', N_bf)

#mf.analyze()
##MO energies

Natm = mol.natm #number of atoms
MOe = mf.mo_energy  #an array of MO energies, in Hartree
#print (MOe)

#mf.mo_occ is an array of occupance [2,2,2,2,2,0,0,0,0.....]
occupied = len(np.where(mf.mo_occ > 0)[0])
#occupied is number of occupied orbitals
# print ('number of occupied orbitals: ', occupied)

virtual = N_bf - occupied
#virtual is number of virtual orbitals
# print ('number of virtual orbitals: ', virtual)


# mf.mo_coeff is the coefficient matrix
# mf.get_ovlp()is basis overlap matrix
# or S = np.dot(np.linalg.inv(C.T), np.linalg.inv(C))

def matrix_power (S,a):
    s,ket = np.linalg.eigh(S)
    s = s**0.5
    X = np.linalg.multi_dot([ket,np.diag(s),ket.T])
    #X == S^1/2
    return X

def orthonormalize (C):
    X = matrix_power(mf.get_ovlp(), 0.5)
    C = np.dot(X,C)
    return C
# now C is orthonormalized coefficient matrix
#np.dot(C.T,C) is a identity matrix
C = mf.mo_coeff
C = orthonormalize (C)
#function to generate q matrix for a certain atom
def generateQ (atom_id):
    q = np.zeros ([N_bf, N_bf])
    for r in range (0, N_bf):
        for s in range (0, N_bf):
            #for a certain element q[r,s]
            #first two loops is to iterate all ith and pth column of C
            for mu in range (0, N_bf):
                if AO[mu] == atom_id:
                    #collect all basis functions centered on atom_id
                    # the last loop is to sum up all C_mui*C_mup, calculate element q[i,p]
                    q[r,s] += C[mu,r]*C[mu,s]
    return q

Qmatrix = [(generateQ(atom_id)) for atom_id in range (0, Natm)]
#a list of q matrix, make them ready to use

#function to return chemical hardness from dictionary HARDNESS
def Hardness (atom_id):
    atom = mol.atom_pure_symbol(atom_id)
    #the symbol of atom looked at
    return HARDNESS[atom]
#mol.atom_pure_symbol(atom_id) is to return pure element symbol only, no special characters

def eta (atom_A_id, atom_B_id):
    eta = (Hardness(atom_A_id) + Hardness(atom_B_id))*0.5
    return eta

R = pyscf.gto.mole.inter_distance(mol, coords = None)
#Inter-particle distance array, unit == ’Bohr’

a_x = 0.25
beta1=0.2
beta2=1.83
alpha1=1.42
alpha2=0.48
beta = beta1 + beta2 * a_x
alpha = alpha1 + alpha2 * a_x
# print ('beta =', beta, 'alpha =', alpha)

# a_x = 0.38
# beta1= 1.86
# beta2=0.00
# alpha1= 0.90
# alpha2=0.00
# beta = beta1 + beta2 * a_x
# alpha = alpha1 + alpha2 * a_x
# print ('beta =', beta, 'alpha =', alpha)


#define gammaJ and gammaK values
def gammaJ(atom_A_id, atom_B_id):
    gamma_A_B_J = (1/(R[atom_A_id, atom_B_id]**beta + (a_x*eta(atom_A_id, atom_B_id))**(-beta)))**(1/beta)
    return gamma_A_B_J

def gammaK(atom_A_id, atom_B_id):
    gamma_A_B_K = (1/(R[atom_A_id, atom_B_id]**alpha \
                   + eta(atom_A_id, atom_B_id)**(-alpha)))\
                    **(1/alpha)
    return gamma_A_B_K

#store gammaJ and gammaK as matrix, make them ready to use
GammaJ = np.zeros([Natm, Natm])
for i in range (0, Natm):
    for j in range (0, Natm):
        GammaJ[i,j] = gammaJ (i,j)

GammaK = np.zeros([Natm, Natm])
for i in range (0, Natm):
    for j in range (0, Natm):
        GammaK[i,j] = gammaK (i,j)

#define two electron intergeral (pq|rs)
def ele_intJ (i,j,a,b):
    ijab = 0
    for atom_A_id in range (0, Natm):
        for atom_B_id in range (0, Natm):
            ijab += Qmatrix[atom_A_id][i,j] * Qmatrix[atom_B_id][a,b] * GammaJ[atom_A_id, atom_B_id]
    return ijab

def ele_intK (i,a,j,b):
    iajb = 0
    for atom_A_id in range (0, Natm):
        for atom_B_id in range (0, Natm):
            iajb += Qmatrix[atom_A_id][i,a] * Qmatrix[atom_B_id][j,b] * GammaK[atom_A_id, atom_B_id]
            #(ia|jb)
    return iajb

#build A matrix
def build_A ():
    A = np.zeros ([occupied*virtual, occupied*virtual])
    # print ('shape of A-sTDA is: ', np.shape(A))
    m = -1
    for i in range (0, occupied):
        for a in range (occupied, N_bf):
            m += 1
            #for each ia pair, it corresponds to a certain row
            n = -1
            for j in range (0, occupied):
                for b in range (occupied, N_bf):
                    n += 1
                #for each jb pair, it corresponds to a certain column
                    if i==j and a==b:
                        A[m,n] = (MOe[a]-MOe[i]) + 2*ele_intK(i,a,j,b) - ele_intJ(i,j,a,b)
                    else:
                        A[m,n] = 2*ele_intK(i,a,j,b) - ele_intJ(i,j,a,b)
    return A

start = time.time()
A = build_A ()
end = time.time()
#print (m,n)
print ('sTDA_A building time =', round (end - start, 2))

#check whether A is symmetric
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)
#print ('symmetry of A_sTDA matrix :', check_symmetric(A, tol=1e-8))



# print (np.round(eigv[:6]*27.21138624598853,5))
#first few eigenvalues

#davidson blcok

# print ('size of A is', n)

vind, hdiag = td.gen_vind(mf)
n = len(hdiag)
#davidson block
def sTDA_davidson (vind, k): # function vind and how many eignvalues to solve

    start = time.time()
    tol = 1e-12      # Convergence tolerance
    max = 90      # Maximum number of iterations
    V = np.zeros((n,30*k)) #array of zeros, a container to hold guess vectors
    W = np.zeros((n,30*k)) #array of zeros, a container to hold transformed guess vectors
    I = np.eye(n)
    # Begin iterations
    m = 2*k  # m is size of subspace Hamiltonian, amount of initial guesses   #m=k works for H2, m=4k works for H2O
    for i in range(0, max):
        if i == 0:
            #first step
            # sort = hdiag.argsort()
            # for j in range(0,m):
            #     V[int(np.argwhere(sort == j)),j] = 1
            #     #positions with lowest values set as 1



            # eigv,eigk = np.linalg.eigh(A)
            eigv,eigk = original_davidson.davidson_A_matrix (A, m)


            for j in range(0,m):
                V[:, j] = eigk [:, j]
    # for i in range (0,m):
    #     V[:, i] == eigenkets[:, i]
    for i in range (0, max):
        for j in range(0, m):
            W[:, j] = vind (V[:,j])   #Hv, create transformed guess vectors
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
                # norm > tol means we didn't find correct eigenkets, so we create new guess vectors
                # d = hdiag-theta[i]
                # #preconditioner
                # d[(d<1e-8)&(d>=0)] = 1e-8
                # d[(d>-1e-8)&(d<0)] = -1e-8
                # #kick out all small values
                # new_vec = residual/d

                new_vec = np.dot(np.linalg.inv(A - theta[x]*I),residual)
                #new guess vectors, core step of Davidson method
                #print (np.shape (new_vec))
                new_vec = new_vec/np.linalg.norm (new_vec) #normalize before GS

                for y in range (0, m + lasit_newvec):  #orthornormalize the new vector against all vectors
                    new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]   # np.linalg.norm(V[:,i])) = 1, should be after np.dot

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
    print ('sTDA as init Davidson time:', round(end-start,4))
    return (theta[:k])

print (27.21138624598853 * sTDA_davidson(A,5))
