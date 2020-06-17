import time
import numpy as np
import pyscf
import matplotlib.pylab as plt
from pyscf import gto, scf, dft, tddft, data


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


HARDNESS = dict(zip(elements,hardness))
#print (HARDNESS)
#create a dictionary by mappig two iteratable subject

mol = gto.Mole()
mol.build(atom = 'O 0 0 0; H 0 0 1; H 0 0 -1 ', basis = 'def2-SVP', symmetry = True)
#print (mol.atom)
# mol.atom is the atoms and coordinates!
# type(mol.atom) is a class <str>

#print (mol.atom_coords())
#the molecular geometry, in matrix form

#print (mol.atom_symbol(0))
# mol.atom_symbol(0) is to return the element symbol, including special characters

#print (mol.atom_pure_symbol(0))
#to return pure element symbol only, no special characters

#print (mol.atom_coord(0))
#to return the coordinates of a certain atom, it is a ndarray, vector

#print (mol.natm)
#number of atoms # therefore, atom_id is in rnage (0, mol.natm)



mf = dft.RKS(mol) #mf is ground density?
mf.xc = 'b3lyp'
mf.kernel()  #single point energy

a_x = 0.3
# a_x is hybrid parameter

#mf.analyze()
##MO energies

mf.mulliken_pop_meta_lowdin_ao()
#population analysis

#nocc = mol.nelectron
# mol.nelectron is number of electrons, not necessary the number of MOs

C = pyscf.lo.orth.orth_ao(mol, method = 'lowdin')
#C = mf.mo_coeff
#C_orthog = np.dot(S**0.5,C)
# mf.mo_coeff is the coefficient matrix

# S = mf.get_ovlp() #.get_ovlp()  is not basis overlap matrix
# s,ket = np.linalg.eigh(S)
# s = s **0.5
# S = np.linalg.multi_dot([ket.T,np.diag(s),ket])

#print (np.round(C,2))

#mol.ao_labels() #a list of each arom's atomic bobitals
AO = [int(i.split(' ',1)[0]) for i in mol.ao_labels()]
# .split(' ',1) is to split each element by space, split once.
# mol.ao_labels() it is a list of all AOs
N_bf = len(AO)
#print (AO)
def generate_q (atom_id):
    q = np.zeros([N_bf, N_bf])
    #q is same size with C
    for i in range (0, N_bf):
        for p in range (0, N_bf):
            #q[i,p]
            #first two loops is to iterate all ith row and pth column of C
            for x in range (0, N_bf):
                # the last loop is to sum up all C_mui*C_mup
                if AO[x] == atom_id:
                    q[i,p] += C[x,i]*C[x,p]
    return q


for atom_id in range (0, mol.natm):
    locals()['q_atom_' + str(atom_id)] = generate_q (atom_id)
    #name = 'q_atom_' + str(atom_id)
    #to create a serial name: q_atom_1, q_atom_2, q_atom_3....


# start = time.time()
# td = tddft.TDA(mf)   #TDA is turned on
# td.kernel()    #compute first few excited states.
# end = time.time()
#
# print ('Pyscf time =', round(end-start,4))
#
#
# vind, hdiag = td.gen_vind(mf)

#this block is to reprocude the TDA A matrix
# n = len(hdiag)
# print ('size of H is', n)
# A = np.zeros((n,n))
# I = np.eye(n)
# for i in range (0,n):
#     A[:,i] = vind (I[:,i])
# plt.matshow(A)
# plt.show() #visualize the matrix



# # davidson block
# def davidson(vind, k): # fucntion vind and how many eignvalues to solve
#     start = time.time()
#     tol = 1e-5      # Convergence tolerance
#     max = 90      # Maximum number of iterations
#     V = np.zeros((n,30*k)) #array of zeros. a container to hold guess vectors
#     W = np.zeros((n,30*k)) #array of zeros. a container to hold transformed guess vectors
#
#
#     # Begin iterations
#     Iteration   = 0
#     m = 4*k  # m is size of subspace Hamiltonian, amount of initial guesses   #m=k works for H2, m=4k works for H2O
#     for i in range(0, max):
#         Iteration = Iteration + 1
#
#         if i == 0:              #first step
#             sort = hdiag.argsort()
#             for j in range(0,m):
#                 V[int(np.argwhere(sort == j)),j] = 1   #positions with lowest values set as 1
#         for j in range(0,m):
#             W[:, j] = vind (V[:,j])   #Hv, create transformed guess vectors
#         T = np.dot(V[:,:m].T, W[:,:m])  # T is subspace Hmailtonian
#         THETA,S = np.linalg.eigh(T)  #Diagonalize the subspace Hamiltonian.
#         idx = THETA.argsort()
#         theta = THETA[idx]    #eigenvalues
#         s = S[:,idx]          #eigenkets, m*m
#
#         sum_convec = 0
#         lasit_newvec = 0  #it records amount of new vector added in last iteration, ranging from 1 to k
#         for x in range(0,k):      #looking at first k vecrors one by one, check if they are roots
#             residual = np.dot((W[:,:m]- theta[x]*V[:,:m]), s[:,x])
#             norm = np.linalg.norm(residual)
#
#             if norm > tol:         # norm > tol means we didn't find correct eigenkets, so we create new guess vectors
#                 # d = hdiag-theta[i]
#                 # d[(d<1e-8)&(d>=0)] = 1e-9
#                 # d[(d>-1e-8)&(d<0)] = -1e-9   #kick out all small values
#
#
#                 new_vec = np.dot(np.linalg.inv(A-theta[i]), residual)          #new guess vectors, core step of Davidson method
#                 new_vec = new_vec/np.linalg.norm (new_vec) #normalize before GS
#
#                 for y in range (0, m + lasit_newvec):  #orthornormalize the new vector against all vectors
#                     new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]   #/ np.linalg.norm(V[:,i])) = 1 should be after np.dot
#
#                 norm = np.linalg.norm (new_vec)
#                 if norm > 1e-16:
#                     new_vec = new_vec/norm
#                     V[:, m + lasit_newvec] = new_vec
#                     lasit_newvec += 1
#             else:
#                 sum_convec += 1
#
#
#         m += lasit_newvec   #now m is size of subspace Hamiltonian in next iteration
#         if sum_convec == k:
#             break
#
#
#     print ('Iteration =', Iteration)
#
#     end = time.time()
#     Eigenkets = np.dot(V[:,:m], s[:, :k])
#     print ('Davidson time:', round(end-start,4))
#     return (theta[:k])
#
# print (27.21138624598853 * davidson(vind,3))




#I = np.eye(n)
#Z = np.zeros((n,n))
#for i in range (0, n):
#    Z[:,i] = vind (I[:, i]) #Z is Hamiltonian rebuild from vind function. It will be a hermition if we call TDA methos, rather than tddft method.
#print ('Hamiltonian built')
#check whether Z is symmetric
#def check_symmetric(a, tol=1e-8):
#    return np.all(np.abs(a-a.T) < tol)
#print (check_symmetric(Z, tol=1e-8))

#E,vec = np.linalg.eig(Z)
#idx = E.argsort()
#e = E[idx]
#print ('numpy =', 27.2114 * e[:3])  # Standard eigenvalues

#print (27.211386245988 * davidson4.davidson (Z,3)) # my own codes
