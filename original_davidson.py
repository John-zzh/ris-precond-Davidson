import time
import numpy as np
from pyscf import gto, scf, dft, tddft
import davidson4
mol = gto.Mole()
# mol.build(atom = 'O 0 0 0; H 0 0 1; H 0 1 0', basis = '631g')

mol.build(atom = 'C         -4.89126        3.29770        0.00029;\
O         -3.49307        3.28429       -0.00328;\
H         -5.28213        2.58374        0.75736;\
H         -5.28213        3.05494       -1.01161;\
H         -5.23998        4.31540        0.27138;\
H         -3.22959        2.35981       -0.24953', basis = 'def2-SVP', symmetry = True)

# mf = dft.RKS(mol)
# mf.xc = 'b3lyp'
# mf.kernel()  #single point energy


mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

td = tddft.TDA(mf)
td.nstates = 5
td.conv_tol = 1e-13

start = time.time()
td.kernel()    #compute first few excited states.
end = time.time()
print ('Pyscf time =', round(end-start,4))


vind, hdiag = td.gen_vind(mf)
n = len(hdiag)
# print ('size of H is', n)

#davidson block
def davidson_vind (vind, k): # fucntion vind and how many eignvalues to solve
    start = time.time()
    tol = 1e-12      # Convergence tolerance
    max = 90      # Maximum number of iterations
    V = np.zeros((n,30*k)) #array of zeros. a container to hold guess vectors
    W = np.zeros((n,30*k)) #array of zeros. a container to hold transformed guess vectors, Av


    # Begin iterations
    Iteration   = 0
    m = 4*k  # m is size of subspace Hamiltonian, amount of initial guesses   #m=k works for H2, m=4k works for H2O
    for i in range(0, max):
        Iteration = Iteration + 1

        if i == 0:              #first step
            sort = hdiag.argsort()
            for j in range(0,m):
                V[int(np.argwhere(sort == j)),j] = 1   #positions with lowest values set as 1
        for j in range(0,m):
            W[:, j] = vind (V[:,j])   # W = Av, create transformed guess vectors
        T = np.dot(V[:,:m].T, W[:,:m])  # T is subspace Hamiltonian
        THETA,S = np.linalg.eigh(T)  #Diagonalize the subspace Hamiltonian.
        idx = THETA.argsort()
        theta = THETA[idx]    #eigenvalues
        s = S[:,idx]          #eigenkets, m*m

        sum_convec = 0
        lasit_newvec = 0  #it records amount of new vector added in last iteration, ranging from 1 to k
        for x in range(0,k):      #looking at first k vecrors one by one, check if they are roots
            residual = np.dot((W[:,:m]- theta[x]*V[:,:m]), s[:,x])
            # np.dotV([:,:m])s[:,x]) can transform the subspace-eigenket back into full space eigenket
            norm = np.linalg.norm(residual)

            if norm > tol:         # norm > tol means we didn't find correct eigenkets, so we create new guess vectors
                d = hdiag-theta[x]
                d[(d<1e-8)&(d>=0)] = 1e-8
                d[(d>-1e-8)&(d<0)] = -1e-8   #kick out all small values
                new_vec = residual/d          #new guess vectors, core step of Davidson method
                #print (np.shape (new_vec))
                new_vec = new_vec/np.linalg.norm (new_vec) #normalize before GS

                for y in range (0, m + lasit_newvec):  #orthornormalize the new vector against all vectors
                    new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]   #/ np.linalg.norm(V[:,i])) = 1 should be after np.dot

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

    print ('Iteration =', Iteration)

    end = time.time()
    Eigenkets = np.dot(V[:,:m], s[:, :k])
    print ('Diagonal Davidson time:', round(end-start,4))
    return (theta[:k], Eigenkets[:,:k])

eigenvalues, eigenkets = davidson_vind(vind,5)
# print (27.21138624598853 * eigenvalues)




I = np.eye(n)
A = np.zeros((n,n))
for i in range (0, n):
   A[:,i] = vind (I[:, i]) #A is Hamiltonian rebuild from vind function. It will be a hermition if we call TDA methos, rather than tddft method.


def davidson_A_matrix (A, k): # matrix A and how many eignvalues to solve
    start = time.time()
    n = np.shape(A)[0]
    tol = 1e-12      # Convergence tolerance
    max = 90      # Maximum number of iterations
    V = np.zeros((n,30*k)) #array of zeros. a container to hold guess vectors
    W = np.zeros((n,30*k)) #array of zeros. a container to hold transformed guess vectors, Av

    # Begin iterations
    Iteration   = 0
    m = 4*k  # m is size of subspace Hamiltonian, amount of initial guesses   #m=k works for H2, m=4k works for H2O
    for i in range(0, max):
        Iteration = Iteration + 1
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

        sum_convec = 0
        lasit_newvec = 0  #it records amount of new vector added in last iteration, ranging from 1 to k
        for x in range(0,k):      #looking at first k vecrors one by one, check if they are roots
            residual = np.dot((W[:,:m]- theta[x]*V[:,:m]), s[:,x])
            # np.dotV([:,:m])s[:,x]) can transform the subspace-eigenket back into full space eigenket
            norm = np.linalg.norm(residual)

            if norm > tol:         # norm > tol means we didn't find correct eigenkets, so we create new guess vectors
                d = np.diag(A)-theta[x]
                d[(d<1e-8)&(d>=0)] = 1e-8
                d[(d>-1e-8)&(d<0)] = -1e-8   #kick out all small values
                new_vec = residual/d          #new guess vectors, core step of Davidson method
                #print (np.shape (new_vec))
                new_vec = new_vec/np.linalg.norm (new_vec) #normalize before GS

                for y in range (0, m + lasit_newvec):  #orthornormalize the new vector against all vectors
                    new_vec = new_vec - np.dot(V[:,y], new_vec) * V[:,y]   #/ np.linalg.norm(V[:,i])) = 1 should be after np.dot

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

    print ('Iteration =', Iteration)

    end = time.time()
    Eigenkets = np.dot(V[:,:m], s[:, :k])
    print ('Davidson time (no vind function):', round(end-start,4))
    return (theta[:k], Eigenkets[:,:k])

eigenvalues, eigenkets = davidson_A_matrix(A,5)
# print (27.21138624598853 * eigenvalues)




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
