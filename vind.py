import time
import numpy as np
from pyscf import gto, scf, dft, tddft
import davidson4
mol = gto.Mole()
mol.build(atom = 'H 0 0 0; F 0 0 1.3', basis = '631g', symmetry = True)
mf = dft.RKS(mol) #RKS, restrict the geometry, no optimization.
   #mf is ground density?
mf.xc = 'b3lyp'
mf.kernel()  #single point energy
td = tddft.TDA(mf)
start = time.time()
td.kernel()    #compute first few excited states.
end = time.time()
print ('Pyscf time =', round(end-start,4))
#mytd = tddft.TDDFT(mf)  #TDDFT would not use full matrix
#mytd.kernel()

vind, hdiag = td.gen_vind(mf)
n = len(hdiag) #n is size of Hamiltonian
print ('size of H is', n)

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


#Gram-Schimidt block,
def orthonormal (v1, v2):
    v2 = v2 - (np.dot(v1, v2) / np.linalg.norm(v1)) * v1
    v2 = v2/np.linalg.norm(v2)
    return v2

#davidson block
def davidson(vind, eig): # matrix A and how many eignvalues to solve
    start = time.time()

    tol = 1e-11      # Convergence tolerance
    k = eig         # number of initial guess vectors
    mmax = 90      # Maximum number of iterations
    #print ('Amount of Eigenvalues we want:', eig)

    t = np.eye(n,k) # [initial guess vectors]. they should be orthonormal.
    V = np.zeros((n,40*eig)) #array of zeros. a container to hold guess vectors
    W = np.zeros((n,40*eig)) #array of zeros. a container to hold transformed guess vectors


    # Begin iterations
    Iteration = 0
    for m in range(k,mmax,k):
        Iteration = Iteration + 1
        #print ('Iteration =', Iteration)
        if m == k:
            for j in range(0,k):
                V[:,j] = t[:,j]

        for i in range(m-k,m):
            W[:, i] = vind (V[:,i])

        T = np.dot(V[:,:m].T, W[:,:m])
        THETA,S = np.linalg.eigh(T)  #Diagonalize the subspace Hamiltonian.
        idx = THETA.argsort()
        theta = THETA[idx]    #eigenvalues
        s = S[:,idx]          #eigenkets, m*m
        sum_norm = 0
        for j in range(0,k):
            residual = np.dot((W[:,:m]- theta[j] * V[:,:m]), s[:,j])
            norm = np.linalg.norm(residual)
            d = hdiag-theta[j]
            d[d<1.0e-8] = 1.0e-8
            new_vec = residual/d
            V[:,(m+j)] = new_vec
            if norm < tol:
                sum_norm = sum_norm +1
        if sum_norm == k:
                #print ('All', sum_norm, 'Guess Vectors Converged')
                break
        #Gram-Schimidt block,
        for p in range(0, k):
            for q in range (0, m+p):
                V[:,m+p] = orthonormal(V[:,q], V[:,m+p])
    #print (sum_norm)
    print ('Iteration =', Iteration)
    end = time.time()
    Eigenkets = np.dot(V[:,:m], s[:, :eig])
    print ('Davidson time:', round(end-start,4))
    return (theta[:eig])

print (27.21138624598853 * davidson(vind,3))
