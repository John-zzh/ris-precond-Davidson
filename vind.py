import time
import numpy as np
from pyscf import gto, scf, dft, tddft
import davidson4
mol = gto.Mole()
mol.build(atom = 'H 0 0 0; F 0 0 1.2', basis = '631g', symmetry = True)
mf = dft.RKS(mol) #RKS, restrict the geometry, no optimization #mf is ground density?
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
n = len(hdiag)
print ('size of H is', n)

#davidson block
def davidson(vind, k): # fucntion vind and how many eignvalues to solve
    start = time.time()
    tol = 1e-8      # Convergence tolerance
    max = 90      # Maximum number of iterations
    V = np.zeros((n,30*k)) #array of zeros. a container to hold guess vectors
    W = np.zeros((n,30*k)) #array of zeros. a container to hold transformed guess vectors
  
    # Begin iterations
    Iteration   = 0
    
    m = 2*k  # m is size of initial subspace Hamiltonian
    lasit_newvec = 0  #amount of new vector added in last iteration
    for i in range(0, max):
        Iteration = Iteration + 1
        m += lasit_newvec #m is size of current subspace Hamiltonian
       
        if i == 0:              #first step
            sort = hdiag.argsort()
            for i in range(0,m):
                V[int(np.argwhere(sort == i)),i] = 1   #positions with lowest values set as 1
        for i in range(0,m):
            W[:, i] = vind (V[:,i])   #Hv, create transformed guess vectors
        T = np.dot(V[:,:m].T, W[:,:m])  # T is subspace Hmailtonian
        THETA,S = np.linalg.eigh(T)  #Diagonalize the subspace Hamiltonian.
        idx = THETA.argsort()
        theta = THETA[idx]    #eigenvalues
        s = S[:,idx]          #eigenkets, m*m
        
        sum_convec = 0
        lasit_newvec = 0
        for i in range(0,k):      #looking at first k vecrors one by one, check if they are roots
            residual = np.dot((W[:,:m]- theta[i]*V[:,:m]), s[:,i])
            norm = np.linalg.norm(residual)
                   # number of added new guess vectors from last iteration.
            if norm > tol:         # norm > tol means we didn't find correct eigenkets, so we create new guess vectors
                d = hdiag-theta[i]
                d[(d<1e-8)&(d>=0)] = 1e-8
                d[(d>-1e-8)&(d<0)] = -1e-8   #kick out all small values
                new_vec = residual/d          #new guess vectors, core step of Davidson method
                for i in range (0, m + lasit_newvec):
                    new_vec = new_vec - np.dot(V[:,i], new_vec) * V[:,i]   #/ np.linalg.norm(v1)) = 1 should be after np.dot
                norm = np.linalg.norm (new_vec)
                if norm > 1e-15:
                    new_vec = new_vec/norm
                    V[:, m + lasit_newvec] = new_vec
                    lasit_newvec += 1
            else:
                sum_convec += 1
        print ('sum_convec =', sum_convec)
        print ('lasit_newvec =', lasit_newvec)
        if sum_convec == k:
            break
                    
                   
    #print ('Iteration =', Iteration)
    
    end = time.time()
    Eigenkets = np.dot(V[:,:m], s[:, :k])
    print ('Davidson time:', round(end-start,4))
    return (theta[:k])

print (27.21138624598853 * davidson(vind,3))




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
