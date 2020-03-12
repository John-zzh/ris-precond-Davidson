#!/usr/bin/env python
import numpy as np
import time
np.set_printoptions (linewidth=300)


def orthonormal (v1, v2):
    v2 = v2 - (np.dot(v1, v2) / np.linalg.norm(v1)) * v1
    v2 = v2/np.linalg.norm(v2)
    return v2
    
def davidson(A, eig): # matrix A and how many eignvalues to solve
    start = time.time()
    n = np.shape(A)
    n = n[0]
    tol = 0.000001      # Convergence tolerance
    k = 2 * eig         # number of initial guess vectors
    mmax = 90      # Maximum number of iterations
    #print ('Amount of Eigenvalues we want:', eig)
   
    t = np.eye(n,k) # [initial guess vectors]. they should be orthonormal. (np.eye(n) Return a 2-D identity matrix: array with ones on the diagonal and zeros elsewhere) set of k unit vectors as guess, k is amount of columns shown in the output. While np.eye(3, k=1) return a one-line-up-shifted identity matrix.
    V = np.zeros((n,n)) # array of zeros to hold guess vec
    

    # Begin iterations
    Iteration = 0
    for m in range(k,mmax,k):
        Iteration = Iteration + 1
        #print ('Iteration =', Iteration)
        if m == k:
            for j in range(0,k):
                V[:,j] = t[:,j]
               
        W = np.dot(A, V[:,:m])
        T = np.dot(V[:,:m].T, W) #T is m*m matrix
        #print ('shape of T:', np.shape(T))
        
        THETA,S = np.linalg.eig(T)  #Diagonalize the subspace Hamiltonian.
        idx = THETA.argsort()
        theta = THETA[idx]    #eigenvalues
        s = S[:,idx]          #eigenkets, m*m
       
        
        sum_norm = 0
        for j in range(0,k):
            residual = np.dot((W- theta[j]*V[:,:m]), s[:,j])
            norm = np.linalg.norm(residual)
            new_vec = residual/(np.diag(A)-theta[j])
#            for p in range (0, m+j-1):
#                new_vec = orthonormal(V[:,p], new_vec)
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
        
    end = time.time()
    print ('Davidson2 time (seconds):', round(end-start,4))
    return (theta[:eig])
       
