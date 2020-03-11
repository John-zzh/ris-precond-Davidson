#!/usr/bin/env python
import numpy as np
import time
np.set_printoptions (linewidth=300)


def davidson(A, eig): # matrix A and how many eignvalues to solve
    start = time.time()
    n = np.shape(A)
    n = n[0]
    tol = 0.00000001      # Convergence tolerance
    k = 2 * eig         # number of initial guess vectors
    mmax = 90      # Maximum number of iterations
    #print ('Amount of Eigenvalues we want:', eig)
   
    t = np.eye(n,k) # [initial guess vectors]. (np.eye(n) Return a 2-D identity matrix: array with ones on the diagonal and zeros elsewhere) set of k unit vectors as guess, k is amount of columns shown in the output. While np.eye(3, k=1) return a one-line-up-shifted identity matrix.
    V = np.zeros((n,n)) # array of zeros to hold guess vec
    I = np.eye(n) # identity matrix same dimension as A

    # Begin iterations
    Iteration = 0
    for m in range(k,mmax,k):
        Iteration = Iteration + 1
        #print ('Iteration =', Iteration)
        if m == k:
            for j in range(0,k):
                V[:,j] = t[:,j]/np.linalg.norm(t[:,j])  # V[:,j] is the jth column of V. norm returns the length of a vector
        V,R = np.linalg.qr(V) #np.shape(V) is n*n
        #print (np.round(V,2))
        T = np.linalg.multi_dot([V[:,:m].T,A,V[:,:m]])  #first step, T is left up m*m block of A. (m+1 is not included).  T is m*m matrix, the projected Hamiltonian in the subspace defined by guess vectors.
        THETA,S = np.linalg.eig(T)  #Diagonalize the subspace Hamiltonian. S is eigenkets of T, THETA is eigenvalues (in form of row vector).
        idx = THETA.argsort()  #idx is increasing eigtenvalues's indexes in original THETA. For exapmle, if THETA = [7,5,6], then idx = [1,2,0], '1' measns the senond value '5'
        theta = THETA[idx] #eigenvalues
        s = S[:,idx]       #eigenkets, m*m
        #rearrange eigenvalues from smallest to largest, and corresponding eigenkets #shape of s is actually 'm'
        sum_norm = 0
        for j in range(0,k):
            residual = np.linalg.multi_dot([(A - theta[j]*I),V[:,:m],s[:,j]])  # V*s projects the eigenvector back into the original space.
            norm = np.linalg.norm(residual)
            #new_vec = np.dot(np.diag(1/np.diag(np.diag((np.diag(A)-theta[j])))), residual)
            new_vec = residual/(np.diag(A)-theta[j])
            V[:,(m+j)] = new_vec
            if norm < tol:
                sum_norm = sum_norm +1
        #print ('Number of converged guess vectors:',sum_norm)
        if sum_norm == k:
            #print ('All', sum_norm, 'Guess Vectors Converged')
            break
    end = time.time()
    print ('Davidson1 time (seconds):', round(end-start,2))
    return (theta[:eig])
       
