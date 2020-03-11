#!/usr/bin/env python
import numpy as np
import davidson1, davidson2
np.set_printoptions(precision = 3)
a = 10000 #dimension of H
sparsity = 0.001
A = np.zeros((a,a))
for i in range(0,a):
    A[i,i] = i + 1
A = A + sparsity*np.random.randn(a,a)
A = (A.T + A)/2   # A is a random Hermition


print ('Dimension of H:', a)
print ('Amount of eigenvalues we want:', 2)
#print ('Davidson1 =',davidson1.davidson (A,2))  # eigenvalues by Davidson1
print ('Davidson2 =',davidson2.davidson (A,2))  # eigenvalues by Davidson2


#E,vec = np.linalg.eig(A)
#idx = E.argsort()
#e = E[idx]
#print (' numpy   =', e[:2])  # Standard eigenvalues
