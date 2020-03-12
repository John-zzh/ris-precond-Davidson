#!/usr/bin/env python
import numpy as np
import davidson1, davidson2, davidson3
np.set_printoptions(precision = 5)
a = 3000 #dimension of H
b = 2
sparsity = 0.001
A = np.zeros((a,a))
for i in range(0,a):
    A[i,i] = i + 1
A = A + sparsity*np.random.randn(a,a)
A = (A.T + A)/2   # A is a random Hermition


print ('Dimension of H:', a)
print ('Amount of eigenvalues we want:', b)
#print ('Davidson2 =',davidson2.davidson (A,b))  # eigenvalues by Davidson1
#print ('Davidson3 =',davidson3.davidson (A,b))  # eigenvalues by Davidson2


davidson2.davidson (A,b)
davidson3.davidson (A,b)


#E,vec = np.linalg.eig(A)
#idx = E.argsort()
#e = E[idx]
#print (' numpy   =', e[:2])  # Standard eigenvalues
