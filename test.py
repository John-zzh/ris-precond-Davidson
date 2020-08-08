#!/usr/bin/env python
import numpy as np
import davidson1, davidson2, davidson3, davidson4
np.set_printoptions(precision = 7)
a = 1000 #dimension of H
b = 3
sparsity = 0.01
A = np.zeros((a,a))
for i in range(0,a):
    A[i,i] = i + 1
A = A + sparsity*np.random.randn(a,a)
A = (A.T + A)/2   # A is a random Hermition


print ('Dimension of H:', a)
print ('Amount of eigenvalues we want:', b)
print ('Davidson2 =',davidson2.davidson (A,b))
print ('Davidson3 =',davidson3.davidson (A,b))
print ('Davidson4 =',davidson4.davidson (A,b))

#davidson1.davidson (A,b)
#davidson2.davidson (A,b)
#davidson3.davidson (A,b)
#davidson4.davidson (A,b)

E,vec = np.linalg.eig(A)
idx = E.argsort()
e = E[idx]
print (' numpy   =', e[:3])  # Standard eigenvalues
