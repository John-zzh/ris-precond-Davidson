import numpy as np
np.set_printoptions(precision=2)
#Q = 0*H
n=10
Q = np.random.rand(20,n)

#built-in function of np approaching, V is the a orthonnormalized matrix
V,R = np.linalg.qr(Q)
print (V)

def coefficient(v1, v2):
    return np.dot(v1.T, v2) / np.dot(v1.T, v1) #get v1 component of v2

def orth(v1, v2):
    return (v2 - coefficient(v1, v2) * v1) #make v2 orthogonal to v1

#orthogonal all the vectors, make sure they are perpendicular to each other
for i in range (0,n-1):
    for j in range (i+1, n):
        Q[:,j] = orth (Q[:,i], Q[:,j])
        
#then normalize all the vectors
for i in range (n):
    Q[:,i] = Q[:,i]/np.linalg.norm(Q[:,i])

print (Q)
#print ("%.2f"% np.dot(Q[:,0],Q[:,2]))
