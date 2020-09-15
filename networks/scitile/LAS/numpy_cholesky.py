import numpy as np
import sys
import time

# Fast Cholesky Decomposition
def fast_cholesky(A):
    L = np.zeros((N,N))
    for c in range(N):
        if c > 0:
            S = np.matmul(R, R.T)
            A[c:,c:] -= S
        Lcc = np.sqrt(A[c,c])
        R = A[c+1:,c:c+1]/Lcc

        L[c,c] = Lcc
        L[c+1:,c:c+1] = R
    return L

# Adapted Cholesky Decomposition (each iteration takes same-sized slice)
def cholesky(A):
    L = np.zeros((N,N))
    R = np.zeros((N,1))
    for c in range(N):
        if c > 0:
            A -= np.matmul(R, R.T)
        Lcc = np.sqrt(A[c,c])
        R[c,:] = 0
        R[c+1:,:] = A[c+1:,c:c+1]/Lcc
        
        L[c,c] = Lcc
        L[:,c:c+1] += R
    return L

N = int(sys.argv[1])

#Generate a well-conditioned test matrix
R = np.random.rand(N,N)
A = R @ R.T + N*np.eye(N)

t0 = time.time()
L = np.linalg.cholesky(A)
np_tm = time.time() - t0

t0 = time.time()
fast_cholesky(A.copy())
fst_tm = time.time()-t0

t0 = time.time()
step_L = cholesky(A.copy())
step_tm = time.time()-t0

print("N: ", N)
print("Numpy Linalg Solution Time: ", np_tm)
print("Fast Cholesky Solution Time: ", fst_tm)
print("Constant-slice Cholesky Solution Time: ", step_tm)
print("Avg. Error (Constant-slice Cholesky): ", np.mean(np.abs((step_L[abs(L)>1e-4]-L[abs(L)>1e-4])/L[abs(L)>1e-4])))