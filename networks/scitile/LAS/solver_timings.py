import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve as densolve
import sys
import time
import conjgrad as pmlcsolver


# Stepping function for conjugate gradient
def step(A, b, x, r, p, rsold):
    Ap = A @ p
    alpha = rsold / (p.T @ Ap)
    x += alpha * p
    r -= alpha * Ap
    rsnew = r.T @ r
    p = r + p * (rsnew / rsold)
    return x, r, p, rsnew


# Computes solution to Ax = b using conjugate gradient method
def conjgrad(A, b, x):
    r = b - A @ x
    p = r.copy()
    rsold = r.T @ r
    for i in range(N):
        x, r, p, rsold = step(A, b, x, r, p, rsold)
        if rsold < 1e-7:
            break
    return x


testnames = ['10x10', '20x20', '40x40', '80x80']
labels = [
    'Numpy Linalg', 'Numpy Conjgrad', 'SciPy Dense', 'SciPy Sparse', 'PlaidML CG Dense',
    'PlaidML CG Sparse'
]
Ns = np.zeros(len(labels))
times = np.zeros((len(testnames), len(labels)))

for i in range(len(testnames)):
    # Load input
    print("Loading Input " + testnames[i])
    testname = testnames[i]
    testdat = np.load('networks/scitile/LAS/test_data/' + testname +
                      '.npz')  # networks/scitile/LAS/
    A, b = testdat['A'], testdat['b']
    testdat.close()
    N = len(b)
    b = np.reshape(b, (N, 1))
    x0 = np.zeros((N, 1))

    Ns[i] = N

    # Numpy black-box solver
    t0 = time.time()
    np_x = np.linalg.solve(A, b)
    times[i, 0] = time.time() - t0

    # Numpy conjugate gradient method
    t0 = time.time()
    cg_x = conjgrad(A, b, x0)
    times[i, 1] = time.time() - t0

    # Scipy black-box dense solver
    t0 = time.time()
    spd_x = densolve(A, b)
    times[i, 2] = time.time() - t0

    # Scipy black-box sparse solver
    sp_A = csc_matrix(A)
    t0 = time.time()
    sps_x = spsolve(sp_A, b)
    times[i, 3] = time.time() - t0

    # PlaidML dense conjugate gradient solver
    b = np.squeeze(b)
    t0 = time.time()
    dpld_x = pmlcsolver.cg_solve(A, b)
    times[i, 4] = time.time() - t0

    # PlaidML sparse conjugate gradient solver
    spA = np.zeros((N, 18))
    C = np.zeros(18 * N, dtype=int)
    for n in range(N):
        filt = A[n, :] != 0
        Ai = A[n, filt]
        conn = np.where(filt)[0]
        spA[n, :len(Ai)] = Ai
        C[18 * n:18 * n + len(conn)] = conn
    t0 = time.time()
    spld_x = pmlcsolver.cg_solve_sp(spA, C, b)
    times[i, 5] = time.time() - t0

print(' ,' + str(labels).replace('[', '').replace(']', '').replace('\'', ''))
for i in range(len(testnames)):
    print(
        str(Ns[i]) + ',' + str(times[i, :]).replace('[', '').replace(']', '').replace(
            '\'', '').replace('  ', ' ').replace(' ', ','))
