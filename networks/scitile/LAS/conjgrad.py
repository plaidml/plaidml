import numpy as np
import sys
import time

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve as densolve

import plaidml
import plaidml.edsl as edsl
import plaidml.op as op
import plaidml.exec as plaidml_exec

RESIDUAL_THRESHOLD = 1e-7


def matmul(A, B):
    I, J = edsl.TensorDims(2)
    i, j = edsl.TensorIndexes(2)
    A.bind_dims(I, J)
    B.bind_dims(J)
    C = edsl.TensorOutput(I)
    C[i] += A[i, j] * B[j]
    return C


def cg_solve(np_A, np_r, timing=False, resthrsh=RESIDUAL_THRESHOLD):
    N = len(np_r)

    np_x = np.zeros(N)
    np_rsq = np.matmul(np_r, np_r)
    np_p = np_r.copy()

    dtype = plaidml.DType.FLOAT32

    A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
    X = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
    R = edsl.Tensor(edsl.LogicalShape(dtype, np_r.shape))
    P = edsl.Tensor(edsl.LogicalShape(dtype, np_p.shape))
    RSQ = edsl.Tensor(edsl.LogicalShape(dtype, np_rsq.shape))

    Ap = matmul(A, P)
    alpha = RSQ / op.sum(P * Ap, axis=0)
    OX = X + alpha * P
    OR = R - alpha * Ap
    RSQN = op.sum(OR * OR, axis=0)
    OP = OR + RSQN / RSQ * P

    Is = [A, P, RSQ, X, R]
    Os = [OX, OR, RSQN, OP]
    program = edsl.Program('las_step', Os)
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()

    binder.input(A).copy_from_ndarray(np_A)
    binder.input(P).copy_from_ndarray(np_p)
    binder.input(RSQ).copy_from_ndarray(np_rsq)
    binder.input(X).copy_from_ndarray(np_x)
    binder.input(R).copy_from_ndarray(np_r)

    t0 = time.time()
    for tt in range(N):
        executable.run()

        binder.input(X).copy_from_ndarray(binder.output(OX).as_ndarray())
        binder.input(P).copy_from_ndarray(binder.output(OP).as_ndarray())
        binder.input(R).copy_from_ndarray(binder.output(OR).as_ndarray())
        rsq = binder.output(RSQN).as_ndarray()
        if rsq < resthrsh:
            break
        else:
            binder.input(RSQ).copy_from_ndarray(rsq)

    tm = time.time() - t0

    np_x = binder.output(OX).as_ndarray()
    if timing:
        return np_x, tm
    return np_x


if __name__ == '__main__':
    # Load test case
    testname = sys.argv[1]
    testdat = np.load('networks/scitile/LAS/test_data/' + testname +
                      '.npz')  # networks/scitile/LAS/
    A, b = testdat['A'], testdat['b']
    testdat.close()

    print("Test case data loaded")

    np_x, tm = cg_solve(A, b, True)

    print("PlaidML Time: ", tm)

    sp_A = csc_matrix(A)
    t0 = time.time()
    true_sol = spsolve(sp_A, b)
    scpysp_tm = time.time() - t0

    print("SciPy Sparse Time: ", scpysp_tm)

    thrs = 0.01 * np.mean(true_sol)
    err = 100 * np.mean(
        np.abs((np_x[abs(true_sol) > thrs] - true_sol[abs(true_sol) > thrs]) /
               true_sol[abs(true_sol) > thrs]))
    print("Avg. % Error: ", err)
