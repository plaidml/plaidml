import numpy as np
import sys
import time

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve as densolve

import plaidml
import plaidml.op as op
import plaidml.tile as tile

RESIDUAL_THRESHOLD = 1e-7


def cg_solve(np_A, np_r, timing=False, resthrsh=RESIDUAL_THRESHOLD):
    N = len(np_r)

    np_x = np.zeros(N)
    np_rsq = np.matmul(np_r, np_r)
    np_p = np_r.copy()
    dtype = plaidml.DType.FLOAT32
    ctx = plaidml.Context()

    with plaidml.open_first_device(ctx) as dev:
        A = tile.Value.from_ndims(2)  #input shape placeholder
        P = tile.Value.from_ndims(1)  #Placeholder
        RSQ = tile.Value.from_ndims(0)
        X = tile.Value.from_ndims(1)
        R = tile.Value.from_ndims(1)

        Ap = op.matmul(A, P)
        alpha = RSQ / op.matmul(P, Ap)
        OX = X + alpha * P
        OR = R - alpha * Ap
        RSQN = op.matmul(OR, OR)
        OP = OR + RSQN / RSQ * P

        func = tile.compose(ctx,
                            dev,
                            inputs=[('A', A), ('P', P), ('RSQ', RSQ), ('X', X), ('R', R)],
                            outputs=[('OX', OX), ('OR', OR), ('RSQN', RSQN), ('OP', OP)])
        invoker = plaidml.Invoker(ctx, func)

        pld_A = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_A.shape))
        pld_P = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_p.shape))
        pld_RSQ = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_rsq.shape))
        pld_X = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_x.shape))
        pld_R = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_r.shape))

        with pld_A.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_A)
            view.writeback()
        with pld_P.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_p)
            view.writeback()
        with pld_RSQ.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_rsq)
            view.writeback()
        with pld_X.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_x)
            view.writeback()
        with pld_R.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_r)
            view.writeback()

        invoker.set_input('A', pld_A)
        invoker.set_input('P', pld_P)
        invoker.set_input('RSQ', pld_RSQ)
        invoker.set_input('X', pld_X)
        invoker.set_input('R', pld_R)

        invoker.set_output('OX', pld_X)
        invoker.set_output('OR', pld_R)
        invoker.set_output('RSQN', pld_RSQ)
        invoker.set_output('OP', pld_P)

        t0 = time.time()
        for tt in range(N):
            invoker.invoke()
            with pld_RSQ.mmap_current() as view:
                if view[0] < resthrsh:
                    break
        tm = time.time() - t0

        with pld_X.mmap_current() as view:
            view.copy_to_ndarray(np_x)

    if timing:
        return np_x, tm
    return np_x


def cg_solve_sp(np_A, np_C, np_r, timing=False, resthrsh=RESIDUAL_THRESHOLD):
    N = len(np_r)

    np_x = np.zeros(N)
    np_rsq = np.matmul(np_r, np_r)
    np_p = np_r.copy()
    dtype = plaidml.DType.FLOAT32
    ctx = plaidml.Context()

    with plaidml.open_first_device(ctx) as dev:
        A = tile.Value.from_ndims(2)  #input shape placeholder
        C = tile.Value.from_ndims(1)
        P = tile.Value.from_ndims(1)  #Placeholder
        RSQ = tile.Value.from_ndims(0)
        X = tile.Value.from_ndims(len(np_x.shape))
        R = tile.Value.from_ndims(1)

        gP = op.gather(P, C)
        grP = op.reshape(gP, (N, 18))
        AgrP = A * grP
        Ap = op.summation(AgrP, axes=1)
        alpha = RSQ / op.matmul(P, Ap)
        OX = X + alpha * P
        OR = R - alpha * Ap
        RSQN = op.matmul(OR, OR)
        OP = OR + RSQN / RSQ * P

        func = tile.compose(ctx,
                            dev,
                            inputs=[('A', A), ('C', C), ('P', P), ('RSQ', RSQ), ('X', X),
                                    ('R', R)],
                            outputs=[('OX', OX), ('OR', OR), ('RSQN', RSQN), ('OP', OP)])
        invoker = plaidml.Invoker(ctx, func)

        pld_A = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_A.shape))
        pld_C = plaidml.Tensor(dev, plaidml.Shape(ctx, plaidml.DType.INT32, *np_C.shape))
        pld_P = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_p.shape))
        pld_RSQ = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_rsq.shape))
        pld_X = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_x.shape))
        pld_R = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_r.shape))

        with pld_A.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_A)
            view.writeback()
        with pld_C.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_C)
            view.writeback()
        with pld_P.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_p)
            view.writeback()
        with pld_RSQ.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_rsq)
            view.writeback()
        with pld_X.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_x)
            view.writeback()
        with pld_R.mmap_discard(ctx) as view:
            view.copy_from_ndarray(np_r)
            view.writeback()

        invoker.set_input('A', pld_A)
        invoker.set_input('C', pld_C)
        invoker.set_input('P', pld_P)
        invoker.set_input('RSQ', pld_RSQ)
        invoker.set_input('X', pld_X)
        invoker.set_input('R', pld_R)

        invoker.set_output('OX', pld_X)
        invoker.set_output('OR', pld_R)
        invoker.set_output('RSQN', pld_RSQ)
        invoker.set_output('OP', pld_P)

        invoker.invoke()

        t0 = time.time()
        for tt in range(N):
            invoker.invoke()
            with pld_RSQ.mmap_current() as view:
                if view[0] < resthrsh:
                    break
        tm = time.time() - t0

        with pld_X.mmap_current() as view:
            view.copy_to_ndarray(np_x)

    if timing:
        return np_x, tm
    return np_x


def dense_to_sparse(A):
    N = A.shape[0]
    spA = np.zeros((N, 18))
    C = np.zeros(18 * N, dtype=int)
    for i in range(N):
        filt = A[i, :] != 0
        Ai = A[i, filt]
        conn = np.where(filt)[0]
        spA[i, :len(Ai)] = Ai
        C[18 * i:18 * i + len(conn)] = conn
    return spA, C


if __name__ == '__main__':
    # Load test case
    testname = sys.argv[1]
    testdat = np.load('networks/scitile/LAS/test_data/' + testname +
                      '.npz')  # networks/scitile/LAS/
    A, b = testdat['A'], testdat['b']
    testdat.close()

    print("Test case data loaded")

    spA, C = dense_to_sparse(A)
    np_x, tm = cg_solve_sp(spA, C, b, True)
    # np_x, tm = cg_solve(A, b, True)

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
