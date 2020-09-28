import numpy as np
import sys
import time

import plaidml
import plaidml.op as op
import plaidml.tile as tile

N = int(sys.argv[1])

#Generate a well-conditioned test matrix
R = np.random.rand(N, N)
np_A = R @ R.T + N * np.eye(N)

np_L = np.zeros((N, N))
np_R = np.zeros((N, 1))

dtype = plaidml.DType.FLOAT32
ctx = plaidml.Context()
with plaidml.open_first_device(ctx) as dev:
    # Define Plaid Functions
    A = tile.Value.from_ndims(2)  #N x N
    R = tile.Value.from_ndims(2)  #N x 1
    LCC = tile.Value.from_ndims(0)  #1
    C = tile.Value.from_ndims(0)  #1

    # Update A Function
    OA = A - op.matmul(R, op.reshape(R, [1, N]))
    updA = tile.compose(ctx, dev, inputs=[('A', A), ('R', R)], outputs=[('OA', OA)])
    updA_inv = plaidml.Invoker(ctx, updA)

    # Update R Function
    OR = tile.Operation(
        """function (A[I,J], LCC, C) -> (OR) {
                                TMP[i: I] = +(A[i,C]);
                                OR = TMP/LCC;
                            }
                            """, [('A', A), ('LCC', LCC), ('C', C)],
        [('OR', tile.Shape(dtype, np_R.shape))]).outputs['OR']
    updR = tile.compose(ctx, dev, inputs=[('A', A), ('LCC', LCC), ('C', C)], outputs=[('OR', OR)])
    updR_inv = plaidml.Invoker(ctx, updR)

    # Update C Function
    # OC = C + 1
    # updC = tile.compose(ctx, dev, inputs=[('C', C)], outputs=[('OC', OC)])
    # updC_inv = plaidml.Invoker(ctx, updC)

    # Bind numpy inputs
    pld_A = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_A.shape))
    pld_R = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *np_R.shape))
    pld_LCC = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *()))
    # pld_C = plaidml.Integer(0)

    with pld_A.mmap_discard(ctx) as view:
        view.copy_from_ndarray(np_A)
        view.writeback()
    with pld_R.mmap_discard(ctx) as view:
        view.copy_from_ndarray(np_R)
        view.writeback()

    # Bind pmlc inputs & outputs to functions
    updA_inv.set_input('A', pld_A)
    updA_inv.set_input('R', pld_R)
    updA_inv.set_output('OA', pld_A)

    updR_inv.set_input('A', pld_A)
    updR_inv.set_input('LCC', pld_LCC)
    # updR_inv.set_input('C', pld_C)
    updR_inv.set_output('OR', pld_R)

    # updC_inv.set_input('C', pld_C)
    # updC_inv.set_output('OC', pld_C)

    # Actual Decomposition
    t0 = time.time()
    for c in range(N):
        pld_C = plaidml.Integer(c)
        stpt0 = time.time()
        if c > 0:
            updA_inv.invoke()
        with pld_A.mmap_current() as view:
            Lcc = (view[c * N + c])**0.5
        with pld_LCC.mmap_discard(ctx) as view:
            view[0] = Lcc
            view.writeback()
        updR_inv.set_input('C', plaidml.Integer(c))
        updR_inv.invoke()
        with pld_R.mmap_current() as view:
            view.copy_to_ndarray(np_L[:, c:c + 1])
        np_L[c, c] = Lcc
        # print(time.time()-stpt0)
        # updC_inv.invoke()

    print("Time: ", time.time() - t0)

    true_chol = np.linalg.cholesky(np_A)
    err = 100 * np.mean(
        np.abs((np_L[abs(true_chol) > 1e-2] - true_chol[abs(true_chol) > 1e-2]) /
               true_chol[abs(true_chol) > 1e-2]))
    print("Error: ", err)
