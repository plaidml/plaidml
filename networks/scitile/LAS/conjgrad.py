import numpy as np
import sys
import time

import plaidml
import plaidml.op as op
import plaidml.tile as tile



N = int(sys.argv[1])

# Generate a reasonable test system
np_A = np.empty((N,N))
for i in range(N):
    for j in range(N):
        np_A[i,j] = N - abs(i-j)
# b = np.random.rand(N)
b = np.zeros(N)
b[-1] = 1

np_x = np.zeros(N)
np_r = b - np.matmul(np_A, np_x)
np_rsq = np.matmul(np_r, np_r)
np_p = np_r.copy()

dtype = plaidml.DType.FLOAT32
ctx = plaidml.Context()
with plaidml.open_first_device(ctx) as dev:
    A = tile.Value.from_ndims(2) #input shape placeholder
    P = tile.Value.from_ndims(1) #Placeholder
    RSQ = tile.Value.from_ndims(0)
    X = tile.Value.from_ndims(1)
    R = tile.Value.from_ndims(1)

    Ap = op.matmul(A, P)
    alpha = RSQ/op.matmul(P, Ap)
    OX = X + alpha*P
    OR = R - alpha*Ap
    RSQN = op.matmul(OR, OR)
    OP = OR + RSQN/RSQ * P

    func = tile.compose(ctx, dev, inputs=[('A', A), ('P', P), ('RSQ', RSQ), ('X', X), ('R', R)], 
                                  outputs=[('OP', OP), ('RSQN', RSQN), ('OX', OX), ('OR', OR)])
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

    invoker.set_output('OP', pld_P)
    invoker.set_output('RSQN', pld_RSQ)
    invoker.set_output('OX', pld_X)
    invoker.set_output('OR', pld_R)
    
    t0 = time.time()
    for tt in range(N):
        invoker.invoke()

    with pld_X.mmap_current() as view:
        view.copy_to_ndarray(np_x)

    print("Time: ",time.time()-t0)

    true_sol = np.linalg.solve(np_A, b)
    err = 100*np.mean(np.abs((np_x[abs(true_sol)>1e-2]-true_sol[abs(true_sol)>1e-2])/true_sol[abs(true_sol)>1e-2]))
    print("Error: ",err)