import numpy as np
import sys
import time

import plaidml
import plaidml.edsl as edsl
import plaidml.op as op
import plaidml.exec as plaidml_exec
from plaidml import Program

RESIDUAL_THRESHOLD = 1e-7


def matmul(A, B):
    I, J = edsl.TensorDims(2)
    i, j = edsl.TensorIndexes(2)
    A.bind_dims(I, J)
    B.bind_dims(J)
    return edsl.Contraction().outShape(I).outAccess(i).sum(A[i, j] * B[j]).build()


def cg_solve(np_A, np_r, timing=False, resthrsh=RESIDUAL_THRESHOLD):
    N = len(np_r)

    np_x = np.zeros(N)
    np_rsq = np.matmul(np_r, np_r)
    np_p = np_r.copy()

    dtype = plaidml.DType.FLOAT32

    A = edsl.Placeholder(dtype, np_A.shape)
    X = edsl.Placeholder(dtype, np_x.shape)
    R = edsl.Placeholder(dtype, np_r.shape)
    P = edsl.Placeholder(dtype, np_p.shape)
    RSQ = edsl.Placeholder(dtype, np_rsq.shape)

    Ap = matmul(A, P)
    alpha = RSQ / op.sum(P * Ap, axis=0)
    OX = X + alpha * P
    OR = R - alpha * Ap
    RSQN = op.sum(OR * OR, axis=0)
    OP = OR + RSQN / RSQ * P

    Is = [A, P, RSQ, X, R]
    Os = [OX, OR, RSQN, OP]
    program = Program('las_step', Is, Os)

    t0 = time.time()
    inputs = [np_A, np_p, np_rsq, np_x, np_r]
    for tt in range(1):
        outputs = plaidml.exec.run(program, inputs)
        if outputs[2] < resthrsh:
            break
        else:
            inputs[1] = outputs[3]
            inputs[2] = outputs[2]
            inputs[3] = outputs[0]
            inputs[4] = outputs[1]

    tm = time.time() - t0

    np_x = outputs[0]
    if timing:
        return np_x, tm
    return np_x
