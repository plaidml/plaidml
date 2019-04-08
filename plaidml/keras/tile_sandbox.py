from collections import OrderedDict
import numpy as np
import plaidml
import plaidml.tile as tile
import plaidml.keras
plaidml.keras.install_backend()
import keras.backend as K


class SandboxOp(tile.Operation):

    def __init__(self, code, a, b, output_shape):
        super(SandboxOp, self).__init__(code, [('A', a), ('B', b)], [('O', output_shape)])


def main(code, tensor_A, tensor_B, output_shape):
    print(K.backend())
    op = SandboxOp(code, tensor_A, tensor_B, tile.Shape(plaidml.DType.FLOAT32, output_shape))
    print(op.sole_output().shape)
    print(op.sole_output().eval())


if __name__ == '__main__':
    plaidml._internal_set_vlog(1)
    A = K.variable(np.arange(12).reshape(4, 3))
    B = K.variable(np.arange(3).reshape(3))
    code = """function (A[N, M], B[M]) -> (O) {
        O[i, j: N, M] = =(A[i, j] + B[j]), i/2 + j/2 + 1/2 < 2;
    }"""
    out_shape = (2, 3)
    main(code, A, B, out_shape)
