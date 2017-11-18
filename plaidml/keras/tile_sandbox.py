from collections import OrderedDict
import numpy as np
import plaidml
import plaidml.keras
plaidml.keras.install_backend()
import keras.backend as K


def main(code, tensor_A, tensor_B, output_shape):
    print(K.backend())
    op = K._Op('sandbox_op', A.dtype, output_shape, code,
               OrderedDict([('A', tensor_A), ('B', tensor_B)]), ['O'])
    print(op.eval())


if __name__ == '__main__':
    plaidml._internal_set_vlog(3)
    A = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
    B = K.variable(np.array([-7., -1., 2.]))
    #    code = """function (A[N, M], B[M]) -> (O) {
    #                  O[i, j: N, M] = =(A[i, j] + B[j]), i/2 + j/2 + 1/2 < 2;
    #              }"""
    #    out_shape = (2, 3)
    code = """function (A[N, M], B[M]) -> (O) {
                  O[i: N] = +(A[i - j, 0] + B[0]), j < N;
              }"""
    out_shape = (3,)
    main(code, A, B, out_shape)
