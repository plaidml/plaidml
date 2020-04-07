#   This example illustrates how to use eDSL to write the reorg layer employed in YOLO-v2
#   --------------------------Op description-----------------------------------
#   The reorg layer is employed in YOLO-v2 or Darknet to combine midlevel and
#   high level features. The reorg layer reshapes the output tensor so that
#   the height and width are aligned with the output tensor of a different
#   layer in the network.
#   inputs:
#            Input Tensor:
#                   dimensions N,C,H,W
#            forward:
#                   boolean argument
#                              forward = true -> channel decrease
#                                        [N,C,H,W] -> [N, C/(s^2), H*s, W*s]
#                              forward = false ->channel increase
#                                        [N,C,H,W] -> [N, C*(s^2), H/s, W/s]
#    output:
#            Output Tensor:
#                   dimentions [N, C*(s^2), H/s, W/s] OR [N, C/(s^2), H*s, W*s]
#   -------------------------- Testing method-------------------------------------
#   The reorg function provided here:
#   https://gist.github.com/leimao/ece7217b5d07fe4e685c47af5e76744a
#   is currently being used for local testing
# TODO: iron out issues with channel increase (forward = false )
# TODO: write python unittests
# TODO: write backend_test style test against pytorch implementation
# TODO: remove test functions and python test code
# TODO: write a note on handling the modulus operator in eDSL
# TODO: write a note on how floor division works in eDSL

import numpy as np

import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec
import plaidml2.op as plaidml_op


def reorgyolo_comparison(arrayIn, batch, C, H, W, stride, forward=False):
    arrayLen = len(arrayIn)
    arrayOut = np.zeros(arrayLen)
    print("C is " + str(C) + "stride is " + str(stride))
    out_c = C // (stride * stride)
    for b in range(batch):
        for k in range(C):
            for j in range(H):
                for i in range(W):
                    in_index = i + W * (j + H * (k + C * b))
                    c2 = k % out_c
                    offset = k // out_c
                    w2 = i * stride + offset % stride
                    h2 = j * stride + offset // stride
                    out_index = int(w2 + W * stride * (h2 + H * stride * (c2 + out_c * b)))
                    if forward:
                        arrayOut[out_index] = arrayIn[in_index]
                    else:
                        arrayOut[in_index] = arrayIn[out_index]
    return arrayOut


def reorgyolo_comparison_nodivmod(arrayIn, batch, C, H, W, stride, forward=False):
    arrayLen = len(arrayIn)
    arrayOut = np.zeros(arrayLen)
    print("C is " + str(C) + "stride is " + str(stride))
    out_c = C // (stride * stride)
    _c1_quotient_range = int(C // (out_c))
    for n1 in range(batch):
        for w1 in range(W):
            for h1 in range(H):
                for c2 in range(out_c):
                    for _w2_quotient in range(_c1_quotient_range // stride):
                        for _w2 in range(stride):
                            _c1 = _w2 + _w2_quotient * stride
                            c1 = c2 + (_c1 * out_c)
                            in_index = w1 + W * (h1 + H * (c1 + C * n1))
                            w2 = w1 * stride + _w2
                            h2 = h1 * stride + _w2_quotient
                            out_index = int(w2 + W * stride * (h2 + H * stride *
                                                               (c2 + out_c * n1)))
                            if forward:
                                arrayOut[out_index] = arrayIn[in_index]
                            else:
                                arrayOut[in_index] = arrayIn[out_index]
    return arrayOut


def reorgyolo(I, stride, forward):
    dims = I.shape.int_dims
    N = dims[0]
    C = dims[1]
    H = dims[2]
    W = dims[3]

    C_decrease = int(C // (stride * stride))
    _c1_quotient_range = int(C // (C_decrease))
    _w2_quotient_range = int(_c1_quotient_range // stride)

    N_in, C_in, H_in, W_in = edsl.TensorDims(4)
    n1, w1, h1, c2, _w2_quotient, _w2 = edsl.TensorIndexes(6)
    I.bind_dims(N_in, C_in, H_in, W_in)

    if forward:
        O = edsl.TensorOutput(N, C_decrease, H * stride, W * stride)
        O[n1, c2, h1 * stride + _w2_quotient, w1 * stride +
          _w2] = I[n1, c2 + ((_w2 + _w2_quotient * stride) * C_decrease), h1, w1]
        O.add_constraint(c2 < C_decrease)
    else:
        C_increase = int(C * (stride * stride))
        O = edsl.TensorOutput(N, C_increase, int(H / stride), int(W / stride))
        O[n1, c2 +
          ((_w2 + _w2_quotient * stride) * C), h1, w1] = I[n1, c2, h1 * stride +
                                                           _w2_quotient, w1 * stride + _w2]
        O.add_constraint(c2 < C)

    O.add_constraint(_w2 < stride)
    O.add_constraint(_w2_quotient < _w2_quotient_range)
    return O


def main():
    n_i = 1
    c_i = 4
    h_i = 6
    w_i = 6
    stride = 2
    forward = False

    I_data_linear = np.array(list(range(n_i * c_i * h_i * w_i))).astype(np.int)
    I_data = np.reshape(I_data_linear, (n_i, c_i, h_i, w_i))

    print("_______________________________________________")
    print("starting matrix")
    print("_______________________________________________")
    print(I_data)
    print("_______________________________________________")

    I = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_data.shape))
    O = reorgyolo(I, stride, forward)
    if forward:
        c_o = c_i // (stride * stride)
        h_o = h_i * stride
        w_o = w_i * stride
    else:
        c_o = c_i * (stride * stride)
        h_o = h_i // stride
        w_o = w_i // stride

    program = edsl.Program('reorgyolo', [O])
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()

    def run():
        binder.input(I).copy_from_ndarray(I_data)
        executable.run()
        return binder.output(O).as_ndarray()

    result = run()

    print("_______________________________________________")
    print("eDSL computed result: \n{}".format(result))
    print("_______________________________________________")

    O_l = reorgyolo_comparison_nodivmod(I_data_linear,
                                        batch=n_i,
                                        C=c_i,
                                        H=h_i,
                                        W=w_i,
                                        stride=stride,
                                        forward=forward)
    O_new = np.reshape(O_l, (n_i, c_o, h_o, w_o))

    print("_______________________________________________")
    print("new result: \n{}".format(O_new))
    print("_______________________________________________")

    O_l = reorgyolo_comparison(I_data_linear,
                               batch=n_i,
                               C=c_i,
                               H=h_i,
                               W=w_i,
                               stride=stride,
                               forward=forward)
    O_exp = np.reshape(O_l, (n_i, c_o, h_o, w_o))

    print("_______________________________________________")
    print("expected result: \n{}".format(O_exp))
    print("_______________________________________________")


if __name__ == '__main__':
    main()
