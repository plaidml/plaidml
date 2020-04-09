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
# TODO: handle cases where the spatial dims aren't evenly divided by stride
# TODO: neaten up the reorgyolo c_o>c_in issue in tester function
# TODO: either figure out how to reference leimao blog code or get rid of it
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
    out_c = C // (stride * stride)
    for n1 in range(batch):
        for w1 in range(W):
            for h1 in range(H):
                for c2 in range(out_c):
                    for h_jump in range(stride):
                        for w_jump in range(stride):
                            _c1 = w_jump + h_jump * stride
                            c1 = c2 + (_c1 * out_c)
                            in_index = w1 + W * (h1 + H * (c1 + C * n1))
                            w2 = w1 * stride + w_jump
                            h2 = h1 * stride + h_jump
                            out_index = int(w2 + W * stride * (h2 + H * stride *
                                                               (c2 + out_c * n1)))
                            if forward:
                                arrayOut[out_index] = arrayIn[in_index]
                            else:
                                arrayOut[in_index] = arrayIn[out_index]
    return arrayOut


def reorgyolo(I, stride, decrease_C):
    N, C, H, W = edsl.TensorDims(4)
    n, w1, h1, c2, h_jump, w_jump = edsl.TensorIndexes(6)
    I.bind_dims(N, C, H, W)
    h2 = h1 * stride + h_jump
    w2 = w1 * stride + w_jump
    if decrease_C:
        c1 = c2 + ((w_jump + h_jump * stride) * (C // (stride * stride)))
        O = edsl.TensorOutput(N, C // (stride * stride), H * stride, W * stride)
        O[n, c2, h2, w2] = I[n, c1, h1, w1]
    else:
        c1 = c2 + ((w_jump + h_jump * stride) * C)
        O = edsl.TensorOutput(N, C * (stride * stride), H // stride, W // stride)
        O[n, c1, h1, w1] = I[n, c2, h2, w2]
    O.add_constraint(h_jump < stride)
    O.add_constraint(w_jump < stride)

    return O


def main():
    n_i = 1
    c_i = 4
    h_i = 6
    w_i = 6
    stride = 2
    decrease = False

    I_data_linear = np.array(list(range(n_i * c_i * h_i * w_i))).astype(np.int)
    I_data = np.reshape(I_data_linear, (n_i, c_i, h_i, w_i))

    print("_______________________________________________")
    print("starting matrix")
    print("_______________________________________________")
    print(I_data)
    print("_______________________________________________")

    I = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_data.shape))
    O = reorgyolo(I, stride, decrease)
    if decrease:
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
    binder.input(I).copy_from_ndarray(I_data)
    executable.run()
    result = binder.output(O).as_ndarray()

    print("_______________________________________________")
    print("eDSL computed result: \n{}".format(result))
    print("_______________________________________________")

    O_l = reorgyolo_comparison_nodivmod(I_data_linear,
                                        batch=n_i,
                                        C=c_i,
                                        H=h_i,
                                        W=w_i,
                                        stride=stride,
                                        forward=decrease)
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
                               forward=decrease)

    if (c_o > c_i):
        expected_result_l = reorgyolo_comparison(I_data_linear,
                                                 batch=n_i,
                                                 C=c_o,
                                                 H=h_o,
                                                 W=w_o,
                                                 stride=stride,
                                                 forward=decrease)
    else:
        expected_result_l = reorgyolo_comparison(I_data_linear,
                                                 batch=n_i,
                                                 C=c_i,
                                                 H=h_i,
                                                 W=w_i,
                                                 stride=stride,
                                                 forward=decrease)
    O_exp = np.reshape(expected_result_l, (n_i, c_o, h_o, w_o))
    print("_______________________________________________")
    print("expected result: \n{}".format(O_exp))
    print("_______________________________________________")


if __name__ == '__main__':
    main()
