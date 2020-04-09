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
# TODO: write python unittests
# TODO: write backend_test style test against pytorch implementation
# TODO: remove test functions and python test code
# TODO: write a note on handling the modulus operator in eDSL
# TODO: write a note on how floor division works in eDSL

import numpy as np

import plaidml as plaidml
import plaidml.edsl as edsl
import plaidml.exec as plaidml_exec
import plaidml.op as plaidml_op


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
    C_out = C // (stride * stride)
    x_range = C // C_out // stride
    for b in range(batch):
        for i in range(W):
            for j in range(H):
                for k in range(C_out):
                    for x in range(x_range):
                        for y in range(stride):
                            c = k + ((y + x * stride) * C_out)
                            in_index = i + W * (j + H * (c + C * b))
                            h = j * stride + x
                            w = i * stride + y
                            out_index = w + W * stride * (h + H * stride * (k + C_out * b))
                            if forward:
                                arrayOut[out_index] = arrayIn[in_index]
                            else:
                                arrayOut[in_index] = arrayIn[out_index]
    return arrayOut


def reorgyolo(I, S, decrease):
    N, C, H, W = edsl.TensorDims(4)
    I.bind_dims(N, C, H, W)

    b, i, j, k, x, y = edsl.TensorIndexes(6)
    h = j * S + x
    w = i * S + y

    if decrease:
        C_out = C // (S * S)
        O = edsl.TensorOutput(N, C_out, H * S, W * S)
        c = k + ((y + x * S) * C_out)
        O[b, k, h, w] = I[b, c, j, i]
        O.add_constraint(x < C // C_out // S)
    else:
        C_out = C * (S * S)
        O = edsl.TensorOutput(N, C_out, H // S, W // S)
        c = k + ((y + x * S) * C)
        O[b, c, j, i] = I[b, k, h, w]

    O.add_constraint(y < S)

    return O


def main():
    n_i = 1
    c_i = 4
    h_i = 6
    w_i = 6
    stride = 2
    decrease = True
    if decrease:
        c_o = c_i // (stride * stride)
        h_o = h_i * stride
        w_o = w_i * stride
    else:
        c_o = c_i * (stride * stride)
        h_o = h_i // stride
        w_o = w_i // stride

    I_data_linear = np.array(list(range(n_i * c_i * h_i * w_i))).astype(np.float)
    I_data = np.reshape(I_data_linear, (n_i, c_i, h_i, w_i))

    print("_______________________________________________")
    print("starting matrix")
    print("_______________________________________________")
    print(I_data)
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
    O_exp = np.reshape(O_l, (n_i, c_o, h_o, w_o))

    print("_______________________________________________")
    print("expected result: \n{}".format(O_exp))
    print("_______________________________________________")

    I = edsl.Placeholder(plaidml.DType.FLOAT32, dims=I_data.shape)
    O = reorgyolo(I, stride, decrease)

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


if __name__ == '__main__':
    main()
