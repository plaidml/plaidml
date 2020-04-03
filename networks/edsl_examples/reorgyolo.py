import numpy as np

import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec
import plaidml2.op as plaidml_op

# Notes:
# the reorg function provided here https://gist.github.com/leimao/ece7217b5d07fe4e685c47af5e76744a is being used for local testing
# a number of approaches were tried out in EDSL none of them have led to the correct result so far
# handling the offset is an issue
# no implementaton exists in keras but a recommended approach is described here : https://github.com/thtrieu/darkflow/issues/173 suggests permute
# might be good to have
# TODO: write python unittests
# TODO: write backend_test style test against pytorch implementation


def reorgyolo(I, stride, forward=False):

    #forward = false ->chennel increase [N,C,H,W] -> [N, C*(s^2), H/s, W/s]
    #forward = true -> channel decrease [N,C,H,W] -> [N, C/(s^2), H*s, W*s]

    #get input tensor dimensions
    dims = I.shape.int_dims
    N = dims[0]
    C = dims[1]
    H = dims[2]
    W = dims[3]

    if forward == False:
        N_out = N
        C_out = int(C // (stride * stride))
        H_out = int(H * stride)
        W_out = int(W * stride)
        print(str(N_out) + "," + str(C_out) + "," + str(H_out) + "," + str(W_out))
        N, C, H, W = edsl.TensorDims(4)
        n, c, c2, h, w, offset = edsl.TensorIndexes(6)
        I.bind_dims(N, C, H, W)
        O = edsl.TensorOutput(N_out, C_out, H_out, W_out)
        O[n, c2, (h * stride) + offset, (w * stride) + offset] = I[n, c, h, w]
        #mod support might be useful TODO: figure out a way to use constraints to accomplish this
        O.add_constraint(c2 < C_out)
        O.add_constraint(offset < stride)
    elif forward == True:
        N_out = N
        C_out = int(C * (stride * stride))
        H_out = int(H / stride)
        W_out = int(W / stride)
        print(str(N_out) + "," + str(C_out) + "," + str(H_out) + "," + str(W_out))
        N, C, H, W = edsl.TensorDims(4)
        n, c, c2, h, w, offset = edsl.TensorIndexes(6)
        I.bind_dims(N, C, H, W)
        O = edsl.TensorOutput(N_out, C_out, H_out, W_out)
        O[n, c2, h, w] = I[n, c, (h * stride) + offset, (w * stride) + offset]
        #mod support might be useful TODO: figure out a way to use constraints to accomplish this
        O.add_constraint(c2 < C_out)
        O.add_constraint(offset < stride)

    return O


def main():
    n_i = 2
    c_i = 4
    h_i = 6
    w_i = 6
    stride = 2

    I_data_linear = np.array(list(range(n_i * c_i * h_i * w_i))).astype(np.int)
    I_data = np.reshape(I_data_linear, (n_i, c_i, h_i, w_i))

    print("_______________________________________________")
    print("starting matrix")
    print("_______________________________________________")
    print(I_data)
    print("_______________________________________________")

    I = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_data.shape))
    O = reorgyolo(I, stride, True)

    c_o = c_i * stride * stride
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
    print("computed result: {}".format(result))
    print("_______________________________________________")


if __name__ == '__main__':
    main()
