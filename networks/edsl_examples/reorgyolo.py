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
# TODO: test against pytorch implementation


def reorgyolo(I, s, forward=False):

    #forward = false ->chennel increase [N,C,H,W] -> [N, C*(s^2), H/s, W/s]
    #forward = true -> channel decrease [N,C,H,W] -> [N, C/(s^2), H*s, W*s]

    #get input tensor dimensions
    dims = I.shape.int_dims
    N = dims[0]
    C = dims[1]
    H = dims[2]
    W = dims[3]
    total_dims = dims[0] * dims[1] * dims[2] * dims[3]
    N_out = N
    C_out = C * (s * s)
    H_out = int(H / s)
    W_out = int(W / s)

    if forward == False:
        #print(str(dims[0]) + "," + str(dims[1]*(s*s)) + "," + str(int(dims[2]/s))+ "," + str(int(dims[3]/s)))
        O_linear = plaidml_op.reshape(I, [total_dims])
        O = edsl.TensorOutput(total_dims)
        i = edsl.TensorIndex()
        n, c, h, w = edsl.TensorIndexes(4)
        no, co, ho, wo = edsl.TensorIndexes(4)
        O[w * s + W * s * ((h * s) + H * s * (c + C_out * n))] += O_linear[w + W * (h + H *
                                                                                    (c + C * n))]

        # dims_out = edsl.TensorDims(1)
        # #O_linear.bind_dims(*dims)
        # O = edsl.TensorOutput(*dims)
        # O[i] = O_linear[i+1]

        # N,C,H,W = edsl.TensorDims(4)
        # n,c,h,w = edsl.TensorIndexes(4)
        # no,co,ho,wo = edsl.TensorIndexes(4)
        # I.bind_dims(N,C,H,W)
        # O = edsl.TensorOutput(N_out,C_out,H_out,W_out)
        # O[no,co+(c%C_out),ho*s,wo*s]=I[n,c,h,w] #mod support might be useful TODO: figure out a way to use constraints to accomplish this
        # O = plaidml_op.reshape(I,[dims[0],int (dims[1]*(s*s)),int(dims[2]/s),int(dims[3]/s)])
    elif forward == True:
        #print(str(dims[0]) + "," + str(int(dims[1]/(s*s))) + "," + str(dims[2]*s)+ "," + str(dims[3]*s))
        O = plaidml_op.reshape(
            I,
            [dims[0], int(dims[1] / (s * s)),
             int(dims[2] * s), int(dims[3] * s)])

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
    O = reorgyolo(I, stride, False)

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
