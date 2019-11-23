import timeit
from collections import OrderedDict

import numpy as np
import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec


def sq(X):
    return X * X


def sum(R):
    idx = edsl.TensorIndex()
    O = edsl.TensorOutput()
    O[()] += R[idx]
    return O


#n: number of darts thrown in 2x2 square
#   The function computes the ratio of points that fall inside a circle
#   of radius 1 (inscribed with-in the 2X2 square region) to the number of
#   points that fall outside. This ratio is used this to compute the area
#   of the inscribed circle ~ pi
def monte_carlo_pi(n, benchmark=False):
    np.random.seed(2)
    values_x = np.random.rand(n)
    values_y = np.random.rand(n)

    X_data = values_x.reshape(n)
    Y_data = values_y.reshape(n)

    X = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, X_data.shape))
    Y = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, Y_data.shape))

    Z = sq(X) + sq(Y)

    Z = edsl.select(Z <= 1, 1, 0)
    O = sum(Z)

    program = edsl.Program('monte_carlo_pi', [O])
    executable = plaidml_exec.Executable(program, [X, Y])

    def run():
        outputs = executable([X_data, Y_data])
        return outputs[0].as_ndarray()

    if benchmark:
        # the first run will compile and run
        print('compiling...')
        result = run()

        # subsequent runs should not include compile time
        print('running...')
        ITERATIONS = 100
        elapsed = timeit.timeit(run, number=ITERATIONS)
        print('runtime:', elapsed / ITERATIONS)
    else:
        result = run()

    return 4 * (result / n)


def main():
    #number of points
    #n = 1000        # 3.208
    #n = 10000  # 3.1436
    #n = 100000  # 3.14352
    n = 1000000  #3.139804
    #n = 10000000  # 3.1405264
    #n = 100000000   # 3.14152228
    #n = 100000000   # 3.14152228
    #n = 1000000000  # 1.570816452 !

    result = monte_carlo_pi(n, True)
    print("computed result: {}".format(result))


if __name__ == '__main__':
    main()
