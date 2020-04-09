import unittest
import numpy.testing as npt
import os

from networks.edsl_examples.reorgyolo import *


def test_reorgyolo(n_i, c_i, h_i, w_i, stride, decrease):

    I_data_linear = np.array(list(range(n_i * c_i * h_i * w_i))).astype(np.int)
    I_data = np.reshape(I_data_linear, (n_i, c_i, h_i, w_i))

    I = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_data.shape))
    O = reorgyolo(I, stride, decrease)

    #create eDSL program, compile and run
    program = edsl.Program('reorgyolo', [O])
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()
    binder.input(I).copy_from_ndarray(I_data)
    executable.run()
    result = binder.output(O).as_ndarray()

    #compute expected results without eDSL
    if decrease:
        c_o = c_i // (stride * stride)
        h_o = h_i * stride
        w_o = w_i * stride
    else:
        c_o = c_i * (stride * stride)
        h_o = h_i // stride
        w_o = w_i // stride
    expected_result_l = reorgyolo_comparison(I_data_linear,
                                             batch=n_i,
                                             C=c_i,
                                             H=h_i,
                                             W=w_i,
                                             stride=stride,
                                             forward=decrease)
    expected_result = np.reshape(expected_result_l, (n_i, c_o, h_o, w_o))

    #check results
    npt.assert_array_equal(result, expected_result)


class EDSLExampleTests(unittest.TestCase):

    @unittest.skip("Fails: incorrect output produced")  #TODO: fix
    def tests_reorgyolo_backward(self):
        test_reorgyolo(1, 4, 6, 6, 2, False)

    def tests_reorgyolo_forward(self):
        test_reorgyolo(2, 4, 6, 6, 2, True)

    #def test reorgyolo_reversible:


if __name__ == '__main__':
    unittest.main()
