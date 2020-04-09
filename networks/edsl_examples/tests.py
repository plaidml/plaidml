import unittest
import numpy.testing as npt
import os

from networks.edsl_examples.reorgyolo import *


class EDSLExampleTests(unittest.TestCase):

    def test_reorgyolo_forward_channel_decrease(self):
        n_i = 1
        c_i = 4
        h_i = 6
        w_i = 6
        stride = 2
        forward = True
        I_data_linear = np.array(list(range(n_i * c_i * h_i * w_i))).astype(np.int)
        I_data = np.reshape(I_data_linear, (n_i, c_i, h_i, w_i))

        I = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_data.shape))
        O = reorgyolo(I, stride, forward)

        #create eDSL program, compile and run
        program = edsl.Program('reorgyolo', [O])
        binder = plaidml_exec.Binder(program)
        executable = binder.compile()
        binder.input(I).copy_from_ndarray(I_data)
        executable.run()
        result = binder.output(O).as_ndarray()

        #compute expected results without eDSL
        if forward:
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
                                                 forward=forward)
        expected_result = np.reshape(expected_result_l, (n_i, c_o, h_o, w_o))

        #check results
        npt.assert_array_equal(result, expected_result)

    #def test_reorgyolo_backward_channel_increase:
    #def test reorgyolo_reversible:


if __name__ == '__main__':
    unittest.main()
