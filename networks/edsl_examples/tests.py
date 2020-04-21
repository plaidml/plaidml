import unittest
import numpy.testing as npt

from networks.edsl_examples.reorgyolo import *


def create_reorgyolo_data(n_i, c_i, h_i, w_i):
    I_data_linear = np.array(list(range(n_i * c_i * h_i * w_i))).astype(np.int)
    I_data = np.reshape(I_data_linear, (n_i, c_i, h_i, w_i))
    return I_data


def create_and_run_reorgyolo_program(I_data, stride, decrease):
    I = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_data.shape))
    O = reorgyolo(I, stride, decrease)
    program = edsl.Program('reorgyolo', [O])
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()
    binder.input(I).copy_from_ndarray(I_data)
    executable.run()
    result = binder.output(O).as_ndarray()
    return result


class EDSLExampleTests(unittest.TestCase):

    def tests_reorgyolo_backward(self):
        n_i = 2  # number of batches
        c_i = 4  # number of channels
        h_i = 6  # height of each channel
        w_i = 6  # width of each channel
        stride = 2  # stride
        decrease = False
        I_data = create_reorgyolo_data(n_i, c_i, h_i, w_i)
        result = create_and_run_reorgyolo_program(I_data, stride, decrease)
        expected_result = reorgyolo_comparison_non_linear(I_data,
                                                          batch=n_i,
                                                          C=c_i,
                                                          H=h_i,
                                                          W=w_i,
                                                          stride=stride,
                                                          forward=decrease)

        #check results
        npt.assert_array_equal(result, expected_result)

    def tests_reorgyolo_forward(self):
        n_i = 2  # number of batches
        c_i = 4  # number of channels
        h_i = 6  # height of each channel
        w_i = 6  # width of each channel
        stride = 2  # stride
        decrease = True
        I_data = create_reorgyolo_data(n_i, c_i, h_i, w_i)
        result = create_and_run_reorgyolo_program(I_data, stride, decrease)
        expected_result = reorgyolo_comparison_non_linear(I_data,
                                                          batch=n_i,
                                                          C=c_i,
                                                          H=h_i,
                                                          W=w_i,
                                                          stride=stride,
                                                          forward=decrease)

        #check results
        npt.assert_array_equal(result, expected_result)

    def test_reorgyolo_reversible(self):
        n_i = 2  # number of batches
        c_i = 4  # number of channels
        h_i = 6  # height of each channel
        w_i = 6  # width of each channel
        stride = 2  # stride
        decrease = True

        I_data = create_reorgyolo_data(n_i, c_i, h_i, w_i)
        result = create_and_run_reorgyolo_program(I_data, stride, decrease)
        result_I = create_and_run_reorgyolo_program(result, stride, not decrease)
        npt.assert_array_equal(result_I, I_data)


if __name__ == '__main__':
    unittest.main()
