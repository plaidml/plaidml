import numpy as np

import plaidml
import plaidml.exec


def _create_buffer(value):
    dtype = plaidml.DType.from_numpy(value.dtype)
    shape = plaidml.TensorShape(dtype, value.shape)
    buffer = plaidml.Buffer(shape, data=value)
    return shape, buffer


def make_conv_test_buffers():
    ins = []
    outs = []
    # TODO
    # Initialize inputs with 1s
    ins.append(_create_buffer(np.ones([1, 56, 56, 64], np.float32)))
    ins.append(_create_buffer(np.ones([1, 1, 64, 256], np.float32)))
    # Initialize outputs with 0s
    outs.append(_create_buffer(np.zeros([1, 56, 56, 256], np.float32)))

    return ins, outs


# TODO: Non-silly `src_path`
def make_program(src_path='/home/tim/plaidml/pmlc/conversion/linalg_to_pxa/test/conv.mlir'):
    code = None
    with open(src_path) as code_file:
        code = code_file.read()
    name = 'test_conv'
    program = plaidml.Program.load(code, name)
    program.compile()
    return program


def execute(program, inputs, outputs):
    executable = plaidml.exec.Executable(program)
    executable.run(inputs, outputs)
    return outputs


if __name__ == '__main__':
    program = make_program()  # TODO: Filename
    in_buffers, out_buffers = make_conv_test_buffers()
    result = execute(program, in_buffers, out_buffers)
    print("Result is:\n{}".format(result))
