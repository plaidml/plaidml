# Copyright 2021 Intel Corporation.
# Note:
#    This is a temp solution for testing the python code in examples

import unittest

import plaidml
import plaidml.exec
from plaidml.edsl import *
from plaidml import Program

DEFAULT_DEVICE = 'llvm_cpu.0'
DEFAULT_TARGET = 'llvm_cpu'

from plaidml.edsl.examples.complex_conv_2d import *
from plaidml.edsl.examples.conv_1d import *
from plaidml.edsl.examples.conv_2d_dilated import *
from plaidml.edsl.examples.gemm import *
from plaidml.edsl.examples.gemv import *
from plaidml.edsl.examples.max_pool_1d import *
from plaidml.edsl.examples.quantize import *


class TestEdslExamples(unittest.TestCase):

    def runProgram(self, program):
        program.compile()
        input_buffers = [plaidml.Buffer(shape) for shape in program.inputs]
        output_buffers = [plaidml.Buffer(shape) for shape in program.outputs]
        executable = plaidml.exec.Executable(program)
        executable.run(input_buffers, output_buffers)

    def checkProgram(self, program, inputs, expected):
        outputs = plaidml.exec.run(program, inputs)
        for out, exp in zip(outputs, expected):
            self.assertEqual(out.tolist(), exp)

    def test_complex_conv_2d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 224, 3, 3])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 3, 3, 32])
        O = complex_conv_2d(I, K, 1, 2, 1, 2)
        program = Program('complex_conv_2d', [I, K], [O])
        self.runProgram(program)

    def test_conv_1d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 3])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 1])
        O = conv_1d(I, K)
        program = Program('conv_1d', [I, K], [O])
        self.runProgram(program)

    def test_conv_2d_dilated(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 224, 1])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 1, 32])
        O = conv_2d_dilated(I, K)
        program = Program('conv_2d_dilated', [I, K], [O])
        self.runProgram(program)

    def test_gemm(self):
        A = Placeholder(plaidml.DType.FLOAT32, [7, 7])
        B = Placeholder(plaidml.DType.FLOAT32, [7, 7])
        C = Placeholder(plaidml.DType.FLOAT32, [7, 7])
        O = gemm(A, B, C)
        program = Program('gemm', [A, B, C], [O])
        self.runProgram(program)

    def test_gemv(self):
        A = Placeholder(plaidml.DType.FLOAT32, [7, 7])
        B = Placeholder(plaidml.DType.FLOAT32, [7])
        C = Placeholder(plaidml.DType.FLOAT32, [7])
        O = gemv(A, B, C)
        program = Program('gemv', [A, B, C], [O])
        self.runProgram(program)

    def test_gemv2(self):
        A = Placeholder(plaidml.DType.FLOAT32, [1, 7])
        B = Placeholder(plaidml.DType.FLOAT32, [7])
        C = Placeholder(plaidml.DType.FLOAT32, [7])
        O = gemv2(A, B, C, 4, 5)
        program = Program('gemv', [A, B, C], [O])
        self.runProgram(program)

    def test_max_pool_1d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [10])
        O = max_pool_1d(I)
        program = Program('max_pool_1d', [I], [O])
        self.runProgram(program)

    def test_wrong_max_pool_1d_odd(self):
        I = Placeholder(plaidml.DType.FLOAT32, [10])
        O = wrong_max_pool_1d(I)
        program = Program('max_pool_1d_odd', [I], [O])

    def test_quantize(self):
        A = Placeholder(plaidml.DType.FLOAT32, [1, 7])
        O = quantize_float32_to_int8(A, 256, 0)
        program = Program('quantize', [A], [O])
        self.runProgram(program)


if __name__ == '__main__':
    plaidml.settings.set('PLAIDML_DEVICE', DEFAULT_DEVICE)
    plaidml.settings.set('PLAIDML_TARGET', DEFAULT_TARGET)
    unittest.main()
