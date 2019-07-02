# Copyright 2019, Intel Corporation.

import argparse
import os
import platform
import sys
import unittest

import plaidml2.bridge.pytorch as plaidml_pytorch
import skimage
import torch
import torch.nn.functional as F
from plaidml2.bridge.pytorch.test_utils import TestBase


class TestOps(TestBase):

    def test_basic(self):

        shape = (512, 512, 3)
        x = torch.randn(512, 512, 3)
        y = torch.randn(512, 512, 3)
        z = torch.randn(512, 512, 3)

        @torch.jit.script
        def foo(a, b, c):
            return a * b + c

        jit_out, pml_out = self._run_both(foo, [x, y, z])
        torch.testing.assert_allclose(jit_out, pml_out, rtol=0.01, atol=0.01)

    def test_mul(self):
        x = torch.randn(512)
        y = torch.randn(512)
        z = torch.randn(512)

        def mul(a, b, c):
            return a * b * c

        jit_out, pml_out = self._run_both(mul, [x, y, z])
        assert torch.allclose(jit_out, pml_out)

    def test_conv_simple(self):
        shape = (1, 3, 224, 224)
        kernel_size = 7
        num_kernels = 64
        # NCHW
        X = torch.rand(shape)
        W = torch.rand((num_kernels, shape[1], kernel_size, kernel_size))
        bias = torch.rand(num_kernels)

        def conv(a, b):
            return F.conv2d(a + a, b)

        ref_out, pml_out = self._run_both(conv, [X, W])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

        def conv_bias(a, b, c):
            return F.conv2d(a + a, b, c)

        ref_out, pml_out = self._run_both(conv_bias, [X, W, bias])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_conv1(self):
        # convolution
        #   I: fp32(1, 3, 224, 224):(150528, 50176, 224, 1):588 KiB
        #   K: ParamExpr{fp32(64, 3, 7, 7):(147, 49, 7, 1):36.75 KiB}
        #   bias: None
        #   strides: (2, 2)
        #   padding: (3, 3)
        #   dilation: (1, 1)
        #   is_transposed: 0
        #   output_padding: (0, 0)
        #   groups: 1
        shape = (1, 3, 224, 224)
        kernel_size = 7
        num_kernels = 64
        # NCHW
        X = torch.rand(shape)
        W = torch.rand(num_kernels, shape[1], kernel_size, kernel_size)

        def conv(a, b):
            return F.conv2d(a, b, stride=(2, 2), padding=(3, 3))

        ref_out, pml_out = self._run_both(conv, [X, W])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_conv2(self):
        # convolution
        #   I: fp32(1, 64, 56, 56):(200704, 3136, 56, 1):784 KiB
        #   K: ParamExpr{fp32(128, 64, 3, 3):(576, 9, 3, 1):288 KiB}
        #   bias: None
        #   strides: (2, 2)
        #   padding: (1, 1)
        #   dilation: (1, 1)
        #   is_transposed: 0
        #   output_padding: (0, 0)
        #   groups: 1
        shape = (1, 64, 56, 56)
        kernel_size = 3
        num_kernels = 128
        # NCHW
        X = torch.rand(shape)
        W = torch.rand(num_kernels, shape[1], kernel_size, kernel_size)

        def conv(a, b):
            return F.conv2d(a, b, stride=(2, 2), padding=(1, 1))

        ref_out, pml_out = self._run_both(conv, [X, W])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_conv3(self):
        # convolution
        #   I: fp32(1, 128, 28, 28):(100352, 784, 28, 1):392 KiB
        #   K: ParamExpr{fp32(128, 128, 3, 3):(1152, 9, 3, 1):576 KiB}
        #   bias: None
        #   strides: (1, 1)
        #   padding: (1, 1)
        #   dilation: (1, 1)
        #   is_transposed: 0
        #   output_padding: (0, 0)
        #   groups: 1
        shape = (1, 128, 28, 28)
        kernel_size = 3
        num_kernels = 128
        # NCHW
        X = torch.rand(shape)
        W = torch.rand(num_kernels, shape[1], kernel_size, kernel_size)

        def conv(a, b):
            return F.conv2d(a, b, stride=(1, 1), padding=(1, 1))

        ref_out, pml_out = self._run_both(conv, [X, W])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_conv4(self):
        # convolution
        #   I: fp32(1, 64, 56, 56):(200704, 3136, 56, 1):784 KiB
        #   K: ParamExpr{fp32(128, 64, 1, 1):(64, 1, 1, 1):32 KiB}
        #   bias: None
        #   strides: (2, 2)
        #   padding: (0, 0)
        #   dilation: (1, 1)
        #   is_transposed: 0
        #   output_padding: (0, 0)
        #   groups: 1
        shape = (1, 64, 56, 56)
        kernel_size = 1
        num_kernels = 128
        # NCHW
        X = torch.rand(shape)
        W = torch.rand(num_kernels, shape[1], kernel_size, kernel_size)

        def conv(a, b):
            return F.conv2d(a, b, stride=(2, 2))

        ref_out, pml_out = self._run_both(conv, [X, W])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_batch_norm(self):
        # batch_norm
        #   I: fp32(1, 64, 112, 112):(802816, 12544, 112, 1):3136 KiB
        #   Weight: ParamExpr{fp32(64):(1):256 bytes}
        #   Bias: ParamExpr{fp32(64):(1):256 bytes}
        #   Mean: ParamExpr{fp32(64):(1):256 bytes}
        #   Var: ParamExpr{fp32(64):(1):256 bytes}
        #   is_training: 0
        #   momentum: 0.1
        #   epsilon: 1e-05
        # shape = (1, 64, 112, 112)
        shape = (1, 128, 28, 28)
        a = torch.rand(shape)
        b = torch.rand(shape[1])
        c = torch.rand(shape[1])
        d = torch.rand(shape[1])
        e = torch.rand(shape[1])

        def batch_norm(a, b, c, d, e):
            return F.batch_norm(a, b, c, weight=d, bias=e)

        ref_out, pml_out = self._run_both(batch_norm, [a, b, c, d, e])
        assert torch.allclose(ref_out, pml_out, rtol=0.05, atol=0.01)

    def test_relu(self):
        # relu
        #   I: fp32(1, 64, 112, 112):(802816, 12544, 112, 1):3136 KiB
        X = torch.rand((1, 64, 112, 112))

        def relu(a):
            return F.relu(F.relu(a))
            # return F.relu(a)

        ref_out, pml_out = self._run_both(relu, [X])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_avg_pool2d(self):
        # avg_pool2d
        #   I: fp32(1, 512, 7, 7):(25088, 49, 7, 1):98 KiB
        #   kernel_size: (7, 7)
        #   strides: (7, 7)
        #   padding: (0, 0)
        X = torch.rand((1, 512, 7, 7))

        def avg_pool2d(a):
            return F.avg_pool2d(a, 2)

        ref_out, pml_out = self._run_both(avg_pool2d, [X])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

        def avg_pool2d_strides(a):
            return F.avg_pool2d(a, 7, stride=(7, 7))

        ref_out, pml_out = self._run_both(avg_pool2d_strides, [X])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_adaptive_avg_pool2d(self):
        # adaptive_avg_pool2d
        #   I: fp32(1, 512, 7, 7):(25088, 49, 7, 1):98 KiB
        #   output_size: (1, 1)
        # avg_pool2d
        #   I: fp32(1, 512, 7, 7):(25088, 49, 7, 1):98 KiB
        #   kernel_size: (7, 7)
        #   strides: (7, 7)
        #   padding: (0, 0)
        X = torch.rand((1, 512, 7, 7))

        def adaptive_avg_pool2d(a):
            return F.adaptive_avg_pool2d(a, 1)

        ref_out, pml_out = self._run_both(adaptive_avg_pool2d, [X])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_max_pool2d(self):
        # max_pool2d
        #   I: fp32(1, 64, 112, 112):(802816, 12544, 112, 1):3136 KiB
        #   kernel_size: (3, 3)
        #   strides: (2, 2)
        #   padding: (1, 1)
        #   dilation: (1, 1)
        #   ceil_mode: 0
        X = torch.rand((1, 64, 112, 112))

        def max_pool2d(a):
            return F.max_pool2d(a, 3) + 2.0

        ref_out, pml_out = self._run_both(max_pool2d, [X])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

        def max_pool2d_strides_padding(a):
            return F.max_pool2d(a, 3, stride=(2, 2), padding=1)

        ref_out, pml_out = self._run_both(max_pool2d_strides_padding, [X])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_linear(self):
        # linear
        #   input: fp32(1, 512):(512, 1):2 KiB
        #   weight: ParamExpr{fp32(1000, 512):(512, 1):2000 KiB}
        #   bias: ParamExpr{fp32(1000):(1):3.90625 KiB}
        X = torch.rand(1, 512)
        W = torch.rand(1000, 512)
        B = torch.rand(1000)

        def linear(a, b, c):
            return F.linear(a + a, b, bias=c)

        ref_out, pml_out = self._run_both(linear, [X, W, B])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_reshape(self):
        shape = (64, 64)
        input = torch.rand(shape)

        # def reshape(input):
        #     return torch.reshape(input, (-1,))

        # ref_out, pml_out = self._run_both(reshape, [input])
        # assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

        def reshape(input):
            return torch.reshape(input, (1, 1, *shape))

        ref_out, pml_out = self._run_both(reshape, [input])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

        def reshape(input):
            return torch.reshape(input, (1, -1))

        ref_out, pml_out = self._run_both(reshape, [input])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

        def reshape(input):
            return torch.reshape(input, (shape[0], 1, 1, shape[1]))

        ref_out, pml_out = self._run_both(reshape, [input])
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    def test_resnet_layers(self):
        # convolution
        #   I: fp32(1, 3, 224, 224):(150528, 50176, 224, 1):588 KiB
        #   K: ParamExpr{fp32(64, 3, 7, 7):(147, 49, 7, 1):36.75 KiB}
        #   bias: None
        #   strides: (2, 2)
        #   padding: (3, 3)
        #   dilation: (1, 1)
        #   is_transposed: 0
        #   output_padding: (0, 0)
        #   groups: 1
        # batch_norm
        #   I: fp32(1, 64, 112, 112):(802816, 12544, 112, 1):3136 KiB
        #   Weight: ParamExpr{fp32(64):(1):256 bytes}
        #   Bias: ParamExpr{fp32(64):(1):256 bytes}
        #   Mean: ParamExpr{fp32(64):(1):256 bytes}
        #   Var: ParamExpr{fp32(64):(1):256 bytes}
        #   is_training: 0
        #   momentum: 0.1
        #   epsilon: 1e-05
        # relu
        #   I: fp32(1, 64, 112, 112):(802816, 12544, 112, 1):3136 KiB
        # max_pool2d
        #   I: fp32(1, 64, 112, 112):(802816, 12544, 112, 1):3136 KiB
        #   kernel_size: (3, 3)
        #   strides: (2, 2)
        #   padding: (1, 1)
        #   dilation: (1, 1)
        #   ceil_mode: 0
        args = [
            torch.rand(1, 3, 224, 224),  # input
            torch.rand(64, 3, 7, 7),  # weight_1
            torch.rand(64),  # bn_weight_1
            torch.rand(64),  # bn_bias_1
            torch.rand(64),  # bn_mean_1
            torch.rand(64),  # bn_var_1
        ]

        def model(input, weights_1, bn_weight_1, bn_bias_1, bn_mean_1, bn_var_1):
            conv_1 = F.conv2d(input, weights_1, stride=(2, 2), padding=(3, 3))
            bn_1 = F.batch_norm(conv_1, bn_mean_1, bn_var_1, weight=bn_weight_1, bias=bn_bias_1)
            relu_1 = F.relu(bn_1)
            pool_1 = F.max_pool2d(relu_1, 3, stride=(2, 2), padding=1)
            return pool_1

        ref_out, pml_out = self._run_both(model, args)
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)

    @unittest.skipIf(platform.system() == 'Darwin', "Fails on metal")
    def test_downsample(self):
        args = [
            torch.rand(1, 128, 28, 28),  # I
            torch.rand(128, 128, 3, 3),  # W2
            torch.rand(128),  # bn_W2
            torch.rand(128),  # bn_B2
            torch.rand(128),  # bn_M2
            torch.rand(128),  # bn_V2
        ]

        def model(x1, x7, x8, x9, x10, x11):
            x = F.conv2d(x1, x7, stride=(1, 1), padding=(1, 1))
            x = F.batch_norm(x, x8, x9, weight=x10, bias=x11)
            return x

        ref_out, pml_out = self._run_both(model, args)
        print('ref_out:', ref_out)
        print('pml_out:', pml_out)
        assert torch.allclose(ref_out, pml_out, rtol=0.01, atol=0.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', type=int, default=0)
    args, remainder = parser.parse_known_args()

    # print('compiled_with_cxx11_abi: {}'.format(torch.compiled_with_cxx11_abi()))

    plaidml_pytorch.set_vlog(args.verbose)

    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
