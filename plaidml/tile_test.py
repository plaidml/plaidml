# Copyright 2018 Intel Corporation.
"""Tile program construction tests."""

from __future__ import print_function

import argparse
import sys
import unittest

import plaidml
import plaidml.context
import plaidml.exceptions
import plaidml.op as op
import plaidml.tile as tile
import testing.plaidml_config


class TestTile(unittest.TestCase):
    """Tile operation library tests."""

    @classmethod
    def setUpClass(cls):
        testing.plaidml_config.unittest_config()
        cls._ctx = plaidml.Context()
        cls._dev = plaidml.Device(cls._ctx, plaidml.devices(cls._ctx, limit=10)[0])

    @classmethod
    def tearDownClass(cls):
        cls._dev.close()

    def make_inited_tensor(self, dims, dtype=plaidml.DType.FLOAT32, start=1.0, step=1.0):
        """Builds an initialized tensor.

        Args:
            dtype (plaidml.DType): The type of the tensor elements.
            start (number): The value of the initial (flattened) tensor element.
            step (number): The increment to add to `start` for each subsequent element.
            dims ((int)): The sizes of each dimension of the tensor.

        Returns:
            plaidml.Tensor: The initialized tensor.
        """
        shape = plaidml.Shape(self._ctx, dtype, *dims)
        tensor = plaidml.Tensor(self._dev, shape)
        with tensor.mmap_discard(self._ctx) as view:
            for idx in range(len(view)):
                view[idx] = start + (idx * step)
            view.writeback()
        return tensor

    def make_output_tensor(self, shape):
        """Builds an uninitialized output tensor.

        Args:
            shape (plaidml.Shape): The shape of the tensor.

        Returns:
            plaidml.Tensor: The uninitialized tensor.
        """
        return plaidml.Tensor(self._dev, shape)

    def test_tuple_deriv(self):
        """Test tuples work via derivatives"""
        A = tile.Value.from_ndims(2)
        B = tile.Value.from_ndims(2)
        out_dims = (A.shape.dims[0], B.shape.dims[1])
        out_shape = tile.Shape(tile.common_dtype(A.shape.dtype, B.shape.dtype), out_dims)
        out = tile.Operation(
            """
            function (A[I, K], B[K, J]) -> (O) {
                T = tuple(A, B);
                C = element(T, 0);
                D = element(T, 1);
                O[i, j : I, J] = +(C[i, k] * D[k, j]);
            }
            """, [('A', A), ('B', B)], [('O', out_shape)]).outputs['O']
        tot = op.summation(out, [0, 1])
        dA = op.gradients(tot, [A])[0]
        func = tile.compose(
            self._ctx, self._dev, inputs=[('A', A), ('B', B)], outputs=[('DA', dA)])
        invoker = plaidml.Invoker(self._ctx, func)
        invoker.set_input('A', self.make_inited_tensor((3, 3)))
        invoker.set_input('B', self.make_inited_tensor((3, 3)))
        output = self.make_output_tensor(invoker.get_output_shape('DA'))
        invoker.set_output('DA', output)
        invoker.invoke()

    def test_matmul_relu(self):
        """Tests that matrix multiply can be combined with a simple relu."""
        lhs = tile.Value.from_ndims(2)
        rhs = tile.Value.from_dimensions((3, None))
        out = op.relu(op.matmul(lhs, rhs))
        func = tile.compose(
            self._ctx, self._dev, inputs=[('lhs', lhs), ('rhs', rhs)], outputs=[('out', out)])

        invoker = plaidml.Invoker(self._ctx, func)
        invoker.set_input('lhs', self.make_inited_tensor((3, 3)))
        invoker.set_input('rhs', self.make_inited_tensor((3, 3)))
        output = self.make_output_tensor(invoker.get_output_shape('out'))
        invoker.set_output('out', output)
        invoker.invoke()

        with output.mmap_current() as view:
            self.assertEqual(view[0], 1.0 + 8.0 + 21.0)
            self.assertEqual(view[1], 2.0 + 10.0 + 24.0)
            self.assertEqual(view[2], 3.0 + 12.0 + 27.0)
            self.assertEqual(view[(1, 0)], 4.0 + 20.0 + 42.0)
            self.assertEqual(view[(1, 1)], 8.0 + 25.0 + 48.0)
            self.assertEqual(view[(1, 2)], 12.0 + 30.0 + 54.0)
            self.assertEqual(view[6], 7.0 + 32.0 + 63.0)
            self.assertEqual(view[7], 14.0 + 40.0 + 72.0)
            self.assertEqual(view[8], 21.0 + 48.0 + 81.0)

    def test_equals_argmax(self):
        """Validates that the =(argmax, argmax) composition works."""
        lhs = tile.Value.from_ndims(2)
        rhs = tile.Value.from_ndims(2)
        out = op.equal(op.argmax(lhs), op.argmax(rhs))
        func = tile.compose(
            self._ctx, self._dev, inputs=[('lhs', lhs), ('rhs', rhs)], outputs=[('out', out)])

        invoker = plaidml.Invoker(self._ctx, func)
        invoker.set_input('lhs', self.make_inited_tensor((3, 4)))
        invoker.set_input('rhs', self.make_inited_tensor((3, 4)))
        output = self.make_output_tensor(invoker.get_output_shape('out'))
        invoker.set_output('out', output)
        invoker.invoke()

        with output.mmap_current() as view:
            for dim in range(0, 2):
                self.assertEqual(view[(dim, 0)], True)


def main():
    """The test runner entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    flags, remainder = parser.parse_known_args()
    plaidml._internal_set_vlog(flags.verbose)
    unittest.main(argv=sys.argv[:1] + remainder, verbosity=flags.verbose + 1)


if __name__ == '__main__':
    main()
