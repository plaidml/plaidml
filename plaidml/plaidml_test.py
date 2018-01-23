from __future__ import print_function

import argparse
import cProfile
import os
import sys
import unittest

import numpy as np
import plaidml
import plaidml.context
import plaidml.exceptions
import testing.plaidml_config


class TestPlaidML(unittest.TestCase):

    def setUp(self):
        testing.plaidml_config.unittest_config()

    def testVersion(self):
        # From https://www.python.org/dev/peps/pep-0440/
        self.assertRegexpMatches(
            plaidml.__version__,
            r'^([1-9]\d*!)?(0|[1-9]\d*)(\.(0|[1-9]\d*))*((a|b|rc)(0|[1-9]\d*))?(\.post(0|[1-9]\d*))?(\.dev(0|[1-9]\d*))?$'
        )

    def testDeviceEnumerator(self):
        ctx = plaidml.Context()
        for conf in plaidml.devices(ctx, limit=100):
            pass

    def testDeviceEnumeratorWithNoDevices(self):
        ctx = plaidml.Context()
        with self.assertRaises(plaidml.exceptions.PlaidMLError):
            plaidml.settings.config = """{
                  "platform": {
                    "@type": "type.vertex.ai/vertexai.tile.local_machine.proto.Platform",
                    "hals": [
                      {
                        "@type": "type.vertex.ai/vertexai.tile.hal.opencl.proto.Driver",
                      }
                    ]
                  }
                }"""
            for conf in plaidml.devices(ctx):
                pass

    def testDeviceEnumeratorInvalidConfig(self):
        ctx = plaidml.Context()
        with self.assertRaises(plaidml.exceptions.InvalidArgument):
            plaidml.settings.config = 'An invalid configuration'
            for conf in plaidml.devices(ctx):
                pass

    def testBufferRanges(self):
        ctx = plaidml.Context()
        with plaidml.open_first_device(ctx) as dev:
            buf = plaidml.Tensor(dev, plaidml.Shape(ctx, plaidml.DType.FLOAT32, 10))
            with buf.mmap_current() as view:
                self.assertEqual(len(view), 10)
                view[0] = 1
                with self.assertRaises(IndexError):
                    view[10] = 0
                view[9] = 2
                view[-1] = 4
                self.assertEqual(view[9], 4)
                view[0:10:3] = (1, 2, 3, 4)
                self.assertEqual(view[3], 2)
                self.assertSequenceEqual(view[0:10:3], (1, 2, 3, 4))

    def testManualReshape(self):
        ctx = plaidml.Context()
        reshape = plaidml.Function(
            "function (I) -> (O) { F[3*j + k: 4 * 3] = >(I[j,k]); O[p,q : 6,2] = >(F[2*p + q]);}")
        iShape = plaidml.Shape(ctx, plaidml.DType.FLOAT32, 4, 3)
        oShape = plaidml.Shape(ctx, plaidml.DType.FLOAT32, 6, 2)
        with plaidml.open_first_device(ctx) as dev:
            I = plaidml.Tensor(dev, iShape)
            with I.mmap_discard(ctx) as view:
                view[0] = 1.0
                view[1] = 2.0
                view[2] = 3.0
                view[3] = 4.0
                view[4] = 5.0
                view[5] = 6.0
                view[6] = 7.0
                view[7] = 8.0
                view[8] = 9.0
                view[9] = 10.0
                view[10] = 11.0
                view[11] = 12.0
                view.writeback()

            O = plaidml.Tensor(dev, oShape)
            plaidml.run(ctx, reshape, inputs={"I": I}, outputs={"O": O})
            with O.mmap_current() as view:
                self.assertEqual(view[0], 1.0)
                self.assertEqual(view[1], 2.0)
                self.assertEqual(view[2], 3.0)
                self.assertEqual(view[3], 4.0)
                self.assertEqual(view[4], 5.0)
                self.assertEqual(view[5], 6.0)
                self.assertEqual(view[6], 7.0)
                self.assertEqual(view[7], 8.0)
                self.assertEqual(view[8], 9.0)
                self.assertEqual(view[9], 10.0)
                self.assertEqual(view[10], 11.0)
                self.assertEqual(view[11], 12.0)

    def runMatrixMultiply(self, ctx, dev):
        matmul = plaidml.Function(
            "function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }")
        shape = plaidml.Shape(ctx, plaidml.DType.FLOAT32, 3, 3)
        b = plaidml.Tensor(dev, shape)
        with b.mmap_discard(ctx) as view:
            view[0] = 1.0
            view[1] = 2.0
            view[2] = 3.0
            view[3] = 4.0
            view[4] = 5.0
            view[5] = 6.0
            view[6] = 7.0
            view[7] = 8.0
            view[8] = 9.0
            view.writeback()

        c = plaidml.Tensor(dev, shape)
        with c.mmap_discard(ctx) as view:
            view[(0, 0)] = 1.0
            view[(0, 1)] = 2.0
            view[(0, 2)] = 3.0
            view[(1, 0)] = 4.0
            view[(1, 1)] = 5.0
            view[(1, 2)] = 6.0
            view[(2, 0)] = 7.0
            view[(2, 1)] = 8.0
            view[(2, 2)] = 9.0
            view.writeback()

        a = plaidml.Tensor(dev, shape)

        plaidml.run(ctx, matmul, inputs={"B": b, "C": c}, outputs={"A": a})

        with a.mmap_current() as view:
            self.assertEqual(view[0], 1.0 + 8.0 + 21.0)
            self.assertEqual(view[1], 2.0 + 10.0 + 24.0)
            self.assertEqual(view[2], 3.0 + 12.0 + 27.0)
            self.assertEqual(view[(1, 0)], 4.0 + 20.0 + 42.0)
            self.assertEqual(view[(1, 1)], 8.0 + 25.0 + 48.0)
            self.assertEqual(view[(1, 2)], 12.0 + 30.0 + 54.0)
            self.assertEqual(view[6], 7.0 + 32.0 + 63.0)
            self.assertEqual(view[7], 14.0 + 40.0 + 72.0)
            self.assertEqual(view[8], 21.0 + 48.0 + 81.0)

    def testMatrixMultiply(self):
        ctx = plaidml.Context()
        with plaidml.open_first_device(ctx) as dev:
            self.runMatrixMultiply(ctx, dev)

    @unittest.skip("T1193: Skip until there is a fake HAL or use single device is implmemented.")
    def testLargeConfigValuesNoCrash(self):
        ctx = plaidml.Context()
        plaidml.settings.config = testing.plaidml_config.very_large_values_config()
        with plaidml.open_first_device(ctx) as dev:
            self.runMatrixMultiply(ctx, dev)

    def testTransferLargeNDArray(self):
        size = 3000000
        shape = (size,)
        dtype = plaidml.DType.FLOAT32

        ctx = plaidml.Context()
        with plaidml.open_first_device(ctx) as dev:
            expected = np.random.uniform(low=0, high=100, size=size)
            tensor = plaidml.Tensor(dev, plaidml.Shape(ctx, dtype, *shape))
            actual = np.ndarray(shape, dtype='f4')

            pr = cProfile.Profile()
            pr.enable()

            with tensor.mmap_discard(ctx) as view:
                view.copy_from_ndarray(expected)
                view.writeback()

            with tensor.mmap_current() as view:
                view.copy_to_ndarray(actual)

            pr.disable()
            pr.print_stats()

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args, remainder = parser.parse_known_args()
    plaidml._internal_set_vlog(args.verbose)
    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
