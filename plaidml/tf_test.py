# Copyright 2018 Intel Corporation.
"""Tensorflow like operation tests."""

from __future__ import print_function

import argparse
import sys
import unittest

import numpy as np
import numpy.testing as npt
# Tensorflow needs some code called directly
import tensorflow
# Hack to avoid using the plaidml.keras submodule when trying to load keras
sys.path = sys.path[1:] + [sys.path[0]]
from keras.backend import tensorflow_backend as tf
from keras.backend import floatx
sys.path = [sys.path[-1]] + sys.path[:-1]

import plaidml
from plaidml.keras import backend as pkb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args, remainder = parser.parse_known_args()

    plaidml._internal_set_vlog(args.verbose)
    if args.fp16:
        pkb.set_floatx('float16')
        DEFAULT_TOL = 1e-2
        DEFAULT_ATOL = 1e-5
    else:
        pkb.set_floatx('float32')
        DEFAULT_TOL = 1e-3
        DEFAULT_ATOL = 1e-8


def opTest(in_data,
           tol=DEFAULT_TOL,
           atol=DEFAULT_ATOL,
           skip_tensorflow=False,
           verbose=False,
           input_shapes=None):
    # If using with non-tensor parameters, all tensor params must appear before
    # all non-tensor params
    def run_one_backend(self, data, test_func, b, shapes=None, *run_args):
        tf_session = tensorflow.Session()
        tf.set_session(tf_session)
        results = []
        with tf_session.as_default():
            if shapes:
                x = [b.placeholder(shape=t) for t in shapes]
            else:
                x = [b.placeholder(shape=t.shape) for t in data if hasattr(t, 'shape')]
            xv = [b.variable(t, dtype=floatx()) for t in data if hasattr(t, 'shape')]
            ps = [t for t in data if not hasattr(t, 'shape')]
            grad_funcs = test_func(self, b, *(x + ps + list(run_args)))
            funcs = test_func(self, b, *(xv + ps + list(run_args)))
            tf_session.run(tensorflow.global_variables_initializer())
            for gf, f in zip(grad_funcs, funcs):
                df = b.gradients(b.mean(gf), x)
                gfn = b.function(x, df, updates=[])
                fr = f.eval()
                gr = gfn([t for t in data if hasattr(t, 'shape')])
                try:
                    values = gr[0].values
                    dense_shape = gr[0].dense_shape
                    indices = gr[0].indices
                    result = np.zeros(dense_shape)
                    for i in indices:
                        result[i] += values[i]
                    gr[0] = result
                except AttributeError:
                    # This wasn't an IndexedSlices object, do nothing
                    pass
                if args.verbose or verbose:
                    print('data: {}'.format(data))
                    print('fr: {}'.format(fr))
                    print('gr: {}'.format(gr))
                results.append((fr, gr))
        tf_session.close()
        return results

    def apply(test_func):

        def output(self, *args):
            for didx, data in enumerate(in_data):
                shapes = None
                if input_shapes:
                    shapes = input_shapes[didx]
                if not skip_tensorflow:
                    tensorflow_results = run_one_backend(self,
                                                         data,
                                                         test_func,
                                                         tf,
                                                         *args,
                                                         shapes=shapes)
                plaidml_results = run_one_backend(self, data, test_func, pkb, *args, shapes=shapes)
                if not skip_tensorflow:
                    for idx, (pmlr, tfr) in enumerate(zip(plaidml_results, tensorflow_results)):
                        idx = idx + 1
                        npt.assert_allclose(
                            pmlr[0],
                            tfr[0],
                            rtol=tol,
                            atol=atol,
                            err_msg='ERR: datum={}, test={}, x=plaidml, y=tensorflow'.format(
                                didx, idx))
                        for x in range(0, len(pmlr[1])):
                            npt.assert_allclose(
                                pmlr[1][x],
                                tfr[1][x],
                                rtol=tol,
                                atol=atol,
                                err_msg='ERR: datum={}, test={}, grad, x=plaidml, y=tensorflow'.
                                format(didx, idx))

        return output

    return apply


class TestTF(unittest.TestCase):
    """Tensorflow like operation tests."""

    @opTest([
        [np.random.random((5, 60, 10, 3)), [[0, 0], [5, 2], [5, 3], [0, 0]]],
        [np.random.random((5, 60, 10, 3)), [[4, 0], [0, 2], [5, 3], [0, 1]]],
        [np.random.random((60, 10)), [[0, 0], [0, 0]]],
    ],
            input_shapes=[
                [[
                    None,
                ] * 4],
                [[
                    None,
                ] * 4],
                [[
                    None,
                ] * 2],
            ])
    def testReflectionPadding(self, b, *args):
        if b == tf:
            args = list(args) + ['REFLECT']
            func = tensorflow.pad
        else:
            func = plaidml.op.reflection_padding
        return [func(*args)]


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    #plaidml._internal_set_vlog(5)
    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
